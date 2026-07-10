"""
SQLite persistence layer.

Replaces the previous "read a CSV, mutate a DataFrame in memory, throw it away
per request" pattern with a real, transactional, indexed store.

Design choices for reliability under Render's single-instance model:
  * WAL journal mode  -> concurrent reads while a write is in progress.
  * busy_timeout      -> transient lock contention retries instead of erroring.
  * A short-lived connection per operation (thread-safe), foreign keys on.
  * All queries parameterised (no string interpolation -> no SQL injection).
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Iterable

from . import settings
from .observability import get_logger

log = get_logger("zintoo.db")
_init_lock = threading.Lock()
_initialised = False


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email          TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    role           TEXT NOT NULL,
    password_hash  TEXT NOT NULL,
    salt           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id      INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    master_category TEXT NOT NULL,
    sub_category    TEXT NOT NULL,
    article_type    TEXT NOT NULL,
    color           TEXT NOT NULL,
    gender          TEXT NOT NULL,
    season          TEXT NOT NULL,
    usage           TEXT NOT NULL,
    description     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_products_gender ON products(gender);
CREATE INDEX IF NOT EXISTS idx_products_master ON products(master_category);

CREATE TABLE IF NOT EXISTS inventory (
    warehouse_id      TEXT NOT NULL,
    sku               TEXT NOT NULL,
    pincode           TEXT NOT NULL,
    current_stock     INTEGER NOT NULL,
    reorder_threshold INTEGER NOT NULL,
    max_capacity      INTEGER NOT NULL,
    PRIMARY KEY (warehouse_id, sku)
);
CREATE INDEX IF NOT EXISTS idx_inventory_sku ON inventory(sku);
CREATE INDEX IF NOT EXISTS idx_inventory_wh ON inventory(warehouse_id);

CREATE TABLE IF NOT EXISTS demand_history (
    sku       TEXT NOT NULL,
    pincode   TEXT NOT NULL,
    ts        TEXT NOT NULL,           -- ISO timestamp (hourly)
    demand    REAL NOT NULL,
    temp_c    REAL,
    is_weekend INTEGER NOT NULL DEFAULT 0,
    hour      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_demand_sku_pin ON demand_history(sku, pincode);

CREATE TABLE IF NOT EXISTS transfers (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    sku            TEXT NOT NULL,
    from_warehouse TEXT NOT NULL,
    to_warehouse   TEXT NOT NULL,
    quantity       INTEGER NOT NULL,
    priority       TEXT NOT NULL,
    reason         TEXT NOT NULL,
    success        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transfers_ts ON transfers(ts);

CREATE TABLE IF NOT EXISTS orchestration_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    status      TEXT NOT NULL,
    transfers   INTEGER NOT NULL,
    units_moved INTEGER NOT NULL,
    cost        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@contextmanager
def connect():
    """Yield a configured sqlite3 connection with row access by name."""
    conn = sqlite3.connect(settings.DB_PATH, timeout=10.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema once. Idempotent + thread-safe."""
    global _initialised
    with _init_lock:
        if _initialised:
            return
        with connect() as conn:
            conn.executescript(SCHEMA)
        _initialised = True
        log.info("database initialised at %s", settings.DB_PATH)


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    with connect() as conn:
        cur = conn.execute(sql, tuple(params))
        return cur.rowcount


def executemany(sql: str, seq: Iterable[Iterable[Any]]) -> None:
    with connect() as conn:
        conn.executemany(sql, [tuple(p) for p in seq])


@contextmanager
def transaction():
    """Explicit transaction for multi-statement atomic writes."""
    with connect() as conn:
        try:
            conn.execute("BEGIN;")
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise


def table_count(table: str) -> int:
    # table name is from a fixed internal set, never user input
    row = query_one(f"SELECT COUNT(*) AS c FROM {table}")
    return int(row["c"]) if row else 0


def get_meta(key: str, default: str | None = None) -> str | None:
    row = query_one("SELECT value FROM meta WHERE key = ?", (key,))
    return row["value"] if row else default


def set_meta(key: str, value: str) -> None:
    execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
