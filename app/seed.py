"""
Deterministic seeding.

The production deploy shipped no data artifacts, so every intelligence endpoint
failed. This seeds a self-contained, realistic dataset into SQLite on first
boot (idempotent — skips if already populated): demo users, a fashion catalog,
warehouse inventory, and 90 days of hourly demand history with weather + weekly
seasonality baked in so the forecaster has a real signal to learn.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from . import auth, db, settings
from .observability import get_logger

log = get_logger("zintoo.seed")
SEED_VERSION = "1"

# Deterministic vocabulary for a believable catalog.
_CATS = {
    "Apparel": {
        "Topwear": ["Tshirts", "Shirts", "Kurtas", "Sweaters", "Jackets"],
        "Bottomwear": ["Jeans", "Trousers", "Shorts", "Track Pants"],
    },
    "Footwear": {
        "Shoes": ["Casual Shoes", "Sports Shoes", "Formal Shoes"],
        "Sandals": ["Sandals", "Flip Flops"],
    },
    "Accessories": {
        "Bags": ["Backpacks", "Handbags"],
        "Watches": ["Watches"],
    },
}
_COLORS = ["Black", "White", "Blue", "Red", "Green", "Grey", "Navy", "Maroon", "Beige", "Pink"]
_GENDERS = ["Men", "Women", "Unisex"]
_SEASONS = ["Summer", "Winter", "Monsoon", "Fall"]
_USAGES = ["Casual", "Formal", "Sports", "Ethnic", "Party"]
_BRANDS = ["Zintoo", "Urbano", "Metroline", "Kora", "Nomad", "Peak", "Loom", "Verve"]


def _seed_users() -> None:
    demo = [
        ("admin@zintoo.ai", "admin123", "System Admin", "Owner"),
        ("demo@zintoo.ai", "demo123", "Demo User", "Viewer"),
    ]
    for email, pwd, name, role in demo:
        pw_hash, salt = auth.hash_password(pwd)
        db.execute(
            "INSERT INTO users(email, name, role, password_hash, salt) VALUES(?,?,?,?,?) "
            "ON CONFLICT(email) DO NOTHING",
            (email, name, role, pw_hash, salt),
        )


def _seed_catalog(rng: random.Random, n: int = 400) -> list[str]:
    products = []
    skus = []
    for pid in range(1, n + 1):
        master = rng.choice(list(_CATS))
        sub = rng.choice(list(_CATS[master]))
        article = rng.choice(_CATS[master][sub])
        color = rng.choice(_COLORS)
        gender = rng.choice(_GENDERS)
        season = rng.choice(_SEASONS)
        usage = rng.choice(_USAGES)
        brand = rng.choice(_BRANDS)
        name = f"{brand} {color} {article}"
        desc = f"{color} {article.lower()} for {gender.lower()} — {usage.lower()} wear, ideal for {season.lower()}. {sub} by {brand}."
        products.append((pid, name, master, sub, article, color, gender, season, usage, desc))
        # A compact set of SKUs used for inventory + forecasting.
        sku = f"{article.upper().replace(' ', '')[:6]}_{pid % 40:03d}"
        skus.append(sku)
    db.executemany(
        "INSERT OR REPLACE INTO products(product_id, name, master_category, sub_category, "
        "article_type, color, gender, season, usage, description) VALUES(?,?,?,?,?,?,?,?,?,?)",
        products,
    )
    # Distinct SKUs for the operational side.
    unique_skus = sorted(set(skus))[:40]
    return unique_skus


def _seed_inventory(rng: random.Random, skus: list[str]) -> None:
    rows = []
    for sku in skus:
        for wh in settings.WAREHOUSE_IDS:
            pincode = settings.WAREHOUSE_PINCODE_MAP[wh]
            # Intentionally uneven so orchestration has real work to do.
            stock = rng.choice([2, 5, 8, 12, 20, 35, 55, 80])
            rows.append((wh, sku, pincode, stock, settings.REORDER_THRESHOLD, settings.MAX_STOCK_PER_SKU))
    db.executemany(
        "INSERT OR REPLACE INTO inventory(warehouse_id, sku, pincode, current_stock, "
        "reorder_threshold, max_capacity) VALUES(?,?,?,?,?,?)",
        rows,
    )


def _seed_demand(rng: random.Random, skus: list[str], days: int = 90) -> None:
    """90 days of hourly demand with dual peaks, weekend uplift, weather noise."""
    forecast_skus = skus[:12]  # keep history compact for the free tier
    start = datetime.now(timezone.utc) - timedelta(days=days)
    batch = []
    for sku in forecast_skus:
        base = rng.uniform(8, 25)
        for pincode in settings.PIN_CODES[:3]:
            pin_factor = rng.uniform(0.7, 1.3)
            t = start
            for _ in range(days * 24):
                hour = t.hour
                # Two daily peaks (lunch ~12, evening ~19).
                p1 = math.exp(-0.5 * ((hour - 12) / 3) ** 2)
                p2 = math.exp(-0.5 * ((hour - 19) / 2.5) ** 2)
                night = 0.15 if 0 <= hour <= 5 else 1.0
                is_weekend = 1 if t.weekday() >= 5 else 0
                weekend_boost = 1.25 if is_weekend else 1.0
                temp = 27 + 5 * math.sin((t.timetuple().tm_yday / 365) * 2 * math.pi) + rng.uniform(-2, 2)
                temp_effect = 1.0 + max(0, (temp - 30)) * 0.02  # hotter -> slightly more demand
                demand = base * pin_factor * (0.3 + p1 + 0.7 * p2) * night * weekend_boost * temp_effect
                demand = max(0.0, demand + rng.uniform(-2, 2))
                batch.append((sku, pincode, t.isoformat(), round(demand, 2), round(temp, 1), is_weekend, hour))
                t += timedelta(hours=1)
    db.executemany(
        "INSERT INTO demand_history(sku, pincode, ts, demand, temp_c, is_weekend, hour) "
        "VALUES(?,?,?,?,?,?,?)",
        batch,
    )


def _inventory_skus() -> list[str]:
    rows = db.query("SELECT DISTINCT sku FROM inventory ORDER BY sku")
    return [r["sku"] for r in rows]


def seed_if_empty() -> None:
    """Seed only what is actually missing. Never overwrite a real catalogue.

    This is table-by-table rather than all-or-nothing, because `ml/import_catalog.py`
    loads the real 44k Kaggle catalogue into `products` (and optionally `inventory`).
    The previous all-or-nothing guard keyed off `seed_version`, which the importer
    never set — so on the next boot the synthetic 400-item catalogue was inserted
    *on top of* the real one and the real inventory was overwritten.
    """
    db.init_db()
    rng = random.Random(42)  # deterministic

    _seed_users()  # always safe: ON CONFLICT DO NOTHING

    has_real_catalog = db.get_meta("catalog_source") is not None
    seeded = []

    if db.table_count("products") == 0 and not has_real_catalog:
        skus = _seed_catalog(rng)
        seeded.append("products")
    else:
        skus = []
        if has_real_catalog:
            log.info("real catalogue present (%s) — not seeding synthetic products",
                     db.get_meta("catalog_source"))

    if db.table_count("inventory") == 0:
        if not skus:
            # Derive SKUs from the catalogue we have, real or synthetic.
            rows = db.query(
                "SELECT article_type, COUNT(*) c FROM products "
                "GROUP BY article_type ORDER BY c DESC LIMIT 40"
            )
            skus = [f"{r['article_type'].upper().replace(' ', '')[:6]}_{i:03d}"
                    for i, r in enumerate(rows)]
        if skus:
            _seed_inventory(rng, skus)
            seeded.append("inventory")

    if db.table_count("demand_history") == 0:
        # Generate demand for whatever SKUs are actually in inventory, so the
        # forecast page and the inventory page always agree.
        inv_skus = _inventory_skus()
        if inv_skus:
            _seed_demand(rng, inv_skus)
            seeded.append("demand_history")

    db.set_meta("seed_version", SEED_VERSION)
    db.set_meta("seeded_at", datetime.now(timezone.utc).isoformat())

    if seeded:
        log.info(
            "seeded %s -> %d products, %d inventory rows, %d demand rows",
            ", ".join(seeded),
            db.table_count("products"),
            db.table_count("inventory"),
            db.table_count("demand_history"),
        )
    else:
        log.info("database already populated — nothing to seed")
