"""
Step 5 — Load the REAL Kaggle catalogue into the app's SQLite database.

Replaces the deterministic synthetic catalogue (`app/seed.py`) with the genuine
44k-item Fashion Product Images catalogue, so that:

  * text search (`/recommend`) ranks real products,
  * visual search (`/recommend/image`) can hydrate real rows from `product_id`,
  * inventory SKUs map to real article types.

** Mandatory after `ml/build_index.py`. ** The embedding index stores Kaggle
product ids (15970, 39386, ...). If `products` still holds the synthetic seed
(ids 1..400) the ids don't resolve and visual search silently returns an empty
grid. `python -m ml.doctor --serve` checks exactly this.

Uses the stdlib `csv` module, NOT pandas: this script runs on the serving
machine, which deliberately has no training dependencies installed.

It also *recovers* the ~15 malformed rows that `prepare_data` skips. Those rows
have unescaped commas inside `productDisplayName` — the final column — so the
overflow fields belong to the name and can simply be rejoined.

Idempotent and transactional: the swap fully succeeds, or the previous
catalogue is left untouched.

    python -m ml.import_catalog              # products only
    python -m ml.import_catalog --inventory  # also regenerate warehouse stock
"""

from __future__ import annotations

import argparse
import csv
import random
import sys

from app import db, settings
from ml import config

COLUMNS = ["id", "gender", "masterCategory", "subCategory", "articleType",
           "baseColour", "season", "year", "usage", "productDisplayName"]


def load_catalog() -> list[dict]:
    if not config.STYLES_CSV.exists():
        sys.exit(
            f"❌ {config.STYLES_CSV} not found.\n"
            f"   Set ZINTOO_DATASET_DIR to the folder containing styles.csv."
        )

    rows: list[dict] = []
    recovered = skipped = 0
    seen: set[int] = set()

    with config.STYLES_CSV.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            sys.exit("❌ styles.csv is empty")
        idx = {name.strip(): i for i, name in enumerate(header)}
        missing = [c for c in COLUMNS if c not in idx]
        if missing:
            sys.exit(f"❌ styles.csv missing columns: {missing}")

        n_cols = len(header)
        name_pos = idx["productDisplayName"]

        for fields in reader:
            if len(fields) > n_cols:
                # Unescaped commas inside the trailing name column: rejoin the
                # overflow rather than dropping the row.
                fields = fields[:n_cols - 1] + [",".join(fields[n_cols - 1:])]
                recovered += 1
            elif len(fields) < n_cols:
                skipped += 1
                continue

            try:
                pid = int(fields[idx["id"]])
            except (ValueError, IndexError):
                skipped += 1
                continue
            if pid in seen:
                continue

            def get(col: str, default: str = "Unknown") -> str:
                val = fields[idx[col]].strip()
                return val or default

            article = fields[idx["articleType"]].strip()
            master = fields[idx["masterCategory"]].strip()
            if not article or not master:
                skipped += 1
                continue

            colour = get("baseColour")
            name = fields[name_pos].strip() or f"{colour} {article}"
            seen.add(pid)
            rows.append({
                "id": pid,
                "gender": get("gender"),
                "masterCategory": master,
                "subCategory": get("subCategory"),
                "articleType": article,
                "baseColour": colour,
                "season": get("season"),
                "usage": get("usage"),
                "name": name,
            })

    print(f"  parsed {len(rows):,} rows "
          f"({recovered:,} malformed rows recovered, {skipped:,} unusable skipped)")
    if not rows:
        sys.exit("❌ No usable rows in styles.csv")
    return rows


def _description(r: dict) -> str:
    return (
        f"{r['baseColour']} {r['articleType'].lower()} for {r['gender'].lower()} — "
        f"{r['usage'].lower()} wear, ideal for {r['season'].lower()}. "
        f"{r['subCategory']} in {r['masterCategory']}."
    )


def import_products(rows: list[dict]) -> int:
    records = [
        (r["id"], r["name"], r["masterCategory"], r["subCategory"], r["articleType"],
         r["baseColour"], r["gender"], r["season"], r["usage"], _description(r))
        for r in rows
    ]
    with db.transaction() as conn:
        conn.execute("DELETE FROM products")
        conn.executemany(
            "INSERT INTO products(product_id, name, master_category, sub_category, "
            "article_type, color, gender, season, usage, description) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            records,
        )
    return len(records)


def regenerate_inventory(rows: list[dict], top_n: int = 40) -> int:
    """Build warehouse stock over the most common real article types."""
    rng = random.Random(config.SEED)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["articleType"]] = counts.get(r["articleType"], 0) + 1
    top_types = sorted(counts, key=lambda k: counts[k], reverse=True)[:top_n]

    records = []
    for i, article in enumerate(top_types):
        sku = f"{article.upper().replace(' ', '')[:6]}_{i:03d}"
        for wh in settings.WAREHOUSE_IDS:
            records.append((
                wh, sku, settings.WAREHOUSE_PINCODE_MAP[wh],
                rng.choice([2, 5, 8, 12, 20, 35, 55, 80]),
                settings.REORDER_THRESHOLD, settings.MAX_STOCK_PER_SKU,
            ))
    with db.transaction() as conn:
        conn.execute("DELETE FROM inventory")
        conn.executemany(
            "INSERT INTO inventory(warehouse_id, sku, pincode, current_stock, "
            "reorder_threshold, max_capacity) VALUES(?,?,?,?,?,?)",
            records,
        )
    return len(records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", action="store_true", help="also regenerate warehouse stock")
    args = ap.parse_args()

    db.init_db()
    print(f"reading {config.STYLES_CSV}")
    rows = load_catalog()

    n = import_products(rows)
    print(f"✅ imported {n:,} products into {settings.DB_PATH}")

    if args.inventory:
        m = regenerate_inventory(rows)
        print(f"✅ regenerated {m:,} inventory rows over real article types")

    db.set_meta("catalog_source", "kaggle:fashion-product-images-dataset")
    db.set_meta("catalog_rows", str(n))
    print("\nNext: python -m ml.doctor --serve   (verify the index ids resolve)")


if __name__ == "__main__":
    main()
