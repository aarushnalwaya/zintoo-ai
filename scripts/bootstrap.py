"""
Container cold-start bootstrap.

Managed hosts (HF Spaces, Render free, Fly machines) give you an ephemeral
filesystem: the SQLite DB is gone on every restart. This runs once before
uvicorn and makes the instance self-sufficient:

  * creates the schema
  * imports the real Kaggle catalogue if `styles.csv` is present and the
    `products` table is empty (so visual-search ids resolve — see ml/doctor.py)
  * otherwise leaves the app to seed its synthetic demo catalogue

Deliberately non-fatal: a bootstrap failure logs and exits 0, so the web
process still starts and can report its state on /health rather than crash-
looping and giving you no way to debug it.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    try:
        from app import db, settings
        db.init_db()

        n = db.table_count("products")
        source = db.get_meta("catalog_source")
        if n > 0 and source:
            print(f"[bootstrap] catalogue already present ({n:,} products, {source})")
            return 0

        # ml/config resolves styles.csv from ZINTOO_DATASET_DIR.
        from ml import config
        if not config.STYLES_CSV.exists():
            print(f"[bootstrap] no styles.csv at {config.STYLES_CSV} — "
                  f"app will seed its synthetic demo catalogue")
            return 0

        from ml import import_catalog
        rows = import_catalog.load_catalog()
        count = import_catalog.import_products(rows)
        import_catalog.regenerate_inventory(rows)
        db.set_meta("catalog_source", "kaggle:fashion-product-images-dataset")
        db.set_meta("catalog_rows", str(count))
        print(f"[bootstrap] imported {count:,} products into {settings.DB_PATH}")

        # Warn loudly if the embedding index won't resolve against this catalogue.
        ids_path = settings.VISUAL_IDS_PATH
        if ids_path.exists():
            import json
            ids = json.loads(ids_path.read_text())[:200]
            ph = ",".join("?" for _ in ids)
            hit = db.query(f"SELECT product_id FROM products WHERE product_id IN ({ph})", ids)
            rate = len(hit) / max(1, len(ids))
            if rate < 0.9:
                print(f"[bootstrap] WARNING: only {rate:.0%} of visual-index ids resolve. "
                      f"Visual search will return empty results.")
            else:
                print(f"[bootstrap] visual index ids resolve ({rate:.0%})")
        return 0
    except Exception:
        print("[bootstrap] FAILED (continuing anyway so /health is reachable):")
        traceback.print_exc()
        return 0


if __name__ == "__main__":
    sys.exit(main())
