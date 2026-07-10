"""
Preflight check — run this BEFORE training, and again after.

Catches the mistakes that otherwise cost you two hours of GPU time or produce a
silently empty results grid:

  * dataset path wrong / nested one level deeper than you think
  * styles.csv unreadable, or images/ missing
  * ids in styles.csv with no image file
  * (post-train) artifacts missing or mismatched
  * (post-train) the visual index references product_ids that are NOT in the
    products table -> classification works, similar-products grid is empty

    python -m ml.doctor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ml import config

OK, WARN, FAIL = "✅", "⚠️ ", "❌"
problems: list[str] = []


def check_dataset() -> None:
    print("\n[1] Dataset")
    if not config.DATASET_DIR.exists():
        print(f"  {FAIL} ZINTOO_DATASET_DIR does not exist: {config.DATASET_DIR}")
        problems.append("dataset dir missing")
        # Kaggle nests the folder; help them find it.
        for base in (Path("/kaggle/input"), Path.cwd()):
            if base.exists():
                hits = list(base.rglob("styles.csv"))[:5]
                if hits:
                    print(f"  {WARN} found styles.csv elsewhere — try one of these as ZINTOO_DATASET_DIR:")
                    for h in hits:
                        print(f"        {h.parent}")
        return
    print(f"  {OK} DATASET_DIR = {config.DATASET_DIR}")

    if not config.STYLES_CSV.exists():
        print(f"  {FAIL} styles.csv not found at {config.STYLES_CSV}")
        problems.append("styles.csv missing")
    else:
        size = config.STYLES_CSV.stat().st_size / 1e6
        print(f"  {OK} styles.csv ({size:.1f} MB)")

    if not config.IMAGES_DIR.exists():
        print(f"  {FAIL} images/ not found at {config.IMAGES_DIR}")
        problems.append("images dir missing")
        return

    imgs = list(config.IMAGES_DIR.glob("*.jpg"))
    print(f"  {OK} images/ contains {len(imgs):,} jpg files")
    if len(imgs) < 1000:
        print(f"  {WARN} that looks low — expected ~44,000 for the full dataset")

    if imgs:
        from PIL import Image
        with Image.open(imgs[0]) as im:
            w, h = im.size
        print(f"  {OK} sample image size: {w}x{h}")
        if min(w, h) < 100:
            print(f"  {WARN} very low resolution — this looks like the 'small' (60x80) variant.")
            print(f"       224x224 training on upscaled 60x80 images will underperform badly.")
        elif min(w, h) > 600:
            print(f"  {WARN} large images: run `python -m ml.resize_cache` first or epochs will crawl.")

    # cross-check csv ids vs image files
    if config.STYLES_CSV.exists():
        try:
            import pandas as pd
        except ImportError:
            print(f"  {WARN} pandas not installed — skipping csv/image cross-check.")
            print(f"       (Only needed for training. On a serving box use `--serve`.)")
            return
        df = pd.read_csv(config.STYLES_CSV, on_bad_lines="skip", engine="python")
        have = {p.stem for p in imgs}
        ids = df["id"].astype(str)
        missing = (~ids.isin(have)).sum()
        print(f"  {OK if missing < len(df) * 0.2 else WARN} {len(df):,} csv rows, {missing:,} without an image file")


def check_artifacts() -> None:
    print("\n[2] Trained artifacts")
    art = config.ARTIFACTS_DIR
    files = {
        "fashion_classifier.onnx": art / "fashion_classifier.onnx",
        "labels.json": art / "labels.json",
        "catalog_embeddings.npy": art / "catalog_embeddings.npy",
        "catalog_ids.json": art / "catalog_ids.json",
    }
    any_missing = False
    for name, p in files.items():
        if p.exists():
            print(f"  {OK} {name} ({p.stat().st_size/1e6:.1f} MB)")
        else:
            print(f"  {WARN} {name} missing — not trained/exported yet")
            any_missing = True
    if any_missing:
        return

    labels = json.loads(files["labels.json"].read_text())
    print(f"  {OK} tasks: " + ", ".join(f"{t}({len(c)})" for t, c in labels["tasks"].items()))
    metrics = labels.get("metrics")
    if isinstance(metrics, dict):
        for task, m in metrics.items():
            if not isinstance(m, dict):
                continue
            score = m.get("accuracy", m.get("top1"))
            if isinstance(score, (int, float)):
                print(f"       {task:<16} top1={score:.4f}")

    import numpy as np
    emb = np.load(files["catalog_embeddings.npy"], mmap_mode="r")
    ids = json.loads(files["catalog_ids.json"].read_text())
    if emb.shape[0] != len(ids):
        print(f"  {FAIL} index corrupt: {emb.shape[0]} embeddings vs {len(ids)} ids")
        problems.append("index corrupt")
    else:
        print(f"  {OK} index: {emb.shape[0]:,} x {emb.shape[1]}")
    if emb.shape[1] != labels["embedding_dim"]:
        print(f"  {FAIL} embedding_dim mismatch: index {emb.shape[1]} vs labels {labels['embedding_dim']}")
        problems.append("embedding dim mismatch")


def check_linkage() -> None:
    """The silent killer: index ids must exist in the products table."""
    print("\n[3] Catalogue linkage (index ids <-> products table)")
    ids_path = config.ARTIFACTS_DIR / "catalog_ids.json"
    if not ids_path.exists():
        print(f"  {WARN} no index yet — skipping")
        return
    from app import db

    db.init_db()
    n_products = db.table_count("products")
    if n_products == 0:
        print(f"  {FAIL} products table is empty")
        problems.append("empty products")
        return

    ids = json.loads(ids_path.read_text())
    sample = ids[:500]
    placeholders = ",".join("?" for _ in sample)
    found = db.query(f"SELECT product_id FROM products WHERE product_id IN ({placeholders})", sample)
    hit_rate = len(found) / len(sample)
    print(f"  products in DB: {n_products:,}")
    print(f"  index ids found in DB: {len(found)}/{len(sample)} ({hit_rate:.0%})")
    if hit_rate < 0.9:
        print(f"  {FAIL} Visual search will return an EMPTY grid.")
        print(f"       The index references Kaggle product ids, but the DB holds a different set.")
        print(f"       FIX:  python -m ml.import_catalog --inventory")
        problems.append("catalogue linkage broken")
    else:
        print(f"  {OK} ids line up — visual search will hydrate real products")


def main() -> None:
    ap = argparse.ArgumentParser(description="Zintoo vision preflight check")
    ap.add_argument(
        "--serve", action="store_true",
        help="Serving machine: skip dataset checks (you don't need the 25 GB "
             "image set to run the app — only the artifacts and the catalogue).",
    )
    args = ap.parse_args()

    print("=" * 62)
    print("ZINTOO VISION — PREFLIGHT CHECK" + (" (serving mode)" if args.serve else ""))
    print("=" * 62)
    if args.serve:
        print("\n[1] Dataset — skipped (--serve)")
    else:
        check_dataset()
    check_artifacts()
    try:
        check_linkage()
    except Exception as exc:  # noqa: BLE001
        print(f"  {WARN} linkage check skipped: {exc}")

    print("\n" + "=" * 62)
    if problems:
        print(f"{FAIL} {len(problems)} problem(s): " + "; ".join(problems))
        sys.exit(1)
    print(f"{OK} No blocking problems found.")


if __name__ == "__main__":
    main()
