"""
Step 1a (STRONGLY recommended) — pre-resize the images once.

The full Kaggle dataset ships 1800x2400 JPEGs (~25 GB). Decoding one of those
and shrinking it to 224x224 costs ~40-60 ms of CPU. With 44k images per epoch
and only 2-4 dataloader workers on Kaggle, the GPU sits idle waiting for JPEG
decode: you get maybe 25 img/s and a 30-minute epoch on a T4 that should take 3.

Resizing once to a 256px shorter side (a few hundred MB) makes every subsequent
epoch I/O-trivial. This pays for itself before the first epoch finishes.

    python -m ml.resize_cache
    # then point training at the cache:
    export ZINTOO_DATASET_DIR=<cache_parent>

Idempotent: already-resized files are skipped, so it is safe to re-run after an
interrupted Kaggle session.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from ml import config

TARGET_SHORT = 256
QUALITY = 90


def _resize_one(args) -> str:
    src, dst = args
    if dst.exists():
        return "skip"
    try:
        with Image.open(src) as img:
            img = img.convert("RGB")
            w, h = img.size
            scale = TARGET_SHORT / min(w, h)
            if scale < 1.0:  # never upscale
                img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
            img.save(dst, "JPEG", quality=QUALITY, optimize=True)
        return "ok"
    except Exception:
        return "fail"


def _pick_output_root() -> Path:
    """Choose a writable cache location.

    The dataset dir is often READ-ONLY (`/kaggle/input`), so writing a sibling
    folder next to it fails. Preference order:
      1. $ZINTOO_RESIZE_OUT (explicit wins)
      2. <dataset>-resized, if the dataset's parent is genuinely writable
      3. ./data/fashion-dataset-resized (always writable)
    """
    explicit = os.getenv("ZINTOO_RESIZE_OUT")
    if explicit:
        return Path(explicit)

    parent = config.DATASET_DIR.parent
    probe = parent / ".zintoo_write_test"
    try:                      # os.access() can lie (root, NFS) — prove it instead
        probe.touch()
        probe.unlink()
        return Path(str(config.DATASET_DIR) + "-resized")
    except OSError:
        fallback = config.PROJECT_ROOT / "data" / "fashion-dataset-resized"
        print(f"  (dataset dir is read-only — caching to {fallback} instead)")
        return fallback


def main() -> None:
    src_dir = config.IMAGES_DIR
    if not src_dir.exists():
        sys.exit(f"❌ {src_dir} not found. Set ZINTOO_DATASET_DIR (run `python -m ml.doctor`).")

    out_root = _pick_output_root()
    out_dir = out_root / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.jpg"))
    if not files:
        sys.exit(f"❌ No .jpg files in {src_dir}")
    print(f"resizing {len(files):,} images -> {out_dir} (shorter side {TARGET_SHORT}px)")

    jobs = [(f, out_dir / f.name) for f in files]
    counts = {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:   # PIL releases the GIL on decode
        for i, res in enumerate(pool.map(_resize_one, jobs), 1):
            counts[res] += 1
            if i % 2000 == 0:
                rate = i / max(1e-6, time.time() - t0)
                print(f"  {i:>7,}/{len(files):,}  ({rate:.0f} img/s)")

    # styles.csv must sit next to images/ for prepare_data to find it.
    styles_dst = out_root / "styles.csv"
    if config.STYLES_CSV.exists() and not styles_dst.exists():
        styles_dst.write_bytes(config.STYLES_CSV.read_bytes())

    size_mb = sum(f.stat().st_size for f in out_dir.glob("*.jpg")) / 1e6
    print(f"\n✅ {counts['ok']:,} resized, {counts['skip']:,} skipped, {counts['fail']:,} failed")
    print(f"   {size_mb:,.0f} MB in {time.time()-t0:.0f}s")
    print(f"\nNow run training against the cache:\n   export ZINTOO_DATASET_DIR={out_root}")


if __name__ == "__main__":
    main()
