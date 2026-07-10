"""
Step 4 — Precompute catalogue embeddings for visual similarity search.

Deliberately uses the **exported ONNX model**, not the PyTorch checkpoint, and
the **same preprocessing module the server uses**. If the index were built with
a different code path than the queries, every similarity score would be subtly
wrong and nobody would notice — the results would just be mediocre.

Stores float16 (halves the file, costs ~1e-3 of cosine precision) and an ids
sidecar. 44k x 256 float16 = ~22 MB.

    python -m ml.build_index
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.vision_preprocess import CROP_SIZE, resize_and_crop, to_tensor
from ml import config

BATCH = 64


def _load_batch(paths: list[str]) -> np.ndarray:
    from PIL import Image

    tensors = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(to_tensor(resize_and_crop(img))[0])
        except Exception:
            tensors.append(np.zeros((3, CROP_SIZE, CROP_SIZE), dtype=np.float32))
    return np.stack(tensors)


def main() -> None:
    import onnxruntime as ort

    model_path = config.ARTIFACTS_DIR / "fashion_classifier.onnx"
    if not model_path.exists():
        sys.exit(f"❌ {model_path} missing. Run `python -m ml.export_onnx` first.")

    # Index every catalogue item we know about (all splits).
    frames = [
        pd.read_csv(config.MANIFEST_DIR / f"{s}.csv") for s in ("train", "val", "test")
        if (config.MANIFEST_DIR / f"{s}.csv").exists()
    ]
    if not frames:
        sys.exit("❌ No manifests. Run `python -m ml.prepare_data` first.")
    df = pd.concat(frames).drop_duplicates("id").reset_index(drop=True)
    print(f"indexing {len(df):,} catalogue items")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0  # use all cores for the offline build
    sess = ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    embs, ids, t0 = [], [], time.time()
    for start in range(0, len(df), BATCH):
        chunk = df.iloc[start:start + BATCH]
        x = _load_batch(chunk["image_path"].tolist())
        emb = sess.run(["embedding"], {in_name: x})[0]
        embs.append(emb.astype(np.float32))
        ids.extend(int(i) for i in chunk["id"])
        if start % (BATCH * 20) == 0:
            done = start + len(chunk)
            rate = done / max(1e-6, time.time() - t0)
            print(f"  {done:>7,}/{len(df):,}  ({rate:.0f} img/s)")

    matrix = np.concatenate(embs)
    # Defensive re-normalisation: the graph already does it, but float16 storage
    # rounds, and search assumes unit vectors.
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
    matrix = matrix.astype(np.float16)

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = config.ARTIFACTS_DIR / "catalog_embeddings.npy"
    ids_path = config.ARTIFACTS_DIR / "catalog_ids.json"
    np.save(emb_path, matrix)
    ids_path.write_text(json.dumps(ids))

    print(f"\n✅ {matrix.shape[0]:,} x {matrix.shape[1]} embeddings "
          f"-> {emb_path} ({emb_path.stat().st_size/1e6:.1f} MB)")
    print(f"✅ ids -> {ids_path}")
    print(f"   took {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
