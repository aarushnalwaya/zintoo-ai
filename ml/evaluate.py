"""
Step 6 — Evaluate the EXPORTED model (not the checkpoint) on the test split.

Evaluating the ONNX artifact rather than the PyTorch checkpoint is the point:
it's the thing that actually serves traffic. Reports:

  * top-1 / top-5 accuracy and macro-F1 per task
  * the 10 worst classes by F1 (where your model actually fails)
  * end-to-end latency percentiles on one CPU thread, which is what a client
    experiences

    python -m ml.evaluate
"""

from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from app.vision_preprocess import CROP_SIZE, resize_and_crop, to_tensor
from ml import config

BATCH = 64


def _batch(paths):
    from PIL import Image

    out = []
    for p in paths:
        try:
            out.append(to_tensor(resize_and_crop(Image.open(p).convert("RGB")))[0])
        except Exception:
            out.append(np.zeros((3, CROP_SIZE, CROP_SIZE), dtype=np.float32))
    return np.stack(out)


def main() -> None:
    import onnxruntime as ort

    model_path = config.ARTIFACTS_DIR / "fashion_classifier.onnx"
    if not model_path.exists():
        sys.exit(f"❌ {model_path} missing. Run `python -m ml.export_onnx`.")

    label_maps = json.loads((config.MANIFEST_DIR / "label_maps.json").read_text())
    test = pd.read_csv(config.MANIFEST_DIR / "test.csv")
    print(f"evaluating {len(test):,} test images\n")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    preds = {t: [] for t in config.TASKS}
    top5 = {t: [] for t in config.TASKS}
    for start in range(0, len(test), BATCH):
        chunk = test.iloc[start:start + BATCH]
        x = _batch(chunk["image_path"].tolist())
        outs = dict(zip(out_names, sess.run(None, {in_name: x})))
        for t in config.TASKS:
            logits = outs[f"logits_{t}"]
            preds[t].append(logits.argmax(1))
            k = min(5, logits.shape[1])
            top5[t].append(np.argsort(-logits, axis=1)[:, :k])

    results = {}
    for t in config.TASKS:
        p = np.concatenate(preds[t])
        g = test[t].to_numpy()
        t5 = np.concatenate(top5[t])
        acc = float((p == g).mean())
        acc5 = float(np.mean([g[i] in t5[i] for i in range(len(g))]))
        f1 = float(f1_score(g, p, average="macro", zero_division=0))
        results[t] = {"top1": acc, "top5": acc5, "macro_f1": f1}
        print(f"{t:<16} top1={acc:.4f}  top5={acc5:.4f}  macro_f1={f1:.4f}")

        if t == config.PRIMARY_TASK:
            names = label_maps[t]
            rep = classification_report(g, p, output_dict=True, zero_division=0)
            per_class = [
                (names[int(k)], v["f1-score"], int(v["support"]))
                for k, v in rep.items() if k.isdigit()
            ]
            per_class.sort(key=lambda x: x[1])
            print(f"\n  10 weakest {t} classes (fix these first):")
            for name, f, sup in per_class[:10]:
                print(f"    {name:<28} f1={f:.3f}  (n={sup})")

    # ─── Latency: one thread, one image, warm session ─────────────────
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    s1 = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    x1 = _batch([test.iloc[0]["image_path"]])
    for _ in range(5):
        s1.run(None, {in_name: x1})     # warm arenas

    lat = []
    for _ in range(50):
        t0 = time.perf_counter()
        s1.run(None, {in_name: x1})
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    latency = {
        "p50_ms": round(statistics.median(lat), 2),
        "p95_ms": round(lat[int(0.95 * len(lat))], 2),
        "p99_ms": round(lat[int(0.99 * len(lat))], 2),
    }
    print(f"\ninference latency (1 thread, batch=1): "
          f"p50={latency['p50_ms']} ms  p95={latency['p95_ms']} ms  p99={latency['p99_ms']} ms")

    out = config.ARTIFACTS_DIR / "eval_report.json"
    out.write_text(json.dumps({"tasks": results, "latency": latency}, indent=2))
    print(f"\n✅ report -> {out}")


if __name__ == "__main__":
    main()
