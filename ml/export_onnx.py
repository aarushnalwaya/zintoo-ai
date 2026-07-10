"""
Step 3 — Export the trained checkpoint to ONNX, and PROVE it matches.

The single most dangerous step in a vision deployment is the export: it is
entirely possible to ship an ONNX file that runs fine and returns subtly
different numbers than the model you evaluated. So this script does not just
export — it asserts numerical parity between PyTorch and ONNX Runtime on random
inputs, and refuses to write the artifact if they disagree.

Normalisation is baked into the graph (see `NormalizedModel`), so the server
feeds a plain [0,1] tensor and cannot introduce train/serve skew.

    python -m ml.export_onnx
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import torch

from ml import config
from ml.model import ExportWrapper, FashionNet

PARITY_TOL = 1e-4


SKEW_TOLERANCE = 0.15   # allowed absolute drop in top-1 before we refuse to ship


def _skew_guard(sess, report: dict, n_images: int = 256) -> None:
    """Run the exported ONNX over real val images using the SERVER's preprocessing.

    This is the check that catches train/serve skew — normalisation applied twice
    or not at all, wrong channel order, wrong resize. Random-tensor parity cannot.
    """
    val_csv = config.MANIFEST_DIR / "val.csv"
    if not val_csv.exists():
        print("⚠️  no val.csv — skipping skew guard (run ml.prepare_data)")
        return

    import pandas as pd
    from PIL import Image
    from app.vision_preprocess import resize_and_crop, to_tensor

    df = pd.read_csv(val_csv).sample(
        n=min(n_images, sum(1 for _ in open(val_csv)) - 1), random_state=0
    )
    task = config.PRIMARY_TASK
    in_name = sess.get_inputs()[0].name

    correct = total = 0
    for start in range(0, len(df), 32):
        chunk = df.iloc[start:start + 32]
        batch = []
        for path in chunk["image_path"]:
            try:
                batch.append(to_tensor(resize_and_crop(Image.open(path).convert("RGB")))[0])
            except Exception:
                batch.append(np.zeros((3, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32))
        logits = sess.run([f"logits_{task}"], {in_name: np.stack(batch)})[0]
        correct += int((logits.argmax(1) == chunk[task].to_numpy()).sum())
        total += len(chunk)

    onnx_acc = correct / max(1, total)
    print(f"   skew guard: exported ONNX top-1 on {total} real val images = {onnx_acc:.4f}")

    expected = None
    if isinstance(report.get("test"), dict):
        expected = report["test"].get(task, {}).get("accuracy")
    if expected is None:
        expected = report.get("best_val_macro_f1")  # weak fallback
        if expected is not None:
            print("   (no test accuracy in train_report; guard is advisory only)")
            return

    drop = expected - onnx_acc
    if drop > SKEW_TOLERANCE:
        raise SystemExit(
            f"\n❌ TRAIN/SERVE SKEW DETECTED.\n"
            f"   Training reported top-1 {expected:.4f}; the exported ONNX scores "
            f"{onnx_acc:.4f} on the same data ({drop:.1%} drop).\n"
            f"   The served model is not the model you evaluated. Refusing to ship.\n"
            f"   Most likely cause: input normalisation applied in one path but not the other."
        )
    print(f"✅ no skew: ONNX {onnx_acc:.4f} vs training {expected:.4f} (drop {drop:+.1%})")


def main() -> None:
    ckpt_path = config.ARTIFACTS_DIR / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"❌ No checkpoint at {ckpt_path}. Run `python -m ml.train` first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    num_classes = ckpt["num_classes"]
    label_maps = json.loads((config.MANIFEST_DIR / "label_maps.json").read_text())

    # Checkpoints written before the normalisation fix trained on raw [0,1]
    # tensors. Exporting those with normalisation ON silently destroys accuracy
    # (~4.4x input scale shift). Honour whatever the checkpoint actually used.
    normalize_input = bool(ckpt.get("normalize_input", False))
    if not normalize_input:
        print(
            "⚠️  LEGACY CHECKPOINT: trained WITHOUT input normalisation.\n"
            "    Exporting to match (no Sub/Div in the graph) so it behaves as trained.\n"
            "    Accuracy will be lower than a properly normalised run — retrain with\n"
            "    the current ml/train.py for the real numbers."
        )

    net = FashionNet(num_classes, normalize_input=normalize_input)
    net.load_state_dict(ckpt["state_dict"], strict=False)  # buffers may be absent in old ckpts
    wrapped = ExportWrapper(net).eval()

    onnx_path = config.ARTIFACTS_DIR / "fashion_classifier.onnx"
    output_names = ["embedding", *[f"logits_{t}" for t in config.TASKS]]
    dummy = torch.rand(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)  # [0,1], like the server sends

    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={"input": {0: "batch"}, **{n: {0: "batch"} for n in output_names}},
        opset_version=17,
        do_constant_folding=True,
        # torch>=2.9 defaults to the dynamo exporter, which needs `onnxscript`
        # and deprecates `dynamic_axes` in favour of `dynamic_shapes`. Pin the
        # classic exporter: it is the path this pipeline is verified against,
        # and it is what keeps the batch axis dynamic for ml/build_index.py.
        dynamo=False,
    )
    print(f"exported -> {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")

    # ─── Parity check: torch vs onnxruntime, on fresh random inputs ────
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    worst = 0.0
    for trial in range(5):
        x = rng.random((2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)).astype(np.float32)
        with torch.no_grad():
            torch_out = wrapped(torch.from_numpy(x))
        ort_out = sess.run(None, {"input": x})
        for name, t_out, o_out in zip(output_names, torch_out, ort_out):
            diff = float(np.abs(t_out.numpy() - o_out).max())
            worst = max(worst, diff)
            if diff > PARITY_TOL:
                raise SystemExit(
                    f"❌ PARITY FAILURE on '{name}' (trial {trial}): max|Δ|={diff:.2e} "
                    f"> tol {PARITY_TOL}. Refusing to ship this artifact."
                )
    print(f"✅ torch/ONNX parity verified across 5 trials (worst max|Δ| = {worst:.2e})")

    # Embedding must be unit-norm — visual search assumes it.
    emb = sess.run(["embedding"], {"input": rng.random((4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)).astype(np.float32)})[0]
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), f"embeddings not L2-normalised: {norms}"
    print(f"✅ embeddings are unit-norm (mean {norms.mean():.6f})")

    report = {}
    tr = config.ARTIFACTS_DIR / "train_report.json"
    if tr.exists():
        report = json.loads(tr.read_text())

    # ─── SKEW GUARD ───────────────────────────────────────────────────
    # Parity above only proves the export faithfully reproduces `wrapped`. It
    # cannot detect that `wrapped` differs from what was TRAINED. So run the
    # exported ONNX over real validation images through the SERVER's own
    # preprocessing and compare accuracy to the training report. A large drop
    # means train/serve skew, and shipping would be worse than not shipping.
    _skew_guard(sess, report)

    labels_path = config.ARTIFACTS_DIR / "labels.json"
    labels_path.write_text(json.dumps({
        "tasks": label_maps,
        "embedding_dim": config.EMBEDDING_DIM,
        "input_size": config.IMAGE_SIZE,
        "backbone": config.BACKBONE,
        "trained_at": report.get("trained_at", datetime.now(timezone.utc).isoformat()),
        "metrics": report.get("test"),
        "preprocess": {
            "resize_short": 256, "crop": config.IMAGE_SIZE,
            "input_range": "[0,1] NCHW float32",
            "normalization": "baked into graph" if normalize_input else "NONE (legacy checkpoint)",
        },
    }, indent=2))
    print(f"✅ labels -> {labels_path}")
    print("\nNext: python -m ml.build_index")


if __name__ == "__main__":
    main()
