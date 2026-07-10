"""
Vision inference service (ONNX Runtime).

Why ONNX and not PyTorch at serve time:
  * `import torch` alone costs ~300 MB RSS and ~800 MB on disk. That does not
    fit a 512 MB instance — it is exactly what turned the original deploy into
    a facade.
  * onnxruntime-cpu is ~50 MB and runs MobileNetV3-Small at 224x224 in roughly
    10-30 ms on a single CPU core. That is genuinely real-time.

Operational properties:
  * Lazy load  — the session is built on first use, not at import, so an
    instance without a model (or with vision disabled) still boots.
  * Thread-safe — ORT sessions are thread-safe for `run()`; construction is
    guarded by a lock so concurrent first-requests don't build twice.
  * Bounded    — intra/inter-op threads pinned to 1. On a shared 0.1-CPU
    instance, letting ORT spawn N threads causes contention, not speed.
  * Degrades honestly — if the model file is absent, `available()` is False and
    the API returns 503 with a clear reason rather than pretending.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from . import settings
from .observability import get_logger, metrics
from .vision_preprocess import ImageDecodeError, preprocess

log = get_logger("zintoo.vision")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)   # numerically stable
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class VisionModel:
    """Wraps an exported multi-head ONNX model + its label sidecar."""

    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self._session = None
        self._lock = threading.Lock()
        self._meta: dict = {}
        self._load_error: str | None = None

    # ─── lifecycle ───────────────────────────────────────────────────
    def available(self) -> bool:
        return self.model_path.exists() and self.labels_path.exists()

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:      # double-checked
                return self._session
            if not self.available():
                raise FileNotFoundError(
                    f"Vision model not found. Expected {self.model_path} and {self.labels_path}. "
                    "Train and export it (see VISION.md), or set ZINTOO_VISION_ENABLED=false."
                )
            import onnxruntime as ort  # imported lazily: ~40 MB RSS

            opts = ort.SessionOptions()
            # Pin threads: on small shared instances more threads = more contention.
            opts.intra_op_num_threads = settings.VISION_THREADS
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.enable_mem_pattern = True

            t0 = time.perf_counter()
            session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self._meta = json.loads(self.labels_path.read_text())
            self._session = session
            log.info(
                "vision model loaded in %.0f ms (tasks=%s, embedding_dim=%s)",
                (time.perf_counter() - t0) * 1000,
                list(self._meta.get("tasks", {})),
                self._meta.get("embedding_dim"),
            )
            self._warmup()
            return self._session

    def _warmup(self) -> None:
        """First ORT run allocates arenas and is 5-10x slower. Pay it at load."""
        try:
            size = int(self._meta.get("input_size", 224))
            dummy = np.zeros((1, 3, size, size), dtype=np.float32)
            self._session.run(None, {self._input_name(): dummy})
            log.info("vision warmup complete")
        except Exception as exc:  # noqa: BLE001
            log.warning("vision warmup failed (non-fatal): %s", exc)

    def _input_name(self) -> str:
        return self._session.get_inputs()[0].name

    # ─── inference ───────────────────────────────────────────────────
    def infer(self, image_bytes: bytes, top_k: int = 3) -> dict:
        """bytes -> {predictions: {task: [{label, confidence}]}, embedding: [...]}"""
        session = self._ensure_session()

        t0 = time.perf_counter()
        tensor = preprocess(image_bytes)     # may raise ImageDecodeError
        t_pre = time.perf_counter() - t0

        t1 = time.perf_counter()
        outputs = session.run(None, {self._input_name(): tensor})
        t_infer = time.perf_counter() - t1

        names = [o.name for o in session.get_outputs()]
        out = dict(zip(names, outputs))

        tasks: dict[str, list[str]] = self._meta.get("tasks", {})
        predictions: dict[str, list[dict]] = {}
        for task, labels in tasks.items():
            key = f"logits_{task}"
            if key not in out:
                continue
            probs = _softmax(out[key][0].astype(np.float32))
            k = min(top_k, len(labels))
            idx = np.argsort(-probs)[:k]
            predictions[task] = [
                {"label": labels[i], "confidence": round(float(probs[i]), 4)} for i in idx
            ]

        embedding = None
        if "embedding" in out:
            vec = out["embedding"][0].astype(np.float32)
            norm = float(np.linalg.norm(vec)) or 1e-9
            embedding = (vec / norm).tolist()   # L2-normalised -> cosine == dot

        metrics.inc("vision_inferences_total")
        metrics.observe_latency("vision.preprocess", t_pre)
        metrics.observe_latency("vision.infer", t_infer)

        return {
            "predictions": predictions,
            "embedding": embedding,
            "timing_ms": {
                "preprocess": round(t_pre * 1000, 2),
                "inference": round(t_infer * 1000, 2),
                "total": round((t_pre + t_infer) * 1000, 2),
            },
        }

    def info(self) -> dict:
        return {
            "available": self.available(),
            "loaded": self._session is not None,
            "model_path": str(self.model_path),
            "tasks": {t: len(v) for t, v in self._meta.get("tasks", {}).items()},
            "embedding_dim": self._meta.get("embedding_dim"),
            "input_size": self._meta.get("input_size"),
            "trained_at": self._meta.get("trained_at"),
            "metrics": self._meta.get("metrics"),
        }


model = VisionModel(settings.VISION_MODEL_PATH, settings.VISION_LABELS_PATH)

__all__ = ["model", "VisionModel", "ImageDecodeError"]
