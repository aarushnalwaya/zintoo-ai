"""
Visual similarity index.

Classification alone tells a client "this is a Blue Casual Shirt". A retailer
actually wants "show me the 10 items in my catalogue that look like this". That
needs nearest-neighbour search over embeddings, which is what the original
project wanted FAISS for.

We don't need FAISS at this scale. With ~44k catalogue items and a 576-d
embedding, an exact brute-force cosine search is a single matmul:
    44,000 x 576 float32 = ~100 MB  ->  stored float16 = ~50 MB
A float32 matmul over that is a few milliseconds in NumPy. Exact results, no
index build, no extra dependency. Above ~1M items you'd want an ANN index.

Embeddings are stored L2-normalised, so cosine similarity == plain dot product.
The array is memory-mapped so we don't copy 50 MB into RSS on every boot.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from . import db, settings
from .observability import get_logger

log = get_logger("zintoo.visual_index")


class VisualIndex:
    def __init__(self, emb_path: Path, ids_path: Path) -> None:
        self.emb_path = Path(emb_path)
        self.ids_path = Path(ids_path)
        self._emb: np.ndarray | None = None
        self._ids: list[int] = []
        self._lock = threading.Lock()

    def available(self) -> bool:
        return self.emb_path.exists() and self.ids_path.exists()

    def _ensure(self) -> None:
        if self._emb is not None:
            return
        with self._lock:
            if self._emb is not None:
                return
            if not self.available():
                raise FileNotFoundError(
                    f"Visual index not found ({self.emb_path}). Run `python -m ml.build_index`."
                )
            # mmap: pages are faulted in on demand, not copied up-front.
            self._emb = np.load(self.emb_path, mmap_mode="r")
            self._ids = json.loads(self.ids_path.read_text())
            if self._emb.shape[0] != len(self._ids):
                raise ValueError(
                    f"Index corrupt: {self._emb.shape[0]} embeddings vs {len(self._ids)} ids"
                )
            log.info("visual index loaded: %d x %d", *self._emb.shape)

    def search(self, embedding: list[float] | np.ndarray, top_k: int = 10) -> list[dict]:
        """Cosine top-k. `embedding` must already be L2-normalised."""
        self._ensure()
        q = np.asarray(embedding, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self._emb.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch: got {q.shape}, index expects {self._emb.shape[1]}"
            )

        # float16 storage -> float32 compute. One matmul over the whole catalogue.
        sims = np.asarray(self._emb, dtype=np.float32) @ q
        k = min(top_k, sims.shape[0])
        # argpartition is O(n); full sort only over the k survivors.
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]

        product_ids = [int(self._ids[i]) for i in top]
        scores = {int(self._ids[i]): float(sims[i]) for i in top}
        return self._hydrate(product_ids, scores)

    def _hydrate(self, product_ids: list[int], scores: dict[int, float]) -> list[dict]:
        """Join ids back to catalogue rows, preserving similarity order."""
        if not product_ids:
            return []
        placeholders = ",".join("?" for _ in product_ids)
        rows = db.query(
            f"SELECT product_id, name, master_category, sub_category, article_type, "
            f"color, gender, season, usage FROM products WHERE product_id IN ({placeholders})",
            product_ids,
        )
        by_id = {r["product_id"]: dict(r) for r in rows}
        out = []
        for rank, pid in enumerate(product_ids, start=1):
            row = by_id.get(pid)
            if row is None:
                continue  # catalogue row deleted since the index was built
            row["rank"] = rank
            row["similarity_score"] = round(scores[pid], 4)
            out.append(row)
        return out

    def info(self) -> dict:
        return {
            "available": self.available(),
            "loaded": self._emb is not None,
            "size": int(self._emb.shape[0]) if self._emb is not None else None,
            "dim": int(self._emb.shape[1]) if self._emb is not None else None,
        }


index = VisualIndex(settings.VISUAL_EMB_PATH, settings.VISUAL_IDS_PATH)
