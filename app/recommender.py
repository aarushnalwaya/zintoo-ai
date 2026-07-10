"""
Text recommendation — real retrieval within the free-tier budget.

FashionCLIP (~600 MB) + torch + a FAISS index cannot fit in Render's 512 MB
free instance, which is why the original was stripped to a facade. Instead of
faking results with random stock photos, this builds a compact in-memory TF-IDF
index over the seeded catalog and returns genuine ranked matches with cosine
scores. It loads in milliseconds and uses a few MB of RAM.

Retrieval details that matter in practice:
  * Light suffix stemming so singular queries hit plural catalog terms
    ("jacket" -> "jackets", "shoe" -> "shoes", "watch" -> "watches").
  * Per-field term weighting: colour / article type / gender / usage carry more
    signal than free-text description, so "black casual shoes" ranks black ones
    first instead of drowning the colour in nine equally-weighted fields.
  * Gender filtering treats "Unisex" as available to everyone, which is how a
    real catalogue behaves.

Image / multimodal search still needs a GPU-class model; rather than fake it,
those endpoints return an explicit `requires_vision_tier` response so the
frontend can degrade honestly.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict

from . import db
from .observability import get_logger

log = get_logger("zintoo.recommender")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no retrieval signal in a product catalogue.
_STOPWORDS = frozenset(
    "a an the for with and or of in on to by is are my me i want need "
    "show find get some any please looking".split()
)

# Fields weighted by how much they should influence ranking.
_FIELD_WEIGHTS: dict[str, int] = {
    "article_type": 4,
    "color": 4,
    "name": 3,
    "gender": 2,
    "usage": 2,
    "sub_category": 2,
    "master_category": 1,
    "season": 1,
    "description": 1,
}


def _stem(word: str) -> str:
    """Very light, symmetric plural stemmer. Applied to BOTH docs and queries,
    so consistency matters more than linguistic correctness."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"          # accessories -> accessory
    if len(word) > 3 and word.endswith("es") and word[:-2].endswith(("ch", "sh", "x", "s", "z")):
        return word[:-2]                # watches -> watch, dresses -> dress
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]                # jackets -> jacket, shoes -> shoe
    return word


def _tokenize(text: str) -> list[str]:
    return [
        _stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
    ]


class TfidfIndex:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._built = False
        self.docs: list[dict] = []
        self.vectors: list[dict[str, float]] = []
        self.norms: list[float] = []
        self.idf: dict[str, float] = {}

    def _doc_terms(self, doc: dict) -> Counter:
        """Weighted term counts: a term in `color` counts more than in `description`."""
        counts: Counter = Counter()
        for field, weight in _FIELD_WEIGHTS.items():
            for term in _tokenize(str(doc.get(field, ""))):
                counts[term] += weight
        return counts

    def build(self) -> None:
        with self._lock:
            rows = db.query(
                "SELECT product_id, name, master_category, sub_category, "
                "article_type, color, gender, season, usage, description FROM products"
            )
            self.docs = [dict(r) for r in rows]

            doc_counts: list[Counter] = []
            df: Counter = Counter()
            for doc in self.docs:
                counts = self._doc_terms(doc)
                doc_counts.append(counts)
                for term in counts:
                    df[term] += 1

            n = max(1, len(self.docs))
            self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

            self.vectors, self.norms = [], []
            for counts in doc_counts:
                total = sum(counts.values()) or 1
                vec = {t: (w / total) * self.idf.get(t, 0.0) for t, w in counts.items()}
                self.vectors.append(vec)
                self.norms.append(math.sqrt(sum(v * v for v in vec.values())) or 1e-9)

            self._built = True
            log.info(
                "TF-IDF index built over %d products, %d terms", len(self.docs), len(self.idf)
            )

    def ensure(self) -> None:
        if not self._built:
            self.build()

    def _passes_filters(self, doc: dict, filters: dict) -> bool:
        gender = filters.get("gender")
        if gender:
            g = gender.strip().lower()
            dg = doc["gender"].lower()
            # "Unisex" stock is available to any gender request.
            if g in ("men", "women") and dg not in (g, "unisex"):
                return False
            if g == "unisex" and dg != "unisex":
                return False
        cat = filters.get("master_category")
        if cat and doc["master_category"].lower() != cat.strip().lower():
            return False
        return True

    def search(self, query: str, top_k: int, filters: dict | None = None) -> list[dict]:
        self.ensure()
        q_tokens = _tokenize(query or "")
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        total = sum(q_tf.values())
        q_vec = {t: (f / total) * self.idf.get(t, 0.0) for t, f in q_tf.items()}
        # Drop query terms absent from the whole catalogue (idf lookup gave 0.0).
        q_vec = {t: v for t, v in q_vec.items() if v > 0.0}
        if not q_vec:
            return []
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1e-9

        filters = filters or {}
        scored: list[tuple[float, dict]] = []
        for i, doc in enumerate(self.docs):
            if not self._passes_filters(doc, filters):
                continue
            vec = self.vectors[i]
            dot = sum(val * vec.get(term, 0.0) for term, val in q_vec.items())
            if dot <= 0:
                continue
            scored.append((dot / (q_norm * self.norms[i]), doc))

        scored.sort(key=lambda x: (-x[0], x[1]["product_id"]))  # stable ties

        results = []
        for rank, (score, doc) in enumerate(scored[:top_k], start=1):
            results.append(
                {
                    "rank": rank,
                    "product_id": doc["product_id"],
                    "name": doc["name"],
                    "similarity_score": round(float(score), 4),
                    "master_category": doc["master_category"],
                    "sub_category": doc["sub_category"],
                    "article_type": doc["article_type"],
                    "color": doc["color"],
                    "gender": doc["gender"],
                    "season": doc["season"],
                    "usage": doc["usage"],
                    "image_path": "",
                    "description": doc["description"],
                }
            )
        return results


index = TfidfIndex()
