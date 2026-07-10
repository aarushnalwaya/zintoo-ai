"""
╔══════════════════════════════════════════════════════════════╗
║  🤖 ML ENGINEER AGENT — Recommendation Engine               ║
║  Text, image, and multimodal product recommendations         ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import faiss
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH, PRODUCT_MAP_PATH, EMBEDDINGS_DIR, TOP_K_DEFAULT,
)


class RecommendationEngine:
    """
    Multimodal fashion recommendation engine.

    Supports three modes:
    1. Text query → find products matching description
    2. Image query → find visually similar products
    3. Multimodal → weighted fusion of text + image
    """

    def __init__(self, embedder=None):
        print("=" * 60)
        print("🤖 ML ENGINEER AGENT: Initializing Recommendation Engine")
        print("=" * 60)

        # Load FAISS index
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run clip_embeddings.py first to build the index."
            )

        self.index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"   ✅ FAISS index loaded: {self.index.ntotal} vectors")

        # Load product map
        with open(PRODUCT_MAP_PATH, "rb") as f:
            product_map = pickle.load(f)

        self.product_ids = product_map["product_ids"]
        self.catalog = pd.DataFrame(product_map["catalog"])
        print(f"   ✅ Product catalog loaded: {len(self.catalog)} items")

        # Load or create embedder
        if embedder is not None:
            self.embedder = embedder
        else:
            from models.clip_embeddings import FashionCLIPEmbedder
            self.embedder = FashionCLIPEmbedder()

        print("   ✅ Recommendation engine ready!")

    def recommend_by_text(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        """
        Find products matching a text description.

        Args:
            query: Natural language query (e.g., "casual kurta for college fest")
            top_k: Number of results to return
            filters: Optional filters like {"gender": "Men", "masterCategory": "Apparel"}

        Returns:
            List of product dicts with similarity scores
        """
        # Encode query
        query_embedding = self.embedder.encode_text([query])

        # Search
        scores, indices = self.index.search(query_embedding, top_k * 3)  # Over-fetch for filtering

        results = self._format_results(scores[0], indices[0], filters, top_k)
        return results

    def recommend_by_image(
        self,
        image_input: Union[str, Path, Image.Image],
        top_k: int = TOP_K_DEFAULT,
        filters: Optional[dict] = None,
        exclude_self: bool = True,
    ) -> List[dict]:
        """
        Find visually similar products.

        Args:
            image_input: Image path, URL, or PIL Image
            top_k: Number of results
            filters: Optional category filters
            exclude_self: Skip exact match (useful when querying catalog image)

        Returns:
            List of product dicts with similarity scores
        """
        # Encode image
        image_embedding = self.embedder.encode_single_image(image_input)

        # Search (over-fetch for filtering + self-exclusion)
        scores, indices = self.index.search(image_embedding, top_k * 3 + 1)

        results = self._format_results(
            scores[0], indices[0], filters, top_k,
            exclude_score_threshold=0.999 if exclude_self else None
        )
        return results

    def recommend_multimodal(
        self,
        text: Optional[str] = None,
        image_input: Optional[Union[str, Path, Image.Image]] = None,
        alpha: float = 0.5,
        top_k: int = TOP_K_DEFAULT,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        """
        Multimodal recommendation: weighted fusion of text + image embeddings.

        Args:
            text: Text query
            image_input: Image input
            alpha: Weight for text (1-alpha for image). Default 0.5 = equal weight.
            top_k: Number of results
            filters: Optional filters

        Returns:
            List of product dicts with similarity scores
        """
        embeddings = []

        if text is not None:
            text_emb = self.embedder.encode_text([text])
            embeddings.append((alpha, text_emb))

        if image_input is not None:
            img_emb = self.embedder.encode_single_image(image_input)
            embeddings.append((1 - alpha, img_emb))

        if not embeddings:
            raise ValueError("At least one of text or image_input must be provided")

        # Weighted fusion
        if len(embeddings) == 1:
            combined = embeddings[0][1]
        else:
            combined = sum(w * e for w, e in embeddings)
            # Re-normalize
            combined = combined / np.linalg.norm(combined, axis=1, keepdims=True)

        combined = combined.astype("float32")

        # Search
        scores, indices = self.index.search(combined, top_k * 3)

        results = self._format_results(scores[0], indices[0], filters, top_k)
        return results

    def _format_results(self, scores, indices, filters, top_k, exclude_score_threshold=None):
        """Format search results into product dicts."""
        results = []

        for idx, score in zip(indices, scores):
            if idx < 0 or idx >= len(self.catalog):
                continue

            # Skip near-exact matches (self)
            if exclude_score_threshold and score > exclude_score_threshold:
                continue

            product = self.catalog.iloc[idx]

            # Apply filters
            if filters:
                skip = False
                for key, value in filters.items():
                    if key in product and str(product[key]).lower() != str(value).lower():
                        skip = True
                        break
                if skip:
                    continue

            results.append({
                "rank": len(results) + 1,
                "product_id": int(product["id"]),
                "name": product["productDisplayName"],
                "similarity_score": round(float(score), 4),
                "master_category": product.get("masterCategory", ""),
                "sub_category": product.get("subCategory", ""),
                "article_type": product.get("articleType", ""),
                "color": product.get("baseColour", ""),
                "gender": product.get("gender", ""),
                "season": product.get("season", ""),
                "usage": product.get("usage", ""),
                "image_path": product.get("image_path", ""),
                "description": product.get("description", ""),
            })

            if len(results) >= top_k:
                break

        return results

    def get_product_by_id(self, product_id: int) -> Optional[dict]:
        """Get a single product by ID."""
        match = self.catalog[self.catalog["id"] == product_id]
        if len(match) > 0:
            return match.iloc[0].to_dict()
        return None


def demo_recommendations():
    """Demo the recommendation engine with example queries."""
    engine = RecommendationEngine()

    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATION DEMO")
    print("=" * 60)

    # Text queries
    queries = [
        "casual kurta for a college fest",
        "formal black shoes for office",
        "summer floral dress for women",
        "sporty running shoes",
        "ethnic wear for wedding",
    ]

    for query in queries:
        print(f"\n📝 Query: '{query}'")
        print("-" * 50)
        results = engine.recommend_by_text(query, top_k=5)
        for r in results:
            print(f"   {r['rank']}. [{r['similarity_score']:.4f}] {r['name']}")
            print(f"      Category: {r['master_category']} > {r['article_type']}")
            print(f"      Color: {r['color']} | Gender: {r['gender']}")

    # Image query (use first catalog image)
    print(f"\n\n🖼️  Image Query: Using first product image")
    print("-" * 50)
    first_product = engine.catalog.iloc[0]
    print(f"   Source: {first_product['productDisplayName']}")
    results = engine.recommend_by_image(first_product["image_path"], top_k=5)
    for r in results:
        print(f"   {r['rank']}. [{r['similarity_score']:.4f}] {r['name']}")
        print(f"      Category: {r['master_category']} > {r['article_type']}")

    # Multimodal query
    print(f"\n\n🔀 Multimodal Query: Text + Image")
    print("-" * 50)
    results = engine.recommend_multimodal(
        text="blue casual shirt",
        image_input=first_product["image_path"],
        alpha=0.6,
        top_k=5,
    )
    for r in results:
        print(f"   {r['rank']}. [{r['similarity_score']:.4f}] {r['name']}")
        print(f"      Category: {r['master_category']} > {r['article_type']}")

    return engine


if __name__ == "__main__":
    demo_recommendations()
