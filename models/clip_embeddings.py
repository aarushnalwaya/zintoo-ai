"""
╔══════════════════════════════════════════════════════════════╗
║  🤖 ML ENGINEER AGENT — FashionCLIP Embeddings + FAISS      ║
║  Extracts image embeddings with FashionCLIP,                 ║
║  builds a FAISS index for fast similarity search             ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CLIP_MODEL_NAME, EMBEDDING_DIM, BATCH_SIZE,
    FAISS_INDEX_PATH, PRODUCT_MAP_PATH,
    DATASET_DIR, IMAGES_DIR, EMBEDDINGS_DIR,
)


class FashionCLIPEmbedder:
    """
    FashionCLIP-based feature extractor for fashion product images.

    Uses the patrickjohncyh/fashion-clip model which is fine-tuned
    on fashion e-commerce data for superior fashion understanding.
    """

    def __init__(self, model_name=CLIP_MODEL_NAME, device=None):
        print("=" * 60)
        print("🤖 ML ENGINEER AGENT: Loading FashionCLIP Model")
        print("=" * 60)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {self.device}")

        print(f"   Loading model: {model_name}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        print("   ✅ Model loaded successfully!")
        print(f"   Embedding dimension: {EMBEDDING_DIM}")

    @torch.no_grad()
    def encode_images(self, image_paths, batch_size=BATCH_SIZE):
        """
        Encode a list of image paths into embeddings.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (N, EMBEDDING_DIM)
        """
        all_embeddings = []

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i + batch_size]
            images = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                except Exception:
                    # Use a blank image as fallback
                    images.append(Image.new("RGB", (224, 224), (128, 128, 128)))

            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.get_image_features(**inputs)
            # L2 normalize for cosine similarity
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings).astype("float32")

    @torch.no_grad()
    def encode_text(self, texts):
        """
        Encode text queries into embeddings.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (N, EMBEDDING_DIM)
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.get_text_features(**inputs)
        embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype("float32")

    @torch.no_grad()
    def encode_single_image(self, image_input):
        """
        Encode a single image (path or PIL Image).

        Args:
            image_input: str path or PIL.Image

        Returns:
            numpy array of shape (1, EMBEDDING_DIM)
        """
        if isinstance(image_input, str) or isinstance(image_input, Path):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")

        inputs = self.processor(images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.get_image_features(**inputs)
        embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype("float32")


def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast similarity search.

    Uses IndexFlatIP (inner product) since embeddings are L2-normalized,
    making inner product equivalent to cosine similarity.
    """
    print("\n🔍 Building FAISS Index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim for normalized vectors
    index.add(embeddings)
    print(f"   ✅ FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def extract_and_index(catalog_df=None):
    """
    Full pipeline: load catalog → extract embeddings → build FAISS index.

    Returns:
        (faiss_index, product_ids, catalog_df)
    """
    print("\n🚀 Starting Feature Extraction Pipeline\n")

    # Load catalog
    if catalog_df is None:
        catalog_path = DATASET_DIR / "catalog.csv"
        if catalog_path.exists():
            catalog_df = pd.read_csv(catalog_path)
        else:
            # Run preprocessing first
            from data.preprocess import preprocess_pipeline
            catalog_df = preprocess_pipeline()

    print(f"\n📊 Catalog: {len(catalog_df)} products")

    # Initialize embedder
    embedder = FashionCLIPEmbedder()

    # Extract image embeddings
    image_paths = catalog_df["image_path"].tolist()
    print(f"\n🖼️  Extracting embeddings for {len(image_paths)} images...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Estimated time: ~{len(image_paths) // BATCH_SIZE * 2}s on GPU, ~{len(image_paths) // BATCH_SIZE * 10}s on CPU")

    embeddings = embedder.encode_images(image_paths)
    print(f"   ✅ Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save index and product mapping
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"   💾 FAISS index saved to {FAISS_INDEX_PATH}")

    product_map = {
        "product_ids": catalog_df["id"].tolist(),
        "catalog": catalog_df.to_dict("records"),
    }
    with open(PRODUCT_MAP_PATH, "wb") as f:
        pickle.dump(product_map, f)
    print(f"   💾 Product map saved to {PRODUCT_MAP_PATH}")

    # Save raw embeddings too (for analysis)
    np.save(EMBEDDINGS_DIR / "embeddings.npy", embeddings)

    print("\n" + "=" * 60)
    print("✅ Feature extraction and indexing complete!")
    print("=" * 60)

    return index, catalog_df, embedder


if __name__ == "__main__":
    index, catalog, embedder = extract_and_index()

    # Quick test
    print("\n🧪 Quick Test: Text query search")
    query = "casual blue t-shirt for men"
    query_emb = embedder.encode_text([query])
    scores, indices = index.search(query_emb, 5)

    print(f"\n   Query: '{query}'")
    print(f"   Top 5 results:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
        product = catalog.iloc[idx]
        print(f"   {rank+1}. [{score:.4f}] {product['productDisplayName']} ({product['masterCategory']})")
