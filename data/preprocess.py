"""
╔══════════════════════════════════════════════════════════════╗
║  🔧 DATA ENGINEER AGENT — Data Preprocessing                ║
║  Cleans styles.csv, validates images, builds product catalog ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STYLES_CSV, IMAGES_DIR, DATASET_DIR


def load_and_clean_styles(csv_path=None):
    """
    Load styles.csv and clean it.

    Returns:
        pd.DataFrame with columns:
        [id, gender, masterCategory, subCategory, articleType,
         baseColour, season, year, usage, productDisplayName, image_path]
    """
    csv_path = csv_path or STYLES_CSV

    print("=" * 60)
    print("🔧 DATA ENGINEER AGENT: Preprocessing Data")
    print("=" * 60)

    # Load CSV (skip bad lines)
    print(f"\n📊 Loading styles.csv from {csv_path}...")
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"   Raw rows: {len(df)}")

    # ── Column cleanup ──
    expected_cols = [
        "id", "gender", "masterCategory", "subCategory",
        "articleType", "baseColour", "season", "year",
        "usage", "productDisplayName",
    ]
    # Keep only expected columns (some CSVs have extra)
    available = [c for c in expected_cols if c in df.columns]
    df = df[available].copy()

    # ── Drop rows with missing critical fields ──
    critical = ["id", "masterCategory", "subCategory", "articleType", "productDisplayName"]
    critical = [c for c in critical if c in df.columns]
    before = len(df)
    df.dropna(subset=critical, inplace=True)
    print(f"   After dropping missing critical fields: {len(df)} (removed {before - len(df)})")

    # ── Fix dtypes ──
    df["id"] = df["id"].astype(int)
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)

    # ── Fill remaining NaN ──
    for col in ["gender", "baseColour", "season", "usage"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # ── Add image path ──
    df["image_path"] = df["id"].apply(lambda x: str(IMAGES_DIR / f"{x}.jpg"))

    return df


def validate_images(df, images_dir=None):
    """
    Check which images actually exist and are valid.

    Returns:
        Filtered DataFrame with only valid images.
    """
    images_dir = images_dir or IMAGES_DIR

    print(f"\n🖼️  Validating images in {images_dir}...")

    valid_mask = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        img_path = Path(row["image_path"])
        if img_path.exists():
            try:
                with Image.open(img_path) as img:
                    img.verify()
                valid_mask.append(True)
            except Exception:
                valid_mask.append(False)
        else:
            valid_mask.append(False)

    df_valid = df[valid_mask].reset_index(drop=True)
    print(f"   Valid images: {len(df_valid)} / {len(df)}")

    return df_valid


def build_product_catalog(df):
    """
    Build a rich product catalog from the cleaned DataFrame.
    Adds a 'description' field combining all metadata.
    """
    print("\n📦 Building product catalog...")

    # Create a rich text description for each product
    def make_description(row):
        parts = []
        if row.get("productDisplayName"):
            parts.append(str(row["productDisplayName"]))
        if row.get("gender") and row["gender"] != "Unknown":
            parts.append(f"for {row['gender']}")
        if row.get("baseColour") and row["baseColour"] != "Unknown":
            parts.append(f"in {row['baseColour']}")
        if row.get("season") and row["season"] != "Unknown":
            parts.append(f"({row['season']} season)")
        if row.get("usage") and row["usage"] != "Unknown":
            parts.append(f"- {row['usage']} wear")
        return " ".join(parts)

    df["description"] = df.apply(make_description, axis=1)

    # Add a category hierarchy string
    df["category_path"] = (
        df["masterCategory"].astype(str) + " > " +
        df["subCategory"].astype(str) + " > " +
        df["articleType"].astype(str)
    )

    print(f"   ✅ Catalog built: {len(df)} products")
    print(f"\n   📊 Category Distribution:")
    print(df["masterCategory"].value_counts().head(10).to_string())
    print(f"\n   🎨 Top Colors:")
    print(df["baseColour"].value_counts().head(10).to_string())

    return df


def preprocess_pipeline():
    """Run the full preprocessing pipeline."""
    # Load and clean
    df = load_and_clean_styles()

    # Validate images (can be slow, skip if needed)
    # For faster setup, we just check file existence
    print(f"\n🖼️  Quick image check (file existence only)...")
    df["image_exists"] = df["image_path"].apply(lambda p: Path(p).exists())
    n_missing = (~df["image_exists"]).sum()
    print(f"   Missing images: {n_missing}")
    df = df[df["image_exists"]].drop(columns=["image_exists"]).reset_index(drop=True)

    # Build catalog
    df = build_product_catalog(df)

    # Save processed catalog
    output_path = DATASET_DIR / "catalog.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved processed catalog to {output_path}")
    print(f"   Total products: {len(df)}")

    return df


if __name__ == "__main__":
    catalog = preprocess_pipeline()
    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)
    print(f"\nSample products:")
    print(catalog[["id", "productDisplayName", "masterCategory", "baseColour"]].head(10).to_string())
