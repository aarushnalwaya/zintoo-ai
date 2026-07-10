"""
╔══════════════════════════════════════════════════════════════╗
║  🔧 DATA ENGINEER AGENT — Kaggle API Setup                  ║
║  Sets up Kaggle credentials and downloads the dataset        ║
╚══════════════════════════════════════════════════════════════╝

Usage:
  - In Google Colab: Run this file as a cell
  - Locally: python setup_kaggle.py
"""

import os
import sys
import json
import zipfile
from pathlib import Path


def setup_kaggle_colab():
    """Setup Kaggle API in Google Colab environment."""
    print("=" * 60)
    print("🔧 DATA ENGINEER AGENT: Kaggle API Setup")
    print("=" * 60)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # Check if already configured
    if kaggle_json.exists():
        print("✅ kaggle.json already exists at", kaggle_json)
        return True

    # Check if running in Colab
    try:
        from google.colab import files
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        print("\n📁 Please upload your kaggle.json file:")
        print("   (Download from https://www.kaggle.com/settings → API → Create New Token)")
        uploaded = files.upload()

        if "kaggle.json" not in uploaded:
            print("❌ ERROR: No kaggle.json uploaded. Please try again.")
            return False

        # Save to ~/.kaggle/
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json, "wb") as f:
            f.write(uploaded["kaggle.json"])
        os.chmod(kaggle_json, 0o600)
        print(f"✅ kaggle.json saved to {kaggle_json}")

    else:
        # Local environment
        if "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ:
            print("✅ Kaggle credentials found in environment variables")
            return True

        print("\n❌ kaggle.json not found!")
        print("   Option 1: Place kaggle.json at ~/.kaggle/kaggle.json")
        print("   Option 2: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("   Download from: https://www.kaggle.com/settings → API → Create New Token")
        return False

    return True


def download_dataset():
    """Download the Fashion Product Images (Small) dataset."""
    print("\n" + "=" * 60)
    print("📥 DATA ENGINEER AGENT: Downloading Dataset")
    print("=" * 60)

    # Import config
    sys.path.insert(0, str(Path(__file__).parent))
    from config import DATA_DIR, DATASET_DIR, STYLES_CSV, IMAGES_DIR

    # Check if already downloaded
    if STYLES_CSV.exists() and IMAGES_DIR.exists():
        import pandas as pd
        df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
        num_images = len(list(IMAGES_DIR.glob("*.jpg")))
        print(f"✅ Dataset already exists!")
        print(f"   Styles CSV: {len(df)} rows")
        print(f"   Images: {num_images} files")
        return True

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle
        print("📦 Downloading 'fashion-product-images-small' from Kaggle...")
        print("   (This is ~280MB, may take a few minutes)")

        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "paramaggarwal/fashion-product-images-small",
            path=str(DATA_DIR),
            unzip=False,
        )

        # Unzip
        zip_path = DATA_DIR / "fashion-product-images-small.zip"
        if zip_path.exists():
            print("📂 Extracting dataset...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(DATA_DIR)
            zip_path.unlink()  # Remove zip after extraction

        # Verify
        if not IMAGES_DIR.exists():
            # Sometimes the extraction creates a nested folder
            possible = list(DATA_DIR.glob("**/images"))
            if possible:
                actual_root = possible[0].parent
                if actual_root != DATASET_DIR:
                    actual_root.rename(DATASET_DIR)

        if STYLES_CSV.exists():
            import pandas as pd
            df = pd.read_csv(STYLES_CSV, on_bad_lines="skip")
            num_images = len(list(IMAGES_DIR.glob("*.jpg")))
            print(f"\n✅ Dataset downloaded and extracted successfully!")
            print(f"   📊 Styles CSV: {len(df)} products")
            print(f"   🖼️  Images: {num_images} files")
            print(f"   📁 Location: {DATASET_DIR}")
            return True
        else:
            print("❌ styles.csv not found after extraction!")
            print(f"   Contents of {DATA_DIR}:")
            for p in DATA_DIR.rglob("*"):
                if p.is_file():
                    print(f"      {p.relative_to(DATA_DIR)}")
            return False

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Make sure kaggle.json is properly configured")
        print("   2. Check your internet connection")
        print("   3. Try: pip install kaggle --upgrade")
        print(f"\n   Full error: {type(e).__name__}: {e}")
        return False


def main():
    """Main setup flow."""
    if setup_kaggle_colab():
        download_dataset()
    else:
        print("\n⚠️  Cannot download dataset without Kaggle credentials.")
        print("   Please configure Kaggle API and re-run this script.")


if __name__ == "__main__":
    main()
