"""Training-side configuration. Kept separate from `app.settings` (serving)."""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where you unzipped the Kaggle dataset. Expect:
#   $DATASET_DIR/styles.csv
#   $DATASET_DIR/images/{id}.jpg
DATASET_DIR = Path(os.getenv("ZINTOO_DATASET_DIR", str(PROJECT_ROOT / "data" / "fashion-dataset")))
STYLES_CSV = DATASET_DIR / "styles.csv"
IMAGES_DIR = DATASET_DIR / "images"

ARTIFACTS_DIR = Path(os.getenv("ZINTOO_MODELS_DIR", str(PROJECT_ROOT / "models_artifacts")))
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"

# Multi-head targets. articleType is the primary task (used for early stopping).
TASKS = ["articleType", "baseColour", "gender", "masterCategory"]
PRIMARY_TASK = "articleType"

# A class must have at least this many examples to be learnable. The dataset's
# articleType has a brutal long tail (some classes have a single image).
MIN_SAMPLES_PER_CLASS = {
    "articleType": 50,
    "baseColour": 50,
    "gender": 20,
    "masterCategory": 20,
}

IMAGE_SIZE = 224
EMBEDDING_DIM = 256
BATCH_SIZE = int(os.getenv("ZINTOO_BATCH_SIZE", "64"))
EPOCHS = int(os.getenv("ZINTOO_EPOCHS", "12"))
LR = float(os.getenv("ZINTOO_LR", "3e-4"))
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
SEED = 42
VAL_FRAC, TEST_FRAC = 0.10, 0.10
BACKBONE = os.getenv("ZINTOO_BACKBONE", "mobilenet_v3_small")  # or efficientnet_b0
NUM_WORKERS = int(os.getenv("ZINTOO_NUM_WORKERS", "4"))
