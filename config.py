"""
╔══════════════════════════════════════════════════════════════╗
║          ZINTOO — Central Configuration                     ║
║   AI-Powered Hyper-Local Fashion Intelligence Platform      ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
from pathlib import Path

# ─── Project Paths ────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Dataset paths (after download)
DATASET_DIR = DATA_DIR / "fashion-product-images-small"
STYLES_CSV = DATASET_DIR / "styles.csv"
IMAGES_DIR = DATASET_DIR / "images"

# ─── Model Configuration ─────────────────────────────────────
CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"
EMBEDDING_DIM = 512
BATCH_SIZE = 64
TOP_K_DEFAULT = 10

# FAISS index paths
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "fashion_index.faiss"
PRODUCT_MAP_PATH = EMBEDDINGS_DIR / "product_map.pkl"

# ─── Demand Forecasting ──────────────────────────────────────
FORECAST_HORIZON_HOURS = 24
HISTORY_DAYS = 90

# ─── Hyper-Local Configuration ───────────────────────────────
# Mumbai metropolitan area pin codes
PIN_CODES = ["400001", "400002", "400003", "400004", "400005"]

# Pin code → (latitude, longitude) for weather API
PIN_CODE_COORDS = {
    "400001": (18.9398, 72.8355),   # Fort, Mumbai
    "400002": (18.9535, 72.8336),   # Kalbadevi
    "400003": (18.9592, 72.8308),   # Mandvi
    "400004": (18.9685, 72.8332),   # Girgaon
    "400005": (18.9432, 72.8235),   # Colaba
}

# ─── Warehouse Configuration ─────────────────────────────────
NUM_WAREHOUSES = 5
WAREHOUSE_IDS = [f"W{i+1}" for i in range(NUM_WAREHOUSES)]

# Map warehouses to pin codes
WAREHOUSE_PINCODE_MAP = {
    "W1": "400001",
    "W2": "400002",
    "W3": "400003",
    "W4": "400004",
    "W5": "400005",
}

# Inventory thresholds
REORDER_THRESHOLD = 10
MAX_STOCK_PER_SKU = 100
TRANSFER_COST_PER_UNIT = 5.0  # INR
SLA_MINUTES = 60  # 60-minute delivery SLA

# ─── Sample SKUs for Forecasting ─────────────────────────────
# We'll pick top-selling SKU categories for forecasting
NUM_SKUS_FOR_FORECAST = 20

# ─── API Configuration ───────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Weather API ──────────────────────────────────────────────
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ─── Evaluation ───────────────────────────────────────────────
EVAL_K_VALUES = [5, 10, 20]

# Ensure output directories exist
for d in [EMBEDDINGS_DIR, FORECASTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
