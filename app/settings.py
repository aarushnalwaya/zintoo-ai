"""
Zintoo AI — Centralised, environment-driven settings.

Everything that differs between local / staging / production is read from
environment variables here, with safe defaults for local development.
Nothing secret is hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Real environment variables always win, so `set FOO=bar` in a shell still
    overrides the file. Implemented with the stdlib — no python-dotenv needed,
    which keeps the serving image lean.
    """
    if not path.is_file():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            # strip surrounding quotes and any trailing inline comment
            if value and value[0] in "\"'" and value[-1] == value[0] and len(value) > 1:
                value = value[1:-1]
            elif " #" in value:
                value = value.split(" #", 1)[0].strip()
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


# Load .env before any getenv() below. Real env vars take precedence.
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "dashboard" / "static"

# On Render free tier the filesystem is ephemeral (resets on redeploy/restart)
# but stable within an instance's lifetime — fine for a self-seeding DB.
# Point ZINTOO_DB_PATH at a mounted disk (paid plan) for durable state.
DATA_DIR = Path(os.getenv("ZINTOO_DATA_DIR", str(BASE_DIR / "runtime")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("ZINTOO_DB_PATH", str(DATA_DIR / "zintoo.db")))


# ─── App / server ─────────────────────────────────────────────────────
APP_NAME = "Zintoo AI Fashion Intelligence"
APP_VERSION = "2.0.0"
ENV = os.getenv("ZINTOO_ENV", "development")  # development | production
IS_PROD = ENV == "production"
LOG_LEVEL = os.getenv("ZINTOO_LOG_LEVEL", "INFO").upper()
# JSON logs are easier to ship to Render/Datadog/Grafana; pretty logs locally.
JSON_LOGS = _bool("ZINTOO_JSON_LOGS", IS_PROD)


# ─── Security ─────────────────────────────────────────────────────────
# HMAC secret used to sign session tokens. MUST be set in production.
# A random per-boot secret is used if unset so tokens simply don't survive a
# restart in dev — never silently insecure.
SECRET_KEY = os.getenv("ZINTOO_SECRET_KEY") or os.urandom(32).hex()
SECRET_KEY_IS_EPHEMERAL = os.getenv("ZINTOO_SECRET_KEY") is None
TOKEN_TTL_SECONDS = int(os.getenv("ZINTOO_TOKEN_TTL", str(60 * 60 * 12)))  # 12h

# CORS. Default: same-origin only (the dashboard is served by this app).
# Set ZINTOO_CORS_ORIGINS="https://a.com,https://b.com" to allow cross-origin.
CORS_ORIGINS = _csv("ZINTOO_CORS_ORIGINS", [])

# Simple per-IP rate limiting (token bucket). Generous defaults.
RATE_LIMIT_ENABLED = _bool("ZINTOO_RATE_LIMIT", True)
RATE_LIMIT_RPS = float(os.getenv("ZINTOO_RATE_LIMIT_RPS", "20"))
RATE_LIMIT_BURST = int(os.getenv("ZINTOO_RATE_LIMIT_BURST", "40"))


# ─── Domain configuration (Mumbai hyper-local demo) ───────────────────
PIN_CODE_COORDS: dict[str, tuple[float, float]] = {
    "400001": (18.9398, 72.8355),  # Fort
    "400002": (18.9535, 72.8336),  # Kalbadevi
    "400003": (18.9592, 72.8308),  # Mandvi
    "400004": (18.9685, 72.8332),  # Girgaon
    "400005": (18.9432, 72.8235),  # Colaba
}
PIN_CODES = list(PIN_CODE_COORDS.keys())

WAREHOUSE_PINCODE_MAP = {
    "W1": "400001",
    "W2": "400002",
    "W3": "400003",
    "W4": "400004",
    "W5": "400005",
}
WAREHOUSE_IDS = list(WAREHOUSE_PINCODE_MAP.keys())

REORDER_THRESHOLD = 10
MAX_STOCK_PER_SKU = 100
TRANSFER_COST_PER_UNIT = 5.0  # INR
SLA_MINUTES = 60

# ─── Weather (Open-Meteo, free, no key) ───────────────────────────────
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT = float(os.getenv("ZINTOO_WEATHER_TIMEOUT", "4.0"))
WEATHER_CACHE_TTL = int(os.getenv("ZINTOO_WEATHER_CACHE_TTL", "900"))  # 15 min
WEATHER_ENABLED = _bool("ZINTOO_WEATHER_ENABLED", True)


# ─── Vision (image classification + visual search) ────────────────────
# Disabled by default: onnxruntime + the model add ~150-250 MB RSS, which is
# tight on a 512 MB free instance. Enable once you've sized the instance.
VISION_ENABLED = _bool("ZINTOO_VISION_ENABLED", False)
MODELS_DIR = Path(os.getenv("ZINTOO_MODELS_DIR", str(BASE_DIR / "models_artifacts")))
VISION_MODEL_PATH = Path(os.getenv("ZINTOO_VISION_MODEL", str(MODELS_DIR / "fashion_classifier.onnx")))
VISION_LABELS_PATH = Path(os.getenv("ZINTOO_VISION_LABELS", str(MODELS_DIR / "labels.json")))
VISUAL_EMB_PATH = Path(os.getenv("ZINTOO_VISUAL_EMB", str(MODELS_DIR / "catalog_embeddings.npy")))
VISUAL_IDS_PATH = Path(os.getenv("ZINTOO_VISUAL_IDS", str(MODELS_DIR / "catalog_ids.json")))
# ORT threads. 1 is correct on small shared vCPU instances.
VISION_THREADS = int(os.getenv("ZINTOO_VISION_THREADS", "1"))
# Preload at startup (slower boot, no first-request latency spike) vs lazy.
VISION_PRELOAD = _bool("ZINTOO_VISION_PRELOAD", False)

# Optional: serve the real product photos at /images/{product_id}.jpg.
# Point this at a folder of {id}.jpg files (the RESIZED cache is ideal, ~300 MB;
# the raw 25 GB set is not). If unset or missing, the dashboard falls back to
# placeholder imagery automatically — nothing breaks.
PRODUCT_IMAGES_DIR = os.getenv("ZINTOO_IMAGES_DIR")
PRODUCT_IMAGES_PATH = Path(PRODUCT_IMAGES_DIR) if PRODUCT_IMAGES_DIR else None


def public_config() -> dict:
    """Non-secret config safe to expose at /health for debugging."""
    return {
        "env": ENV,
        "version": APP_VERSION,
        "db_path": str(DB_PATH),
        "cors_origins": CORS_ORIGINS or "same-origin",
        "rate_limit": RATE_LIMIT_ENABLED,
        "weather_enabled": WEATHER_ENABLED,
        "vision_enabled": VISION_ENABLED,
        "secret_ephemeral": SECRET_KEY_IS_EPHEMERAL,
    }
