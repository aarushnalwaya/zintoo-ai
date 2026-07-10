"""
╔══════════════════════════════════════════════════════════════╗
║  🌐 BACKEND AGENT — FastAPI Application                     ║
║  Unified API for recommendations, forecasting,               ║
║  and inventory orchestration                                 ║
║  + Dashboard static file serving                             ║
╚══════════════════════════════════════════════════════════════╝

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import io
import traceback
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.schemas import (
    RecommendRequest, RecommendResponse, ProductCard,
    ForecastResponse, OrchestrationResponse, HealthResponse,
)

# ─── App Setup ────────────────────────────────────────────────

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard" / "static"

app = FastAPI(
    title="Zintoo AI Fashion Intelligence API",
    description="AI-Powered Hyper-Local Fashion Intelligence Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Dashboard Static Files ──────────────────────────────────

app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")

# Serve product images from dataset (if downloaded)
IMAGES_DIR = Path(__file__).parent.parent / "data" / "fashion-product-images-small" / "images"
if IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.get("/images/{image_name}", include_in_schema=False)
async def serve_product_image(image_name: str):
    """Serve product images — returns placeholder if not found."""
    img_path = IMAGES_DIR / image_name
    if img_path.exists():
        return FileResponse(str(img_path))
    # Return a 404 so the frontend can use its CSS fallback
    raise HTTPException(status_code=404, detail="Image not found")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serve the dashboard SPA."""
    return FileResponse(str(DASHBOARD_DIR / "index.html"))


# ─── Authentication ──────────────────────────────────────────

import hashlib
import secrets
from pydantic import BaseModel


class LoginRequest(BaseModel):
    email: str
    password: str


# Demo users (for presentation purposes)
DEMO_USERS = {
    "admin@zintoo.ai": {
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "name": "System Admin",
        "role": "Owner",
    },
    "demo@zintoo.ai": {
        "password_hash": hashlib.sha256("demo123".encode()).hexdigest(),
        "name": "Demo User",
        "role": "Viewer",
    },
}


@app.post("/api/login")
async def login(req: LoginRequest):
    """Authenticate user and return token."""
    user = DEMO_USERS.get(req.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email address")

    pwd_hash = hashlib.sha256(req.password.encode()).hexdigest()
    if pwd_hash != user["password_hash"]:
        raise HTTPException(status_code=401, detail="Incorrect password")

    token = secrets.token_hex(32)
    return {
        "success": True,
        "token": token,
        "user": {"name": user["name"], "role": user["role"], "email": req.email},
    }


# ─── Lazy-loaded modules ─────────────────────────────────────

_rec_engine = None
_forecaster = None
_toolkit = None


def get_recommendation_engine():
    global _rec_engine
    if _rec_engine is None:
        from models.recommendation import RecommendationEngine
        _rec_engine = RecommendationEngine()
    return _rec_engine


def get_forecaster():
    global _forecaster
    if _forecaster is None:
        from models.demand_forecast import DemandForecaster
        _forecaster = DemandForecaster()
    return _forecaster


def get_toolkit():
    global _toolkit
    if _toolkit is None:
        from agents.tools import InventoryToolkit
        _toolkit = InventoryToolkit()
    return _toolkit


# ─── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and module availability."""
    try:
        from config import FAISS_INDEX_PATH, DATA_DIR
        modules = {
            "recommendation": FAISS_INDEX_PATH.exists(),
            "forecasting": (DATA_DIR / "demand_history.csv").exists(),
            "orchestration": (DATA_DIR / "warehouse_inventory.csv").exists(),
        }
    except Exception:
        modules = {
            "recommendation": False,
            "forecasting": False,
            "orchestration": False,
        }

    return HealthResponse(
        status="healthy" if all(modules.values()) else "partial",
        modules=modules,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Get fashion product recommendations.

    Supports text-only queries. For image upload, use /recommend/image.
    """
    try:
        engine = get_recommendation_engine()

        filters = {}
        if request.gender_filter:
            filters["gender"] = request.gender_filter
        if request.category_filter:
            filters["masterCategory"] = request.category_filter

        results = engine.recommend_by_text(
            query=request.text_query,
            top_k=request.top_k,
            filters=filters if filters else None,
        )

        return RecommendResponse(
            query=request.text_query or "",
            mode="text",
            results=[ProductCard(**r) for r in results],
            total_results=len(results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend/image", response_model=RecommendResponse)
async def recommend_by_image(
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=50),
    text_query: str = Query(None),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
):
    """
    Get recommendations from an uploaded image (optionally with text).
    """
    try:
        engine = get_recommendation_engine()

        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if text_query:
            # Multimodal
            results = engine.recommend_multimodal(
                text=text_query,
                image_input=image,
                alpha=alpha,
                top_k=top_k,
            )
            mode = "multimodal"
        else:
            # Image only
            results = engine.recommend_by_image(
                image_input=image,
                top_k=top_k,
            )
            mode = "image"

        return RecommendResponse(
            query=text_query or f"[Image: {file.filename}]",
            mode=mode,
            results=[ProductCard(**r) for r in results],
            total_results=len(results),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/{sku}/{pincode}", response_model=ForecastResponse)
async def get_forecast(
    sku: str,
    pincode: str,
    hours: int = Query(24, ge=1, le=168),
):
    """
    Get demand forecast for a SKU at a specific pin code.
    """
    try:
        forecaster = get_forecaster()
        forecast_df, metrics = forecaster.get_demand_prediction(
            sku=sku,
            pincode=pincode,
            hours=hours,
        )

        return ForecastResponse(
            sku=sku,
            pincode=pincode,
            hours=hours,
            predicted_total_demand=int(forecast_df["predicted_demand"].sum()),
            peak_hour_demand=int(forecast_df["predicted_demand"].max()),
            peak_hour=str(forecast_df.loc[forecast_df["predicted_demand"].idxmax(), "timestamp"]),
            hourly_forecast=forecast_df.to_dict("records"),
            metrics=metrics,
            source="prophet",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orchestrate", response_model=OrchestrationResponse)
async def run_orchestration():
    """
    Trigger an autonomous inventory orchestration cycle.
    """
    try:
        from agents.inventory_agent import InventoryOrchestrationAgent

        toolkit = get_toolkit()
        agent = InventoryOrchestrationAgent(toolkit=toolkit)
        report = agent.run_orchestration_cycle()

        return OrchestrationResponse(
            status=report["status"],
            transfers=report.get("transfers", []),
            decision_log=report.get("decision_log", []),
            reflection=report.get("reflection"),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/summary")
async def inventory_summary():
    """Get current inventory summary across all warehouses."""
    try:
        toolkit = get_toolkit()
        summary = toolkit.get_inventory_summary()
        return JSONResponse(content={
            "warehouses": summary.to_dict("records"),
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/skus")
async def list_skus():
    """List available SKUs."""
    try:
        forecaster = get_forecaster()
        return {"skus": forecaster.skus, "pincodes": forecaster.pincodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
