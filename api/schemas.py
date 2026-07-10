"""
╔══════════════════════════════════════════════════════════════╗
║  🌐 BACKEND AGENT — Pydantic Schemas                        ║
║  Request and response models for the FastAPI backend         ║
╚══════════════════════════════════════════════════════════════╝
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ─── Recommendation Schemas ──────────────────────────────────

class RecommendRequest(BaseModel):
    text_query: Optional[str] = Field(None, description="Natural language query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Text weight for multimodal (1-alpha for image)")
    gender_filter: Optional[str] = None
    category_filter: Optional[str] = None


class ProductCard(BaseModel):
    rank: int
    product_id: int
    name: str
    similarity_score: float
    master_category: str = ""
    sub_category: str = ""
    article_type: str = ""
    color: str = ""
    gender: str = ""
    season: str = ""
    usage: str = ""
    image_path: str = ""
    description: str = ""


class RecommendResponse(BaseModel):
    query: str
    mode: str  # "text", "image", "multimodal"
    results: List[ProductCard]
    total_results: int


# ─── Forecast Schemas ────────────────────────────────────────

class HourlyForecast(BaseModel):
    timestamp: str
    predicted_demand: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    sku: str
    pincode: str
    hours: int
    predicted_total_demand: int
    peak_hour_demand: int
    peak_hour: Optional[str] = None
    hourly_forecast: Optional[List[dict]] = None
    metrics: Optional[dict] = None
    source: str = "prophet"


# ─── Orchestration Schemas ───────────────────────────────────

class TransferRecord(BaseModel):
    sku: str
    from_warehouse: str
    to_warehouse: str
    quantity: int
    priority: str = "medium"
    reason: str = ""
    success: bool = True


class OrchestrationResponse(BaseModel):
    status: str
    transfers: List[dict]
    decision_log: List[dict]
    reflection: Optional[dict] = None


# ─── Health Check ────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    modules: dict
    timestamp: str
