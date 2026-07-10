"""
Zintoo AI — FastAPI application (production build).

Wires together persistence, auth, real-time (SSE + WebSocket), the working
recommendation/forecast/orchestration endpoints, observability, and static
dashboard serving. Backward-compatible with the original frontend routes,
plus new /events (SSE), /ws (WebSocket), /metrics, /readiness, and auth routes.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import (
    Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import (
    auth, db, forecast, orchestrator, recommender, seed, security, settings,
    vision, visual_index,
)
from .vision_preprocess import ImageDecodeError
from .events import bus, sse_stream
from .middleware import RequestContextMiddleware, unhandled_exception_handler
from .observability import configure_logging, get_logger, metrics

configure_logging()
log = get_logger("zintoo.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("starting %s v%s (env=%s)", settings.APP_NAME, settings.APP_VERSION, settings.ENV)
    if settings.IS_PROD and settings.SECRET_KEY_IS_EPHEMERAL:
        log.warning("ZINTOO_SECRET_KEY not set in production — tokens won't survive restarts.")
    # Seed + build index in a worker thread so startup stays responsive.
    await asyncio.to_thread(seed.seed_if_empty)
    await asyncio.to_thread(recommender.index.build)
    if settings.VISION_ENABLED and settings.VISION_PRELOAD and vision.model.available():
        # Pay the ~1s session-build + warmup cost at boot instead of on the
        # first client request. Non-fatal if it fails.
        try:
            await asyncio.to_thread(vision.model._ensure_session)
        except Exception as exc:  # noqa: BLE001
            log.warning("vision preload failed: %s", exc)
    log.info("startup complete — ready to serve")
    yield
    log.info("shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(RequestContextMiddleware)
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
app.add_exception_handler(Exception, unhandled_exception_handler)


# ─── Schemas ──────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


class RecommendRequest(BaseModel):
    text_query: str | None = Field(None)
    top_k: int = Field(10, ge=1, le=50)
    gender_filter: str | None = None
    category_filter: str | None = None


# ─── Health / readiness / metrics ─────────────────────────────────────
@app.get("/health")
async def health():
    """Liveness + subsystem status reflecting REAL state."""
    try:
        modules = {
            "recommendation": db.table_count("products") > 0,
            "forecasting": db.table_count("demand_history") > 0,
            "orchestration": db.table_count("inventory") > 0,
        }
        ok = all(modules.values())
    except Exception:  # noqa: BLE001
        modules, ok = {"recommendation": False, "forecasting": False, "orchestration": False}, False
    return {
        "status": "healthy" if ok else "degraded",
        "modules": modules,
        "realtime": {"subscribers": bus.subscriber_count},
        "config": settings.public_config(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/readiness")
async def readiness():
    try:
        db.query_one("SELECT 1")
        return {"ready": True}
    except Exception:  # noqa: BLE001
        return JSONResponse(status_code=503, content={"ready": False})


@app.get("/metrics")
async def metrics_endpoint():
    return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")


# ─── Auth ─────────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(req: LoginRequest, request: Request):
    security.check_rate_limit(request)
    row = db.query_one("SELECT * FROM users WHERE email = ?", (req.email.lower().strip(),))
    if not row or not auth.verify_password(req.password, row["password_hash"], row["salt"]):
        metrics.inc("login_failed_total")
        # Same message for both cases -> no user enumeration.
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = auth.issue_token(row["email"], row["role"])
    metrics.inc("login_success_total")
    await bus.publish("system.login", {"message": f"{row['name']} signed in", "role": row["role"]})
    return {
        "success": True,
        "token": token,
        "user": {"name": row["name"], "role": row["role"], "email": row["email"]},
    }


@app.get("/api/me")
async def me(user: dict = Depends(security.current_user)):
    return {"email": user["sub"], "role": user["role"], "expires": user["exp"]}


# ─── Recommendations ──────────────────────────────────────────────────
@app.post("/recommend")
async def recommend(req: RecommendRequest, request: Request):
    security.check_rate_limit(request)
    if not req.text_query:
        raise HTTPException(status_code=422, detail="text_query is required")
    filters = {}
    if req.gender_filter:
        filters["gender"] = req.gender_filter
    if req.category_filter:
        filters["master_category"] = req.category_filter
    results = await asyncio.to_thread(recommender.index.search, req.text_query, req.top_k, filters)
    metrics.inc("recommend_requests_total")
    await bus.publish("search.query", {"message": f'Search: "{req.text_query}" → {len(results)} results'})
    return {"query": req.text_query, "mode": "text", "results": results, "total_results": len(results)}


@app.post("/recommend/image")
async def recommend_image(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(10, ge=1, le=50),
    text_query: str | None = Query(None, description="Optional text to fuse with the image"),
    alpha: float = Query(0.5, ge=0.0, le=1.0, description="Weight of the image vs text signal"),
):
    """Classify an uploaded garment and return visually similar catalogue items.

    Pipeline: decode -> ONNX (multi-head classify + embedding) -> cosine search.
    Runs in roughly 20-60 ms end-to-end on one CPU core.
    """
    security.check_rate_limit(request)

    if not settings.VISION_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Vision is disabled on this instance. Set ZINTOO_VISION_ENABLED=true "
                   "(needs ~250 MB RAM and a trained model). See VISION.md.",
        )
    if not vision.model.available():
        raise HTTPException(
            status_code=503,
            detail="Vision model artifact not found. Train and export it (see VISION.md).",
        )

    data = await file.read()
    try:
        result = await asyncio.to_thread(vision.model.infer, data, 3)
    except ImageDecodeError as exc:
        # The client's fault, not ours -> 400, with the actual reason.
        raise HTTPException(status_code=400, detail=str(exc))

    similar: list[dict] = []
    if result["embedding"] and visual_index.index.available():
        try:
            similar = await asyncio.to_thread(
                visual_index.index.search, result["embedding"], top_k
            )
        except Exception as exc:  # noqa: BLE001 — search failure shouldn't kill classification
            log.warning("visual search failed: %s", exc)

    # Optional multimodal fusion: blend visual neighbours with text matches.
    mode = "image"
    if text_query:
        mode = "multimodal"
        text_hits = await asyncio.to_thread(recommender.index.search, text_query, top_k * 2, None)
        similar = _fuse(similar, text_hits, alpha, top_k)

    metrics.inc("vision_requests_total")
    top = result["predictions"].get("articleType", [{}])[0]
    await bus.publish("vision.classify", {
        "message": f"Image classified: {top.get('label', '?')} "
                   f"({top.get('confidence', 0):.0%}) in {result['timing_ms']['total']:.0f} ms",
        "prediction": top,
        "latency_ms": result["timing_ms"]["total"],
    })

    return {
        "query": text_query or f"[Image: {file.filename}]",
        "mode": mode,
        "predictions": result["predictions"],
        "timing_ms": result["timing_ms"],
        "results": similar,
        "total_results": len(similar),
    }


def _fuse(visual: list[dict], textual: list[dict], alpha: float, top_k: int) -> list[dict]:
    """Weighted score fusion over product_id. alpha=1 -> pure image, 0 -> pure text."""
    scores: dict[int, float] = {}
    rows: dict[int, dict] = {}
    for r in visual:
        pid = r["product_id"]
        scores[pid] = scores.get(pid, 0.0) + alpha * r["similarity_score"]
        rows[pid] = r
    for r in textual:
        pid = r["product_id"]
        scores[pid] = scores.get(pid, 0.0) + (1 - alpha) * r["similarity_score"]
        rows.setdefault(pid, r)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
    out = []
    for rank, (pid, score) in enumerate(ranked, start=1):
        row = dict(rows[pid])
        row["rank"] = rank
        row["similarity_score"] = round(score, 4)
        out.append(row)
    return out


@app.get("/vision/health")
async def vision_health():
    """Introspect the vision subsystem: artifact presence, labels, metrics."""
    return {
        "enabled": settings.VISION_ENABLED,
        "model": vision.model.info(),
        "visual_index": visual_index.index.info(),
    }


@app.get("/catalog")
async def catalog(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
    rows = db.query(
        "SELECT product_id, name, master_category, sub_category, article_type, color, "
        "gender, season, usage FROM products ORDER BY product_id LIMIT ? OFFSET ?",
        (limit, offset),
    )
    total = db.table_count("products")
    return {"total": total, "limit": limit, "offset": offset, "products": [dict(r) for r in rows]}


# ─── Forecasting ──────────────────────────────────────────────────────
@app.get("/forecast/{sku}/{pincode}")
async def get_forecast(sku: str, pincode: str, hours: int = Query(24, ge=1, le=168)):
    try:
        result = await asyncio.to_thread(forecast.forecast, sku, pincode, hours)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    metrics.inc("forecast_requests_total")
    await bus.publish("forecast.run", {
        "message": f"Forecast {sku}@{pincode}: {result['predicted_total_demand']} units / {hours}h",
    })
    return result


@app.get("/skus")
async def list_skus():
    return {"skus": forecast.available_skus(), "pincodes": forecast.available_pincodes()}


# ─── Inventory / orchestration ────────────────────────────────────────
@app.get("/inventory/summary")
async def inventory_summary():
    return {
        "warehouses": orchestrator.inventory_summary(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/inventory/stock")
async def inventory_stock():
    return {"stock": orchestrator.stock_matrix()}


@app.get("/inventory/transfers")
async def inventory_transfers(limit: int = Query(25, ge=1, le=100)):
    return {"transfers": orchestrator.recent_transfers(limit)}


@app.post("/orchestrate")
async def orchestrate(user: dict = Depends(security.require_role("Owner", "Viewer"))):
    """Run a rebalancing cycle. Requires auth (was previously unprotected)."""
    report = await orchestrator.run_cycle()
    return report


@app.get("/orchestrate/stats")
async def orchestrate_stats():
    return orchestrator.run_stats()


# ─── Real-time: SSE + WebSocket ───────────────────────────────────────
@app.get("/events")
async def events(request: Request):
    """Server-Sent Events stream of live domain events."""
    queue = await bus.subscribe()
    metrics.inc("sse_connections_total")
    return StreamingResponse(
        sse_stream(request, queue),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (Render/nginx)
        },
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue = await bus.subscribe()
    metrics.inc("ws_connections_total")
    try:
        for event in bus.recent(10):
            await ws.send_text(event.to_json())
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
                await ws.send_text(event.to_json())
            except asyncio.TimeoutError:
                await ws.send_text('{"type":"heartbeat"}')
    except WebSocketDisconnect:
        pass
    finally:
        await bus.unsubscribe(queue)


# ─── Static dashboard ─────────────────────────────────────────────────
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Optional: real product photos at /images/{product_id}.jpg. The dashboard
# requests these and falls back to placeholders on 404, so this is purely an
# upgrade — never a dependency.
if settings.PRODUCT_IMAGES_PATH and settings.PRODUCT_IMAGES_PATH.is_dir():
    app.mount("/images", StaticFiles(directory=str(settings.PRODUCT_IMAGES_PATH)), name="images")
    log.info("serving product images from %s", settings.PRODUCT_IMAGES_PATH)
else:
    @app.get("/images/{filename}", include_in_schema=False)
    async def _no_images(filename: str):
        # Answer fast and quietly; the frontend's onerror swaps in a placeholder.
        return Response(status_code=404)


@app.get("/", include_in_schema=False)
async def dashboard():
    index = settings.STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "Zintoo AI API", "docs": "/api/docs"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
