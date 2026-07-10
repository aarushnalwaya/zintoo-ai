"""
FastAPI middleware and the global exception handler.

Kept separate from `app.observability` so that logging/metrics stay
framework-agnostic and testable without importing FastAPI.
"""

from __future__ import annotations

import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .observability import get_logger, metrics, request_id_ctx

log = get_logger("zintoo.access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request, record metrics, add headers."""

    _QUIET_PREFIXES = ("/health", "/readiness", "/metrics", "/events", "/ws", "/images", "/static")

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_ctx.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - start
            metrics.inc("requests_total")
            metrics.inc("requests_errors_total")
            log.exception("unhandled error %s %s (%.1fms)", request.method, request.url.path, elapsed * 1000)
            request_id_ctx.reset(token)
            raise
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        metrics.inc("requests_total")
        metrics.observe_latency(route_path, elapsed)
        if response.status_code >= 500:
            metrics.inc("requests_errors_total")
        response.headers["x-request-id"] = rid
        response.headers["x-response-time-ms"] = f"{elapsed * 1000:.1f}"
        response.headers.setdefault("x-content-type-options", "nosniff")
        response.headers.setdefault("x-frame-options", "SAMEORIGIN")
        response.headers.setdefault("referrer-policy", "strict-origin-when-cross-origin")
        if not request.url.path.startswith(self._QUIET_PREFIXES):
            log.info("%s %s -> %s (%.1fms)", request.method, request.url.path, response.status_code, elapsed * 1000)
        request_id_ctx.reset(token)
        return response


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a safe, correlated error — never a raw stack trace to clients."""
    get_logger("zintoo.error").exception("unhandled exception on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred. Reference the request id when reporting.",
            "request_id": request_id_ctx.get(),
        },
    )
