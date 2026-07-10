"""
Observability infrastructure with zero web-framework coupling.

Structured logging, request-id tracing (via contextvar), and in-process
metrics. The FastAPI middleware / exception handler that *use* these live in
`app.middleware` so this module stays framework-agnostic and unit-testable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from contextvars import ContextVar
from threading import Lock

from . import settings

# request id available anywhere in the request lifecycle (incl. log records)
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


# ─── Logging ──────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_ctx.get()
        base = f"{time.strftime('%H:%M:%S')} {record.levelname:<7} [{rid[:8]}] {record.name}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if settings.JSON_LOGS else PrettyFormatter())
    root.addHandler(handler)
    root.setLevel(settings.LOG_LEVEL)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ─── Metrics (Prometheus text exposition, zero deps) ──────────────────
class Metrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = defaultdict(float)
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, float] = defaultdict(float)

    def inc(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += value

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            self.gauges[name] = value

    def observe_latency(self, route: str, seconds: float) -> None:
        with self._lock:
            self._latency_sum[route] += seconds
            self._latency_count[route] += 1

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for name, val in sorted(self.counters.items()):
                lines.append(f"zintoo_{name} {val}")
            for name, val in sorted(self.gauges.items()):
                lines.append(f"zintoo_{name} {val}")
            for route, total in sorted(self._latency_sum.items()):
                count = self._latency_count[route] or 1
                safe = route.replace('"', "")
                lines.append(f'zintoo_request_latency_seconds_avg{{route="{safe}"}} {total / count:.6f}')
        return "\n".join(lines) + "\n"


metrics = Metrics()
