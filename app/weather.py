"""
Weather client (Open-Meteo, free, no API key).

Adds the resilience the original code lacked: bounded timeout, one retry with
backoff, and an in-process TTL cache so a burst of forecast requests doesn't
hammer the upstream or block on a slow network. Fails soft: on any error we
return a neutral seasonal default so forecasting still works.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from threading import Lock

from . import settings
from .observability import get_logger, metrics

log = get_logger("zintoo.weather")

_cache: dict[str, tuple[float, list[float]]] = {}
_cache_lock = Lock()
_NEUTRAL_TEMP = 29.0  # Mumbai-ish default (°C)


def _fetch(lat: float, lon: float, hours: int) -> list[float] | None:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "forecast_days": max(1, min(7, (hours // 24) + 1)),
        "timezone": "auto",
    }
    url = f"{settings.OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=settings.WEATHER_TIMEOUT) as resp:
                import json

                payload = json.loads(resp.read().decode())
            temps = payload.get("hourly", {}).get("temperature_2m")
            if temps:
                metrics.inc("weather_fetch_ok_total")
                return [float(t) for t in temps[:hours]]
            return None
        except Exception as exc:  # noqa: BLE001
            metrics.inc("weather_fetch_error_total")
            log.warning("weather fetch failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(0.4 * (attempt + 1))
    return None


def hourly_temps(pincode: str, hours: int = 24) -> list[float]:
    """Return `hours` hourly temperatures for a pincode. Never raises."""
    if not settings.WEATHER_ENABLED:
        return [_NEUTRAL_TEMP] * hours
    coords = settings.PIN_CODE_COORDS.get(pincode)
    if not coords:
        return [_NEUTRAL_TEMP] * hours

    key = f"{pincode}:{hours}"
    now = time.time()
    with _cache_lock:
        cached = _cache.get(key)
        if cached and now - cached[0] < settings.WEATHER_CACHE_TTL:
            metrics.inc("weather_cache_hit_total")
            return cached[1]

    temps = _fetch(coords[0], coords[1], hours)
    if not temps:
        temps = [_NEUTRAL_TEMP] * hours
    # Pad/trim to exactly `hours`.
    if len(temps) < hours:
        temps = temps + [temps[-1]] * (hours - len(temps))
    temps = temps[:hours]
    with _cache_lock:
        _cache[key] = (now, temps)
    return temps
