"""
Authentication primitives — pure, framework-agnostic, unit-testable.

Fixes the original design (unsalted SHA-256 passwords; a token minted at login
but never validated on any later request).

  * Passwords: PBKDF2-HMAC-SHA256, 200k iterations, per-user random salt.
  * Tokens:    signed `payload.signature` (HMAC-SHA256) with an expiry claim,
               validated on every protected call. Stateless, no server store.

The FastAPI dependencies that consume these live in `app.security`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from threading import Lock

from . import settings

_PBKDF2_ROUNDS = 200_000


# ─── Password hashing ─────────────────────────────────────────────────
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return digest.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)


# ─── Tokens ───────────────────────────────────────────────────────────
def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def issue_token(email: str, role: str) -> str:
    payload = {
        "sub": email,
        "role": role,
        "iat": int(time.time()),
        "exp": int(time.time()) + settings.TOKEN_TTL_SECONDS,
        "jti": secrets.token_hex(8),
    }
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(settings.SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64e(sig)}"


def decode_token(token: str) -> dict | None:
    try:
        body, sig = token.split(".", 1)
    except ValueError:
        return None
    expected = hmac.new(settings.SECRET_KEY.encode(), body.encode(), hashlib.sha256).digest()
    try:
        if not hmac.compare_digest(_b64d(sig), expected):
            return None
        payload = json.loads(_b64d(body))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload


# ─── Rate limiting (per-IP token bucket) ──────────────────────────────
class RateLimiter:
    def __init__(self, rate: float, burst: int) -> None:
        self.rate = rate
        self.burst = burst
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True
