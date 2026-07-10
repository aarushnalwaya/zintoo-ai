"""
FastAPI security dependencies built on the pure primitives in `app.auth`.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import auth, settings

bearer_scheme = HTTPBearer(auto_error=False)
_limiter = auth.RateLimiter(settings.RATE_LIMIT_RPS, settings.RATE_LIMIT_BURST)


async def current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    if creds is None or not creds.credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = auth.decode_token(creds.credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def require_role(*roles: str):
    async def _guard(user: dict = Depends(current_user)) -> dict:
        if user.get("role") not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user

    return _guard


def check_rate_limit(request: Request) -> None:
    if not settings.RATE_LIMIT_ENABLED:
        return
    client = request.client.host if request.client else "unknown"
    if not _limiter.allow(client):
        raise HTTPException(status_code=429, detail="Too many requests, slow down.")
