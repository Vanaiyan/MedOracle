"""
member3_explainability/backend/auth.py
=======================================
JWT access/refresh token utilities + bcrypt password hashing.

Tokens
------
  access  : 15-minute expiry  (sent with every API request)
  refresh : 7-day expiry      (used to silently re-issue access tokens)

Secret key is read from the JWT_SECRET env var.
Algorithm : HS256.


"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SECRET_KEY  = os.getenv("JWT_SECRET", "medoracle-dev-secret-change-in-production")
ALGORITHM   = "HS256"
ACCESS_TTL  = timedelta(minutes=15)
REFRESH_TTL = timedelta(days=7)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# Token creation
# ---------------------------------------------------------------------------

def _create_token(data: dict, ttl: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + ttl
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_id: str, email: str) -> str:
    return _create_token({"sub": user_id, "email": email, "type": "access"}, ACCESS_TTL)


def create_refresh_token(user_id: str) -> str:
    return _create_token({"sub": user_id, "type": "refresh"}, REFRESH_TTL)


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------

def _decode_token(token: str, expected_type: str) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.get("type") != expected_type:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Expected {expected_type} token.",
        )
    return payload


# ---------------------------------------------------------------------------
# FastAPI dependency — resolves the current authenticated user_id
# ---------------------------------------------------------------------------

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """Dependency: extract and validate the Bearer access token → return user_id."""
    payload = _decode_token(credentials.credentials, expected_type="access")
    user_id: Optional[str] = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")
    return user_id
