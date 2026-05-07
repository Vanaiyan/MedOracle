"""
member3_explainability/backend/database.py
==========================================
SQLAlchemy async engine + session factory.

Dev  : SQLite  (no setup needed — auto-created as medoracle.db)
Prod : PostgreSQL via DATABASE_URL env var

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

_RAW_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./medoracle.db")

# SQLite needs check_same_thread=False via connect_args
_CONNECT_ARGS = {"check_same_thread": False} if _RAW_URL.startswith("sqlite") else {}

engine = create_async_engine(
    _RAW_URL,
    echo=False,
    connect_args=_CONNECT_ARGS,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# ---------------------------------------------------------------------------
# Base class for all ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# FastAPI dependency — yields a DB session per request
# ---------------------------------------------------------------------------

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Create all tables on startup
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Called once at application startup to create tables if missing."""
    async with engine.begin() as conn:
        from member3_explainability.backend import models  # noqa: F401 — registers models
        await conn.run_sync(Base.metadata.create_all)
