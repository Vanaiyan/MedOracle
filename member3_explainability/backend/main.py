"""
member3_explainability/backend/main.py
=======================================
FastAPI application entry point.

Run with:
    uvicorn member3_explainability.backend.main:app --reload --port 8000

All routes except /auth/* require a valid JWT Bearer token.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from member3_explainability.backend.database import init_db
from member3_explainability.backend.routers import (
    auth_router,
    predict_router,
    sessions_router,
    chat_router,
    dashboard_router,
)


# ---------------------------------------------------------------------------
# Lifespan: create DB tables on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MedOracle API",
    description=(
        "Multimodal Emotion Recognition System — "
        "SHAP Explainability + LLM Chatbot Backend\n\n"
        "Member 3: Adshaya Balarajah (214024V)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React dev server (port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
    "http://localhost:5176",
    "http://127.0.0.1:5176"
    "http://localhost:3000",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(auth_router.router)
app.include_router(predict_router.router)
app.include_router(sessions_router.router)
app.include_router(chat_router.router)
app.include_router(dashboard_router.router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "MedOracle API"}
