"""
member3_explainability/backend/routers/sessions_router.py
==========================================================
GET /sessions        — list all sessions for authenticated user
GET /sessions/{id}   — detailed session with full SHAP breakdown

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from member3_explainability.backend.database import get_db
from member3_explainability.backend.models import Session, SHAPLog
from member3_explainability.backend.auth import get_current_user_id
from member3_explainability.backend.schemas import (
    SessionListResponse, SessionSummary, SessionDetail, SHAPValuesOut,
)

router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return all sessions for the authenticated user, ordered by timestamp desc."""
    result = await db.execute(
        select(Session)
        .where(Session.user_id == user_id)
        .order_by(Session.timestamp.desc())
    )
    sessions = result.scalars().all()

    return SessionListResponse(
        sessions=[
            SessionSummary(
                session_id=s.session_id,
                timestamp=s.timestamp,
                predicted_emotion=s.predicted_emotion,
                confidence=s.confidence,
            )
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return detailed session data including full SHAP breakdown."""
    result = await db.execute(
        select(Session)
        .options(selectinload(Session.shap_log))
        .where(Session.session_id == session_id, Session.user_id == user_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    log = session.shap_log

    return SessionDetail(
        session_id=session.session_id,
        timestamp=session.timestamp,
        predicted_emotion=session.predicted_emotion,
        confidence=session.confidence,
        modality_weights=session.modality_weights,
        signal_quality=session.signal_quality,
        class_probabilities=session.class_probabilities,
        shap_values=SHAPValuesOut(**log.shap_values) if log else None,
        feature_importance=SHAPValuesOut(**log.feature_importance) if log else None,
        faithfulness_score=log.faithfulness_score if log else None,
        per_modality_predictions=log.per_modality_predictions if log else None,
        ig_attribution=log.ig_attribution if log else None,
    )
