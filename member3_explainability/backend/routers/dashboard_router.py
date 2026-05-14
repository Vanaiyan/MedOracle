"""
member3_explainability/backend/routers/dashboard_router.py
===========================================================
GET /dashboard/summary — aggregated user statistics

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from member3_explainability.backend.database import get_db
from member3_explainability.backend.models import Session, SHAPLog
from member3_explainability.backend.auth import get_current_user_id
from member3_explainability.backend.schemas import DashboardSummary, EmotionTrendPoint

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Return aggregated user statistics:
    - session_count
    - dominant_emotion (most frequent predicted emotion)
    - dominant_modality (avg highest SHAP modality across sessions)
    - avg_confidence
    - emotion_trend (per-session: emotion, fused conf, physio-only conf)
    """
    result = await db.execute(
        select(Session)
        .options(selectinload(Session.shap_log))
        .where(Session.user_id == user_id)
        .order_by(Session.timestamp.asc())
    )
    sessions = result.scalars().all()

    if not sessions:
        return DashboardSummary(
            session_count=0,
            dominant_emotion=None,
            dominant_modality=None,
            avg_confidence=None,
            emotion_trend=[],
        )

    session_count  = len(sessions)
    emotion_counts = Counter(s.predicted_emotion for s in sessions)
    dominant_emotion = emotion_counts.most_common(1)[0][0]

    avg_confidence = sum(s.confidence for s in sessions) / session_count

    # Dominant modality: for each session pick the modality with highest |SHAP|
    modality_votes: list[str] = []
    for s in sessions:
        if s.shap_log:
            fi = s.shap_log.feature_importance or {}
            if fi:
                # Vote for the single signal with highest SHAP importance (EEG, GSR, or video)
                dominant = max(fi, key=fi.get)
                modality_votes.append(dominant)

    dominant_modality: Optional[str] = None
    if modality_votes:
        dominant_modality = Counter(modality_votes).most_common(1)[0][0]

    # Quality weight mapping (same as kernel_shap.py)
    _QUALITY_WEIGHT = {"good": 1.0, "degraded": 0.5, "poor": 0.1}

    # Emotion trend
    trend: list[EmotionTrendPoint] = []
    for s in sessions:
        eeg_conf: Optional[float] = None
        gsr_conf: Optional[float] = None

        pmr = s.shap_log.per_modality_predictions if s.shap_log else None
        sq  = s.signal_quality or {}

        video_conf: Optional[float] = None
        if pmr and "physio" in pmr:
            physio_conf = pmr["physio"].get("confidence") or 0.0
            w_eeg = _QUALITY_WEIGHT.get(sq.get("eeg", "good"), 1.0)
            w_gsr = _QUALITY_WEIGHT.get(sq.get("gsr", "good"), 1.0)
            total = w_eeg + w_gsr if (w_eeg + w_gsr) > 0 else 1.0
            eeg_conf = round(physio_conf * (w_eeg / total), 4)
            gsr_conf = round(physio_conf * (w_gsr / total), 4)
        if pmr and "video" in pmr:
            video_conf = pmr["video"].get("confidence")

        trend.append(EmotionTrendPoint(
            session_id=s.session_id,
            timestamp=s.timestamp,
            predicted_emotion=s.predicted_emotion,
            fused_confidence=s.confidence,
            eeg_confidence=eeg_conf,
            gsr_confidence=gsr_conf,
            video_confidence=video_conf,
        ))

    return DashboardSummary(
        session_count=session_count,
        dominant_emotion=dominant_emotion,
        dominant_modality=dominant_modality,
        avg_confidence=round(avg_confidence, 4),
        emotion_trend=trend,
    )
