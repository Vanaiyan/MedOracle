"""
member3_explainability/backend/routers/predict_router.py
=========================================================
POST /predict        — run SHAP pipeline on prediction_output, store to DB
GET  /explain/{id}   — fetch stored SHAP values for a session

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from member3_explainability.backend.database import get_db
from member3_explainability.backend.models import Session, SHAPLog
from member3_explainability.backend.auth import get_current_user_id
from member3_explainability.backend.schemas import (
    PredictRequest, PredictResponse, ExplainResponse, SHAPValuesOut,
)
from member3_explainability.shap.shap_output_builder import build_shap_output
from member3_explainability.shap.synthetic_data import generate_prediction_output, EMOTIONS

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a prediction_output dict, run SHAP + faithfulness pipeline,
    store results in DB, and return the full shap_output.
    """
    # Convert Pydantic model → plain dict (matches M2 interface contract)
    prediction_output = {
        "predicted_emotion":        body.predicted_emotion,
        "confidence":               body.confidence,
        "class_probabilities":      body.class_probabilities,
        "modality_weights":         body.modality_weights,
        "signal_quality":           body.signal_quality.model_dump(),
        "per_modality_predictions": {
            k: v.model_dump() for k, v in body.per_modality_predictions.items()
        },
    }

    # Run SHAP
    shap_output = build_shap_output(prediction_output)

    # Persist session
    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        user_id=user_id,
        timestamp=datetime.utcnow(),
        predicted_emotion=shap_output["predicted_emotion"],
        confidence=shap_output["confidence"],
        modality_weights=shap_output["modality_weights"],
        signal_quality=shap_output["signal_quality"],
        class_probabilities=shap_output["class_probabilities"],
    )
    db.add(session)
    await db.flush()

    # Persist SHAP log
    shap_log = SHAPLog(
        session_id=session_id,
        shap_values=shap_output["shap_values"],
        feature_importance=shap_output["feature_importance"],
        faithfulness_score=shap_output["faithfulness_score"],
        signal_reliability=shap_output["signal_reliability"],
        coalition_values=shap_output.get("coalition_values"),
        per_modality_predictions=shap_output.get("per_modality_predictions"),
    )
    db.add(shap_log)

    sv = shap_output["shap_values"]
    fi = shap_output["feature_importance"]

    return PredictResponse(
        session_id=session_id,
        predicted_emotion=shap_output["predicted_emotion"],
        confidence=shap_output["confidence"],
        class_probabilities=shap_output["class_probabilities"],
        modality_weights=shap_output["modality_weights"],
        signal_quality=shap_output["signal_quality"],
        shap_values=SHAPValuesOut(**sv),
        feature_importance=SHAPValuesOut(**fi),
        faithfulness_score=shap_output["faithfulness_score"],
        coalition_values=shap_output.get("coalition_values", {}),
    )


@router.post("/predict/synthetic", response_model=PredictResponse)
async def predict_synthetic(
    emotion: str = None,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a synthetic fused prediction_output and run the full SHAP pipeline.

    Use this while the real EEG / GSR / video models are not yet integrated.

    Query param:
        emotion (optional) — one of: stress, calm, happy, sad, angry.
                             If omitted, a random emotion is chosen.
    """
    if emotion is not None and emotion not in EMOTIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"emotion must be one of {EMOTIONS}",
        )

    prediction_output = generate_prediction_output(emotion)
    shap_output = build_shap_output(prediction_output)

    session_id = str(uuid.uuid4())
    session = Session(
        session_id=session_id,
        user_id=user_id,
        timestamp=datetime.utcnow(),
        predicted_emotion=shap_output["predicted_emotion"],
        confidence=shap_output["confidence"],
        modality_weights=shap_output["modality_weights"],
        signal_quality=shap_output["signal_quality"],
        class_probabilities=shap_output["class_probabilities"],
    )
    db.add(session)
    await db.flush()

    shap_log = SHAPLog(
        session_id=session_id,
        shap_values=shap_output["shap_values"],
        feature_importance=shap_output["feature_importance"],
        faithfulness_score=shap_output["faithfulness_score"],
        signal_reliability=shap_output["signal_reliability"],
        coalition_values=shap_output.get("coalition_values"),
        per_modality_predictions=shap_output.get("per_modality_predictions"),
    )
    db.add(shap_log)

    sv = shap_output["shap_values"]
    fi = shap_output["feature_importance"]

    return PredictResponse(
        session_id=session_id,
        predicted_emotion=shap_output["predicted_emotion"],
        confidence=shap_output["confidence"],
        class_probabilities=shap_output["class_probabilities"],
        modality_weights=shap_output["modality_weights"],
        signal_quality=shap_output["signal_quality"],
        shap_values=SHAPValuesOut(**sv),
        feature_importance=SHAPValuesOut(**fi),
        faithfulness_score=shap_output["faithfulness_score"],
        coalition_values=shap_output.get("coalition_values", {}),
    )


@router.get("/explain/{session_id}", response_model=ExplainResponse)
async def explain(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Fetch stored SHAP values for a previously completed session."""
    # Verify session belongs to the authenticated user
    result = await db.execute(
        select(Session).where(
            Session.session_id == session_id,
            Session.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    result2 = await db.execute(
        select(SHAPLog).where(SHAPLog.session_id == session_id)
    )
    log = result2.scalar_one_or_none()
    if not log:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="SHAP log not found.")

    return ExplainResponse(
        session_id=session_id,
        shap_values=SHAPValuesOut(**log.shap_values),
        feature_importance=SHAPValuesOut(**log.feature_importance),
        faithfulness_score=log.faithfulness_score,
        signal_reliability=log.signal_reliability,
        coalition_values=log.coalition_values,
        per_modality_predictions=log.per_modality_predictions,
    )
