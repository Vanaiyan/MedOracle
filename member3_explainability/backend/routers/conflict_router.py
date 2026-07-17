"""
member3_explainability/backend/routers/conflict_router.py
=========================================================
POST /explain/conflict  — Member 3's novel contribution surfaced over HTTP.

Accepts a prediction_output dict (the M2 -> M3 contract) and returns the
trust-aware modality-conflict explanation: why the modalities disagreed, how
the quality gate resolved it, which modality to trust, exact 2-modality Shapley
attribution, counterfactuals, and a hallucination-verified natural-language
explanation (RAG-grounded; Claude if an API key is set, else the verified
template).

Stateless compute endpoint (no DB) so it composes cleanly with /predict.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from member3_explainability.backend.auth import get_current_user_id
from member3_explainability.backend.database import get_db
from member3_explainability.backend.models import Session, SHAPLog
from member3_explainability.backend.schemas import (
    PredictRequest, ConflictExplainResponse, NLExplanationOut,
    PredictResponse, SHAPValuesOut,
)
from member3_explainability.conflict.conflict_explainer import explain_conflict
from member3_explainability.conflict.gate import run_gate, physio_quality_of
from member3_explainability.conflict.synthetic_conflict import generate_conflict_case
from member3_explainability.shap.shap_output_builder import build_shap_output
from member3_explainability.evaluation.llm_chatbot import generate_explanation
from member3_explainability.evaluation.knowledge_base import Retriever

router = APIRouter(tags=["Conflict Explanation"])
_retriever = Retriever()
EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]


def _normalize_via_gate(prediction_output: dict) -> dict:
    """
    Re-derive the fused fields from the gate so the explanation is internally
    consistent regardless of the fused values the client sent. The client only
    needs to supply the two per-modality predictions and the signal quality.
    """
    pm = prediction_output["per_modality_predictions"]
    sq = prediction_output["signal_quality"]
    pq = physio_quality_of(sq["eeg"], sq["gsr"])
    trace = run_gate(
        pm["physio"]["class_probabilities"],
        pm["video"]["class_probabilities"],
        pq, sq["video"],
    )
    prediction_output["predicted_emotion"] = trace.fused_emotion
    prediction_output["class_probabilities"] = trace.fused_probs
    prediction_output["modality_weights"] = {"physio": trace.w_physio, "video": trace.w_video}
    prediction_output["confidence"] = trace.fused_confidence
    return prediction_output


def build_conflict_response(prediction_output: dict) -> dict:
    """
    Pure function (no auth / DB) so it is unit-testable:
    prediction_output -> conflict explanation + verified NL explanation.
    """
    prediction_output = _normalize_via_gate(prediction_output)
    exp = explain_conflict(prediction_output)
    nl = generate_explanation(exp, retriever=_retriever)
    sc = nl["scores"]
    return {
        "is_conflict":              exp["is_conflict"],
        "physio_emotion":           exp["physio_emotion"],
        "video_emotion":            exp["video_emotion"],
        "fused_emotion":            exp["fused_emotion"],
        "agrees_with":              exp["agrees_with"],
        "gate":                     exp["gate"],
        "modality_shapley":         exp["modality_shapley"],
        "trust":                    exp["trust"],
        "counterfactuals":          exp["counterfactuals"],
        "losing_modality_recovery": exp["losing_modality_recovery"],
        "rationale":                exp["rationale"],
        "explanation": {
            "text":               nl["text"],
            "source":             nl["source"],
            "verified":           nl["verified"],
            "faithfulness":       sc["faithfulness"],
            "hallucination_rate": sc["hallucination_rate"],
            "clinical_relevance": sc["clinical_relevance"],
        },
    }


@router.post("/explain/conflict", response_model=ConflictExplainResponse)
async def explain_conflict_endpoint(
    body: PredictRequest,
    user_id: str = Depends(get_current_user_id),
):
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
    return build_conflict_response(prediction_output)


@router.get("/explain/conflict/{session_id}", response_model=ConflictExplainResponse)
async def explain_conflict_for_session(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Explain the conflict for a STORED session (the one shown in Session Detail)."""
    session = await db.get(Session, session_id)
    if session is None or session.user_id != user_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    shap_log = (
        await db.execute(select(SHAPLog).where(SHAPLog.session_id == session_id))
    ).scalar_one_or_none()
    per_modality = shap_log.per_modality_predictions if shap_log else None
    if not per_modality or "physio" not in per_modality or "video" not in per_modality:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="This session has no per-modality predictions, so a conflict "
                   "explanation cannot be produced (e.g. a video-only session).",
        )

    prediction_output = {
        "predicted_emotion":        session.predicted_emotion,
        "confidence":               session.confidence,
        "class_probabilities":      session.class_probabilities,
        "modality_weights":         session.modality_weights,
        "signal_quality":           session.signal_quality,
        "per_modality_predictions": per_modality,
    }
    return build_conflict_response(prediction_output)


@router.post("/predict/conflict", response_model=PredictResponse)
async def create_conflict_session(
    physio_emotion: str = None,
    video_emotion: str = None,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a STORED session in which physiology and video DISAGREE, so the
    dashboard has a genuine conflict to explain. Optional query params fix the
    two emotions; otherwise they are randomised (guaranteed distinct).
    """
    for name, val in (("physio_emotion", physio_emotion), ("video_emotion", video_emotion)):
        if val is not None and val not in EMOTIONS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{name} must be one of {EMOTIONS}",
            )
    if physio_emotion and video_emotion and physio_emotion == video_emotion:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="physio_emotion and video_emotion must differ for a conflict",
        )

    prediction_output = generate_conflict_case(
        physio_emotion=physio_emotion, video_emotion=video_emotion,
    )
    per_modality = prediction_output["per_modality_predictions"]
    prediction_output.pop("_ground_truth", None)   # drop eval-only key before SHAP

    shap_output = build_shap_output(prediction_output)

    session_id = str(uuid.uuid4())
    db.add(Session(
        session_id=session_id, user_id=user_id, timestamp=datetime.utcnow(),
        predicted_emotion=shap_output["predicted_emotion"],
        confidence=shap_output["confidence"],
        modality_weights=shap_output["modality_weights"],
        signal_quality=shap_output["signal_quality"],
        class_probabilities=shap_output["class_probabilities"],
    ))
    await db.flush()
    db.add(SHAPLog(
        session_id=session_id,
        shap_values=shap_output["shap_values"],
        feature_importance=shap_output["feature_importance"],
        faithfulness_score=shap_output["faithfulness_score"],
        signal_reliability=shap_output["signal_reliability"],
        coalition_values=shap_output.get("coalition_values"),
        per_modality_predictions=per_modality,   # guarantee it's stored for conflict
    ))

    return PredictResponse(
        session_id=session_id,
        predicted_emotion=shap_output["predicted_emotion"],
        confidence=shap_output["confidence"],
        class_probabilities=shap_output["class_probabilities"],
        modality_weights=shap_output["modality_weights"],
        signal_quality=shap_output["signal_quality"],
        shap_values=SHAPValuesOut(**shap_output["shap_values"]),
        feature_importance=SHAPValuesOut(**shap_output["feature_importance"]),
        faithfulness_score=shap_output["faithfulness_score"],
        coalition_values=shap_output.get("coalition_values", {}),
    )
