"""
member3_explainability/backend/routers/predict_router.py
=========================================================
POST /predict        — run SHAP pipeline on prediction_output, store to DB
GET  /explain/{id}   — fetch stored SHAP values for a session

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import io
import logging
import uuid
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
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
# NOTE: member2_video_fusion.inference (predict_video) is imported lazily inside
# the /predict/video handler so the API can boot without Member 2's video stack
# (torchvision etc.) installed. Only the video endpoint requires it.

_ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm"}

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


async def _persist_and_respond(
    prediction_output: dict, user_id: str, db: AsyncSession,
    ig_attribution: Optional[dict] = None,
) -> "PredictResponse":
    """Run SHAP on a prediction_output, store the session + SHAP log, return the
    response. Shared by the fusion endpoints (currently /predict/multimodal).

    ig_attribution : real, fused Integrated Gradients result from
        attribution/fused_ig.explain_fused() -- only computed by callers that
        have BOTH the real EEG/GSR AND video tensors (see /predict/multimodal).
    """
    shap_output = build_shap_output(prediction_output)

    session_id = str(uuid.uuid4())
    db.add(Session(
        session_id=session_id,
        user_id=user_id,
        timestamp=datetime.utcnow(),
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
        per_modality_predictions=shap_output.get("per_modality_predictions"),
        ig_attribution=ig_attribution,
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
        ig_attribution=ig_attribution,
    )


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


@router.post("/predict/video", response_model=PredictResponse)
async def predict_from_video(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a video file upload, run the trained VideoEmotionModel,
    pipe through SHAP, persist to DB, and return the full shap_output.

    Operates in video-only mode (no physio input — graceful degradation).
    Supported formats: mp4, avi, mov, mkv, flv, webm.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_VIDEO_SUFFIXES)}",
        )

    # Write upload to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        try:
            try:
                from member2_video_fusion.inference import predict_video
            except ImportError as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=("Video inference is unavailable: Member 2's video stack "
                            f"is not installed ({exc}). Install torchvision + "
                            "member2_video_fusion to enable /predict/video."),
                )
            prediction_output = predict_video(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

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


def _load_signal(upload: UploadFile, expected_shape: tuple) -> np.ndarray:
    """Parse an uploaded EEG/GSR file (.npy or .csv) into a float32 array.

    Raises HTTP 422 if the file is unreadable or has the wrong shape.
    """
    raw = upload.file.read()
    name = (upload.filename or "").lower()
    try:
        if name.endswith((".csv", ".txt")):
            arr = np.loadtxt(io.StringIO(raw.decode("utf-8")), delimiter=",")
        else:  # .npy (default)
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not read '{upload.filename}' as .npy or .csv ({exc}).",
        )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape != expected_shape:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{upload.filename}' must have shape {expected_shape}, got {arr.shape}.",
        )
    return arr


@router.post("/predict/multimodal", response_model=PredictResponse)
async def predict_multimodal(
    file: UploadFile = File(..., description="Video clip (required)"),
    eeg:  Optional[UploadFile] = File(None, description="EEG window .npy/.csv, shape (32,512)"),
    gsr:  Optional[UploadFile] = File(None, description="GSR window .npy/.csv, shape (512,)"),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Real multimodal inference: the user uploads a **video** plus (optionally) an
    **EEG** and **GSR** window. Both models run (M1 physio + M2 video), the gated
    fusion combines them, then the fused prediction goes through SHAP + is stored.

    - Provide all three files → full multimodal fusion.
    - Provide only the video (omit eeg/gsr) → video-only (graceful degradation).
    - EEG must be shape (32, 512); GSR shape (512,); both .npy or .csv.
    (See data/synced_samples/ for ready-made subject_XX/{eeg.npy, gsr.npy, video}.)
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported video type '{suffix}'. Allowed: {sorted(_ALLOWED_VIDEO_SUFFIXES)}",
        )

    # Parse physio uploads (both or neither)
    if (eeg is None) != (gsr is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide BOTH eeg and gsr files, or neither (video-only).",
        )
    eeg_arr = _load_signal(eeg, (32, 512)) if eeg is not None else None
    gsr_arr = _load_signal(gsr, (512,))    if gsr is not None else None

    # Write the video upload to a temp file for OpenCV
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    ig_attribution = None
    try:
        try:
            from member2_video_fusion.pipeline import run_full_pipeline
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=("Multimodal inference unavailable: Member 2's stack is not "
                        f"installed ({exc})."),
            )
        prediction_output = run_full_pipeline(
            eeg=eeg_arr, gsr=gsr_arr, video_path=tmp_path
        )

        # Real, fused Integrated Gradients -- only possible when we have BOTH
        # real modalities' raw tensors (not just video-only). Best-effort: if
        # torch/captum aren't installed, or the video can't be re-decoded, or
        # anything else goes wrong, log and continue WITHOUT IG rather than
        # failing the whole prediction -- SHAP explainability must not depend
        # on this working.
        if eeg_arr is not None and gsr_arr is not None:
            try:
                from member3_explainability.attribution.fused_ig import explain_fused
                from member2_video_fusion.inference import _process_video

                video_frames, _quality = _process_video(tmp_path)
                if video_frames is not None:
                    sq = prediction_output["signal_quality"]
                    # 32 steps keeps the CPU IG pass to ~1-2 min per request (the
                    # 128 default is ~7 min through ResNet50+BiLSTM) while still
                    # giving a solid attribution estimate for the demo.
                    ig_attribution = explain_fused(
                        eeg=eeg_arr, gsr=gsr_arr, video_frames=video_frames,
                        eeg_quality=sq["eeg"], gsr_quality=sq["gsr"],
                        video_quality=sq["video"],
                    )
            except ImportError as exc:
                logger.warning("IG unavailable (torch/captum not installed?): %s", exc)
            except Exception as exc:
                logger.warning("IG computation failed, continuing without it: %s", exc)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Fusion inference failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    return await _persist_and_respond(prediction_output, user_id, db, ig_attribution=ig_attribution)


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
