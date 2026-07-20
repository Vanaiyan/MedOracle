"""
member3_explainability/backend/schemas.py
==========================================
Pydantic v2 request / response schemas for all 9 endpoints.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email:        EmailStr
    password:     str = Field(min_length=6)
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    user_id:      str
    email:        str
    display_name: Optional[str]
    created_at:   datetime


# ---------------------------------------------------------------------------
# Prediction  (POST /predict)
# ---------------------------------------------------------------------------

class SignalQualityIn(BaseModel):
    eeg:   Literal["good", "degraded", "poor"] = "good"
    gsr:   Literal["good", "degraded", "poor"] = "good"
    video: Literal["good", "degraded", "poor"] = "good"


class ModalityPredictionIn(BaseModel):
    predicted_emotion:   str
    confidence:          float
    class_probabilities: Dict[str, float]


class PredictRequest(BaseModel):
    """
    Mirrors the M2 → M3 prediction_output dict.
    In production, /predict would run the full pipeline and generate this
    internally. For the API layer, clients may also POST a pre-computed
    prediction_output (useful for demo / testing).
    """
    predicted_emotion:        str
    confidence:               float
    class_probabilities:      Dict[str, float]
    modality_weights:         Dict[str, float]   # {physio, video}
    signal_quality:           SignalQualityIn
    per_modality_predictions: Dict[str, ModalityPredictionIn]

    @field_validator("confidence")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return v


class SHAPValuesOut(BaseModel):
    EEG:   float
    GSR:   float
    video: float


class PredictResponse(BaseModel):
    session_id:          str
    predicted_emotion:   str
    confidence:          float
    class_probabilities: Dict[str, float]
    modality_weights:    Dict[str, float]
    signal_quality:      Dict[str, str]
    shap_values:         SHAPValuesOut
    feature_importance:  SHAPValuesOut
    faithfulness_score:  float
    coalition_values:    Dict[str, float]


# ---------------------------------------------------------------------------
# Explain  (GET /explain/{session_id})
# ---------------------------------------------------------------------------

class ExplainResponse(BaseModel):
    session_id:          str
    shap_values:         SHAPValuesOut
    feature_importance:  SHAPValuesOut
    faithfulness_score:  float
    signal_reliability:  Dict[str, str]
    coalition_values:    Optional[Dict[str, float]]
    per_modality_predictions: Optional[Dict]


# ---------------------------------------------------------------------------
# Conflict Explanation  (POST /explain/conflict)  — Member 3 novel contribution
# ---------------------------------------------------------------------------

class NLExplanationOut(BaseModel):
    text:     str
    source:   Literal["llm", "template"]
    verified: bool
    faithfulness:       float
    hallucination_rate: float
    clinical_relevance: float


class ConflictExplainResponse(BaseModel):
    is_conflict:              bool
    physio_emotion:           str
    video_emotion:            str
    fused_emotion:            str
    agrees_with:              str
    gate:                     Dict
    modality_shapley:         Dict[str, float]
    trust:                    Dict
    counterfactuals:          List[Dict]
    losing_modality_recovery: Optional[Dict]
    rationale:                str
    explanation:              NLExplanationOut


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    session_id:        str
    timestamp:         datetime
    predicted_emotion: str
    confidence:        float

class SessionDetail(BaseModel):
    session_id:               str
    timestamp:                datetime
    predicted_emotion:        str
    confidence:               float
    modality_weights:         Dict[str, float]
    signal_quality:           Dict[str, str]
    class_probabilities:      Dict[str, float]
    shap_values:              Optional[SHAPValuesOut]
    feature_importance:       Optional[SHAPValuesOut]
    faithfulness_score:       Optional[float]
    per_modality_predictions: Optional[Dict]


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]
    total:    int


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: str
    message:    str = Field(min_length=1, max_length=2000)


class ChatMessageOut(BaseModel):
    message_id: str
    role:       Literal["user", "assistant"]
    content:    str
    timestamp:  datetime


class ChatResponse(BaseModel):
    message_id: str
    response:   str
    timestamp:  datetime


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages:   List[ChatMessageOut]


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class EmotionTrendPoint(BaseModel):
    session_id:           str
    timestamp:            datetime
    predicted_emotion:    str
    fused_confidence:     float
    eeg_confidence:       Optional[float]
    gsr_confidence:       Optional[float]
    video_confidence:     Optional[float]


class DashboardSummary(BaseModel):
    session_count:      int
    dominant_emotion:   Optional[str]
    dominant_modality:  Optional[str]
    avg_confidence:     Optional[float]
    emotion_trend:      List[EmotionTrendPoint]
