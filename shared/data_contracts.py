"""
shared/data_contracts.py
========================
MedOracle — Shared Interface Contracts

Defines the exact data structures that flow between pipeline members.
All members MUST produce/consume these exact shapes.

  M1  →  physiological_prediction_dict  →  M2
  M2  →  prediction_output              →  M3
  M3  →  shap_output                    →  FastAPI

Author: MedOracle Team (214206G · 214024V · 214215H)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

EMOTION_CLASSES: Dict[str, int] = {
    "stress": 0,
    "calm":   1,
    "happy":  2,
    "sad":    3,
    "angry":  4,
}
"""
The five discrete emotion classes used throughout the system.
ALL class_probabilities dicts must use exactly these keys.
"""

EmotionLabel = Literal["stress", "calm", "happy", "sad", "angry"]
SignalQuality = Literal["good", "degraded", "poor"]


# ---------------------------------------------------------------------------
# SynchronizedInput  (raw input to the full pipeline)
# ---------------------------------------------------------------------------

@dataclass
class SynchronizedInput:
    """
    A single 4-second trial's worth of raw, synchronised multimodal input.

    Shapes
    ------
    eeg   : (32, 512)  — 32 EEG channels × 512 samples @ 128 Hz
    gsr   : (512,)     — 1 GSR channel, resampled to 128 Hz (4 Hz × 128)
    video : (16, 224, 224, 3) — 16 uniformly sampled RGB frames
    """
    eeg:        np.ndarray        # shape (32, 512)
    gsr:        np.ndarray        # shape (512,)
    video:      np.ndarray        # shape (16, 224, 224, 3)
    trial_id:   str = ""
    subject_id: str = ""

    def validate(self) -> None:
        """Raise AssertionError if any array has the wrong shape."""
        assert self.eeg.shape   == (32, 512),         \
            f"EEG shape mismatch: expected (32, 512), got {self.eeg.shape}"
        assert self.gsr.shape   == (512,),             \
            f"GSR shape mismatch: expected (512,), got {self.gsr.shape}"
        assert self.video.shape == (16, 224, 224, 3),  \
            f"Video shape mismatch: expected (16, 224, 224, 3), got {self.video.shape}"


# ---------------------------------------------------------------------------
# M1 → M2  :  physiological_prediction_dict
# ---------------------------------------------------------------------------

def make_physiological_prediction(
    predicted_emotion: EmotionLabel,
    confidence: float,
    class_probabilities: Dict[str, float],
    eeg_quality: SignalQuality,
    gsr_quality: SignalQuality,
) -> dict:
    """
    Build a validated physiological_prediction_dict.

    Parameters
    ----------
    predicted_emotion    : one of the five emotion labels
    confidence           : entropy-based confidence in [0, 1]
    class_probabilities  : {"stress": p, "calm": p, "happy": p, "sad": p, "angry": p}
    eeg_quality          : "good" | "degraded" | "poor"
    gsr_quality          : "good" | "degraded" | "poor"

    Returns
    -------
    dict matching the M1 → M2 interface contract
    """
    _validate_emotion_label(predicted_emotion)
    _validate_confidence(confidence)
    _validate_class_probs(class_probabilities)
    _validate_quality(eeg_quality, "eeg_quality")
    _validate_quality(gsr_quality, "gsr_quality")

    return {
        "predicted_emotion":   predicted_emotion,
        "confidence":          float(confidence),
        "class_probabilities": {k: float(class_probabilities[k]) for k in EMOTION_CLASSES},
        "signal_quality": {
            "eeg": eeg_quality,
            "gsr": gsr_quality,
        },
    }


# ---------------------------------------------------------------------------
# M2 → M3  :  prediction_output
# ---------------------------------------------------------------------------

def make_prediction_output(
    predicted_emotion: EmotionLabel,
    confidence: float,
    class_probabilities: Dict[str, float],
    modality_weights: Dict[Literal["physio", "video"], float],
    signal_quality: Dict[Literal["eeg", "gsr", "video"], SignalQuality],
    per_modality_predictions: Dict[str, dict],
) -> dict:
    """
    Build a validated prediction_output dict.

    Parameters
    ----------
    predicted_emotion        : fused prediction label
    confidence               : fused entropy-based confidence
    class_probabilities      : fused P_fused[5]
    modality_weights         : {"physio": w1, "video": w2}  — must sum to 1
    signal_quality           : {"eeg": ..., "gsr": ..., "video": ...}
    per_modality_predictions : {"physio": {pred_emotion, confidence, class_probs},
                                "video":  {pred_emotion, confidence, class_probs}}

    Returns
    -------
    dict matching the M2 → M3 interface contract
    """
    _validate_emotion_label(predicted_emotion)
    _validate_confidence(confidence)
    _validate_class_probs(class_probabilities)

    w = modality_weights
    assert set(w.keys()) == {"physio", "video"}, \
        "modality_weights must have exactly keys 'physio' and 'video'"
    assert abs(w["physio"] + w["video"] - 1.0) < 1e-4, \
        f"modality_weights must sum to 1, got {w['physio'] + w['video']}"

    for k in ("eeg", "gsr", "video"):
        assert k in signal_quality, f"signal_quality missing key '{k}'"
        _validate_quality(signal_quality[k], k)

    for modality in ("physio", "video"):
        assert modality in per_modality_predictions, \
            f"per_modality_predictions missing key '{modality}'"

    return {
        "predicted_emotion":        predicted_emotion,
        "confidence":               float(confidence),
        "class_probabilities":      {k: float(class_probabilities[k]) for k in EMOTION_CLASSES},
        "modality_weights":         {"physio": float(w["physio"]), "video": float(w["video"])},
        "signal_quality":           signal_quality,
        "per_modality_predictions": per_modality_predictions,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_emotion_label(label: str) -> None:
    assert label in EMOTION_CLASSES, \
        f"Invalid emotion label '{label}'. Must be one of {list(EMOTION_CLASSES.keys())}"

def _validate_confidence(conf: float) -> None:
    assert 0.0 <= conf <= 1.0, \
        f"Confidence must be in [0, 1], got {conf}"

def _validate_class_probs(probs: Dict[str, float]) -> None:
    assert set(probs.keys()) == set(EMOTION_CLASSES.keys()), \
        f"class_probabilities must have exactly these keys: {list(EMOTION_CLASSES.keys())}"
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-3, \
        f"class_probabilities must sum to ~1.0, got {total}"

def _validate_quality(val: str, name: str = "quality") -> None:
    assert val in ("good", "degraded", "poor"), \
        f"'{name}' must be 'good', 'degraded', or 'poor', got '{val}'"
