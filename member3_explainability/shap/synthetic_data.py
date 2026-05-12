"""
member3_explainability/shap/synthetic_data.py
=============================================
Generates realistic synthetic prediction_output dicts that mimic
what Member 2's fusion pipeline would produce.

Used by POST /predict/synthetic so the SHAP explainability pipeline
can be demonstrated without the real EEG / GSR / video models.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import random
from typing import Optional

EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]
QUALITIES = ["good", "good", "good", "degraded", "poor"]   # weighted — mostly good


def _make_class_probs(dominant: str, confidence: float) -> dict:
    """
    Build a probability dict where `dominant` gets `confidence` and
    the remaining probability is spread randomly across the other 4 emotions.
    """
    others = [e for e in EMOTIONS if e != dominant]
    remaining = 1.0 - confidence

    # Random split of remaining probability
    cuts = sorted([random.random() for _ in range(len(others) - 1)])
    cuts = [0.0] + cuts + [1.0]
    splits = [cuts[i + 1] - cuts[i] for i in range(len(others))]
    other_probs = [round(remaining * s, 4) for s in splits]

    # Fix rounding so sum == 1.0
    probs = {e: p for e, p in zip(others, other_probs)}
    probs[dominant] = round(confidence, 4)
    total = sum(probs.values())
    probs[dominant] = round(probs[dominant] + (1.0 - total), 4)

    return probs


def generate_prediction_output(emotion: Optional[str] = None) -> dict:
    """
    Generate a synthetic prediction_output dict.

    Parameters
    ----------
    emotion : str, optional
        Target emotion to simulate. If None, a random one is chosen.
        Must be one of: stress, calm, happy, sad, angry.

    Returns
    -------
    dict — matches the M2 → M3 prediction_output contract accepted by /predict.
    """
    if emotion is None:
        emotion = random.choice(EMOTIONS)
    elif emotion not in EMOTIONS:
        raise ValueError(f"emotion must be one of {EMOTIONS}, got '{emotion}'")

    # Fused confidence (how sure the fused model is)
    fused_confidence = round(random.uniform(0.60, 0.92), 4)

    # Per-modality predictions — modalities always agree on the dominant emotion
    # but with different confidence levels to create interesting SHAP spread.
    # Both must predict the same emotion as the fused output to ensure
    # SHAP values are positive and meaningful (not artifacts of inconsistency).
    physio_conf = round(random.uniform(0.45, 0.82), 4)
    video_conf  = round(random.uniform(0.45, 0.82), 4)

    # One modality is stronger than the other to create non-trivial SHAP split
    if random.random() > 0.5:
        physio_conf = min(physio_conf + 0.15, 0.92)   # physio dominates
    else:
        video_conf  = min(video_conf  + 0.15, 0.92)   # video dominates

    physio_probs = _make_class_probs(emotion, physio_conf)
    video_probs  = _make_class_probs(emotion, video_conf)

    # Signal quality — EEG and GSR always get different quality levels
    # and which one is higher is random each time (fair alternation).
    # e.g. sometimes EEG=good/GSR=degraded, sometimes EEG=poor/GSR=good, etc.
    eeg_gsr_levels = random.sample(["good", "degraded", "poor"], 2)
    eeg_quality   = eeg_gsr_levels[0]
    gsr_quality   = eeg_gsr_levels[1]
    video_quality = random.choice(QUALITIES)

    # Modality weights — physio + video sum to 1.0
    physio_weight = round(random.uniform(0.30, 0.70), 4)
    video_weight  = round(1.0 - physio_weight, 4)

    return {
        "predicted_emotion":   emotion,
        "confidence":          fused_confidence,
        "class_probabilities": _make_class_probs(emotion, fused_confidence),
        "modality_weights": {
            "physio": physio_weight,
            "video":  video_weight,
        },
        "signal_quality": {
            "eeg":   eeg_quality,
            "gsr":   gsr_quality,
            "video": video_quality,
        },
        "per_modality_predictions": {
            "physio": {
                "predicted_emotion":   emotion,
                "confidence":          physio_conf,
                "class_probabilities": physio_probs,
            },
            "video": {
                "predicted_emotion":   emotion,
                "confidence":          video_conf,
                "class_probabilities": video_probs,
            },
        },
    }
