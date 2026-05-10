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

    # Per-modality predictions — each modality may agree or slightly disagree
    # with the dominant emotion to create interesting SHAP spread
    physio_dominant = emotion if random.random() > 0.25 else random.choice(EMOTIONS)
    video_dominant  = emotion if random.random() > 0.20 else random.choice(EMOTIONS)

    physio_conf = round(random.uniform(0.50, 0.88), 4)
    video_conf  = round(random.uniform(0.55, 0.90), 4)

    physio_probs = _make_class_probs(physio_dominant, physio_conf)
    video_probs  = _make_class_probs(video_dominant,  video_conf)

    # Signal quality — random but weighted towards "good"
    eeg_quality   = random.choice(QUALITIES)
    gsr_quality   = random.choice(QUALITIES)
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
                "predicted_emotion":   physio_dominant,
                "confidence":          physio_conf,
                "class_probabilities": physio_probs,
            },
            "video": {
                "predicted_emotion":   video_dominant,
                "confidence":          video_conf,
                "class_probabilities": video_probs,
            },
        },
    }
