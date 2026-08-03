"""
member3_explainability/conflict/synthetic_conflict.py
=====================================================
Synthetic *conflict-case* generator.

The existing shap/synthetic_data.py deliberately makes both modalities AGREE
on the dominant emotion.  The conflict-explanation contribution needs the
opposite: cases where physiology and video predict DIFFERENT emotions, so the
quality gate has to resolve a genuine disagreement.

Each generated prediction_output is internally consistent: the fused emotion,
fused confidence, and modality_weights are all computed by re-running the
gate (conflict/gate.py) on the per-modality predictions, so the dict is
exactly what Member 2's pipeline would emit for that conflict.


"""

from __future__ import annotations

import random
from typing import Optional

from member3_explainability.conflict.gate import (
    EMOTIONS, run_gate, physio_quality_of,
)
from shared.data_contracts import make_prediction_output

_QUALITIES = ["good", "degraded", "poor"]


def _class_probs(dominant: str, confidence: float) -> dict:
    """Probability dict with `dominant` = confidence, remainder spread randomly."""
    others = [e for e in EMOTIONS if e != dominant]
    remaining = 1.0 - confidence
    cuts = sorted(random.random() for _ in range(len(others) - 1))
    cuts = [0.0] + cuts + [1.0]
    splits = [cuts[i + 1] - cuts[i] for i in range(len(others))]
    probs = {e: round(remaining * s, 4) for e, s in zip(others, splits)}
    probs[dominant] = round(confidence, 4)
    # fix rounding drift
    probs[dominant] = round(probs[dominant] + (1.0 - sum(probs.values())), 4)
    return probs


def generate_conflict_case(
    physio_emotion: Optional[str] = None,
    video_emotion: Optional[str] = None,
    eeg_quality: Optional[str] = None,
    gsr_quality: Optional[str] = None,
    video_quality: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Generate one conflict prediction_output where physio and video disagree.

    Any argument left as None is randomised.  Guarantees
    physio_emotion != video_emotion (a true conflict).

    Returns
    -------
    dict — a contract-valid M2 -> M3 prediction_output with a built-in
    disagreement, plus an extra top-level key `_ground_truth` recording the
    two modality emotions and qualities (useful for evaluation; ignored by
    the pipeline which only reads the contract keys).
    """
    if seed is not None:
        random.seed(seed)

    physio_emotion = physio_emotion or random.choice(EMOTIONS)
    if video_emotion is None:
        video_emotion = random.choice([e for e in EMOTIONS if e != physio_emotion])
    if physio_emotion == video_emotion:
        raise ValueError("physio_emotion and video_emotion must differ for a conflict case")

    eeg_quality = eeg_quality or random.choice(_QUALITIES)
    gsr_quality = gsr_quality or random.choice(_QUALITIES)
    video_quality = video_quality or random.choice(_QUALITIES)

    # Each modality is fairly confident in its OWN (different) emotion.
    physio_conf = round(random.uniform(0.55, 0.85), 4)
    video_conf = round(random.uniform(0.55, 0.85), 4)
    physio_probs = _class_probs(physio_emotion, physio_conf)
    video_probs = _class_probs(video_emotion, video_conf)

    pq = physio_quality_of(eeg_quality, gsr_quality)
    trace = run_gate(physio_probs, video_probs, pq, video_quality)

    pred = make_prediction_output(
        predicted_emotion=trace.fused_emotion,
        confidence=trace.fused_confidence,
        class_probabilities=trace.fused_probs,
        modality_weights={"physio": trace.w_physio, "video": trace.w_video},
        signal_quality={"eeg": eeg_quality, "gsr": gsr_quality, "video": video_quality},
        per_modality_predictions={
            "physio": {
                "predicted_emotion": physio_emotion,
                "confidence": physio_conf,
                "class_probabilities": physio_probs,
            },
            "video": {
                "predicted_emotion": video_emotion,
                "confidence": video_conf,
                "class_probabilities": video_probs,
            },
        },
    )
    pred["_ground_truth"] = {
        "physio_emotion": physio_emotion,
        "video_emotion": video_emotion,
        "eeg_quality": eeg_quality,
        "gsr_quality": gsr_quality,
        "video_quality": video_quality,
    }
    return pred


def generate_conflict_dataset(n: int = 200, seed: int = 42) -> list[dict]:
    """Generate a list of `n` conflict prediction_output dicts (reproducible)."""
    random.seed(seed)
    return [generate_conflict_case() for _ in range(n)]
