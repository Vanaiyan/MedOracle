"""
member2_video_fusion/fusion.py
==============================
MedOracle — Member 2 (Vanaiyan Kirupagaran, 214215H)

Canonical quality-aware **gated fusion**.

Combines the physiological (M1) and video (M2) per-modality predictions into a
single fused `prediction_output`, using entropy-based confidence weighted by a
signal-quality penalty, L1-normalised modality weights, and graceful degradation
for poor or missing modalities.

Fusion math (CLAUDE.md RESOLVED ISSUE 4; Guo et al., 2017)
---------------------------------------------------------
    c_m   = 1 - H(P_m) / log K            entropy confidence per modality
    alpha : good = 1.0, degraded = 0.5, poor = 0.1    signal-quality penalty
    g_m   = c_m * alpha_m                 gate score
    w_m   = g_m / Σ_m g_m                 L1-normalised weight (NOT softmax)
    P_fused = Σ_m w_m * P_m               weighted-sum fusion
    predicted_emotion = argmax(P_fused)

Graceful degradation
--------------------
    both present         → full gated fusion
    physio missing/None  → video only   (w = {physio:0, video:1})
    video  missing/None  → physio only  (w = {physio:1, video:0})
    both maximally uncertain (Σg ≈ 0) → uniform, flagged unreliable

This module is the **single source of truth** for the gate math. It is
numerically identical to `member3_explainability/conflict/gate.run_gate`
when both modalities are present (verified in tests/test_fusion.py); that
module (and `shap/kernel_shap._weighted_fuse`) can later import from here to
remove the duplicate copies.

Author: Vanaiyan Kirupagaran (214215H)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from shared.data_contracts import EMOTION_CLASSES, make_prediction_output

EMOTIONS = list(EMOTION_CLASSES.keys())   # ["stress","calm","happy","sad","angry"]

# Quality penalty factor α (CLAUDE.md RESOLVED ISSUE 4)
QUALITY_ALPHA: Dict[str, float] = {"good": 1.0, "degraded": 0.5, "poor": 0.1}
_QUALITY_RANK: Dict[str, int]   = {"good": 2, "degraded": 1, "poor": 0}
_EPS = 1e-9   # matches conflict/gate.run_gate for bit-level equivalence


# ---------------------------------------------------------------------------
# Primitives (importable so gate.py / kernel_shap can share one implementation)
# ---------------------------------------------------------------------------

def entropy_confidence(probs: Dict[str, float]) -> float:
    """Entropy-based confidence  c = 1 − H(P)/log K, clipped to [0, 1].

    Returns 1.0 for a certain prediction (all mass on one class) and 0.0 for a
    uniform distribution. Reference: Guo et al. (2017).
    """
    K = len(probs)
    H = -sum(p * math.log(p + _EPS) for p in probs.values())
    return max(0.0, min(1.0, 1.0 - H / math.log(K)))


def quality_alpha(quality: str) -> float:
    """Signal-quality penalty α: good=1.0, degraded=0.5, poor=0.1."""
    return QUALITY_ALPHA.get(quality, 0.1)


def physio_quality_of(eeg_quality: str, gsr_quality: str) -> str:
    """Combined physiological quality = the worse of EEG / GSR."""
    return (eeg_quality
            if _QUALITY_RANK[eeg_quality] <= _QUALITY_RANK[gsr_quality]
            else gsr_quality)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform() -> Dict[str, float]:
    return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}


def _probs_of(pred: dict) -> Dict[str, float]:
    """Extract a clean {emotion: float} probability dict in canonical order."""
    return {e: float(pred["class_probabilities"][e]) for e in EMOTIONS}


def _per_modality(pred: Optional[dict]) -> dict:
    """Build the per-modality sub-dict for the contract.

    A missing modality is represented by a uniform, zero-confidence placeholder.
    """
    if pred is None:
        return {
            "predicted_emotion":   "calm",   # neutral placeholder (physio absent)
            "confidence":          0.0,
            "class_probabilities": _uniform(),
        }
    probs = _probs_of(pred)
    conf  = pred.get("confidence")
    return {
        "predicted_emotion":   pred["predicted_emotion"],
        "confidence":          float(conf if conf is not None else entropy_confidence(probs)),
        "class_probabilities": probs,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gated_fusion(
    physio_pred:   Optional[dict],
    video_pred:    Optional[dict],
    video_quality: str = "poor",
    eeg_quality:   Optional[str] = None,
    gsr_quality:   Optional[str] = None,
) -> dict:
    """
    Fuse physiological + video predictions into a validated `prediction_output`.

    Parameters
    ----------
    physio_pred : M1 contract dict
        {"predicted_emotion", "confidence", "class_probabilities",
         "signal_quality": {"eeg", "gsr"}}  — or None if physio is absent.
    video_pred : video per-modality dict
        {"predicted_emotion", "confidence", "class_probabilities"} — or None.
    video_quality : "good" | "degraded" | "poor"
        Video signal-quality grade (from face_detector / inference).
    eeg_quality, gsr_quality : optional overrides for the EEG/GSR quality
        (otherwise read from physio_pred["signal_quality"]).

    Returns
    -------
    dict — prediction_output, validated by shared.data_contracts.make_prediction_output.
    """
    if physio_pred is None and video_pred is None:
        raise ValueError("gated_fusion requires at least one modality; both are None.")

    # --- resolve signal qualities -------------------------------------------
    pq_dict = (physio_pred or {}).get("signal_quality", {})
    eeg_q = eeg_quality if eeg_quality is not None else pq_dict.get("eeg", "poor")
    gsr_q = gsr_quality if gsr_quality is not None else pq_dict.get("gsr", "poor")
    physio_q = physio_quality_of(eeg_q, gsr_q)

    physio_probs = _probs_of(physio_pred) if physio_pred is not None else None
    video_probs  = _probs_of(video_pred)  if video_pred  is not None else None

    # --- weights + fused distribution ---------------------------------------
    if physio_probs is not None and video_probs is not None:
        # Both present → full gated fusion (identical to gate.run_gate)
        c_p = entropy_confidence(physio_probs)
        c_v = entropy_confidence(video_probs)
        g_p = c_p * quality_alpha(physio_q)
        g_v = c_v * quality_alpha(video_quality)
        g_total = g_p + g_v
        if g_total < _EPS:
            # both maximally uncertain → uniform, unreliable
            w_physio = w_video = 0.5
            fused = _uniform()
        else:
            w_physio = g_p / g_total
            w_video  = g_v / g_total
            fused = {e: w_physio * physio_probs[e] + w_video * video_probs[e]
                     for e in EMOTIONS}
    elif video_probs is not None:
        # physio missing → video only
        w_physio, w_video = 0.0, 1.0
        fused = dict(video_probs)
    else:
        # video missing → physio only
        w_physio, w_video = 1.0, 0.0
        fused = dict(physio_probs)

    fused_emotion = max(fused, key=fused.get)
    fused_conf    = entropy_confidence(fused)

    return make_prediction_output(
        predicted_emotion   = fused_emotion,
        confidence          = fused_conf,
        class_probabilities = fused,
        modality_weights    = {"physio": w_physio, "video": w_video},
        signal_quality      = {"eeg": eeg_q, "gsr": gsr_q, "video": video_quality},
        per_modality_predictions = {
            "physio": _per_modality(physio_pred),
            "video":  _per_modality(video_pred),
        },
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    physio = {
        "predicted_emotion": "stress",
        "confidence": 0.34,
        "class_probabilities": {"stress": 0.6, "calm": 0.1, "happy": 0.1, "sad": 0.1, "angry": 0.1},
        "signal_quality": {"eeg": "good", "gsr": "good"},
    }
    video = {
        "predicted_emotion": "happy",
        "confidence": 0.29,
        "class_probabilities": {"stress": 0.1, "calm": 0.1, "happy": 0.6, "sad": 0.1, "angry": 0.1},
    }

    print("=== both present (physio good, video poor) ===")
    out = gated_fusion(physio, video, video_quality="poor")
    print(f"fused: {out['predicted_emotion']}  weights={out['modality_weights']}")
    print(json.dumps(out["class_probabilities"], indent=2))

    print("\n=== video only (physio missing) ===")
    out = gated_fusion(None, video, video_quality="good")
    print(f"fused: {out['predicted_emotion']}  weights={out['modality_weights']}")

    print("\n=== physio only (video missing) ===")
    out = gated_fusion(physio, None)
    print(f"fused: {out['predicted_emotion']}  weights={out['modality_weights']}")

    print("\n✓ fusion smoke test complete")
