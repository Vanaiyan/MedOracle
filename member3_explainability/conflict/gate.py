"""
member3_explainability/conflict/gate.py
=======================================
A single, shared re-implementation of Member 2's quality-gated fusion,
exposed in a form the conflict layer can introspect.

`_weighted_fuse` in shap/kernel_shap.py fuses probabilities but hides the
intermediate gate scores and weights.  The conflict-explanation layer needs
those intermediates (confidence, alpha, gate score, weight per modality) to
explain *why* one modality won, so this module re-derives them explicitly and
returns everything.

It is deliberately consistent with:
  - _QUALITY_WEIGHT           (alpha per quality tier)   from kernel_shap
  - the entropy confidence     c = 1 - H(P)/log(K)        (Guo et al., 2017)
  - L1-normalised weights      w_m = g_m / sum_m g_m


"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

# Reuse Member-2-aligned alpha map so we never drift from the SHAP layer.
from member3_explainability.shap.kernel_shap import _QUALITY_WEIGHT

EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]
_QUALITY_RANK = {"good": 2, "degraded": 1, "poor": 0}


def entropy_confidence(probs: Dict[str, float]) -> float:
    """Entropy-based confidence  c = 1 - H(P)/log(K),  K = number of classes."""
    K = len(probs)
    H = -sum(p * math.log(p + 1e-9) for p in probs.values())
    H_max = math.log(K)
    return max(0.0, min(1.0, 1.0 - H / H_max))


def alpha(quality: str) -> float:
    """Quality penalty factor alpha (good=1.0, degraded=0.5, poor=0.1)."""
    return _QUALITY_WEIGHT.get(quality, 0.1)


def physio_quality_of(eeg_quality: str, gsr_quality: str) -> str:
    """Physio quality = the worse of EEG / GSR (same rule M2 uses)."""
    return eeg_quality if _QUALITY_RANK[eeg_quality] <= _QUALITY_RANK[gsr_quality] else gsr_quality


@dataclass
class GateTrace:
    """Full, introspectable trace of one gated-fusion decision."""
    physio_conf: float
    video_conf: float
    physio_alpha: float
    video_alpha: float
    g_physio: float
    g_video: float
    w_physio: float
    w_video: float
    fused_probs: Dict[str, float]
    fused_emotion: str
    fused_confidence: float
    physio_quality: str
    video_quality: str

    def dominant_modality(self) -> str:
        return "physio" if self.w_physio >= self.w_video else "video"

    def as_dict(self) -> dict:
        return {
            "physio_conf": round(self.physio_conf, 4),
            "video_conf": round(self.video_conf, 4),
            "physio_alpha": self.physio_alpha,
            "video_alpha": self.video_alpha,
            "g_physio": round(self.g_physio, 4),
            "g_video": round(self.g_video, 4),
            "w_physio": round(self.w_physio, 4),
            "w_video": round(self.w_video, 4),
            "fused_emotion": self.fused_emotion,
            "fused_confidence": round(self.fused_confidence, 4),
            "physio_quality": self.physio_quality,
            "video_quality": self.video_quality,
            "dominant_modality": self.dominant_modality(),
        }


def run_gate(
    physio_probs: Dict[str, float],
    video_probs: Dict[str, float],
    physio_quality: str,
    video_quality: str,
) -> GateTrace:
    """
    Re-run Member 2's gated fusion and return every intermediate value.

    This is the introspectable twin of `_weighted_fuse`; the fused
    probabilities it produces are identical to that function's output.
    """
    c_p = entropy_confidence(physio_probs)
    c_v = entropy_confidence(video_probs)
    a_p = alpha(physio_quality)
    a_v = alpha(video_quality)

    g_p = c_p * a_p
    g_v = c_v * a_v
    g_total = g_p + g_v

    if g_total < 1e-9:
        w_p = w_v = 0.5
        fused = {k: 1.0 / len(physio_probs) for k in physio_probs}
    else:
        w_p = g_p / g_total
        w_v = g_v / g_total
        fused = {k: w_p * physio_probs[k] + w_v * video_probs[k] for k in physio_probs}

    fused_emotion = max(fused, key=fused.get)
    return GateTrace(
        physio_conf=c_p, video_conf=c_v,
        physio_alpha=a_p, video_alpha=a_v,
        g_physio=g_p, g_video=g_v,
        w_physio=w_p, w_video=w_v,
        fused_probs=fused,
        fused_emotion=fused_emotion,
        fused_confidence=entropy_confidence(fused),
        physio_quality=physio_quality,
        video_quality=video_quality,
    )
