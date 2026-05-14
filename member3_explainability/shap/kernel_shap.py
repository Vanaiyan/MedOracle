"""
member3_explainability/shap/kernel_shap.py
==========================================
Kernel SHAP over the MedOracle 2-modality fusion pipeline.

Background
----------
Lundberg & Lee (2017) — "A Unified Approach to Interpreting Model Predictions", NeurIPS.

With M = 2 effective modalities (physiological, video) the number of coalitions
is 2^2 = 4:

    S = {}               → replace BOTH modalities with background means
    S = {physio}         → use real physio, replace video with background mean
    S = {video}          → replace physio with background mean, use real video
    S = {physio, video}  → use BOTH real modalities  (grand coalition)

The Shapley value for each player (modality) is the weighted average of its
marginal contributions over all orderings.  For exactly M = 2 players this
has an exact closed-form solution — no sampling / regression needed:

    φ_physio = ½ × [ (v({physio})       - v({}))
                    + (v({physio,video}) - v({video})) ]

    φ_video  = ½ × [ (v({video})        - v({}))
                    + (v({physio,video}) - v({physio})) ]

The physio φ is then decomposed into separate EEG and GSR contributions
using quality-weighted attribution (since M1's model jointly processes both).

Outputs
-------
shap_values       : {EEG: float, GSR: float, video: float}
                    Signed contributions to the winning-class probability.
feature_importance: {EEG: float, GSR: float, video: float}
                    Absolute SHAP values (magnitude of contribution).

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality → weight mapping (mirrors the gated-fusion α from Member 2)
# ---------------------------------------------------------------------------

_QUALITY_WEIGHT: Dict[str, float] = {
    "good":     1.0,
    "degraded": 0.5,
    "poor":     0.1,
}


# ---------------------------------------------------------------------------
# Background distributions
# ---------------------------------------------------------------------------

# Default background: uniform probability across the 5 emotion classes.
# In production, replace these with the mean class-probability vectors
# computed across your training split.
_BACKGROUND_PHYSIO: Dict[str, float] = {
    "stress": 0.20, "calm": 0.20, "happy": 0.20, "sad": 0.20, "angry": 0.20,
}
_BACKGROUND_VIDEO: Dict[str, float] = {
    "stress": 0.20, "calm": 0.20, "happy": 0.20, "sad": 0.20, "angry": 0.20,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_fuse(
    physio_probs: Dict[str, float],
    video_probs:  Dict[str, float],
    physio_quality: str,
    video_quality:  str,
) -> Dict[str, float]:
    """
    Lightweight re-implementation of Member 2's gated fusion.

    Used *only* inside SHAP to evaluate coalition values without calling
    the full heavy pipeline for each of the 4 coalitions.

    Gate score  g_m = confidence_m × α_m
    Weight      w_m = g_m / (g_physio + g_video)   [L1 normalisation]
    Fused       P   = w_physio × P_physio + w_video × P_video
    """
    def _entropy_confidence(probs: Dict[str, float]) -> float:
        """Entropy-based confidence: c = 1 - H(P) / log(K), K=5."""
        import math
        K = len(probs)
        H = -sum(p * math.log(p + 1e-9) for p in probs.values())
        H_max = math.log(K)
        return max(0.0, min(1.0, 1.0 - H / H_max))

    alpha_physio = _QUALITY_WEIGHT.get(physio_quality, 0.1)
    alpha_video  = _QUALITY_WEIGHT.get(video_quality,  0.1)

    g_physio = _entropy_confidence(physio_probs) * alpha_physio
    g_video  = _entropy_confidence(video_probs)  * alpha_video

    g_total = g_physio + g_video
    if g_total < 1e-9:
        # Both modalities have zero gate score → uniform output
        return {k: 1.0 / 5 for k in physio_probs}

    w_physio = g_physio / g_total
    w_video  = g_video  / g_total

    return {
        k: w_physio * physio_probs[k] + w_video * video_probs[k]
        for k in physio_probs
    }


def _coalition_value(
    target_emotion: str,
    # Which modalities are PRESENT (using real data)
    physio_present: bool,
    video_present:  bool,
    # Real data
    real_physio_probs: Dict[str, float],
    real_video_probs:  Dict[str, float],
    physio_quality: str,
    video_quality:  str,
    # Background distributions
    bg_physio: Dict[str, float],
    bg_video:  Dict[str, float],
) -> float:
    """
    Evaluate the coalition value v(S) = P_fused[target_emotion] for subset S.

    Absent modalities are replaced by their background mean distributions.
    The quality flags for absent modalities are set to 'good' so the background
    is weighted purely by its entropy confidence (it's already calibrated).
    """
    effective_physio = real_physio_probs if physio_present else bg_physio
    effective_video  = real_video_probs  if video_present  else bg_video

    # When a modality is absent we treat it as 'good' quality so the
    # background distribution is not further penalised by an α factor —
    # the background already encodes the average behaviour.
    eff_physio_quality = physio_quality if physio_present else "good"
    eff_video_quality  = video_quality  if video_present  else "good"

    fused = _weighted_fuse(
        effective_physio, effective_video,
        eff_physio_quality, eff_video_quality,
    )
    return fused[target_emotion]


def _split_physio_shap(
    phi_physio: float,
    eeg_quality: str,
    gsr_quality: str,
) -> Dict[str, float]:
    """
    Decompose the joint physio SHAP value into EEG and GSR contributions.

    Member 1's BiAttn network jointly processes EEG and GSR, so their
    combined SHAP value is φ_physio.  We attribute it proportionally to
    each sub-signal's quality weight (a proxy for its effective contribution
    to the physiological prediction).

    If both signals have equal quality, the split is 50 / 50.
    """
    w_eeg = _QUALITY_WEIGHT.get(eeg_quality, 0.1)
    w_gsr = _QUALITY_WEIGHT.get(gsr_quality, 0.1)
    total = w_eeg + w_gsr

    if total < 1e-9:
        # Degenerate: both signals are absent — split equally
        return {"EEG": phi_physio * 0.5, "GSR": phi_physio * 0.5}

    return {
        "EEG": phi_physio * (w_eeg / total),
        "GSR": phi_physio * (w_gsr / total),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class KernelSHAPExplainer:
    """
    Kernel SHAP explainer for the MedOracle 2-modality pipeline.

    Parameters
    ----------
    background_physio : dict, optional
        Mean class-probability vector for the physio modality over the
        training set.  Defaults to the uniform distribution.
    background_video  : dict, optional
        Mean class-probability vector for the video modality over the
        training set.  Defaults to the uniform distribution.

    Usage
    -----
    >>> explainer = KernelSHAPExplainer()
    >>> result = explainer.explain(prediction_output)
    >>> result["shap_values"]        # {EEG: float, GSR: float, video: float}
    >>> result["feature_importance"] # {EEG: float, GSR: float, video: float}
    """

    def __init__(
        self,
        background_physio: Optional[Dict[str, float]] = None,
        background_video:  Optional[Dict[str, float]] = None,
    ):
        self.bg_physio = background_physio or dict(_BACKGROUND_PHYSIO)
        self.bg_video  = background_video  or dict(_BACKGROUND_VIDEO)

    # ------------------------------------------------------------------

    def explain(self, prediction_output: dict) -> dict:
        """
        Compute SHAP values for a single prediction_output dict (M2 → M3 contract).

        Parameters
        ----------
        prediction_output : dict
            The exact dict produced by Member 2's fusion pipeline, with keys:
              - predicted_emotion
              - confidence
              - class_probabilities
              - modality_weights          {physio, video}
              - signal_quality            {eeg, gsr, video}
              - per_modality_predictions  {physio: {...}, video: {...}}

        Returns
        -------
        dict  with keys:
            shap_values        {EEG: float, GSR: float, video: float}
            feature_importance {EEG: float, GSR: float, video: float}
            coalition_values   {empty, physio, video, full} — for audit / faithfulness
            target_emotion     str
            baseline           float  (v({}) — background prediction)
            model_output       float  (v({physio, video}) — full prediction)
        """
        # ── 1. Unpack prediction_output ────────────────────────────────────
        target_emotion  = prediction_output["predicted_emotion"]
        signal_quality  = prediction_output["signal_quality"]    # {eeg, gsr, video}
        per_modality    = prediction_output["per_modality_predictions"]

        physio_probs    = per_modality["physio"]["class_probabilities"]
        video_probs     = per_modality["video"]["class_probabilities"]

        eeg_quality     = signal_quality["eeg"]
        gsr_quality     = signal_quality["gsr"]
        video_quality   = signal_quality["video"]

        # Treat physio quality as the worse of EEG / GSR (same rule M2 uses)
        _quality_rank = {"good": 2, "degraded": 1, "poor": 0}
        physio_quality = (
            eeg_quality
            if _quality_rank[eeg_quality] <= _quality_rank[gsr_quality]
            else gsr_quality
        )

        logger.debug("KernelSHAP: target_emotion=%s", target_emotion)

        # ── 2. Evaluate all 4 coalition values ─────────────────────────────
        #
        #   v({})             → physio=background, video=background
        #   v({physio})       → physio=real,       video=background
        #   v({video})        → physio=background, video=real
        #   v({physio,video}) → physio=real,        video=real

        v_empty  = _coalition_value(
            target_emotion,
            physio_present=False, video_present=False,
            real_physio_probs=physio_probs, real_video_probs=video_probs,
            physio_quality=physio_quality,  video_quality=video_quality,
            bg_physio=self.bg_physio,       bg_video=self.bg_video,
        )
        v_physio = _coalition_value(
            target_emotion,
            physio_present=True,  video_present=False,
            real_physio_probs=physio_probs, real_video_probs=video_probs,
            physio_quality=physio_quality,  video_quality=video_quality,
            bg_physio=self.bg_physio,       bg_video=self.bg_video,
        )
        v_video  = _coalition_value(
            target_emotion,
            physio_present=False, video_present=True,
            real_physio_probs=physio_probs, real_video_probs=video_probs,
            physio_quality=physio_quality,  video_quality=video_quality,
            bg_physio=self.bg_physio,       bg_video=self.bg_video,
        )
        v_full   = _coalition_value(
            target_emotion,
            physio_present=True,  video_present=True,
            real_physio_probs=physio_probs, real_video_probs=video_probs,
            physio_quality=physio_quality,  video_quality=video_quality,
            bg_physio=self.bg_physio,       bg_video=self.bg_video,
        )

        logger.debug(
            "Coalition values: v({})=%.4f  v({physio})=%.4f  "
            "v({video})=%.4f  v({physio,video})=%.4f",
            v_empty, v_physio, v_video, v_full,
        )

        # ── 3. Shapley formula (exact closed-form for M=2) ─────────────────
        #
        #   φ_physio = ½ × [(v({physio}) - v({})) + (v({physio,video}) - v({video}))]
        #   φ_video  = ½ × [(v({video})  - v({})) + (v({physio,video}) - v({physio}))]
        #
        # Efficiency axiom check: φ_physio + φ_video ≈ v(full) - v({})
        # (the SHAP values sum to the difference from baseline)

        phi_physio = 0.5 * ((v_physio - v_empty) + (v_full - v_video))
        phi_video  = 0.5 * ((v_video  - v_empty) + (v_full - v_physio))

        efficiency_gap = abs((phi_physio + phi_video) - (v_full - v_empty))
        if efficiency_gap > 1e-6:
            logger.warning(
                "SHAP efficiency axiom violated: gap=%.2e "
                "(should be ~0 for exact Kernel SHAP with M=2)", efficiency_gap
            )

        # ── 4. Decompose physio SHAP into EEG + GSR ────────────────────────
        eeg_gsr = _split_physio_shap(phi_physio, eeg_quality, gsr_quality)

        shap_values = {
            "EEG":   round(eeg_gsr["EEG"], 6),
            "GSR":   round(eeg_gsr["GSR"], 6),
            "video": round(phi_video,       6),
        }

        feature_importance = {k: abs(v) for k, v in shap_values.items()}

        # ── 5. Return full result dict ─────────────────────────────────────
        return {
            "shap_values":      shap_values,
            "feature_importance": feature_importance,
            # Audit trail — stored in shap_logs table, used by faithfulness metric
            "coalition_values": {
                "empty":  round(v_empty,  6),
                "physio": round(v_physio, 6),
                "video":  round(v_video,  6),
                "full":   round(v_full,   6),
            },
            "target_emotion": target_emotion,
            "baseline":       round(v_empty, 6),
            "model_output":   round(v_full,  6),
        }


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

_default_explainer: Optional[KernelSHAPExplainer] = None


def explain(
    prediction_output: dict,
    background_physio: Optional[Dict[str, float]] = None,
    background_video:  Optional[Dict[str, float]] = None,
) -> dict:
    """
    Convenience wrapper.  Creates a KernelSHAPExplainer (caches it on first
    call) and returns the SHAP result dict.

    Parameters
    ----------
    prediction_output : dict
        Output from Member 2's fusion pipeline (M2 → M3 contract).
    background_physio : dict, optional
        Training-set mean physio probabilities.  Defaults to uniform.
    background_video  : dict, optional
        Training-set mean video probabilities.  Defaults to uniform.

    Returns
    -------
    dict — same as KernelSHAPExplainer.explain()
    """
    global _default_explainer
    if background_physio is not None or background_video is not None:
        # Custom backgrounds: always create a new explainer
        explainer = KernelSHAPExplainer(background_physio, background_video)
    else:
        if _default_explainer is None:
            _default_explainer = KernelSHAPExplainer()
        explainer = _default_explainer

    return explainer.explain(prediction_output)
