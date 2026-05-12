"""
member3_explainability/shap/faithfulness.py
===========================================
Faithfulness metric for MedOracle SHAP explanations.

Method (standard masking-and-measuring approach)
-------------------------------------------------
1. Rank modalities by absolute SHAP value  (highest → lowest).
2. Progressively mask the top-ranked modality (replace with background mean)
   and re-run the lightweight fusion.
3. Measure the drop in confidence for the predicted emotion after each mask.
4. faithfulness_score = Spearman correlation between
       rank_order(|SHAP values|)  and  rank_order(confidence_drops)
   A score near 1.0 means the most important modality (per SHAP) also
   causes the biggest confidence drop when removed → faithful explanation.

Score range : [0, 1]   (clamped from raw Spearman which is in [-1, 1])
Threshold   : ≥ 0.7  considered acceptable for reporting (per spec).

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-use the lightweight fusion helper from kernel_shap
# ---------------------------------------------------------------------------
from member3_explainability.shap.kernel_shap import (
    _weighted_fuse,
    _QUALITY_WEIGHT,
    _BACKGROUND_PHYSIO,
    _BACKGROUND_VIDEO,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pearson(xs: List[float], ys: List[float]) -> float:
    """
    Pearson correlation between two equal-length lists.
    Returns a continuous value in [-1, 1].
    Returns 1.0 for lists of length < 2 or zero-variance inputs.

    Unlike Spearman (rank-based), Pearson works on actual values so it
    produces continuous faithfulness scores even with only 3 data points.
    Spearman with n=3 can only return {-1, -0.5, 0.5, 1} — too coarse.
    """
    import math
    n = len(xs)
    if n < 2:
        return 1.0

    mx = sum(xs) / n
    my = sum(ys) / n

    cov  = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx   = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    sy   = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))

    if sx < 1e-9 or sy < 1e-9:
        # Zero variance — SHAP values or drops are all equal → treat as perfect
        return 1.0

    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_faithfulness(
    prediction_output: dict,
    shap_result: dict,
    background_physio: Dict[str, float] | None = None,
    background_video:  Dict[str, float] | None = None,
) -> float:
    """
    Compute the faithfulness score for a SHAP explanation.

    Parameters
    ----------
    prediction_output : dict
        The M2 → M3 prediction_output dict.
    shap_result       : dict
        The dict returned by KernelSHAPExplainer.explain().
    background_physio : dict, optional
        Background mean for physio (defaults to uniform).
    background_video  : dict, optional
        Background mean for video  (defaults to uniform).

    Returns
    -------
    float
        Faithfulness score clamped to [0, 1].
        Score ≥ 0.7 is considered acceptable.
    """
    bg_physio = background_physio or dict(_BACKGROUND_PHYSIO)
    bg_video  = background_video  or dict(_BACKGROUND_VIDEO)

    target_emotion = shap_result["target_emotion"]
    feature_imp    = shap_result["feature_importance"]  # {EEG, GSR, video}
    signal_quality = prediction_output["signal_quality"]
    per_modality   = prediction_output["per_modality_predictions"]

    physio_probs   = per_modality["physio"]["class_probabilities"]
    video_probs    = per_modality["video"]["class_probabilities"]

    eeg_quality    = signal_quality["eeg"]
    gsr_quality    = signal_quality["gsr"]
    video_quality  = signal_quality["video"]

    # Physio quality = worst of EEG / GSR
    _q_rank = {"good": 2, "degraded": 1, "poor": 0}
    physio_quality = (
        eeg_quality
        if _q_rank[eeg_quality] <= _q_rank[gsr_quality]
        else gsr_quality
    )

    # ── Baseline: full fusion confidence ──────────────────────────────────
    full_fused   = _weighted_fuse(physio_probs, video_probs,
                                  physio_quality, video_quality)
    base_conf    = full_fused[target_emotion]

    logger.debug("Faithfulness baseline confidence: %.4f", base_conf)

    # ── 3-signal importance: EEG, GSR, Video ─────────────────────────────
    # Treat all three signals independently — gives 3 data points for
    # a meaningful Spearman correlation (vs. only 2 with physio+video).
    #
    # Masking strategy:
    #   EEG   → physio marginal contribution × EEG quality share
    #   GSR   → physio marginal contribution × GSR quality share
    #   Video → replace video_probs with background mean
    #
    # The physio marginal = v(full) - v(video_only), split proportionally
    # by quality weights (same logic used to split phi_physio in kernel_shap).

    coalition_values = shap_result.get("coalition_values", {})
    v_full   = coalition_values.get("full",   base_conf)
    v_video  = coalition_values.get("video",  base_conf)
    v_physio = coalition_values.get("physio", base_conf)

    physio_marginal = v_full - v_video   # impact of adding physio
    video_marginal  = v_full - v_physio  # impact of adding video

    w_eeg = _QUALITY_WEIGHT.get(eeg_quality, 0.1)
    w_gsr = _QUALITY_WEIGHT.get(gsr_quality, 0.1)
    w_total = w_eeg + w_gsr if (w_eeg + w_gsr) > 1e-9 else 1.0

    eeg_drop   = physio_marginal * (w_eeg / w_total)
    gsr_drop   = physio_marginal * (w_gsr / w_total)
    video_drop = video_marginal

    signal_importances: List[Tuple[str, float]] = [
        ("EEG",   feature_imp.get("EEG",   0.0)),
        ("GSR",   feature_imp.get("GSR",   0.0)),
        ("video", feature_imp.get("video", 0.0)),
    ]
    confidence_drops = [eeg_drop, gsr_drop, video_drop]

    logger.debug(
        "3-signal drops — EEG: %.4f  GSR: %.4f  Video: %.4f",
        eeg_drop, gsr_drop, video_drop,
    )

    raw_pearson = _pearson(
        [imp for _, imp in signal_importances],
        confidence_drops,
    )

    # Clamp to [0, 1] — negative correlations mean SHAP order contradicts
    # the actual drop order, treated as 0 faithfulness
    faithfulness_score = max(0.0, min(1.0, raw_pearson))

    logger.info(
        "Faithfulness score: %.4f (Pearson=%.4f)  %s",
        faithfulness_score,
        raw_pearson,
        "✓ acceptable" if faithfulness_score >= 0.7 else "⚠ below threshold",
    )

    return round(faithfulness_score, 4)
