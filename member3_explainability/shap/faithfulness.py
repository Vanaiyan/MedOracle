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

def _rank_list(values: List[float]) -> List[int]:
    """
    Return 1-based ranks for a list of values (largest = rank 1).
    Ties share the lower rank (standard competition ranking).
    """
    sorted_idx = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(sorted_idx, start=1):
        ranks[idx] = rank
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    """
    Spearman rank-order correlation between two equal-length lists.
    Returns a value in [-1, 1].  Returns 1.0 for lists of length < 2.
    """
    n = len(xs)
    if n < 2:
        return 1.0

    rx = _rank_list(xs)
    ry = _rank_list(ys)

    d2_sum = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6 * d2_sum) / (n * (n ** 2 - 1))


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

    # ── Modality-level importance (aggregate EEG+GSR back into physio) ────
    # We mask at the modality level (physio vs video) since that's the
    # granularity of the fusion gate.
    physio_importance = feature_imp["EEG"] + feature_imp["GSR"]
    video_importance  = feature_imp["video"]

    modality_importances: List[Tuple[str, float]] = [
        ("physio", physio_importance),
        ("video",  video_importance),
    ]

    # Sort by importance descending (highest SHAP first)
    modality_importances.sort(key=lambda x: -x[1])
    shap_ranks = list(range(1, len(modality_importances) + 1))

    # ── Progressive masking ────────────────────────────────────────────────
    # Start from full prediction; mask one modality at a time.
    confidence_drops: List[float] = []

    current_physio = physio_probs
    current_video  = video_probs
    current_pq     = physio_quality
    current_vq     = video_quality

    for rank, (modality, importance) in enumerate(modality_importances):
        if modality == "physio":
            masked_physio = bg_physio
            masked_video  = current_video
            masked_pq     = "good"   # background treated as 'good'
            masked_vq     = current_vq
        else:
            masked_physio = current_physio
            masked_video  = bg_video
            masked_pq     = current_pq
            masked_vq     = "good"

        masked_fused = _weighted_fuse(
            masked_physio, masked_video, masked_pq, masked_vq
        )
        masked_conf = masked_fused[target_emotion]
        drop = base_conf - masked_conf
        confidence_drops.append(drop)

        logger.debug(
            "  Mask step %d: modality=%s  conf_after=%.4f  drop=%.4f",
            rank, modality, masked_conf, drop,
        )

        # Update current state for next step (cumulative masking)
        current_physio = masked_physio
        current_video  = masked_video
        current_pq     = masked_pq
        current_vq     = masked_vq

    # ── Spearman correlation ───────────────────────────────────────────────
    # shap_ranks is already [1, 2, ...] (ordered by importance)
    # We want to check: do higher SHAP ranks correlate with higher drops?
    raw_spearman = _spearman(
        [imp for _, imp in modality_importances],
        confidence_drops,
    )

    # Clamp to [0, 1] — negative correlations are treated as 0 faithfulness
    faithfulness_score = max(0.0, min(1.0, raw_spearman))

    logger.info(
        "Faithfulness score: %.4f (Spearman=%.4f)  %s",
        faithfulness_score,
        raw_spearman,
        "✓ acceptable" if faithfulness_score >= 0.7 else "⚠ below threshold",
    )

    return round(faithfulness_score, 4)
