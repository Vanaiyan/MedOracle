"""
member3_explainability/conflict/conflict_explainer.py
=====================================================
The core of the novel contribution:

    "Trust-Aware Modality-Conflict Explanation for Quality-Gated
     Multimodal Emotion Recognition."

Given a Member 2 `prediction_output` dict, this module answers three questions
the base system cannot:

  1. WHY do the physiological and video modalities disagree?
     -> compare per-modality predictions; surface the disagreement.

  2. HOW does the quality gate resolve that disagreement?
     -> re-derive the full gate trace (confidence x quality-alpha -> weights)
        and decompose which factor (confidence vs signal quality) decided it.

  3. WHICH modality should be trusted, and how much?
     -> exact 2-modality Shapley attribution of the FUSED winning class,
        plus a trust score.

It also produces COUNTERFACTUALS over the quality/gate variables:
     "if EEG signal quality had been good, the decision would flip to stress."

Design notes
------------
* Depends only on `conflict/gate.py` (+ stdlib). No torch / shap / numpy needed,
  so the whole conflict layer runs on synthetic dicts with zero heavy deps.
* Feature-level attribution (Integrated Gradients / Kernel SHAP) is a SEPARATE
  layer (Phase 2) that plugs in later; this module operates purely on the
  contract dict and the gate arithmetic, which is exact and cheap.


"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional

from member3_explainability.conflict.gate import (
    EMOTIONS,
    GateTrace,
    run_gate,
    entropy_confidence,
    alpha,
    physio_quality_of,
    _QUALITY_RANK,
)

_QUALITY_TIERS = ["good", "degraded", "poor"]


# ---------------------------------------------------------------------------
# 1. Conflict detection
# ---------------------------------------------------------------------------

def detect_conflict(pred: dict) -> dict:
    """
    Determine whether the two modalities disagree on the dominant emotion.

    Returns a small dict:
        is_conflict     : bool  (physio argmax != video argmax)
        physio_emotion  : str
        video_emotion   : str
        fused_emotion   : str
        agrees_with     : "physio" | "video" | "neither"
                          which modality the FUSED decision sided with.
    """
    pm = pred["per_modality_predictions"]
    physio_e = pm["physio"]["predicted_emotion"]
    video_e = pm["video"]["predicted_emotion"]
    fused_e = pred["predicted_emotion"]

    if fused_e == physio_e and fused_e != video_e:
        agrees = "physio"
    elif fused_e == video_e and fused_e != physio_e:
        agrees = "video"
    elif fused_e == physio_e == video_e:
        agrees = "both"
    else:
        agrees = "neither"

    return {
        "is_conflict": physio_e != video_e,
        "physio_emotion": physio_e,
        "video_emotion": video_e,
        "fused_emotion": fused_e,
        "agrees_with": agrees,
    }


# ---------------------------------------------------------------------------
# 2. Exact 2-modality Shapley over the FUSED winning class
# ---------------------------------------------------------------------------

def modality_shapley(
    physio_probs: Dict[str, float],
    video_probs: Dict[str, float],
    fused_probs: Dict[str, float],
    target_class: str,
    n_classes: int = 5,
) -> Dict[str, float]:
    """
    Exact Shapley attribution of the fused probability of `target_class`
    to the two modalities (M = 2 players -> closed form, no sampling).

    Value function v(S) = probability mass assigned to `target_class`
    using only the modalities in coalition S:
        v({})            = 1 / K            (uninformed / uniform prior)
        v({physio})      = physio_probs[c]
        v({video})       = video_probs[c]
        v({physio,video})= fused_probs[c]   (the actual gated fusion)

    phi_physio = 1/2 [ (v_p - v_empty) + (v_both - v_video) ]
    phi_video  = 1/2 [ (v_v - v_empty) + (v_both - v_physio) ]

    Efficiency holds: phi_physio + phi_video == v_both - v_empty.
    """
    c = target_class
    v_empty = 1.0 / n_classes
    v_p = physio_probs[c]
    v_v = video_probs[c]
    v_both = fused_probs[c]

    phi_physio = 0.5 * ((v_p - v_empty) + (v_both - v_v))
    phi_video = 0.5 * ((v_v - v_empty) + (v_both - v_p))
    return {"physio": phi_physio, "video": phi_video}


# ---------------------------------------------------------------------------
# 3. Counterfactuals over the quality / gate variables
# ---------------------------------------------------------------------------

def _fused_emotion_for_quality(
    physio_probs: Dict[str, float],
    video_probs: Dict[str, float],
    eeg_q: str,
    gsr_q: str,
    video_q: str,
) -> str:
    """Re-run the gate under a hypothetical quality assignment; return argmax."""
    pq = physio_quality_of(eeg_q, gsr_q)
    return run_gate(physio_probs, video_probs, pq, video_q).fused_emotion


def single_variable_counterfactuals(
    physio_probs: Dict[str, float],
    video_probs: Dict[str, float],
    eeg_q: str,
    gsr_q: str,
    video_q: str,
) -> List[dict]:
    """
    Enumerate every single-variable change to {eeg, gsr, video} quality and
    report the ones that FLIP the fused decision.

    Each returned item:
        {"variable": "eeg"|"gsr"|"video",
         "from": <tier>, "to": <tier>,
         "new_emotion": <str>, "direction": "upgrade"|"downgrade"}
    """
    baseline = _fused_emotion_for_quality(physio_probs, video_probs, eeg_q, gsr_q, video_q)
    current = {"eeg": eeg_q, "gsr": gsr_q, "video": video_q}
    flips: List[dict] = []

    for var in ("eeg", "gsr", "video"):
        for tier in _QUALITY_TIERS:
            if tier == current[var]:
                continue
            trial = dict(current)
            trial[var] = tier
            new_e = _fused_emotion_for_quality(
                physio_probs, video_probs, trial["eeg"], trial["gsr"], trial["video"]
            )
            if new_e != baseline:
                flips.append({
                    "variable": var,
                    "from": current[var],
                    "to": tier,
                    "new_emotion": new_e,
                    "direction": "upgrade" if _QUALITY_RANK[tier] > _QUALITY_RANK[current[var]]
                                 else "downgrade",
                })
    return flips


# ---------------------------------------------------------------------------
# 4. Trust score for the resolution
# ---------------------------------------------------------------------------

def resolution_trust(trace: GateTrace) -> dict:
    """
    A simple, interpretable trust score for HOW the conflict was resolved.

    Two ingredients:
      * weight_margin  = |w_physio - w_video|   (decisiveness of the gate)
      * winner_quality = alpha of the dominant modality (was the winner clean?)

    trust = weight_margin * winner_quality_alpha, in [0, 1].
    Low trust  => the gate barely preferred a modality, or preferred one whose
                  signal quality was itself poor -> explanation flagged fragile.

    (This is a first, defensible heuristic; Phase 3 replaces/validates it with
     a faithfulness-grounded metric and an expert check.)
    """
    dom = trace.dominant_modality()
    winner_alpha = trace.physio_alpha if dom == "physio" else trace.video_alpha
    weight_margin = abs(trace.w_physio - trace.w_video)
    score = weight_margin * winner_alpha
    if score >= 0.5:
        band = "high"
    elif score >= 0.2:
        band = "moderate"
    else:
        band = "low"
    return {
        "trust_score": round(score, 4),
        "band": band,
        "weight_margin": round(weight_margin, 4),
        "winner_alpha": winner_alpha,
        "dominant_modality": dom,
    }


# ---------------------------------------------------------------------------
# 5. The explainer
# ---------------------------------------------------------------------------

@dataclass
class ConflictExplanation:
    """Structured, machine-readable conflict explanation (LLM/RAG-ready)."""
    is_conflict: bool
    physio_emotion: str
    video_emotion: str
    fused_emotion: str
    agrees_with: str
    gate: dict
    shapley: Dict[str, float]
    trust: dict
    counterfactuals: List[dict]
    rationale: str
    losing_modality_recovery: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "is_conflict": self.is_conflict,
            "physio_emotion": self.physio_emotion,
            "video_emotion": self.video_emotion,
            "fused_emotion": self.fused_emotion,
            "agrees_with": self.agrees_with,
            "gate": self.gate,
            "modality_shapley": {k: round(v, 4) for k, v in self.shapley.items()},
            "trust": self.trust,
            "counterfactuals": self.counterfactuals,
            "losing_modality_recovery": self.losing_modality_recovery,
            "rationale": self.rationale,
        }


class ConflictExplainer:
    """
    Turn a Member 2 `prediction_output` dict into a trust-aware, counterfactual
    conflict explanation. Pure function of the contract dict + gate arithmetic.
    """

    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes

    # -- public API --------------------------------------------------------

    def explain(self, pred: dict) -> ConflictExplanation:
        det = detect_conflict(pred)
        pm = pred["per_modality_predictions"]
        physio_probs = pm["physio"]["class_probabilities"]
        video_probs = pm["video"]["class_probabilities"]
        sq = pred["signal_quality"]
        eeg_q, gsr_q, video_q = sq["eeg"], sq["gsr"], sq["video"]

        # Re-derive the introspectable gate trace (twin of M2's fusion).
        pq = physio_quality_of(eeg_q, gsr_q)
        trace = run_gate(physio_probs, video_probs, pq, video_q)

        # Shapley attribution of the fused winning class.
        shap = modality_shapley(
            physio_probs, video_probs, trace.fused_probs,
            target_class=trace.fused_emotion, n_classes=self.n_classes,
        )

        trust = resolution_trust(trace)
        cfs = single_variable_counterfactuals(
            physio_probs, video_probs, eeg_q, gsr_q, video_q
        )

        recovery = self._losing_modality_recovery(
            det, physio_probs, video_probs, eeg_q, gsr_q, video_q
        )

        rationale = self._render_rationale(det, trace, shap, trust, recovery)

        return ConflictExplanation(
            is_conflict=det["is_conflict"],
            physio_emotion=det["physio_emotion"],
            video_emotion=det["video_emotion"],
            fused_emotion=det["fused_emotion"],
            agrees_with=det["agrees_with"],
            gate=trace.as_dict(),
            shapley=shap,
            trust=trust,
            counterfactuals=cfs,
            rationale=rationale,
            losing_modality_recovery=recovery,
        )

    # -- internals ---------------------------------------------------------

    def _losing_modality_recovery(
        self, det, physio_probs, video_probs, eeg_q, gsr_q, video_q
    ) -> Optional[dict]:
        """
        The headline counterfactual: could the LOSING modality have won if its
        signal quality were good?  Returns the minimal upgrade that flips the
        decision to the losing modality's emotion, if one exists.
        """
        if not det["is_conflict"]:
            return None

        loser = "video" if det["agrees_with"] == "physio" else "physio"
        loser_emotion = det["video_emotion"] if loser == "video" else det["physio_emotion"]

        # Upgrade the loser's quality to "good" and see whether it flips.
        if loser == "video":
            new_e = _fused_emotion_for_quality(
                physio_probs, video_probs, eeg_q, gsr_q, "good"
            )
            change = {"video": "good"}
        else:
            new_e = _fused_emotion_for_quality(
                physio_probs, video_probs, "good", "good", video_q
            )
            change = {"eeg": "good", "gsr": "good"}

        return {
            "losing_modality": loser,
            "losing_emotion": loser_emotion,
            "upgrade": change,
            "would_flip": new_e == loser_emotion,
            "new_fused_emotion": new_e,
        }

    def _render_rationale(self, det, trace: GateTrace, shap, trust, recovery) -> str:
        """
        A templated, plain-English rationale.  This is the SOURCE-OF-TRUTH text
        that the RAG/LLM layer (Phase 3) must stay faithful to — every claim
        here is derived directly from the gate arithmetic.
        """
        if not det["is_conflict"]:
            return (
                f"Both modalities agree on '{det['fused_emotion']}'. "
                f"Physiology weight={trace.w_physio:.2f}, video weight={trace.w_video:.2f}."
            )

        dom = trace.dominant_modality()
        dom_emotion = det["physio_emotion"] if dom == "physio" else det["video_emotion"]
        loser = "video" if dom == "physio" else "physio"
        dom_conf = trace.physio_conf if dom == "physio" else trace.video_conf
        dom_alpha = trace.physio_alpha if dom == "physio" else trace.video_alpha
        dom_qual = trace.physio_quality if dom == "physio" else trace.video_quality

        parts = [
            f"Conflict: physiology predicts '{det['physio_emotion']}' but video "
            f"predicts '{det['video_emotion']}'.",
            f"The quality gate resolved this in favour of {dom} "
            f"('{dom_emotion}'), giving it weight {max(trace.w_physio, trace.w_video):.2f} "
            f"vs {min(trace.w_physio, trace.w_video):.2f}.",
            f"It did so because {dom} had confidence {dom_conf:.2f} and its signal "
            f"quality was '{dom_qual}' (alpha={dom_alpha}), yielding the higher gate score.",
            f"Modality contribution to the final '{trace.fused_emotion}' decision "
            f"(Shapley): physio={shap['physio']:+.3f}, video={shap['video']:+.3f}.",
            f"Resolution trust: {trust['band']} (score={trust['trust_score']}).",
        ]
        if recovery and recovery["would_flip"]:
            parts.append(
                f"Counterfactual: had the {recovery['losing_modality']} signal quality "
                f"been good, the decision would have flipped to "
                f"'{recovery['new_fused_emotion']}'."
            )
        elif recovery:
            parts.append(
                f"Counterfactual: even with good {recovery['losing_modality']} signal "
                f"quality, the decision would remain '{recovery['new_fused_emotion']}'."
            )
        return " ".join(parts)


# Convenience one-shot function -------------------------------------------------

def explain_conflict(pred: dict) -> dict:
    """Explain a single prediction_output dict; return a plain dict."""
    return ConflictExplainer().explain(pred).as_dict()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from member3_explainability.conflict.synthetic_conflict import generate_conflict_case

    # A designed case: physio says 'stress' (good signal), video says 'happy'
    # (poor signal). The gate should trust physiology.
    case = generate_conflict_case(
        physio_emotion="stress", video_emotion="happy",
        eeg_quality="good", gsr_quality="good", video_quality="poor",
        seed=7,
    )
    exp = explain_conflict(case)
    print("=" * 70)
    print("DEMO: designed conflict (physio=stress/good, video=happy/poor)")
    print("=" * 70)
    print(exp["rationale"])
    print()
    print(json.dumps({k: v for k, v in exp.items() if k != "rationale"}, indent=2))
