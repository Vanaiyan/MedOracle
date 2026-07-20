"""
member3_explainability/evaluation/metrics.py
============================================
Deterministic scoring of a natural-language explanation against the STRUCTURED
conflict explanation it is supposed to describe.

The novelty vs off-the-shelf RAG metrics (RAGAS etc.): faithfulness here is
measured against the model's own attribution / gate maths, not against retrieved
documents. Retrieved knowledge only widens what counts as a *grounded* mechanism
claim (the RAG effect we measure).

Claim types checked
-------------------
* emotion      : every emotion word asserted must be one of {fused, physio,
                 video}; any other emotion is an invented (hallucinated) claim.
* dominant     : a "trusted / favoured / higher weight" statement must name the
                 modality the gate actually preferred.
* counterfactual: an "if ... quality had been good ..." statement must agree
                 with the recovery counterfactual.
* mechanism    : a physiological/psychological mechanism term (sympathetic,
                 arousal, alpha, ...) is grounded only if a retrieved fact
                 supports it; otherwise it is unsupported (hallucinated).

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

EMOTIONS = ["stress", "calm", "happy", "sad", "angry"]
_PHYSIO_WORDS = ["physio", "physiolog", "eeg", "gsr"]
_VIDEO_WORDS = ["video", "facial", "face"]
MECHANISM_TERMS = {
    "sympathetic", "arousal", "valence", "alpha", "beta", "theta",
    "circumplex", "entropy", "calibration",
}


def _toks(text: str) -> Set[str]:
    return set(re.findall(r"[a-z_]+", text.lower()))


def _modality_in(text: str, modality: str) -> bool:
    words = _PHYSIO_WORDS if modality == "physio" else _VIDEO_WORDS
    return any(w in text for w in words)


def score_explanation(
    text: str,
    conflict: Dict,
    grounded_terms: Optional[Set[str]] = None,
) -> Dict:
    """
    Score one explanation. `grounded_terms` = terms made legitimate by retrieved
    RAG facts (pass an empty set to simulate RAG OFF).

    Returns faithfulness, hallucination_rate, clinical_relevance in [0,1] plus a
    per-claim breakdown.
    """
    grounded_terms = grounded_terms or set()
    t = text.lower()
    toks = _toks(t)

    fused = conflict["fused_emotion"]
    allowed = {fused, conflict["physio_emotion"], conflict["video_emotion"]}
    dom = conflict["gate"]["dominant_modality"]
    rec = conflict.get("losing_modality_recovery")

    claims: List[tuple] = []   # (type, status)  status in correct/incorrect/unsupported

    # -- emotion claims ----------------------------------------------------
    for e in EMOTIONS:
        if re.search(rf"\b{e}\b", t):
            claims.append(("emotion", "correct" if e in allowed else "unsupported"))

    # -- dominant-modality claim ------------------------------------------
    # Capture the modality named right after a trust/favour/weight cue.
    cue = re.search(
        r"(?:favou?r\w*\s+of|trust\w*(?:\s+the)?|relied\s+on|resolved[^.]*?favou?r\w*\s+of|"
        r"higher\s+weight[^.]*?)\s+(physi\w*|video|facial|face|eeg|gsr)",
        t,
    )
    if cue:
        said = "physio" if cue.group(1)[:3] in ("phy", "eeg", "gsr") else "video"
        claims.append(("dominant", "correct" if said == dom else "incorrect"))

    # -- counterfactual claim ---------------------------------------------
    if rec and re.search(r"\b(if|had)\b", t) and re.search(r"(good|better|been)", t):
        names_loser = _modality_in(t, rec["losing_modality"])
        names_newemotion = bool(re.search(rf"\b{rec['new_fused_emotion']}\b", t))
        if rec["would_flip"]:
            ok = names_loser and names_newemotion
        else:
            ok = names_loser and not names_newemotion or names_loser
        claims.append(("counterfactual", "correct" if ok else "incorrect"))

    # -- mechanism grounding (the RAG-sensitive part) ----------------------
    for term in MECHANISM_TERMS & toks:
        claims.append(("mechanism", "correct" if term in grounded_terms else "unsupported"))

    # -- aggregate ---------------------------------------------------------
    total = len(claims)
    correct = sum(1 for _, s in claims if s == "correct")
    bad = sum(1 for _, s in claims if s in ("incorrect", "unsupported"))
    faithfulness = correct / total if total else 0.0
    hallucination = bad / total if total else 0.0

    clinical = _clinical_relevance(t, conflict, grounded_terms, toks)

    return {
        "faithfulness": round(faithfulness, 4),
        "hallucination_rate": round(hallucination, 4),
        "clinical_relevance": round(clinical, 4),
        "n_claims": total,
        "claims": claims,
    }


def _clinical_relevance(t: str, conflict: Dict, grounded: Set[str], toks: Set[str]) -> float:
    """Rubric: concepts a clinically useful conflict explanation should contain."""
    checks = {
        "names_both_modalities": _modality_in(t, "physio") and _modality_in(t, "video"),
        "notes_disagreement": bool(re.search(r"(conflict|disagree|differ|but|however)", t)),
        "mentions_quality_or_confidence":
            bool(re.search(r"(quality|good|degraded|poor|confidence|weight)", t)),
        "states_correct_emotion": bool(re.search(rf"\b{conflict['fused_emotion']}\b", t)),
        "uses_grounded_mechanism": bool((MECHANISM_TERMS & toks) & grounded),
    }
    return sum(checks.values()) / len(checks)
