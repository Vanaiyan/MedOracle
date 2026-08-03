"""
member3_explainability/evaluation/knowledge_base.py
===================================================
The small, curated RAG knowledge base + a dependency-free retriever.

Scope is deliberately narrow and OWNED by the project: the harmonisation
citations and signal-interpretation facts that ground an emotion explanation.
A generic "RAG over the web" would add nothing defensible; grounding in exactly
this domain knowledge is what lets the LLM cite a source for every claim and is
what we measure (hallucination down, groundedness up, RAG on vs off).

Each fact has: id, text, tags (for retrieval), and a citation.


"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Fact:
    id: str
    text: str
    tags: List[str]
    citation: str


# --- curated facts (project citations + signal-interpretation knowledge) ----
KNOWLEDGE_BASE: List[Fact] = [
    Fact("russell_circumplex",
         "Emotions can be placed on a circumplex of valence and arousal; high "
         "arousal with low valence corresponds to stress/anger, low arousal with "
         "high valence to calm.",
         ["valence", "arousal", "stress", "calm", "angry", "circumplex"],
         "Russell (1980)"),
    Fact("kreibig_sympathetic",
         "Elevated GSR skin-conductance responses reflect sympathetic nervous "
         "system activation, associated with stress and fear states.",
         ["gsr", "gsr_scr", "sympathetic", "stress", "arousal", "physio"],
         "Kreibig (2010)"),
    Fact("eeg_alpha_calm",
         "Increased EEG alpha-band power is associated with relaxed, calm, low-"
         "arousal states.",
         ["eeg", "eeg_alpha", "calm", "arousal", "physio"],
         "Koelstra et al. (2012)"),
    Fact("eeg_beta_arousal",
         "Increased EEG beta-band activity is associated with heightened arousal, "
         "alertness and stress.",
         ["eeg", "eeg_beta", "stress", "arousal", "physio"],
         "Koelstra et al. (2012)"),
    Fact("guo_confidence",
         "Entropy-based confidence (1 - H/logK) calibrates how peaked a model's "
         "probability distribution is; lower entropy means higher confidence.",
         ["confidence", "entropy", "calibration", "gate"],
         "Guo et al. (2017)"),
    Fact("quality_alpha",
         "Signal-quality gating applies a penalty factor alpha (good=1.0, "
         "degraded=0.5, poor=0.1) so that low-quality modalities contribute less "
         "to the fused decision.",
         ["quality", "alpha", "gate", "degraded", "poor", "good", "fusion"],
         "MedOracle gating (Guo et al., 2017)"),
    Fact("facial_valence",
         "Facial action units such as smiles (AU12) and brow raises (AU1/2) index "
         "emotional valence from the video modality.",
         ["video", "au_smile", "au_brow", "face_valence", "happy", "valence"],
         "Cao et al. (2014)"),
    Fact("late_fusion",
         "Late fusion combines independent per-modality predictions; it is robust "
         "when modalities have different noise characteristics and reliability.",
         ["fusion", "late", "modality", "video", "physio", "weight"],
         "Poria et al. (2017)"),
]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z_]+", text.lower())


class Retriever:
    """Tiny keyword-overlap retriever (no external deps)."""

    def __init__(self, facts: List[Fact] = None):
        self.facts = facts or KNOWLEDGE_BASE

    def retrieve(self, query: str, k: int = 3) -> List[Fact]:
        q = set(_tokens(query))
        scored = []
        for f in self.facts:
            score = len(q & set(f.tags)) * 2 + len(q & set(_tokens(f.text)))
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [f for _, f in scored[:k]]

    def grounded_terms(self, facts: List[Fact]) -> set:
        """All tag/keyword terms considered 'grounded' by the retrieved facts."""
        terms = set()
        for f in facts:
            terms |= set(f.tags)
        return terms
