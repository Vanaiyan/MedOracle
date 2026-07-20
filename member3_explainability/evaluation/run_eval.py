"""
member3_explainability/evaluation/run_eval.py
=============================================
Phase 3 harness — runs WITHOUT any LLM key.

It demonstrates two things the evaluation framework must do:

 (A) DISCRIMINATION: a faithful explanation (the gate-derived template rationale)
     scores high faithfulness / low hallucination, while a deliberately
     corrupted explanation (wrong modality, invented emotion, wrong
     counterfactual) scores the opposite. If the metrics could not tell these
     apart they would be worthless.

 (B) RAG EFFECT: the same faithful explanation is scored with retrieval ON vs
     OFF. With grounding, mechanistic terms (alpha, arousal, ...) are supported
     by a cited fact, so measured hallucination drops.

The LLM chatbot (llm_chatbot.py) plugs in as another "explanation source" whose
output is scored by exactly these metrics; here we use the template + a
corrupted variant so the harness is fully runnable offline.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import statistics as stats
from typing import Dict

from member3_explainability.conflict.synthetic_conflict import generate_conflict_dataset
from member3_explainability.conflict.conflict_explainer import explain_conflict
from member3_explainability.evaluation.metrics import score_explanation, EMOTIONS
from member3_explainability.evaluation.knowledge_base import Retriever


def corrupted_explanation(conflict: Dict) -> str:
    """A plausible-sounding but WRONG explanation (hallucinated)."""
    dom = conflict["gate"]["dominant_modality"]
    wrong_mod = "video" if dom == "physio" else "physiology"
    allowed = {conflict["fused_emotion"], conflict["physio_emotion"], conflict["video_emotion"]}
    invented = next(e for e in EMOTIONS if e not in allowed)
    return (
        f"The system trusted the {wrong_mod} modality and the final emotion was "
        f"{invented}. If the {wrong_mod} signal had been good the decision would "
        f"have become {invented}. This reflects the subject's underlying mood."
    )


def _grounded_for(conflict: Dict, retriever: Retriever) -> set:
    query = " ".join([
        conflict["physio_emotion"], conflict["video_emotion"], conflict["fused_emotion"],
        conflict["gate"]["dominant_modality"], "signal quality confidence alpha arousal",
    ])
    return retriever.grounded_terms(retriever.retrieve(query, k=4))


def run(n: int = 150, seed: int = 42) -> Dict:
    data = generate_conflict_dataset(n=n, seed=seed)
    retriever = Retriever()

    rows = {
        "faithful_RAG_on": [],
        "faithful_RAG_off": [],
        "corrupted_RAG_on": [],
    }
    for pred in data:
        exp = explain_conflict(pred)
        if not exp["is_conflict"]:
            continue
        grounded = _grounded_for(exp, retriever)
        faithful_text = exp["rationale"]
        bad_text = corrupted_explanation(exp)

        rows["faithful_RAG_on"].append(score_explanation(faithful_text, exp, grounded))
        rows["faithful_RAG_off"].append(score_explanation(faithful_text, exp, set()))
        rows["corrupted_RAG_on"].append(score_explanation(bad_text, exp, grounded))

    def agg(key, metric):
        vals = [r[metric] for r in rows[key]]
        return stats.mean(vals), stats.pstdev(vals)

    print("=" * 74)
    print(f"PHASE 3 EVALUATION  (n={len(rows['faithful_RAG_on'])} conflict cases)")
    print("=" * 74)
    header = f"{'condition':22s} {'faithfulness':>14s} {'halluc.rate':>13s} {'clin.rel':>10s}"
    print(header); print("-" * len(header))
    for key in rows:
        f_m, f_s = agg(key, "faithfulness")
        h_m, h_s = agg(key, "hallucination_rate")
        c_m, _ = agg(key, "clinical_relevance")
        print(f"{key:22s} {f_m:6.3f}±{f_s:4.2f}   {h_m:6.3f}±{h_s:4.2f}   {c_m:8.3f}")

    print("\nReads:")
    print("  (A) faithful vs corrupted -> metrics separate good from hallucinated.")
    print("  (B) RAG on vs off (faithful row) -> grounding lowers hallucination.")
    return rows


if __name__ == "__main__":
    run()
