"""
member3_explainability/evaluation/check_rag.py
==============================================
A quick, human-readable check that the RAG layer is working.



It shows three things:
  1. RETRIEVAL   — the knowledge base returns relevant cited facts for a query.
  2. GROUNDING   — those facts change what counts as a supported claim.
  3. EFFECT      — the same explanation has LOWER hallucination with RAG ON.

If retrieval returns facts and hallucination(RAG on) < hallucination(RAG off),
RAG is working.


"""

from __future__ import annotations

from member3_explainability.conflict.synthetic_conflict import generate_conflict_case
from member3_explainability.conflict.conflict_explainer import explain_conflict
from member3_explainability.evaluation.knowledge_base import Retriever
from member3_explainability.evaluation.metrics import score_explanation


def main() -> None:
    retriever = Retriever()

    # A conflict case + its (faithful) explanation text.
    case = generate_conflict_case(physio_emotion="stress", video_emotion="happy",
                                  eeg_quality="good", gsr_quality="good",
                                  video_quality="poor", seed=7)
    exp = explain_conflict(case)
    text = exp["rationale"]

    query = " ".join([exp["physio_emotion"], exp["video_emotion"], exp["fused_emotion"],
                      exp["gate"]["dominant_modality"], "signal quality confidence alpha arousal"])
    facts = retriever.retrieve(query, k=4)
    grounded = retriever.grounded_terms(facts)

    print("=" * 70)
    print("1. RETRIEVAL — facts the knowledge base returned for this case")
    print("=" * 70)
    for f in facts:
        print(f"  [{f.citation}] {f.text}")
    print(f"\n  grounded terms enabled by retrieval: {sorted(grounded)}")

    on = score_explanation(text, exp, grounded)      # RAG ON
    off = score_explanation(text, exp, set())        # RAG OFF (no grounding)

    print("\n" + "=" * 70)
    print("2/3. EFFECT — same explanation, scored with RAG ON vs OFF")
    print("=" * 70)
    print(f"  RAG ON  -> faithfulness {on['faithfulness']:.3f} | hallucination {on['hallucination_rate']:.3f}")
    print(f"  RAG OFF -> faithfulness {off['faithfulness']:.3f} | hallucination {off['hallucination_rate']:.3f}")

    working = len(facts) > 0 and on["hallucination_rate"] < off["hallucination_rate"]
    print("\n" + ("RAG IS WORKING  ✅  (retrieval returns facts AND grounding lowers hallucination)"
                  if working else
                  "RAG NOT behaving as expected  ❌"))


if __name__ == "__main__":
    main()
