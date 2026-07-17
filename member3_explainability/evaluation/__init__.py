"""
member3_explainability/evaluation
=================================
Phase 3 — quantitative evaluation of the natural-language conflict explanations.

Three metrics, scored against the STRUCTURED conflict explanation (the gate /
Shapley / counterfactual numbers), NOT against retrieved documents:

  * Faithfulness       — do the text's claims match the underlying maths?
  * Hallucination rate — did it assert things unsupported by the maths or the
                         curated knowledge base?
  * Clinical relevance — does it cover the concepts a clinician needs, with
                         correct terminology?

RAG grounding (knowledge_base.py) is the mechanism that lets the LLM cite a
source for each mechanistic claim; the harness measures hallucination with RAG
ON vs OFF to show the grounding actually helps.

Author : Adshaya Balarajah (214024V)
"""

from member3_explainability.evaluation.knowledge_base import Retriever, KNOWLEDGE_BASE
from member3_explainability.evaluation.metrics import score_explanation

__all__ = ["Retriever", "KNOWLEDGE_BASE", "score_explanation"]
