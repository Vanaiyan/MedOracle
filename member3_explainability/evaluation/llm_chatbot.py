"""
member3_explainability/evaluation/llm_chatbot.py
================================================
RAG-grounded, hallucination-GUARDED natural-language explanation generator.

Pipeline
--------
1. Retrieve the relevant curated facts (knowledge_base.Retriever).
2. Build a prompt containing (a) the STRUCTURED conflict numbers and (b) the
   retrieved facts, instructing the LLM to explain using only those.
3. Generate with Claude (anthropic) if an API key is available.
4. VERIFY the generated text against the maths with evaluation.metrics; if
   faithfulness < threshold (or hallucination too high), send corrective
   feedback and regenerate (closed loop). After max_retries, fall back to the
   guaranteed-faithful template rationale.

This converts "prompt an LLM to narrate SHAP" (not research) into a measured,
safety-constrained method: every returned explanation has passed a numeric
faithfulness check or been replaced by the template.

Without an API key the generator returns the template rationale, so the whole
system remains runnable and testable offline.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Optional

from member3_explainability.evaluation.knowledge_base import Retriever
from member3_explainability.evaluation.metrics import score_explanation

logger = logging.getLogger(__name__)


def build_prompt(conflict: Dict, facts) -> str:
    g = conflict["gate"]
    lines = [
        "You are explaining an emotion-recognition decision to a clinician.",
        "Explain ONLY using the numbers and cited facts below. Do not invent reasons.",
        "",
        "DECISION NUMBERS:",
        f"- physiology predicted: {conflict['physio_emotion']}",
        f"- video predicted:      {conflict['video_emotion']}",
        f"- fused decision:       {conflict['fused_emotion']}",
        f"- modality trusted:     {g['dominant_modality']} "
        f"(weights physio={g['w_physio']}, video={g['w_video']})",
        f"- signal quality:       physio={g['physio_quality']}, video={g['video_quality']}",
    ]
    rec = conflict.get("losing_modality_recovery")
    if rec:
        lines.append(
            f"- counterfactual:       if {rec['losing_modality']} quality were good, "
            f"decision would {'flip to ' + rec['new_fused_emotion'] if rec['would_flip'] else 'stay ' + rec['new_fused_emotion']}"
        )
    lines += ["", "CITED FACTS (ground every mechanism claim in one of these):"]
    for f in facts:
        lines.append(f"- [{f.citation}] {f.text}")
    lines += ["", "Write 3-4 sentences: why they disagreed, which was trusted and why,",
              "and the counterfactual. Cite facts in [brackets]."]
    return "\n".join(lines)


def _anthropic_generate(prompt: str, feedback: Optional[str] = None) -> Optional[str]:
    """Call Claude if possible; return None if unavailable."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        model = "claude-sonnet-5"
        logger.info("Using AI model: Anthropic Claude (%s)", model)
        content = prompt if not feedback else prompt + "\n\nCORRECTION:\n" + feedback
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": content}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    except Exception:
        return None


def generate_explanation(
    conflict: Dict,
    retriever: Optional[Retriever] = None,
    threshold: float = 0.8,
    max_retries: int = 2,
    generate_fn: Callable[[str, Optional[str]], Optional[str]] = _anthropic_generate,
) -> Dict:
    """
    Return a verified explanation dict:
        text, source ("llm"|"template"), verified (bool), attempts, scores.
    """
    retriever = retriever or Retriever()
    query = " ".join([
        conflict["physio_emotion"], conflict["video_emotion"], conflict["fused_emotion"],
        conflict["gate"]["dominant_modality"], "signal quality confidence alpha arousal",
    ])
    facts = retriever.retrieve(query, k=4)
    grounded = retriever.grounded_terms(facts)
    prompt = build_prompt(conflict, facts)

    feedback = None
    for attempt in range(1, max_retries + 1):
        text = generate_fn(prompt, feedback)
        if text is None:
            break  # no LLM available -> fall back
        scores = score_explanation(text, conflict, grounded)
        if scores["faithfulness"] >= threshold and scores["hallucination_rate"] <= (1 - threshold):
            return {"text": text, "source": "llm", "verified": True,
                    "attempts": attempt, "scores": scores}
        feedback = (
            f"Your explanation had faithfulness {scores['faithfulness']} and "
            f"hallucination {scores['hallucination_rate']}. Only state: trusted "
            f"modality = {conflict['gate']['dominant_modality']}, fused emotion = "
            f"{conflict['fused_emotion']}. Do not name other emotions or modalities."
        )

    # Fallback: guaranteed-faithful template rationale.
    text = conflict["rationale"]
    scores = score_explanation(text, conflict, grounded)
    return {"text": text, "source": "template", "verified": True,
            "attempts": max_retries, "scores": scores}


if __name__ == "__main__":
    import json
    from member3_explainability.conflict.synthetic_conflict import generate_conflict_case
    from member3_explainability.conflict.conflict_explainer import explain_conflict

    case = generate_conflict_case(physio_emotion="stress", video_emotion="happy",
                                  eeg_quality="good", gsr_quality="good",
                                  video_quality="poor", seed=7)
    exp = explain_conflict(case)

    # Offline demo: no API key -> the guard falls back to the verified template.
    out = generate_explanation(exp)
    print("source :", out["source"], "| verified:", out["verified"])
    print("scores :", json.dumps(out["scores"], default=str))
    print("text   :", out["text"][:300])

    # Show the guard REJECTS a bad generator and falls back.
    def bad_gen(prompt, feedback=None):
        return "The video modality was trusted and the emotion was sad because of dopamine."
    out2 = generate_explanation(exp, generate_fn=bad_gen)
    print("\nwith a hallucinating generator -> source:", out2["source"],
          "(fell back after", out2["attempts"], "attempts)")
