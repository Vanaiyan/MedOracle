"""
member3_explainability/backend/llm_client.py
=============================================
SHAP-to-Prompt Bridging — LLM chatbot layer.

Primary  : Anthropic Claude (claude-sonnet-4-5)
Fallback  : OpenAI GPT-4o
Fallback2 : Simulated response (if neither API key is set — for offline demo)

Privacy note: only anonymised SHAP values and emotion labels are sent
— NO raw EEG/GSR/video signal data ever leaves the system.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are MedOracle's empathetic health insight assistant.
Your role is to explain emotion recognition results to clinicians and patients
in clear, accessible language.

When explaining results, you MUST:
1. Always cite the dominant modality (physiological or video) that most strongly
   influenced the prediction, and explain in lay terms what that modality measures.
   - Physiological (EEG + GSR): brain electrical activity and skin conductance,
     which reflect internal arousal and stress responses.
   - Video (facial expression): visible facial muscle movements that express emotion.
2. Mention any signal quality caveats (e.g. 'the physiological sensors had
   degraded signal quality, so this result should be interpreted with caution').
3. Use non-clinical, everyday language. Avoid jargon.
4. Never make diagnostic claims. Do not say 'you have X condition'.
5. Be empathetic and supportive in tone.
6. Keep responses concise — 3–5 sentences for auto-explanations,
   up to 8 sentences for follow-up questions."""


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------

def _build_context_block(shap_output: dict) -> str:
    """
    Inject the full shap_output as structured text into the prompt context.
    Only anonymised analytical data is included — no raw signals.
    """
    sv  = shap_output.get("shap_values", {})
    fi  = shap_output.get("feature_importance", {})
    sq  = shap_output.get("signal_quality", {})
    pmr = shap_output.get("per_modality_predictions", {})
    fs  = shap_output.get("faithfulness_score", "N/A")
    mw  = shap_output.get("modality_weights", {})

    physio_pred = pmr.get("physio", {})
    video_pred  = pmr.get("video",  {})

    dominant_feature = max(fi, key=fi.get) if fi else "unknown"
    dominant_modality = "physiological (EEG + GSR)" if dominant_feature in ("EEG", "GSR") else "facial video"

    lines = [
        "=== MedOracle Analysis Results ===",
        f"Predicted emotion     : {shap_output.get('predicted_emotion', 'unknown')}",
        f"Fused confidence      : {shap_output.get('confidence', 0.0):.1%}",
        "",
        "--- SHAP Contributions (signed, to predicted emotion probability) ---",
        f"  EEG (brain activity) : {sv.get('EEG', 0.0):+.4f}",
        f"  GSR (skin response)  : {sv.get('GSR', 0.0):+.4f}",
        f"  Video (facial)       : {sv.get('video', 0.0):+.4f}",
        f"  Dominant feature     : {dominant_feature} → {dominant_modality}",
        "",
        "--- Feature Importance (absolute SHAP) ---",
        f"  EEG   : {fi.get('EEG', 0.0):.4f}",
        f"  GSR   : {fi.get('GSR', 0.0):.4f}",
        f"  Video : {fi.get('video', 0.0):.4f}",
        "",
        "--- Modality Fusion Weights ---",
        f"  Physiological : {mw.get('physio', 0.0):.1%}",
        f"  Video         : {mw.get('video',  0.0):.1%}",
        "",
        "--- Signal Quality ---",
        f"  EEG   : {sq.get('eeg',   'unknown')}",
        f"  GSR   : {sq.get('gsr',   'unknown')}",
        f"  Video : {sq.get('video', 'unknown')}",
        "",
        "--- Per-Modality Predictions ---",
        f"  Physiological alone : {physio_pred.get('predicted_emotion','?')} ({physio_pred.get('confidence', 0):.1%} confidence)",
        f"  Video alone         : {video_pred.get('predicted_emotion','?')} ({video_pred.get('confidence', 0):.1%} confidence)",
        "",
        f"Explanation faithfulness score : {fs} (≥0.7 is acceptable)",
        "===================================",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM callers
# ---------------------------------------------------------------------------

async def _call_claude(
    messages: List[dict],
    system: str,
) -> str:
    """Call Anthropic Claude API."""
    try:
        import anthropic  # type: ignore
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=system,
            messages=messages,
        )
        return response.content[0].text.strip()
    except Exception as exc:
        logger.warning("Claude call failed: %s", exc)
        raise


async def _call_openai(
    messages: List[dict],
    system: str,
) -> str:
    """Call OpenAI GPT-4o API."""
    try:
        from openai import AsyncOpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = AsyncOpenAI(api_key=api_key)
        full_messages = [{"role": "system", "content": system}] + messages
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=full_messages,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("OpenAI call failed: %s", exc)
        raise


def _simulated_response(shap_output: dict, user_message: str) -> str:
    """
    Offline fallback when no LLM API key is configured.
    Generates a rule-based explanation from the SHAP values.
    """
    emotion  = shap_output.get("predicted_emotion", "unknown")
    conf     = shap_output.get("confidence", 0.0)
    fi       = shap_output.get("feature_importance", {})
    sq       = shap_output.get("signal_quality", {})
    mw       = shap_output.get("modality_weights", {})

    dominant = max(fi, key=fi.get) if fi else "EEG"
    dom_label = {
        "EEG":   "brain electrical activity (EEG)",
        "GSR":   "skin conductance response (GSR)",
        "video": "facial expression analysis",
    }.get(dominant, dominant)

    physio_w = mw.get("physio", 0.5)
    video_w  = mw.get("video",  0.5)

    quality_note = ""
    degraded = [k for k, v in sq.items() if v in ("degraded", "poor")]
    if degraded:
        quality_note = (
            f" Note: the {', '.join(degraded)} signal(s) had reduced quality, "
            "so this result should be interpreted with some caution."
        )

    return (
        f"The system detected **{emotion}** with {conf:.0%} confidence. "
        f"The most influential signal was {dom_label}, which contributed the most "
        f"to this prediction according to the SHAP analysis. "
        f"The physiological sensors (EEG + GSR) were weighted at {physio_w:.0%} "
        f"and the facial video at {video_w:.0%} in the final fusion.{quality_note} "
        f"This is an analytical result — please consult a qualified clinician "
        f"for any health-related decisions."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_llm_response(
    shap_output: dict,
    user_message: str,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    """
    Main entry point for the LLM chatbot.

    Parameters
    ----------
    shap_output           : dict — full shap_output from build_shap_output()
    user_message          : str  — the user's typed question or auto-trigger
    conversation_history  : list — previous [{role, content}] messages for multi-turn

    Returns
    -------
    str — plain-English LLM response
    """
    context_block = _build_context_block(shap_output)

    # Build message list: context injection + history + current user message
    messages: List[dict] = []

    # Inject context as first user turn if no history yet
    if not conversation_history:
        messages.append({
            "role":    "user",
            "content": f"Here are the analysis results:\n\n{context_block}\n\nPlease explain these results.",
        })
        messages.append({
            "role":    "assistant",
            "content": _simulated_response(shap_output, "auto-explain"),  # Will be overwritten by LLM
        })

    # Add conversation history
    if conversation_history:
        # Ensure first message contains context block
        first = conversation_history[0]
        if context_block not in first.get("content", ""):
            conversation_history[0]["content"] = (
                f"Here are the analysis results:\n\n{context_block}\n\n"
                + first["content"]
            )
        messages = conversation_history

    # Add current user message
    messages.append({"role": "user", "content": user_message})

    # Try Claude → OpenAI → simulated fallback
    try:
        return await _call_claude(messages, _SYSTEM_PROMPT)
    except Exception:
        pass

    try:
        return await _call_openai(messages, _SYSTEM_PROMPT)
    except Exception:
        pass

    logger.info("Using simulated LLM response (no API keys configured).")
    return _simulated_response(shap_output, user_message)


async def get_auto_explanation(shap_output: dict) -> str:
    """
    Generate the auto-explanation card shown after every prediction.
    Triggered automatically — no user question needed.
    """
    trigger = (
        f"Please provide a brief, empathetic explanation of these emotion analysis results "
        f"for the patient/clinician. Focus on the dominant modality and what it means."
    )
    return await get_llm_response(shap_output, trigger, conversation_history=None)
