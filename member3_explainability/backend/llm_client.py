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

_SYSTEM_PROMPT = """You are MedOracle's empathetic health insight assistant.
Your role is to explain emotion recognition results to clinicians and patients
in clear, accessible language.

When explaining results, you MUST:
1. Always cite the dominant modality (physiological or video) that most strongly
   influenced the prediction, and explain in lay terms what that modality measures.
   - Physiological (EEG + GSR): brain electrical activity and skin conductance,
     which reflect internal arousal and stress responses.
   - Video (facial expression): visible facial muscle movements that express emotion.
2. Mention any signal quality caveats if quality is degraded or poor.
3. Use non-clinical, everyday language. Avoid jargon.
4. Never make diagnostic claims. Do not say 'you have X condition'.
5. Be empathetic and supportive in tone.
6. Keep responses concise — 3-5 sentences for auto-explanations,
   up to 8 sentences for follow-up questions.
7. When answering follow-up questions, directly address what was asked.
   Do NOT repeat the full analysis summary if the user asks a specific question."""


def _build_context_block(shap_output: dict) -> str:
    sv  = shap_output.get("shap_values", {})
    fi  = shap_output.get("feature_importance", {})
    sq  = shap_output.get("signal_quality", {})
    pmr = shap_output.get("per_modality_predictions", {})
    fs  = shap_output.get("faithfulness_score", "N/A")
    mw  = shap_output.get("modality_weights", {})

    physio_pred = pmr.get("physio", {})
    video_pred  = pmr.get("video",  {})

    dominant_feature  = max(fi, key=fi.get) if fi else "unknown"
    dominant_modality = "physiological (EEG + GSR)" if dominant_feature in ("EEG", "GSR") else "facial video"

    lines = [
        "=== MedOracle Analysis Results ===",
        f"Predicted emotion     : {shap_output.get('predicted_emotion', 'unknown')}",
        f"Fused confidence      : {shap_output.get('confidence', 0.0):.1%}",
        "",
        "--- SHAP Contributions ---",
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
        f"Explanation faithfulness score : {fs} (>=0.7 is acceptable)",
        "===================================",
    ]
    return "\n".join(lines)


async def _call_claude(messages: List[dict], system: str) -> str:
    try:
        import anthropic
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


async def _call_openai(messages: List[dict], system: str) -> str:
    try:
        from openai import AsyncOpenAI
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
    """Offline fallback — generates varied responses based on user question."""
    emotion  = shap_output.get("predicted_emotion", "unknown")
    conf     = shap_output.get("confidence", 0.0)
    fi       = shap_output.get("feature_importance", {})
    sq       = shap_output.get("signal_quality", {})
    mw       = shap_output.get("modality_weights", {})

    dominant  = max(fi, key=fi.get) if fi else "EEG"
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
            "so interpret this result with some caution."
        )

    msg_lower = user_message.lower()

    # Answer specific questions differently
    if any(w in msg_lower for w in ["why", "reason", "how come", "what caused"]):
        return (
            f"The {emotion} prediction was primarily driven by {dom_label}, "
            f"which showed the strongest indicators for this emotional state. "
            f"The SHAP analysis assigned it the highest feature importance score "
            f"among all three signals (EEG, GSR, video).{quality_note}"
        )
    elif any(w in msg_lower for w in ["confidence", "sure", "certain", "accurate"]):
        return (
            f"The system is {conf:.0%} confident in the {emotion} prediction. "
            f"This confidence is calculated from the fused probability distribution "
            f"across all five emotion classes. "
            f"A score above 70% is generally considered reliable.{quality_note}"
        )
    elif any(w in msg_lower for w in ["eeg", "brain", "electrical"]):
        return (
            f"EEG (electroencephalography) measures electrical activity in the brain. "
            f"In this session, EEG had a SHAP importance score of {fi.get('EEG', 0):.4f}, "
            f"making it {'the dominant' if dominant == 'EEG' else 'a secondary'} signal. "
            f"Brain activity patterns in specific frequency bands (alpha, beta, theta) "
            f"are strongly linked to emotional arousal and valence."
        )
    elif any(w in msg_lower for w in ["gsr", "skin", "conductance"]):
        return (
            f"GSR (galvanic skin response) measures how much your skin conducts electricity, "
            f"which increases with emotional arousal and stress. "
            f"In this session, GSR had a SHAP importance of {fi.get('GSR', 0):.4f}. "
            f"It was weighted at {physio_w:.0%} combined with EEG in the physiological modality.{quality_note}"
        )
    elif any(w in msg_lower for w in ["video", "face", "facial", "camera"]):
        return (
            f"The video modality analyzes facial muscle movements captured by camera. "
            f"In this session, video had a SHAP importance of {fi.get('video', 0):.4f} "
            f"and was weighted at {video_w:.0%} in the final fusion. "
            f"{'It was the dominant signal.' if dominant == 'video' else 'The physiological signals were more influential this time.'}{quality_note}"
        )
    elif any(w in msg_lower for w in ["quality", "signal", "reliable", "trust"]):
        if degraded:
            return (
                f"The {', '.join(degraded)} signal(s) had degraded or poor quality during this session. "
                f"This means the system automatically reduced their influence in the final prediction. "
                f"For more reliable results, ensure sensors are properly attached and the face is clearly visible."
            )
        else:
            return (
                f"All signals (EEG, GSR, and video) had good quality during this session. "
                f"This means the {emotion} prediction is based on clean, reliable data "
                f"and can be interpreted with higher confidence."
            )
    elif any(w in msg_lower for w in ["shap", "explain", "interpret", "faithfulness"]):
        fs = shap_output.get("faithfulness_score", 0)
        return (
            f"SHAP (SHapley Additive exPlanations) measures how much each signal "
            f"contributed to the final prediction. "
            f"The faithfulness score of {fs:.2f} {'confirms' if fs >= 0.7 else 'suggests'} "
            f"that these explanations {'genuinely reflect' if fs >= 0.7 else 'may not fully reflect'} "
            f"the model's decision-making. "
            f"The dominant contributor was {dom_label}."
        )
    else:
        # Default auto-explanation
        return (
            f"The system detected **{emotion}** with {conf:.0%} confidence. "
            f"The most influential signal was {dom_label}. "
            f"Physiological sensors were weighted at {physio_w:.0%} "
            f"and facial video at {video_w:.0%}.{quality_note} "
            f"This is an analytical result — consult a qualified clinician "
            f"for any health-related decisions."
        )


async def get_llm_response(
    shap_output: dict,
    user_message: str,
    conversation_history: Optional[List[dict]] = None,
) -> str:
    context_block = _build_context_block(shap_output)

    # Build messages — context always injected in first user message
    if not conversation_history:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Here are the MedOracle analysis results:\n\n{context_block}\n\n"
                    f"{user_message}"
                ),
            }
        ]
    else:
        messages = list(conversation_history)
        # Ensure context is in the first message
        first = messages[0]
        if context_block not in first.get("content", ""):
            messages[0] = {
                **first,
                "content": f"Here are the MedOracle analysis results:\n\n{context_block}\n\n{first['content']}",
            }
        # Add current user message
        messages.append({"role": "user", "content": user_message})

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
    trigger = (
        "Please provide a brief, empathetic explanation of these emotion analysis results "
        "for the patient or clinician. Focus on the dominant modality, what it means in "
        "everyday language, and any signal quality caveats."
    )
    return await get_llm_response(shap_output, trigger, conversation_history=None)