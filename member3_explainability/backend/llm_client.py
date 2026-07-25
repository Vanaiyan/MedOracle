"""
member3_explainability/backend/llm_client.py
=============================================
SHAP-to-Prompt Bridging — LLM chatbot layer.

Primary  : Google Gemini 1.5 Flash (free tier — no billing required)
Fallback : Simulated response (if API key not set or quota exceeded)

Topic restriction: the assistant ONLY answers questions related to
emotion recognition, SHAP explainability, EEG/GSR/video signals,
and the MedOracle session results. Off-topic questions are politely refused.

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
# System prompt — defines persona + topic restriction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are MedOracle Assistant, an AI embedded inside a multimodal
emotion recognition system. Your ONLY purpose is to explain emotion analysis results
to clinicians and patients.

STRICT TOPIC RESTRICTION — you must ONLY answer questions about:
- The emotion predicted in this session (stress, calm, happy, sad, angry)
- SHAP explainability values (what EEG, GSR, and video contributed)
- Signal quality (EEG, GSR, video) and what it means
- Physiological signals: EEG (brain electrical activity), GSR (skin conductance)
- Facial video analysis and what it measures
- Modality fusion weights and confidence scores
- Faithfulness scores and what they indicate
- General emotion science directly relevant to the results shown

If the user asks ANYTHING outside these topics (e.g. general health advice,
diagnoses, coding, weather, unrelated science, personal questions, etc.),
you MUST respond with exactly:
"I can only help with questions about your MedOracle emotion analysis results.
Please ask me about the emotion detected, SHAP values, signal quality, or
what the modalities measure."

When explaining results, you MUST:
1. Always cite the dominant modality (EEG, GSR, or video facial) that most
   influenced the prediction, and explain in plain everyday language what it measures.
   - EEG: brain electrical activity reflecting internal arousal and stress.
   - GSR: skin conductance response reflecting emotional arousal.
   - Video: facial muscle movements expressing emotion.
2. Mention signal quality caveats if quality is degraded or poor.
3. Use non-clinical, everyday language. Avoid jargon.
4. Never make diagnostic claims. Do not say 'you have X condition'.
5. Be empathetic and supportive in tone.
6. Keep responses concise — 3-5 sentences for auto-explanations,
   up to 8 sentences for follow-up questions.
7. When answering follow-up questions, directly address what was asked.
   Do NOT repeat the full analysis summary if the user asks a specific question."""


# ---------------------------------------------------------------------------
# Off-topic guard — quick keyword check before calling the LLM
# ---------------------------------------------------------------------------

_EMOTION_KEYWORDS = {
    "emotion", "stress", "calm", "happy", "sad", "angry", "fear", "anxiety",
    "eeg", "gsr", "galvanic", "skin", "brain", "video", "facial", "face",
    "shap", "modality", "physio", "physiological", "signal", "quality",
    "confidence", "prediction", "fusion", "weight", "faithfulness",
    "explain", "analysis", "result", "session", "medoracle", "arousal",
    "electroencephalography", "conductance", "detection", "recognition",
    "why", "what", "how", "dominant", "contribute", "reliable", "trust",
    "interpret", "score", "percentage", "accurate", "feature", "importance",
}

def _is_on_topic(message: str) -> bool:
    """
    Returns True if the message contains at least one emotion/SHAP-related keyword.
    Short messages (e.g. 'yes', 'ok', 'tell me more') are always allowed
    as they are follow-ups in an ongoing conversation.
    """
    words = message.lower().split()
    if len(words) <= 4:
        return True   # Short follow-up — always allow
    return any(w.strip("?.,!") in _EMOTION_KEYWORDS for w in words)


_OFF_TOPIC_REPLY = (
    "I can only help with questions about your MedOracle emotion analysis results. "
    "Please ask me about the emotion detected, SHAP values, signal quality, or "
    "what the modalities measure."
)


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------

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
    dominant_modality = (
        "physiological (EEG + GSR)" if dominant_feature in ("EEG", "GSR")
        else "facial video"
    )

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
        f"  Physiological alone : {physio_pred.get('predicted_emotion','?')} "
        f"({physio_pred.get('confidence', 0):.1%} confidence)",
        f"  Video alone         : {video_pred.get('predicted_emotion','?')} "
        f"({video_pred.get('confidence', 0):.1%} confidence)",
        "",
        f"Explanation faithfulness score : {fs} (>=0.7 is acceptable)",
        "===================================",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini call (free tier)
# ---------------------------------------------------------------------------

async def _call_deepseek(messages: List[dict], system: str) -> str:
    """Call DeepSeek via OpenRouter (OpenAI-compatible)."""
    import httpx

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "system", "content": system}, *messages],
        "temperature": 0.4,
        "max_tokens": 512,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _call_groq(messages: List[dict], system: str) -> str:
    """Call Groq (free, OpenAI-compatible) using GROQ_API_KEY."""
    import httpx

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}, *messages],
        "temperature": 0.7,
        "max_tokens": 700,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:400]}")
        return resp.json()["choices"][0]["message"]["content"].strip()


async def _call_gemini(messages: List[dict], system: str) -> str:
    """Call Google Gemini via REST using GEMINI_API_KEY."""
    import httpx

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    # Gemini roles are "user"/"model"; map "assistant" -> "model".
    contents = [
        {"role": "model" if m.get("role") == "assistant" else "user",
         "parts": [{"text": m.get("content", "")}]}
        for m in messages
    ]
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": contents,
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 700},
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            # Surface Google's actual error text so we can diagnose (bad model
            # name, invalid key, etc.) from the server log.
            raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:400]}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            raise RuntimeError(f"Gemini unexpected response: {str(data)[:400]}")


# ---------------------------------------------------------------------------
# Simulated fallback (no API key needed)
# ---------------------------------------------------------------------------

def _simulated_response(shap_output: dict, user_message: str) -> str:
    """Offline fallback — generates varied responses based on user question.

    Conditions are ordered from MOST SPECIFIC to LEAST SPECIFIC so that
    a question like "Why is EEG lower than GSR?" is caught by the cross-modality
    comparison branch before the generic "why" branch fires.
    """
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

    eeg_fi = fi.get("EEG",   0.0)
    gsr_fi = fi.get("GSR",   0.0)
    vid_fi = fi.get("video", 0.0)

    # Rank modalities by importance for comparison responses
    ranked = sorted(
        [("EEG", eeg_fi), ("GSR", gsr_fi), ("Video/Facial", vid_fi)],
        key=lambda x: x[1], reverse=True,
    )

    quality_note = ""
    degraded = [k for k, v in sq.items() if v in ("degraded", "poor")]
    if degraded:
        quality_note = (
            f" Note: the {', '.join(degraded)} signal(s) had reduced quality, "
            "so interpret this result with some caution."
        )

    msg_lower = user_message.lower()

    # Pre-compute which modalities the user is asking about
    has_eeg = any(w in msg_lower for w in ["eeg", "brain", "electrical"])
    has_gsr = any(w in msg_lower for w in ["gsr", "skin", "conductance", "galvanic"])
    has_vid = any(w in msg_lower for w in ["video", "face", "facial", "camera"])
    modalities_mentioned = sum([has_eeg, has_gsr, has_vid])

    # ── 1. Cross-modality comparison (most specific — check BEFORE "why") ──
    # e.g. "Why is EEG lower than GSR?", "EEG vs GSR", "difference between"
    comparison_words = ["compare", "comparison", "versus", "vs", "difference",
                        "lower than", "higher than", "more than", "less than",
                        "bigger", "smaller", "contrast"]
    if modalities_mentioned >= 2 or any(w in msg_lower for w in comparison_words):
        return (
            f"Here is how each signal contributed to the {emotion} prediction:\n"
            f"  {ranked[0][0]}: {ranked[0][1]:.4f}  ← highest (most influential)\n"
            f"  {ranked[1][0]}: {ranked[1][1]:.4f}\n"
            f"  {ranked[2][0]}: {ranked[2][1]:.4f}  ← lowest influence\n"
            f"The difference reflects both the signal quality and how strongly each "
            f"modality's pattern matched the {emotion} state.{quality_note}"
        )

    # ── 2. Which modality contributes most / dominates ──────────────────────
    # e.g. "which modality contributes the most", "what is the dominant signal"
    if any(w in msg_lower for w in ["which", "most", "highest", "dominant",
                                     "biggest", "largest", "top", "contributes",
                                     "contribute", "mainly", "primarily"]):
        return (
            f"The most influential signal was {dom_label}, "
            f"with a SHAP importance score of {fi.get(dominant, 0.0):.4f}. "
            f"The full ranking was: "
            f"{ranked[0][0]} ({ranked[0][1]:.4f}) > "
            f"{ranked[1][0]} ({ranked[1][1]:.4f}) > "
            f"{ranked[2][0]} ({ranked[2][1]:.4f}).{quality_note}"
        )

    # ── 3. Why / reason (generic causal question) ────────────────────────────
    if any(w in msg_lower for w in ["why", "reason", "how come", "what caused"]):
        return (
            f"The {emotion} prediction was primarily driven by {dom_label}, "
            f"which showed the strongest indicators for this emotional state. "
            f"The SHAP analysis assigned it the highest feature importance score "
            f"among all three signals (EEG, GSR, video).{quality_note}"
        )

    # ── 4. Confidence / certainty ────────────────────────────────────────────
    elif any(w in msg_lower for w in ["confidence", "sure", "certain", "accurate",
                                       "percent", "%", "77", "reliable"]):
        return (
            f"The system is {conf:.0%} confident in the {emotion} prediction. "
            f"This confidence is calculated from the fused probability distribution "
            f"across all five emotion classes. "
            f"A score above 70% is generally considered reliable.{quality_note}"
        )

    # ── 5. Single-modality questions ─────────────────────────────────────────
    elif has_eeg:
        return (
            f"EEG (electroencephalography) measures electrical activity in the brain. "
            f"In this session, EEG had a SHAP importance of {eeg_fi:.4f}, "
            f"making it {'the dominant' if dominant == 'EEG' else 'a secondary'} signal. "
            f"Brain activity patterns in specific frequency bands (alpha, beta, theta) "
            f"are strongly linked to emotional arousal and valence."
        )
    elif has_gsr:
        return (
            f"GSR (galvanic skin response) measures how much your skin conducts "
            f"electricity, which increases with emotional arousal and stress. "
            f"In this session, GSR had a SHAP importance of {gsr_fi:.4f}. "
            f"It is weighted at {physio_w:.0%} combined with EEG in the "
            f"physiological modality.{quality_note}"
        )
    elif has_vid:
        return (
            f"The video modality analyses facial muscle movements captured by camera. "
            f"In this session, video had a SHAP importance of {vid_fi:.4f} "
            f"and was weighted at {video_w:.0%} in the final fusion. "
            f"{'It was the dominant signal.' if dominant == 'video' else 'The physiological signals were more influential this time.'}{quality_note}"
        )

    # ── 6. Signal quality ────────────────────────────────────────────────────
    elif any(w in msg_lower for w in ["quality", "signal", "trust", "noise"]):
        if degraded:
            return (
                f"The {', '.join(degraded)} signal(s) had degraded or poor quality "
                f"during this session. "
                f"This means the system automatically reduced their influence in the "
                f"final prediction. "
                f"For more reliable results, ensure sensors are properly attached "
                f"and the face is clearly visible."
            )
        else:
            return (
                f"All signals (EEG, GSR, and video) had good quality during this session. "
                f"This means the {emotion} prediction is based on clean, reliable data "
                f"and can be interpreted with higher confidence."
            )

    # ── 7. SHAP / explanation overview ───────────────────────────────────────
    elif any(w in msg_lower for w in ["shap", "explain", "interpret", "faithfulness",
                                       "result", "results", "overview", "summary"]):
        fs = shap_output.get("faithfulness_score", 0)
        return (
            f"SHAP (SHapley Additive exPlanations) measures how much each signal "
            f"contributed to the final prediction. "
            f"The faithfulness score of {fs:.2f} "
            f"{'confirms' if fs >= 0.7 else 'suggests'} that these explanations "
            f"{'genuinely reflect' if fs >= 0.7 else 'may not fully reflect'} "
            f"the model's decision-making. "
            f"The dominant contributor was {dom_label}."
        )

    # ── 8. Catch-all ─────────────────────────────────────────────────────────
    else:
        return (
            f"The system detected {emotion} with {conf:.0%} confidence. "
            f"The most influential signal was {dom_label}. "
            f"Physiological sensors were weighted at {physio_w:.0%} "
            f"and facial video at {video_w:.0%}.{quality_note} "
            f"This is an analytical result — consult a qualified clinician "
            f"for any health-related decisions."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_llm_response(
    shap_output: dict,
    user_message: str,
    conversation_history: Optional[List[dict]] = None,
    grounding_facts: Optional[List[str]] = None,
) -> str:
    # ── 1. Off-topic guard ────────────────────────────────────────────────
    if not _is_on_topic(user_message):
        return _OFF_TOPIC_REPLY

    context_block = _build_context_block(shap_output)

    # ── 1b. RAG grounding — inject retrieved sources into the system prompt ─
    system = _SYSTEM_PROMPT
    if grounding_facts:
        facts_txt = "\n".join(f"- {f}" for f in grounding_facts)
        system = system + (
            "\n\n=== GROUNDING SOURCES — you MUST use these ===\n"
            f"{facts_txt}\n\n"
            "CITATION RULES FOR THIS ANSWER (override brevity if needed):\n"
            "1. Explain the science using ONLY these sources.\n"
            "2. You MUST place an inline citation in square brackets IMMEDIATELY after "
            "each scientific claim — e.g. \"elevated GSR reflects sympathetic arousal "
            "[Kreibig 2010]\". Do NOT collect all citations at the end.\n"
            "3. Your answer MUST contain at least two inline [Author Year] citations "
            "drawn from the sources above.\n"
            "4. After each cited fact, connect it to THIS session's numbers (predicted "
            "emotion, trusted modality, signal quality).\n"
            "5. Do not invent facts or citations beyond the sources above."
        )

    # ── 2. Build message list ─────────────────────────────────────────────
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
        # Inject context into the first message if not already there
        first = messages[0]
        if context_block not in first.get("content", ""):
            messages[0] = {
                **first,
                "content": (
                    f"Here are the MedOracle analysis results:\n\n{context_block}\n\n"
                    f"{first['content']}"
                ),
            }
        messages.append({"role": "user", "content": user_message})

    # ── 3. Try live LLMs: Groq, then Gemini, then OpenRouter/DeepSeek ──────
    for caller in (_call_groq, _call_gemini, _call_deepseek):
        try:
            return await caller(messages, system)
        except Exception as exc:
            logger.warning("%s failed (%s: %s)", caller.__name__, type(exc).__name__, exc)

    # ── 4. Simulated fallback (no API key / all providers failed) ──────────
    return _simulated_response(shap_output, user_message)


async def get_auto_explanation(shap_output: dict) -> str:
    trigger = (
        "Please provide a brief, empathetic explanation of these emotion analysis results "
        "for the patient or clinician. Focus on the dominant modality, what it means in "
        "everyday language, and any signal quality caveats."
    )
    return await get_llm_response(shap_output, trigger, conversation_history=None)
