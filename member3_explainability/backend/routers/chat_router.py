"""
member3_explainability/backend/routers/chat_router.py
======================================================
POST /chat                        — send message, get LLM response
GET  /chat/history/{session_id}   — retrieve full chat history

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from member3_explainability.backend.database import get_db
from member3_explainability.backend.models import Session, SHAPLog, ChatMessage
from member3_explainability.backend.auth import get_current_user_id
from member3_explainability.backend.schemas import (
    ChatRequest, ChatResponse, ChatHistoryResponse, ChatMessageOut,
)
from member3_explainability.backend.llm_client import get_llm_response

router = APIRouter(prefix="/chat", tags=["Chatbot"])


@router.post("", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """
    Accept a user message and session_id.
    Loads SHAP context + full conversation history, calls the LLM,
    stores both the user message and assistant reply, returns the reply.
    """
    # Verify session belongs to the user
    result = await db.execute(
        select(Session).where(
            Session.session_id == body.session_id,
            Session.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")

    # Load SHAP output for this session
    shap_result = await db.execute(
        select(SHAPLog).where(SHAPLog.session_id == body.session_id)
    )
    shap_log = shap_result.scalar_one_or_none()
    if not shap_log:
        raise HTTPException(status_code=404, detail="SHAP log not found for this session.")

    shap_output = {
        "predicted_emotion":        session.predicted_emotion,
        "confidence":               session.confidence,
        "modality_weights":         session.modality_weights,
        "signal_quality":           session.signal_quality,
        "shap_values":              shap_log.shap_values,
        "feature_importance":       shap_log.feature_importance,
        "faithfulness_score":       shap_log.faithfulness_score,
        "signal_reliability":       shap_log.signal_reliability,
        "per_modality_predictions": shap_log.per_modality_predictions or {},
    }

    # Load existing conversation history
    history_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == body.session_id)
        .order_by(ChatMessage.timestamp.asc())
    )
    history_rows = history_result.scalars().all()
    conversation_history = [
        {"role": row.role, "content": row.content}
        for row in history_rows
    ]

    # Store user message
    user_msg = ChatMessage(
        session_id=body.session_id,
        user_id=user_id,
        role="user",
        content=body.message,
        timestamp=datetime.utcnow(),
    )
    db.add(user_msg)
    await db.flush()

    # Call LLM
    llm_reply = await get_llm_response(
        shap_output=shap_output,
        user_message=body.message,
        conversation_history=conversation_history if conversation_history else None,
    )

    # Store assistant message
    now = datetime.utcnow()
    assistant_msg_id = str(uuid.uuid4())
    assistant_msg = ChatMessage(
        message_id=assistant_msg_id,
        session_id=body.session_id,
        user_id=user_id,
        role="assistant",
        content=llm_reply,
        timestamp=now,
    )
    db.add(assistant_msg)

    return ChatResponse(
        message_id=assistant_msg_id,
        response=llm_reply,
        timestamp=now,
    )


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Return all chat messages for a given session."""
    # Verify ownership
    result = await db.execute(
        select(Session).where(
            Session.session_id == session_id,
            Session.user_id == user_id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Session not found.")

    history_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp.asc())
    )
    messages = history_result.scalars().all()

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[
            ChatMessageOut(
                message_id=m.message_id,
                role=m.role,
                content=m.content,
                timestamp=m.timestamp,
            )
            for m in messages
        ],
    )
