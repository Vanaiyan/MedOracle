"""
member3_explainability/backend/models.py
=========================================
SQLAlchemy ORM models — 4 tables per spec §7.5.

Author : Adshaya Balarajah (214024V)
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Text, DateTime,
    ForeignKey, Enum as SAEnum,
)
from sqlalchemy.orm import relationship

from member3_explainability.backend.database import Base

# JSON stored as Text + Python-side serialization (works for SQLite + Postgres)
from sqlalchemy import JSON


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    user_id      = Column(String, primary_key=True, default=_uuid)
    email        = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)

    sessions     = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="user", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------

class Session(Base):
    __tablename__ = "sessions"

    session_id          = Column(String, primary_key=True, default=_uuid)
    user_id             = Column(String, ForeignKey("users.user_id"), nullable=False, index=True)
    timestamp           = Column(DateTime, default=datetime.utcnow, nullable=False)
    predicted_emotion   = Column(String, nullable=False)
    confidence          = Column(Float,  nullable=False)
    modality_weights    = Column(JSON,   nullable=False)   # {"physio": float, "video": float}
    signal_quality      = Column(JSON,   nullable=False)   # {"eeg": str, "gsr": str, "video": str}
    class_probabilities = Column(JSON,   nullable=False)   # {"stress": float, ...}

    user      = relationship("User",      back_populates="sessions")
    shap_log  = relationship("SHAPLog",   back_populates="session", uselist=False, cascade="all, delete-orphan")
    chat_messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# shap_logs
# ---------------------------------------------------------------------------

class SHAPLog(Base):
    __tablename__ = "shap_logs"

    log_id             = Column(String, primary_key=True, default=_uuid)
    session_id         = Column(String, ForeignKey("sessions.session_id"), nullable=False, unique=True)
    shap_values        = Column(JSON,  nullable=False)   # {"EEG": float, "GSR": float, "video": float}
    feature_importance = Column(JSON,  nullable=False)   # {"EEG": float, "GSR": float, "video": float}
    faithfulness_score = Column(Float, nullable=False)
    signal_reliability = Column(JSON,  nullable=False)   # {"eeg": str, "gsr": str, "video": str}
    coalition_values   = Column(JSON,  nullable=True)    # {"empty", "physio", "video", "full"}
    per_modality_predictions = Column(JSON, nullable=True)

    session = relationship("Session", back_populates="shap_log")


# ---------------------------------------------------------------------------
# chat_history
# ---------------------------------------------------------------------------

class ChatMessage(Base):
    __tablename__ = "chat_history"

    message_id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("sessions.session_id"), nullable=False, index=True)
    user_id    = Column(String, ForeignKey("users.user_id"),   nullable=False)
    role       = Column(SAEnum("user", "assistant", name="chat_role"), nullable=False)
    content    = Column(Text, nullable=False)
    timestamp  = Column(DateTime, default=datetime.utcnow, nullable=False)

    session = relationship("Session", back_populates="chat_messages")
    user    = relationship("User",    back_populates="chat_messages")
