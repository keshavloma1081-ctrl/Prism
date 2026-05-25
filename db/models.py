"""
PRISM — db/models.py
SQLAlchemy ORM Models

Persistent storage for workflow sessions,
agents, EAT events, and VERDICT scores.

Replaces the in-memory dict store in api/main.py.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Column, String, Float, Boolean, Integer,
    DateTime, Text, ForeignKey, Enum as SAEnum,
    JSON, create_engine
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ─── SESSION ──────────────────────────────────────────────────────────────────

class SessionModel(Base):
    __tablename__ = "sessions"

    session_id          = Column(String(64),  primary_key=True)
    client_id           = Column(String(128), nullable=False, index=True)
    workflow_id         = Column(String(128), nullable=False)
    status              = Column(String(32),  default="ACTIVE")
    is_counterfactual   = Column(Boolean,     default=False)
    parent_session_id   = Column(String(64),  nullable=True)
    novelty_potential   = Column(Float,       nullable=True)
    epistemic_decay_score = Column(Float,     nullable=True)
    coupling_index      = Column(Float,       nullable=True)
    metadata_json       = Column(JSON,        default=dict)
    created_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc))
    updated_at          = Column(DateTime(timezone=True),
                                 default=lambda: datetime.now(timezone.utc),
                                 onupdate=lambda: datetime.now(timezone.utc))
    completed_at        = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    agents  = relationship("AgentModel",  back_populates="session",
                           cascade="all, delete-orphan")
    events  = relationship("EATEventModel", back_populates="session",
                           cascade="all, delete-orphan",
                           order_by="EATEventModel.t")


# ─── AGENT ────────────────────────────────────────────────────────────────────

class AgentModel(Base):
    __tablename__ = "agents"

    agent_id    = Column(String(64),  primary_key=True)
    session_id  = Column(String(64),  ForeignKey("sessions.session_id"),
                         nullable=False, index=True)
    agent_type  = Column(String(16),  nullable=False)  # HUMAN | AI
    name        = Column(String(128), nullable=True)    # Human agents
    role        = Column(String(128), nullable=True)    # Human agents
    model_name  = Column(String(128), nullable=True)    # AI agents
    provider    = Column(String(64),  nullable=True)    # AI agents
    access_level = Column(String(32), default="GREY_BOX")
    client_id   = Column(String(128), nullable=False)
    metadata_json = Column(JSON,      default=dict)
    created_at  = Column(DateTime(timezone=True),
                         default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionModel", back_populates="agents")


# ─── EAT EVENT ────────────────────────────────────────────────────────────────

class EATEventModel(Base):
    __tablename__ = "eat_events"

    event_id         = Column(String(64),  primary_key=True)
    session_id       = Column(String(64),  ForeignKey("sessions.session_id"),
                              nullable=False, index=True)
    agent_id         = Column(String(64),  nullable=False, index=True)
    agent_type       = Column(String(16),  nullable=False)
    event_type       = Column(String(32),  nullable=False)
    t                = Column(Integer,     nullable=False)
    delta_magnitude  = Column(Float,       default=0.0)
    confidence       = Column(Float,       default=1.0)

    # Belief states stored as JSON
    prior_belief     = Column(JSON, nullable=True)
    posterior_belief = Column(JSON, nullable=True)

    # Conceptual graphs stored as JSON
    prior_graph      = Column(JSON, nullable=True)
    posterior_graph  = Column(JSON, nullable=True)

    # Trigger references
    trigger_ref      = Column(String(64),  nullable=True)
    trigger_agent    = Column(String(64),  nullable=True)

    # VERDICT scores
    groundedness       = Column(Float, nullable=True)
    calibration_score  = Column(Float, nullable=True)
    influence_survival = Column(Float, nullable=True)
    novelty_delta      = Column(Float, nullable=True)

    # Evidence
    raw_evidence     = Column(Text,  nullable=True)
    metadata_json    = Column(JSON,  default=dict)

    timestamp = Column(DateTime(timezone=True),
                       default=lambda: datetime.now(timezone.utc))

    session = relationship("SessionModel", back_populates="events")