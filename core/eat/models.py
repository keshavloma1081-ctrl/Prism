"""
PRISM — core/eat/models.py
Epistemic Action Trace (EAT) Event Schema

The single most important file in PRISM.
Every system — PULSE, GHOST, VERDICT, DECAY, ATLAS — reads from this.
Zero external dependencies. Pure Pydantic.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator
import uuid


# ─── ENUMS ────────────────────────────────────────────────────────────────────

class AgentType(str, Enum):
    HUMAN = "HUMAN"
    AI = "AI"


class EATEventType(str, Enum):
    B_UPDATE  = "B_UPDATE"   # Belief update — single agent
    C_UPDATE  = "C_UPDATE"   # Conceptual graph update — single agent
    B_TRIGGER = "B_TRIGGER"  # Belief update caused by another agent
    C_TRIGGER = "C_TRIGGER"  # Conceptual update caused by another agent


class AccessLevel(str, Enum):
    WHITE_BOX = "WHITE_BOX"  # Full activation access
    GREY_BOX  = "GREY_BOX"   # Chain-of-thought + output logits
    BLACK_BOX = "BLACK_BOX"  # Output text only


class WorkflowStatus(str, Enum):
    ACTIVE    = "ACTIVE"
    PAUSED    = "PAUSED"
    COMPLETED = "COMPLETED"
    REPLAYING = "REPLAYING"  # Ghost Runner counterfactual mode


# ─── AGENT MODELS ─────────────────────────────────────────────────────────────

class HumanAgent(BaseModel):
    """Represents a human operator in a PRISM workflow session."""
    agent_id:    str       = Field(default_factory=lambda: f"H-{uuid.uuid4().hex[:8]}")
    agent_type:  AgentType = AgentType.HUMAN
    name:        str
    role:        str                        # e.g. "analyst", "domain-expert", "decision-maker"
    client_id:   str
    session_ids: List[str] = Field(default_factory=list)
    metadata:    Dict[str, Any] = Field(default_factory=dict)


class AIAgent(BaseModel):
    """Represents an AI agent in a PRISM workflow session."""
    agent_id:     str        = Field(default_factory=lambda: f"A-{uuid.uuid4().hex[:8]}")
    agent_type:   AgentType  = AgentType.AI
    model_name:   str                        # e.g. "claude-sonnet-4-6", "gpt-4o"
    provider:     str                        # e.g. "anthropic", "openai", "cohere"
    access_level: AccessLevel = AccessLevel.GREY_BOX
    client_id:    str
    session_ids:  List[str]  = Field(default_factory=list)
    metadata:     Dict[str, Any] = Field(default_factory=dict)


# ─── BELIEF STATE ─────────────────────────────────────────────────────────────

class BeliefState(BaseModel):
    """
    Probability distribution over hypothesis space Ω.
    For humans: elicited via structured belief reporting.
    For AI: approximated via output token probabilities.
    """
    hypotheses:   Dict[str, float]  # hypothesis_id → confidence [0.0, 1.0]
    timestamp:    datetime = Field(default_factory=datetime.utcnow)
    approximated: bool = False       # True for AI agents — not directly observed
    uncertainty:  float = 0.0        # Approximation uncertainty [0.0, 1.0]

    @model_validator(mode="after")
    def validate_probabilities(self) -> BeliefState:
        for h, p in self.hypotheses.items():
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"Confidence for '{h}' must be in [0.0, 1.0], got {p}")
        return self


class ConceptualGraph(BaseModel):
    """
    Directed graph over concept space.
    Nodes are concepts. Edges are inferred relationships.
    For humans: derived from discourse analysis.
    For AI: approximated from attention patterns or CoT traces.
    """
    nodes: Dict[str, str]              # concept_id → concept_label
    edges: List[Dict[str, Any]]        # [{source, target, weight, relation_type}]
    approximated: bool = False
    uncertainty:  float = 0.0


# ─── EAT EVENT ────────────────────────────────────────────────────────────────

class EATEvent(BaseModel):
    """
    Atomic epistemic act. The core measurement unit of PRISM.

    Every belief update, hypothesis revision, causal inference,
    and conceptual connection — from any agent, human or AI —
    is recorded as an EATEvent.

    All seven PRISM systems operate on EATEvent streams.
    """

    # Identity
    event_id:   str      = Field(default_factory=lambda: f"EAT-{uuid.uuid4().hex[:12]}")
    session_id: str
    agent_id:   str
    agent_type: AgentType
    event_type: EATEventType

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    t:         int                           # Discrete time step index

    # State change
    prior_belief:      Optional[BeliefState]    = None
    posterior_belief:  Optional[BeliefState]    = None
    prior_graph:       Optional[ConceptualGraph] = None
    posterior_graph:   Optional[ConceptualGraph] = None

    # Magnitude — computed by validators below
    delta_magnitude: float = 0.0

    # Causation — for TRIGGER events
    trigger_ref:    Optional[str] = None     # event_id of originating event
    trigger_agent:  Optional[str] = None     # agent_id of originating agent

    # Evidence
    confidence:   float = 1.0               # Record accuracy confidence
    raw_evidence: Optional[str] = None      # Pointer to transcript/CoT/activation

    # VERDICT scores — populated by verdict/ system
    groundedness:         Optional[float] = None
    calibration_score:    Optional[float] = None
    influence_survival:   Optional[float] = None
    novelty_delta:        Optional[float] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_event_consistency(self) -> EATEvent:
        # B_UPDATE and B_TRIGGER require belief states
        if self.event_type in (EATEventType.B_UPDATE, EATEventType.B_TRIGGER):
            if self.prior_belief is None or self.posterior_belief is None:
                raise ValueError(
                    f"{self.event_type} requires prior_belief and posterior_belief"
                )

        # C_UPDATE and C_TRIGGER require conceptual graphs
        if self.event_type in (EATEventType.C_UPDATE, EATEventType.C_TRIGGER):
            if self.prior_graph is None or self.posterior_graph is None:
                raise ValueError(
                    f"{self.event_type} requires prior_graph and posterior_graph"
                )

        # TRIGGER events require a trigger reference
        if self.event_type in (EATEventType.B_TRIGGER, EATEventType.C_TRIGGER):
            if self.trigger_ref is None:
                raise ValueError(
                    f"{self.event_type} requires trigger_ref"
                )

        return self


# ─── SESSION ──────────────────────────────────────────────────────────────────

class WorkflowSession(BaseModel):
    """
    Container for a complete PRISM measurement session.
    Holds all agents, all EAT events, and session-level metadata.
    """
    session_id:   str    = Field(default_factory=lambda: f"S-{uuid.uuid4().hex[:12]}")
    client_id:    str
    workflow_id:  str
    status:       WorkflowStatus = WorkflowStatus.ACTIVE

    # Agents
    human_agents: Dict[str, HumanAgent] = Field(default_factory=dict)
    ai_agents:    Dict[str, AIAgent]    = Field(default_factory=dict)

    # EAT stream
    events: List[EATEvent] = Field(default_factory=list)

    # Timing
    created_at:   datetime = Field(default_factory=datetime.utcnow)
    updated_at:   datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Session-level scores — populated by VERDICT + DECAY
    novelty_potential:      Optional[float] = None
    epistemic_decay_score:  Optional[float] = None
    coupling_index:         Optional[float] = None

    # Ghost Runner flag
    is_counterfactual: bool  = False
    parent_session_id: Optional[str] = None   # Original session if counterfactual

    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_event(self, event: EATEvent) -> None:
        self.events.append(event)
        self.updated_at = datetime.now(timezone.utc)

    def get_events_by_agent(self, agent_id: str) -> List[EATEvent]:
        return [e for e in self.events if e.agent_id == agent_id]

    def get_events_by_type(self, event_type: EATEventType) -> List[EATEvent]:
        return [e for e in self.events if e.event_type == event_type]

    def get_human_events(self) -> List[EATEvent]:
        return [e for e in self.events if e.agent_type == AgentType.HUMAN]

    def get_ai_events(self) -> List[EATEvent]:
        return [e for e in self.events if e.agent_type == AgentType.AI]

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def all_agents(self) -> Dict[str, HumanAgent | AIAgent]:
        return {**self.human_agents, **self.ai_agents}