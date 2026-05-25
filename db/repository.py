"""
PRISM — db/repository.py
Database Repository Layer

All database operations go through this layer.
Keeps API endpoints clean and testable.

Pattern: Repository pattern — one class per model.
"""

from __future__ import annotations
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from db.models import SessionModel, AgentModel, EATEventModel
from core.eat.models import (
    WorkflowSession, HumanAgent, AIAgent,
    EATEvent, EATEventType, AgentType,
    BeliefState, ConceptualGraph, WorkflowStatus,
    AccessLevel
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _belief_to_dict(belief: Optional[BeliefState]) -> Optional[Dict]:
    if belief is None:
        return None
    return {
        "hypotheses":   belief.hypotheses,
        "approximated": belief.approximated,
        "uncertainty":  belief.uncertainty
    }


def _graph_to_dict(graph: Optional[ConceptualGraph]) -> Optional[Dict]:
    if graph is None:
        return None
    return {
        "nodes":        graph.nodes,
        "edges":        graph.edges,
        "approximated": graph.approximated,
        "uncertainty":  graph.uncertainty
    }


def _dict_to_belief(d: Optional[Dict]) -> Optional[BeliefState]:
    if d is None:
        return None
    return BeliefState(
        hypotheses   = d.get("hypotheses", {}),
        approximated = d.get("approximated", False),
        uncertainty  = d.get("uncertainty", 0.0)
    )


def _dict_to_graph(d: Optional[Dict]) -> Optional[ConceptualGraph]:
    if d is None:
        return None
    return ConceptualGraph(
        nodes        = d.get("nodes", {}),
        edges        = d.get("edges", []),
        approximated = d.get("approximated", False),
        uncertainty  = d.get("uncertainty", 0.0)
    )


# ─── SESSION REPOSITORY ───────────────────────────────────────────────────────

class SessionRepository:
    """All database operations for workflow sessions."""

    def __init__(self, db: Session):
        self.db = db

    def create(self, session: WorkflowSession) -> SessionModel:
        db_session = SessionModel(
            session_id        = session.session_id,
            client_id         = session.client_id,
            workflow_id       = session.workflow_id,
            status            = session.status.value,
            is_counterfactual = session.is_counterfactual,
            parent_session_id = session.parent_session_id,
            metadata_json     = session.metadata
        )
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        return db_session

    def get(self, session_id: str) -> Optional[SessionModel]:
        return self.db.query(SessionModel).filter(
            SessionModel.session_id == session_id
        ).first()

    def list(
        self,
        client_id:  Optional[str] = None,
        limit:      int = 50,
        offset:     int = 0
    ) -> List[SessionModel]:
        query = self.db.query(SessionModel)
        if client_id:
            query = query.filter(SessionModel.client_id == client_id)
        return query.order_by(
            SessionModel.created_at.desc()
        ).offset(offset).limit(limit).all()

    def update_status(
        self,
        session_id: str,
        status:     str,
        completed_at: Optional[datetime] = None
    ) -> Optional[SessionModel]:
        db_session = self.get(session_id)
        if db_session is None:
            return None
        db_session.status     = status
        db_session.updated_at = datetime.now(timezone.utc)
        if completed_at:
            db_session.completed_at = completed_at
        self.db.commit()
        self.db.refresh(db_session)
        return db_session

    def update_scores(
        self,
        session_id:           str,
        novelty_potential:    Optional[float] = None,
        decay_score:          Optional[float] = None,
        coupling_index:       Optional[float] = None
    ) -> None:
        db_session = self.get(session_id)
        if db_session is None:
            return
        if novelty_potential is not None:
            db_session.novelty_potential = novelty_potential
        if decay_score is not None:
            db_session.epistemic_decay_score = decay_score
        if coupling_index is not None:
            db_session.coupling_index = coupling_index
        db_session.updated_at = datetime.now(timezone.utc)
        self.db.commit()

    def to_domain(self, db_session: SessionModel) -> WorkflowSession:
        """Convert DB model to domain model."""
        session = WorkflowSession(
            client_id         = db_session.client_id,
            workflow_id       = db_session.workflow_id,
            is_counterfactual = db_session.is_counterfactual,
            parent_session_id = db_session.parent_session_id,
            metadata          = db_session.metadata_json or {}
        )
        session.session_id = db_session.session_id
        session.status     = WorkflowStatus(db_session.status)
        return session


# ─── AGENT REPOSITORY ─────────────────────────────────────────────────────────

class AgentRepository:
    """All database operations for agents."""

    def __init__(self, db: Session):
        self.db = db

    def create_human(
        self,
        session_id: str,
        agent:      HumanAgent
    ) -> AgentModel:
        db_agent = AgentModel(
            agent_id   = agent.agent_id,
            session_id = session_id,
            agent_type = "HUMAN",
            name       = agent.name,
            role       = agent.role,
            client_id  = agent.client_id,
            metadata_json = agent.metadata
        )
        self.db.add(db_agent)
        self.db.commit()
        self.db.refresh(db_agent)
        return db_agent

    def create_ai(
        self,
        session_id: str,
        agent:      AIAgent
    ) -> AgentModel:
        db_agent = AgentModel(
            agent_id     = agent.agent_id,
            session_id   = session_id,
            agent_type   = "AI",
            model_name   = agent.model_name,
            provider     = agent.provider,
            access_level = agent.access_level.value,
            client_id    = agent.client_id,
            metadata_json = agent.metadata
        )
        self.db.add(db_agent)
        self.db.commit()
        self.db.refresh(db_agent)
        return db_agent

    def get_by_session(self, session_id: str) -> List[AgentModel]:
        return self.db.query(AgentModel).filter(
            AgentModel.session_id == session_id
        ).all()

    def to_human_domain(self, db_agent: AgentModel) -> HumanAgent:
        agent = HumanAgent(
            name      = db_agent.name or "",
            role      = db_agent.role or "",
            client_id = db_agent.client_id,
            metadata  = db_agent.metadata_json or {}
        )
        agent.agent_id = db_agent.agent_id
        return agent

    def to_ai_domain(self, db_agent: AgentModel) -> AIAgent:
        agent = AIAgent(
            model_name   = db_agent.model_name or "",
            provider     = db_agent.provider or "",
            client_id    = db_agent.client_id,
            access_level = AccessLevel(
                db_agent.access_level or "GREY_BOX"
            ),
            metadata     = db_agent.metadata_json or {}
        )
        agent.agent_id = db_agent.agent_id
        return agent


# ─── EVENT REPOSITORY ─────────────────────────────────────────────────────────

class EventRepository:
    """All database operations for EAT events."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        session_id: str,
        event:      EATEvent
    ) -> EATEventModel:
        db_event = EATEventModel(
            event_id         = event.event_id,
            session_id       = session_id,
            agent_id         = event.agent_id,
            agent_type       = event.agent_type.value,
            event_type       = event.event_type.value,
            t                = event.t,
            delta_magnitude  = event.delta_magnitude,
            confidence       = event.confidence,
            prior_belief     = _belief_to_dict(event.prior_belief),
            posterior_belief = _belief_to_dict(event.posterior_belief),
            prior_graph      = _graph_to_dict(event.prior_graph),
            posterior_graph  = _graph_to_dict(event.posterior_graph),
            trigger_ref      = event.trigger_ref,
            trigger_agent    = event.trigger_agent,
            groundedness     = event.groundedness,
            calibration_score  = event.calibration_score,
            influence_survival = event.influence_survival,
            novelty_delta    = event.novelty_delta,
            raw_evidence     = event.raw_evidence,
            metadata_json    = event.metadata
        )
        self.db.add(db_event)
        self.db.commit()
        self.db.refresh(db_event)
        return db_event

    def get_by_session(
        self,
        session_id: str,
        agent_id:   Optional[str] = None,
        event_type: Optional[str] = None,
        limit:      int = 200
    ) -> List[EATEventModel]:
        query = self.db.query(EATEventModel).filter(
            EATEventModel.session_id == session_id
        )
        if agent_id:
            query = query.filter(EATEventModel.agent_id == agent_id)
        if event_type:
            query = query.filter(EATEventModel.event_type == event_type)
        return query.order_by(EATEventModel.t).limit(limit).all()

    def to_domain(self, db_event: EATEventModel) -> EATEvent:
        """Convert DB model to domain EATEvent."""
        event = EATEvent(
            session_id       = db_event.session_id,
            agent_id         = db_event.agent_id,
            agent_type       = AgentType(db_event.agent_type),
            event_type       = EATEventType(db_event.event_type),
            t                = db_event.t,
            prior_belief     = _dict_to_belief(db_event.prior_belief),
            posterior_belief = _dict_to_belief(db_event.posterior_belief),
            prior_graph      = _dict_to_graph(db_event.prior_graph),
            posterior_graph  = _dict_to_graph(db_event.posterior_graph),
            trigger_ref      = db_event.trigger_ref,
            trigger_agent    = db_event.trigger_agent,
            confidence       = db_event.confidence,
            raw_evidence     = db_event.raw_evidence,
            groundedness     = db_event.groundedness,
            calibration_score  = db_event.calibration_score,
            influence_survival = db_event.influence_survival,
            novelty_delta    = db_event.novelty_delta,
            metadata         = db_event.metadata_json or {}
        )
        event.event_id       = db_event.event_id
        event.delta_magnitude = db_event.delta_magnitude
        return event