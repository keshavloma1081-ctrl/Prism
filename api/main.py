"""
PRISM — api/main.py
FastAPI REST API Layer

The external surface of PRISM.
Every PRISM capability exposed as a clean REST endpoint.

Sessions and events persisted to SQLite/PostgreSQL.
Verdict, Decay, Compass are session-scoped in memory.

Endpoints:
  POST /sessions/              → create new workflow session
  POST /sessions/{id}/agents   → register agents
  POST /sessions/{id}/events   → stream EAT events
  GET  /sessions/{id}/verdict  → real-time VERDICT scores
  GET  /sessions/{id}/decay    → epistemic health check
  GET  /sessions/{id}/atlas    → causal fingerprint
  POST /sessions/{id}/ghost    → run Ghost Runner
  GET  /sessions/{id}/report   → generate Chronicle report
  GET  /sessions/{id}/compass  → optimization recommendations
  GET  /health                 → system health check
"""

from __future__ import annotations
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from db.database import init_db, get_db
from db.repository import SessionRepository, AgentRepository, EventRepository

from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession, HumanAgent, AIAgent,
    BeliefState, ConceptualGraph, WorkflowStatus,
    AccessLevel
)
from core.eat.validators  import validate_eat_event
from core.eat.delta       import compute_delta_magnitude
from verdict.scorer       import VerdictScorer
from decay.detector       import DecayDetector
from atlas.graph          import AtlasGraph
from ghost.replay         import GhostRunner
from chronicle.reporter   import ChronicleReporter
from compass.optimizer    import CompassOptimizer


# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PRISM API",
    description = (
        "Epistemic Observability Platform for Enterprise Human-AI Workflows. "
        "Measures not just what AI produces — but what humans and AI "
        "discover together that neither could alone."
    ),
    version  = "0.1.0",
    docs_url = "/docs",
    redoc_url= "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── STARTUP ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    init_db()


# ─── IN-MEMORY CACHE ──────────────────────────────────────────────────────────
# Sessions + events → SQLite/PostgreSQL via db/repository.py
# Verdict, Decay, Compass → session-scoped, rebuilt on demand

verdicts:  Dict[str, VerdictScorer]    = {}
detectors: Dict[str, DecayDetector]    = {}
compasses: Dict[str, CompassOptimizer] = {}


# ─── REQUEST MODELS ───────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    client_id:   str
    workflow_id: str
    metadata:    Dict[str, Any] = Field(default_factory=dict)


class RegisterHumanAgentRequest(BaseModel):
    name:     str
    role:     str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegisterAIAgentRequest(BaseModel):
    model_name:     str
    provider:       str
    access_level:   str = "GREY_BOX"
    initial_prompt: Optional[str] = None
    metadata:       Dict[str, Any] = Field(default_factory=dict)


class BeliefStateRequest(BaseModel):
    hypotheses:   Dict[str, float]
    approximated: bool  = False
    uncertainty:  float = 0.0


class ConceptualGraphRequest(BaseModel):
    nodes:        Dict[str, str]
    edges:        List[Dict[str, Any]]
    approximated: bool  = False
    uncertainty:  float = 0.0


class SubmitEventRequest(BaseModel):
    agent_id:         str
    agent_type:       str
    event_type:       str
    t:                int
    prior_belief:     Optional[BeliefStateRequest]     = None
    posterior_belief: Optional[BeliefStateRequest]     = None
    prior_graph:      Optional[ConceptualGraphRequest] = None
    posterior_graph:  Optional[ConceptualGraphRequest] = None
    trigger_ref:      Optional[str]  = None
    trigger_agent:    Optional[str]  = None
    confidence:       float          = 1.0
    raw_evidence:     Optional[str]  = None
    groundedness:         Optional[float] = None
    calibration_score:    Optional[float] = None
    influence_survival:   Optional[float] = None
    novelty_delta:        Optional[float] = None
    metadata:         Dict[str, Any] = Field(default_factory=dict)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _load_session(
    session_id: str,
    db:         Session
) -> WorkflowSession:
    """
    Load a WorkflowSession from DB and reconstruct
    its agents and events into the domain model.
    """
    s_repo = SessionRepository(db)
    a_repo = AgentRepository(db)
    e_repo = EventRepository(db)

    db_session = s_repo.get(session_id)
    if db_session is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"Session '{session_id}' not found"
        )

    session = s_repo.to_domain(db_session)

    # Load agents
    for db_agent in a_repo.get_by_session(session_id):
        if db_agent.agent_type == "HUMAN":
            agent = a_repo.to_human_domain(db_agent)
            session.human_agents[agent.agent_id] = agent
        else:
            agent = a_repo.to_ai_domain(db_agent)
            session.ai_agents[agent.agent_id] = agent

    # Load events
    for db_event in e_repo.get_by_session(session_id):
        event = e_repo.to_domain(db_event)
        session.events.append(event)

    return session


def _build_eat_event(
    req:        SubmitEventRequest,
    session_id: str
) -> EATEvent:
    """Convert API request to EATEvent domain model."""

    prior_belief = BeliefState(
        **req.prior_belief.model_dump()
    ) if req.prior_belief else None

    posterior_belief = BeliefState(
        **req.posterior_belief.model_dump()
    ) if req.posterior_belief else None

    prior_graph = ConceptualGraph(
        **req.prior_graph.model_dump()
    ) if req.prior_graph else None

    posterior_graph = ConceptualGraph(
        **req.posterior_graph.model_dump()
    ) if req.posterior_graph else None

    event = EATEvent(
        session_id         = session_id,
        agent_id           = req.agent_id,
        agent_type         = AgentType(req.agent_type),
        event_type         = EATEventType(req.event_type),
        t                  = req.t,
        prior_belief       = prior_belief,
        posterior_belief   = posterior_belief,
        prior_graph        = prior_graph,
        posterior_graph    = posterior_graph,
        trigger_ref        = req.trigger_ref,
        trigger_agent      = req.trigger_agent,
        confidence         = req.confidence,
        raw_evidence       = req.raw_evidence,
        groundedness       = req.groundedness,
        calibration_score  = req.calibration_score,
        influence_survival = req.influence_survival,
        novelty_delta      = req.novelty_delta,
        metadata           = req.metadata
    )
    event.delta_magnitude = compute_delta_magnitude(event)
    return event


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    s_repo = SessionRepository(db)
    return {
        "status":    "healthy",
        "version":   "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sessions":  len(s_repo.list(limit=1000))
    }


# ── Session Management ─────────────────────────────────────────────────────────

@app.post("/sessions/")
def create_session(
    req: CreateSessionRequest,
    db:  Session = Depends(get_db)
):
    session  = WorkflowSession(
        client_id   = req.client_id,
        workflow_id = req.workflow_id,
        metadata    = req.metadata
    )
    s_repo = SessionRepository(db)
    s_repo.create(session)

    verdicts[session.session_id]  = VerdictScorer(session)
    detectors[session.session_id] = DecayDetector(session)
    compasses[session.session_id] = CompassOptimizer(
        client_id   = req.client_id,
        workflow_id = req.workflow_id
    )

    return {
        "session_id":  session.session_id,
        "client_id":   session.client_id,
        "workflow_id": session.workflow_id,
        "status":      session.status.value,
        "created_at":  session.created_at.isoformat()
    }


@app.get("/sessions/{session_id}")
def get_session_info(
    session_id: str,
    db:         Session = Depends(get_db)
):
    session = _load_session(session_id, db)
    return {
        "session_id":   session.session_id,
        "client_id":    session.client_id,
        "workflow_id":  session.workflow_id,
        "status":       session.status.value,
        "total_events": session.total_events,
        "human_agents": len(session.human_agents),
        "ai_agents":    len(session.ai_agents),
        "created_at":   session.created_at.isoformat(),
        "updated_at":   session.updated_at.isoformat()
    }


@app.get("/sessions/")
def list_sessions(
    client_id: Optional[str] = None,
    db:        Session = Depends(get_db)
):
    s_repo   = SessionRepository(db)
    sessions = s_repo.list(client_id=client_id)
    result   = [
        {
            "session_id":   s.session_id,
            "client_id":    s.client_id,
            "workflow_id":  s.workflow_id,
            "status":       s.status,
            "total_events": len(s.events)
        }
        for s in sessions
    ]
    return {"sessions": result, "total": len(result)}


# ── Agent Registration ─────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/agents/human")
def register_human_agent(
    session_id: str,
    req:        RegisterHumanAgentRequest,
    db:         Session = Depends(get_db)
):
    s_repo = SessionRepository(db)
    if s_repo.get(session_id) is None:
        raise HTTPException(status_code=404,
                            detail=f"Session '{session_id}' not found")

    agent  = HumanAgent(
        name      = req.name,
        role      = req.role,
        client_id = s_repo.get(session_id).client_id,
        metadata  = req.metadata
    )
    a_repo = AgentRepository(db)
    a_repo.create_human(session_id, agent)

    # Update in-memory session if cached
    if session_id in verdicts:
        session = _load_session(session_id, db)
        verdicts[session_id]  = VerdictScorer(session)
        detectors[session_id] = DecayDetector(session)

    return {
        "agent_id":   agent.agent_id,
        "agent_type": "HUMAN",
        "name":       agent.name,
        "role":       agent.role
    }


@app.post("/sessions/{session_id}/agents/ai")
def register_ai_agent(
    session_id: str,
    req:        RegisterAIAgentRequest,
    db:         Session = Depends(get_db)
):
    s_repo = SessionRepository(db)
    if s_repo.get(session_id) is None:
        raise HTTPException(status_code=404,
                            detail=f"Session '{session_id}' not found")

    agent  = AIAgent(
        model_name = req.model_name,
        provider   = req.provider,
        client_id  = s_repo.get(session_id).client_id,
        metadata   = req.metadata
    )
    a_repo = AgentRepository(db)
    a_repo.create_ai(session_id, agent)

    if req.initial_prompt and session_id in compasses:
        compasses[session_id].register_agent(
            agent.agent_id, req.initial_prompt
        )

    return {
        "agent_id":   agent.agent_id,
        "agent_type": "AI",
        "model_name": agent.model_name,
        "provider":   agent.provider
    }


# ── Event Streaming ────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/events")
def submit_event(
    session_id: str,
    req:        SubmitEventRequest,
    db:         Session = Depends(get_db)
):
    session = _load_session(session_id, db)

    try:
        event = _build_eat_event(req, session_id)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    validation = validate_eat_event(event, session)
    if not validation.passed:
        raise HTTPException(
            status_code = 422,
            detail      = {
                "message": "EAT event validation failed",
                "errors":  validation.errors
            }
        )

    # Score with VERDICT
    scorer = verdicts.get(session_id)
    if scorer is None:
        scorer = VerdictScorer(session)
        verdicts[session_id] = scorer
    event = scorer.score_event(event)

    # Persist event
    e_repo = EventRepository(db)
    e_repo.create(session_id, event)

    # Check DECAY every 5 events
    detector = detectors.get(session_id)
    if detector is None:
        detector = DecayDetector(session)
        detectors[session_id] = detector

    decay_alerts = []
    if session.total_events % 5 == 0:
        session.add_event(event)
        alerts = detector.check(current_t=event.t)
        decay_alerts = [
            {
                "type":           a.alert_type,
                "severity":       a.severity,
                "recommendation": a.recommendation[:100]
            }
            for a in alerts
        ]
    else:
        session.add_event(event)

    return {
        "event_id":        event.event_id,
        "session_id":      session_id,
        "delta_magnitude": round(event.delta_magnitude, 4),
        "verdict_scores": {
            "groundedness":       event.groundedness,
            "novelty_delta":      event.novelty_delta,
            "influence_survival": event.influence_survival,
            "calibration_score":  event.calibration_score
        },
        "decay_alerts": decay_alerts,
        "warnings":     validation.warnings
    }


@app.get("/sessions/{session_id}/events")
def get_events(
    session_id: str,
    agent_id:   Optional[str] = None,
    event_type: Optional[str] = None,
    limit:      int = 50,
    db:         Session = Depends(get_db)
):
    s_repo = SessionRepository(db)
    if s_repo.get(session_id) is None:
        raise HTTPException(status_code=404,
                            detail=f"Session '{session_id}' not found")

    e_repo = EventRepository(db)
    events = e_repo.get_by_session(
        session_id = session_id,
        agent_id   = agent_id,
        event_type = event_type,
        limit      = limit
    )

    return {
        "session_id": session_id,
        "total":      len(events),
        "events": [
            {
                "event_id":        e.event_id,
                "agent_id":        e.agent_id,
                "agent_type":      e.agent_type,
                "event_type":      e.event_type,
                "t":               e.t,
                "delta_magnitude": round(e.delta_magnitude or 0, 4),
                "timestamp":       e.timestamp.isoformat()
            }
            for e in events
        ]
    }


# ── VERDICT ────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/verdict")
def get_verdict(
    session_id: str,
    db:         Session = Depends(get_db)
):
    session = _load_session(session_id, db)
    scorer  = verdicts.get(session_id, VerdictScorer(session))
    for event in session.events:
        scorer.score_event(event)
    return scorer.session_verdict_summary()


# ── DECAY ──────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/decay")
def get_decay(
    session_id: str,
    db:         Session = Depends(get_db)
):
    session  = _load_session(session_id, db)
    detector = detectors.get(session_id, DecayDetector(session))
    if session.events:
        detector.check(current_t=session.events[-1].t)
    return detector.decay_summary()


# ── ATLAS ──────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/atlas")
def get_atlas(
    session_id: str,
    db:         Session = Depends(get_db)
):
    session = _load_session(session_id, db)
    atlas   = AtlasGraph(session)
    atlas.build_fingerprint()
    high_mag = sorted(
        [e for e in session.events if e.delta_magnitude > 0.3],
        key     = lambda e: e.delta_magnitude,
        reverse = True
    )[:5]
    for event in high_mag:
        atlas.trace_discovery(event.event_id)
    return atlas.export_dict()


# ── GHOST ──────────────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/ghost")
def run_ghost(
    session_id: str,
    db:         Session = Depends(get_db)
):
    session = _load_session(session_id, db)
    if session.total_events < 2:
        raise HTTPException(
            status_code = 400,
            detail      = "Session needs at least 2 events for Ghost Runner"
        )
    ghost = GhostRunner(session)
    sig   = ghost.run()
    return {
        "session_id":      session_id,
        "emergence_score": sig.emergence_score,
        "ai_value_score":  sig.ai_value_score,
        "human_value_score": sig.human_value_score,
        "concept_emergence": list(sig.concept_emergence),
        "ai_unique":       list(sig.ai_unique_concepts),
        "human_unique":    list(sig.human_unique_concepts),
        "entropy_lift":    sig.entropy_lift,
        "magnitude_lift":  sig.magnitude_lift,
        "verdict":         sig.verdict(),
        "recommendation":  sig.recommendation
    }


# ── COMPASS ────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/compass")
def get_compass(session_id: str, db: Session = Depends(get_db)):
    s_repo = SessionRepository(db)
    if s_repo.get(session_id) is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"No session '{session_id}' found"
        )
    compass = compasses.get(session_id)
    if compass is None:
        db_session = s_repo.get(session_id)
        compass = CompassOptimizer(
            client_id   = db_session.client_id,
            workflow_id = db_session.workflow_id
        )
    return compass.optimization_summary()


# ── CHRONICLE ──────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/report")
def generate_report(
    session_id: str,
    db:         Session = Depends(get_db)
):
    session  = _load_session(session_id, db)
    compass  = compasses.get(session_id)
    reporter = ChronicleReporter(session)
    if compass:
        reporter.attach_compass(compass)
    report = reporter.generate()
    return report.to_dict()


@app.post("/sessions/{session_id}/complete")
def complete_session(
    session_id: str,
    db:         Session = Depends(get_db)
):
    s_repo = SessionRepository(db)
    result = s_repo.update_status(
        session_id   = session_id,
        status       = "COMPLETED",
        completed_at = datetime.now(timezone.utc)
    )
    if result is None:
        raise HTTPException(status_code=404,
                            detail=f"Session '{session_id}' not found")
    return {
        "session_id":   session_id,
        "status":       "COMPLETED",
        "completed_at": result.completed_at.isoformat(),
        "total_events": len(result.events)
    }