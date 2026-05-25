"""
PRISM — api/main.py
FastAPI REST API Layer

The external surface of PRISM.
Every PRISM capability exposed as a clean REST endpoint.

An FDE drops this into a client environment and
every workflow session is instantly instrumented,
scored, and reportable via HTTP.

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
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession, HumanAgent, AIAgent,
    BeliefState, ConceptualGraph, WorkflowStatus
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
    version     = "0.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── IN-MEMORY STORE ──────────────────────────────────────────────────────────
# Production: replace with PostgreSQL via SQLAlchemy

sessions:  Dict[str, WorkflowSession]  = {}
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
    model_name:    str
    provider:      str
    access_level:  str = "GREY_BOX"
    initial_prompt: Optional[str] = None
    metadata:      Dict[str, Any] = Field(default_factory=dict)


class BeliefStateRequest(BaseModel):
    hypotheses:   Dict[str, float]
    approximated: bool  = False
    uncertainty:  float = 0.0


class ConceptualGraphRequest(BaseModel):
    nodes: Dict[str, str]
    edges: List[Dict[str, Any]]
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
    metadata:         Dict[str, Any] = Field(default_factory=dict)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def get_session(session_id: str) -> WorkflowSession:
    if session_id not in sessions:
        raise HTTPException(
            status_code = 404,
            detail      = f"Session '{session_id}' not found"
        )
    return sessions[session_id]


def build_eat_event(
    req:        SubmitEventRequest,
    session_id: str
) -> EATEvent:
    """Convert API request to EATEvent model."""

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
        session_id       = session_id,
        agent_id         = req.agent_id,
        agent_type       = AgentType(req.agent_type),
        event_type       = EATEventType(req.event_type),
        t                = req.t,
        prior_belief     = prior_belief,
        posterior_belief = posterior_belief,
        prior_graph      = prior_graph,
        posterior_graph  = posterior_graph,
        trigger_ref      = req.trigger_ref,
        trigger_agent    = req.trigger_agent,
        confidence       = req.confidence,
        raw_evidence     = req.raw_evidence,
        metadata         = req.metadata
    )

    # Compute delta magnitude
    event.delta_magnitude = compute_delta_magnitude(event)
    return event


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status":    "healthy",
        "version":   "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "sessions":  len(sessions)
    }


# ── Session Management ─────────────────────────────────────────────────────────

@app.post("/sessions/")
def create_session(req: CreateSessionRequest):
    """Create a new PRISM workflow session."""
    session = WorkflowSession(
        client_id   = req.client_id,
        workflow_id = req.workflow_id,
        metadata    = req.metadata
    )

    sessions[session.session_id]   = session
    verdicts[session.session_id]   = VerdictScorer(session)
    detectors[session.session_id]  = DecayDetector(session)
    compasses[session.session_id]  = CompassOptimizer(
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
def get_session_info(session_id: str):
    """Get session metadata and current state."""
    session = get_session(session_id)
    return {
        "session_id":    session.session_id,
        "client_id":     session.client_id,
        "workflow_id":   session.workflow_id,
        "status":        session.status.value,
        "total_events":  session.total_events,
        "human_agents":  len(session.human_agents),
        "ai_agents":     len(session.ai_agents),
        "created_at":    session.created_at.isoformat(),
        "updated_at":    session.updated_at.isoformat()
    }


@app.get("/sessions/")
def list_sessions(client_id: Optional[str] = None):
    """List all sessions, optionally filtered by client."""
    result = []
    for sid, session in sessions.items():
        if client_id and session.client_id != client_id:
            continue
        result.append({
            "session_id":   sid,
            "client_id":    session.client_id,
            "workflow_id":  session.workflow_id,
            "status":       session.status.value,
            "total_events": session.total_events
        })
    return {"sessions": result, "total": len(result)}


# ── Agent Registration ─────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/agents/human")
def register_human_agent(session_id: str, req: RegisterHumanAgentRequest):
    """Register a human agent in a session."""
    session = get_session(session_id)
    agent   = HumanAgent(
        name      = req.name,
        role      = req.role,
        client_id = session.client_id,
        metadata  = req.metadata
    )
    session.human_agents[agent.agent_id] = agent
    return {
        "agent_id":   agent.agent_id,
        "agent_type": "HUMAN",
        "name":       agent.name,
        "role":       agent.role
    }


@app.post("/sessions/{session_id}/agents/ai")
def register_ai_agent(session_id: str, req: RegisterAIAgentRequest):
    """Register an AI agent in a session."""
    session = get_session(session_id)
    agent   = AIAgent(
        model_name   = req.model_name,
        provider     = req.provider,
        client_id    = session.client_id,
        metadata     = req.metadata
    )
    session.ai_agents[agent.agent_id] = agent

    # Register with COMPASS if initial prompt provided
    if req.initial_prompt:
        compass = compasses.get(session_id)
        if compass:
            compass.register_agent(agent.agent_id, req.initial_prompt)

    return {
        "agent_id":    agent.agent_id,
        "agent_type":  "AI",
        "model_name":  agent.model_name,
        "provider":    agent.provider
    }


# ── Event Streaming ────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/events")
def submit_event(session_id: str, req: SubmitEventRequest):
    """
    Submit an EAT event to the PULSE stream.
    Validates, scores with VERDICT, checks DECAY.
    Core ingestion endpoint — called for every epistemic act.
    """
    session = get_session(session_id)

    # Build and validate event
    try:
        event = build_eat_event(req, session_id)
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
    scorer  = verdicts.get(session_id)
    if scorer:
        event = scorer.score_event(event)

    # Add to session
    session.add_event(event)

    # Check DECAY every 5 events
    detector = detectors.get(session_id)
    decay_alerts = []
    if detector and session.total_events % 5 == 0:
        alerts = detector.check(current_t=event.t)
        decay_alerts = [
            {
                "type":           a.alert_type,
                "severity":       a.severity,
                "recommendation": a.recommendation[:100]
            }
            for a in alerts
        ]

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
        "decay_alerts":    decay_alerts,
        "warnings":        validation.warnings
    }


@app.get("/sessions/{session_id}/events")
def get_events(
    session_id: str,
    agent_id:   Optional[str] = None,
    event_type: Optional[str] = None,
    limit:      int = 50
):
    """Retrieve events from a session with optional filtering."""
    session = get_session(session_id)
    events  = session.events

    if agent_id:
        events = [e for e in events if e.agent_id == agent_id]
    if event_type:
        events = [e for e in events if e.event_type.value == event_type]

    events = events[-limit:]

    return {
        "session_id": session_id,
        "total":      len(events),
        "events": [
            {
                "event_id":        e.event_id,
                "agent_id":        e.agent_id,
                "agent_type":      e.agent_type.value,
                "event_type":      e.event_type.value,
                "t":               e.t,
                "delta_magnitude": round(e.delta_magnitude, 4),
                "timestamp":       e.timestamp.isoformat()
            }
            for e in events
        ]
    }


# ── VERDICT ────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/verdict")
def get_verdict(session_id: str):
    """Get real-time VERDICT scores for a session."""
    session = get_session(session_id)
    scorer  = verdicts.get(session_id, VerdictScorer(session))
    return scorer.session_verdict_summary()


# ── DECAY ──────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/decay")
def get_decay(session_id: str):
    """Get epistemic health analysis for a session."""
    session  = get_session(session_id)
    detector = detectors.get(session_id, DecayDetector(session))

    # Run final check
    if session.events:
        detector.check(current_t=session.events[-1].t)

    return detector.decay_summary()


# ── ATLAS ──────────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/atlas")
def get_atlas(session_id: str):
    """Get causal discovery fingerprint for a session."""
    session = get_session(session_id)
    atlas   = AtlasGraph(session)
    atlas.build_fingerprint()

    # Trace top discoveries
    high_mag = sorted(
        [e for e in session.events if e.delta_magnitude > 0.3],
        key=lambda e: e.delta_magnitude,
        reverse=True
    )[:5]
    for event in high_mag:
        atlas.trace_discovery(event.event_id)

    return atlas.export_dict()


# ── GHOST ──────────────────────────────────────────────────────────────────────

@app.post("/sessions/{session_id}/ghost")
def run_ghost(session_id: str):
    """
    Run Ghost Runner counterfactual analysis.
    Computes emergence signature for the session.
    """
    session = get_session(session_id)

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
def get_compass(session_id: str):
    """Get COMPASS optimization recommendations for a session."""
    compass = compasses.get(session_id)
    if compass is None:
        raise HTTPException(
            status_code = 404,
            detail      = f"No COMPASS optimizer for session '{session_id}'"
        )
    return compass.optimization_summary()


# ── CHRONICLE ──────────────────────────────────────────────────────────────────

@app.get("/sessions/{session_id}/report")
def generate_report(session_id: str):
    """
    Generate full Chronicle client intelligence report.
    Orchestrates all seven PRISM systems.
    """
    session  = get_session(session_id)
    compass  = compasses.get(session_id)

    reporter = ChronicleReporter(session)
    if compass:
        reporter.attach_compass(compass)

    report = reporter.generate()
    return report.to_dict()


@app.post("/sessions/{session_id}/complete")
def complete_session(session_id: str):
    """Mark a session as completed."""
    session              = get_session(session_id)
    session.status       = WorkflowStatus.COMPLETED
    session.completed_at = datetime.utcnow()
    return {
        "session_id":   session_id,
        "status":       "COMPLETED",
        "completed_at": session.completed_at.isoformat(),
        "total_events": session.total_events
    }