"""
PRISM — core/eat/validators.py
EAT Event Validation + Integrity Engine

Validates every EAT event before it enters the PULSE stream.
Catches bad data at the boundary — before it corrupts
downstream systems (VERDICT, DECAY, ATLAS, GHOST).

Three validation levels:
  SCHEMA    → Pydantic already handles this in models.py
  SEMANTIC  → Does this event make logical sense?
  INTEGRITY → Does this event fit coherently into the session?
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from datetime import datetime
from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession, BeliefState, ConceptualGraph
)
from core.eat.delta import compute_delta_magnitude


# ─── VALIDATION RESULT ────────────────────────────────────────────────────────

class ValidationResult:
    """
    Result of validating a single EAT event.
    Carries pass/fail status, error messages, and warnings.
    """
    def __init__(self):
        self.passed:   bool         = True
        self.errors:   List[str]    = []
        self.warnings: List[str]    = []
        self.confidence_penalty: float = 0.0

    def fail(self, message: str) -> None:
        self.passed = False
        self.errors.append(message)

    def warn(self, message: str, penalty: float = 0.05) -> None:
        self.warnings.append(message)
        self.confidence_penalty += penalty

    def adjusted_confidence(self, base: float) -> float:
        return max(0.0, base - self.confidence_penalty)

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"ValidationResult({status} | "
            f"errors={len(self.errors)} | "
            f"warnings={len(self.warnings)} | "
            f"penalty={self.confidence_penalty:.2f})"
        )


# ─── SEMANTIC VALIDATORS ──────────────────────────────────────────────────────

def validate_belief_state(
    belief: BeliefState,
    result: ValidationResult,
    label: str = "belief"
) -> None:
    """Validate a single belief state for semantic correctness."""

    if not belief.hypotheses:
        result.fail(f"{label}: hypothesis space is empty")
        return

    # Check all confidences in [0, 1]
    for h, p in belief.hypotheses.items():
        if not 0.0 <= p <= 1.0:
            result.fail(f"{label}: confidence {p} for '{h}' out of range [0,1]")

    # Warn if approximated but no uncertainty declared
    if belief.approximated and belief.uncertainty == 0.0:
        result.warn(
            f"{label}: marked approximated but uncertainty=0.0 — "
            f"set uncertainty to reflect approximation error",
            penalty=0.1
        )

    # Warn if all confidences are equal — likely placeholder data
    values = list(belief.hypotheses.values())
    if len(set(values)) == 1 and len(values) > 1:
        result.warn(
            f"{label}: all confidences identical ({values[0]}) — "
            f"possible placeholder data",
            penalty=0.05
        )


def validate_conceptual_graph(
    graph: ConceptualGraph,
    result: ValidationResult,
    label: str = "graph"
) -> None:
    """Validate a conceptual graph for structural integrity."""

    # Check all edge nodes exist in node registry
    node_ids = set(graph.nodes.keys())
    for i, edge in enumerate(graph.edges):
        src = edge.get("source")
        tgt = edge.get("target")
        if src is None or tgt is None:
            result.fail(f"{label} edge[{i}]: missing source or target")
            continue
        if src not in node_ids:
            result.warn(
                f"{label} edge[{i}]: source '{src}' not in node registry",
                penalty=0.05
            )
        if tgt not in node_ids:
            result.warn(
                f"{label} edge[{i}]: target '{tgt}' not in node registry",
                penalty=0.05
            )

        # Check edge weight in [0, 1] if present
        w = edge.get("weight")
        if w is not None and not 0.0 <= float(w) <= 1.0:
            result.warn(
                f"{label} edge[{i}]: weight {w} outside [0,1]",
                penalty=0.02
            )

    # Warn on empty graph — valid but suspicious
    if not graph.nodes:
        result.warn(f"{label}: empty node set", penalty=0.1)


# ─── EVENT-LEVEL VALIDATOR ────────────────────────────────────────────────────

def validate_eat_event(
    event: EATEvent,
    session: Optional[WorkflowSession] = None
) -> ValidationResult:
    """
    Full semantic + integrity validation of a single EAT event.

    Semantic checks: does this event make logical sense in isolation?
    Integrity checks: does this event fit coherently into the session?
    """
    result = ValidationResult()

    # ── 1. Agent exists in session ─────────────────────────────────────────
    if session is not None:
        if event.agent_id not in session.all_agents:
            result.fail(
                f"Agent '{event.agent_id}' not registered in session "
                f"'{session.session_id}'"
            )

    # ── 2. Agent type consistency ──────────────────────────────────────────
    if session is not None and event.agent_id in session.all_agents:
        registered = session.all_agents[event.agent_id]
        if registered.agent_type != event.agent_type:
            result.fail(
                f"Agent type mismatch: event says {event.agent_type}, "
                f"session has {registered.agent_type}"
            )

    # ── 3. Belief state validation ─────────────────────────────────────────
    if event.prior_belief is not None:
        validate_belief_state(event.prior_belief, result, "prior_belief")
    if event.posterior_belief is not None:
        validate_belief_state(event.posterior_belief, result, "posterior_belief")

    # ── 4. Conceptual graph validation ────────────────────────────────────
    if event.prior_graph is not None:
        validate_conceptual_graph(event.prior_graph, result, "prior_graph")
    if event.posterior_graph is not None:
        validate_conceptual_graph(event.posterior_graph, result, "posterior_graph")

    # ── 5. Zero-delta warning ──────────────────────────────────────────────
    magnitude = compute_delta_magnitude(event)
    if magnitude == 0.0:
        result.warn(
            "delta_magnitude is 0.0 — prior and posterior states are identical. "
            "Possible duplicate event or no actual epistemic change occurred.",
            penalty=0.15
        )

    # ── 6. Trigger reference validation ───────────────────────────────────
    if event.event_type in (EATEventType.B_TRIGGER, EATEventType.C_TRIGGER):
        if event.trigger_ref is None:
            result.fail("TRIGGER event missing trigger_ref")
        elif session is not None:
            # Verify the trigger event actually exists in this session
            trigger_ids = {e.event_id for e in session.events}
            if event.trigger_ref not in trigger_ids:
                result.warn(
                    f"trigger_ref '{event.trigger_ref}' not found in session "
                    f"events — may be cross-session trigger or ordering issue",
                    penalty=0.1
                )

    # ── 7. Temporal ordering ───────────────────────────────────────────────
    if session is not None and session.events:
        last_event = session.events[-1]
        if event.t < last_event.t:
            result.warn(
                f"Event t={event.t} is earlier than last session event "
                f"t={last_event.t} — out-of-order event detected",
                penalty=0.05
            )

    # ── 8. AI approximation warning ───────────────────────────────────────
    if event.agent_type == AgentType.AI:
        if event.prior_belief and not event.prior_belief.approximated:
            result.warn(
                "AI agent belief state not marked as approximated — "
                "AI beliefs are never directly observed, set approximated=True",
                penalty=0.1
            )

    # ── 9. Confidence sanity ──────────────────────────────────────────────
    if not 0.0 <= event.confidence <= 1.0:
        result.fail(f"Event confidence {event.confidence} out of range [0,1]")

    return result


# ─── BATCH VALIDATOR ──────────────────────────────────────────────────────────

def validate_session_events(
    session: WorkflowSession
) -> List[Tuple[EATEvent, ValidationResult]]:
    """
    Validate all events in a session in sequence.
    Returns list of (event, result) tuples.
    Used by GHOST Runner before replaying a session.
    """
    results = []
    for event in session.events:
        result = validate_eat_event(event, session)
        results.append((event, result))
    return results


def session_integrity_score(
    session: WorkflowSession
) -> float:
    """
    Scalar integrity score for an entire session [0.0, 1.0].
    1.0 = all events passed validation with no warnings.
    0.0 = critical failures throughout.

    Used by CHRONICLE to flag data quality issues in client reports.
    """
    if not session.events:
        return 1.0

    results = validate_session_events(session)
    scores = []

    for event, result in results:
        if not result.passed:
            scores.append(0.0)
        else:
            adjusted = result.adjusted_confidence(event.confidence)
            scores.append(adjusted)

    import numpy as np
    return float(np.mean(scores))