"""
PRISM — verdict/scorer.py
Four-Dimensional AI Evaluation Engine

Scores every AI epistemic event across four dimensions
the moment it fires into the PULSE stream.

The four dimensions:
  1. GROUNDEDNESS     — is this belief anchored in session evidence?
  2. CALIBRATION      — when AI is confident, is it right more often?
  3. INFLUENCE SURVIVAL — do AI-triggered human beliefs survive?
  4. NOVELTY DELTA    — did AI introduce concepts humans didn't have?

No other eval tool measures these in live production workflows.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession, BeliefState, ConceptualGraph
)
from core.eat.delta import (
    kl_divergence, belief_entropy,
    rsa_similarity, node_novelty_score
)


# ─── DIMENSION 1: GROUNDEDNESS ────────────────────────────────────────────────

def score_groundedness(
    event: EATEvent,
    session_knowledge: Dict[str, float],
    threshold: float = 0.3
) -> float:
    """
    Is this AI belief update anchored in evidence present
    in the current session context — or confabulated from pretraining?

    Measured via overlap between the AI's posterior hypothesis
    confidences and the session knowledge base weights.

    session_knowledge: {hypothesis_id → evidence_weight}
                       built from human-confirmed beliefs and
                       retrieved documents in this session.

    Returns [0.0, 1.0]:
      1.0 → fully grounded in session evidence
      0.0 → no overlap with session knowledge base
    """
    if event.agent_type != AgentType.AI:
        return 1.0  # Groundedness only scored for AI agents

    if event.posterior_belief is None:
        return 0.0

    if not session_knowledge:
        return 0.5  # No knowledge base yet — neutral score

    posterior = event.posterior_belief.hypotheses
    overlap_scores = []

    for hypothesis, ai_confidence in posterior.items():
        session_weight = session_knowledge.get(hypothesis, 0.0)
        # Overlap: both AI and session agree this hypothesis matters
        overlap = min(ai_confidence, session_weight)
        overlap_scores.append(overlap)

    if not overlap_scores:
        return 0.0

    return float(np.mean(overlap_scores))


# ─── DIMENSION 2: CALIBRATION ─────────────────────────────────────────────────

class CalibrationTracker:
    """
    Tracks AI calibration across a session.

    Calibration: when the AI expresses high confidence,
    is it right more often than when it expresses low confidence?

    This is Anthropic's alignment team's most-watched metric.
    PRISM makes it visible to every FDE in production.

    Tracks (confidence_bucket → [correct, total]) per AI agent.
    """

    def __init__(self, n_buckets: int = 5):
        self.n_buckets = n_buckets
        # agent_id → bucket_index → [correct_count, total_count]
        self.records: Dict[str, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0])
        )

    def _bucket(self, confidence: float) -> int:
        """Map confidence [0,1] to bucket index [0, n_buckets-1]."""
        return min(int(confidence * self.n_buckets), self.n_buckets - 1)

    def record_prediction(
        self,
        agent_id: str,
        confidence: float,
        was_correct: bool
    ) -> None:
        """Record a single AI prediction outcome."""
        bucket = self._bucket(confidence)
        self.records[agent_id][bucket][1] += 1  # total
        if was_correct:
            self.records[agent_id][bucket][0] += 1  # correct

    def calibration_score(self, agent_id: str) -> float:
        """
        Expected Calibration Error (ECE) for an agent.
        Lower ECE = better calibrated.
        Returned as 1 - ECE so higher = better (consistent with other scores).
        """
        agent_records = self.records.get(agent_id, {})
        if not agent_records:
            return 0.5  # No data — neutral

        total_predictions = sum(r[1] for r in agent_records.values())
        if total_predictions == 0:
            return 0.5

        ece = 0.0
        bucket_width = 1.0 / self.n_buckets

        for bucket_idx, (correct, total) in agent_records.items():
            if total == 0:
                continue
            bucket_confidence = (bucket_idx + 0.5) * bucket_width
            accuracy = correct / total
            weight = total / total_predictions
            ece += weight * abs(bucket_confidence - accuracy)

        return float(1.0 - ece)

    def all_agent_scores(self) -> Dict[str, float]:
        return {
            agent_id: self.calibration_score(agent_id)
            for agent_id in self.records
        }


# ─── DIMENSION 3: INFLUENCE SURVIVAL ─────────────────────────────────────────

def score_influence_survival(
    trigger_event: EATEvent,
    session: WorkflowSession,
    survival_window: int = 5
) -> float:
    """
    When this AI event triggered a belief change in a human,
    did that belief survive subsequent evidence rounds?

    Survival = the human didn't significantly revert their
    belief back toward the pre-trigger state within
    survival_window time steps.

    High survival rate → AI is adding real epistemic value.
    Low survival rate  → AI sounds confident but humans revert —
                         a key signal of low-quality influence.

    Returns [0.0, 1.0]:
      1.0 → triggered beliefs fully survived
      0.0 → human completely reverted within survival_window
    """
    if trigger_event.event_type not in (
        EATEventType.B_TRIGGER, EATEventType.C_TRIGGER
    ):
        return 1.0  # Not a trigger event

    if trigger_event.posterior_belief is None:
        return 1.0

    # Find subsequent B_UPDATE events from the triggered human agent
    trigger_t = trigger_event.t
    triggered_agent = trigger_event.trigger_agent

    if triggered_agent is None:
        return 1.0

    subsequent_updates = [
        e for e in session.events
        if e.agent_id == triggered_agent
        and e.event_type == EATEventType.B_UPDATE
        and trigger_t < e.t <= trigger_t + survival_window
        and e.prior_belief is not None
        and e.posterior_belief is not None
    ]

    if not subsequent_updates:
        return 1.0  # No updates in window — belief unchanged = survived

    # Measure drift back toward pre-trigger belief
    pre_trigger_belief = trigger_event.prior_belief
    if pre_trigger_belief is None:
        return 1.0

    post_trigger_belief = trigger_event.posterior_belief
    survival_scores = []

    for update in subsequent_updates:
        # Distance from post-trigger state to current state
        drift_to_prior = kl_divergence(
            post_trigger_belief.hypotheses,
            update.posterior_belief.hypotheses
        )
        # Distance from post-trigger state to pre-trigger state
        total_shift = kl_divergence(
            pre_trigger_belief.hypotheses,
            post_trigger_belief.hypotheses
        )

        if total_shift == 0:
            survival_scores.append(1.0)
        else:
            # How much of the original shift survived?
            reversion = min(drift_to_prior / total_shift, 1.0)
            survival_scores.append(1.0 - reversion)

    return float(np.mean(survival_scores)) if survival_scores else 1.0


# ─── DIMENSION 4: NOVELTY DELTA ───────────────────────────────────────────────

def score_novelty_delta(
    event: EATEvent,
    session: WorkflowSession
) -> float:
    """
    Did this AI agent introduce a concept or connection
    that wasn't already present in the human agents'
    conceptual graphs before this event?

    Novelty delta = fraction of posterior concepts that are
    genuinely new to the ensemble — not just new to this agent.

    Returns [0.0, 1.0]:
      1.0 → all concepts introduced are new to the ensemble
      0.0 → all concepts already existed in human agent graphs
    """
    if event.agent_type != AgentType.AI:
        return 0.0

    if event.posterior_graph is None:
        return 0.0

    # Collect all concepts present in human agents' graphs
    # at time steps before this event
    human_concepts: set = set()
    for past_event in session.events:
        if (past_event.agent_type == AgentType.HUMAN
                and past_event.t < event.t
                and past_event.posterior_graph is not None):
            human_concepts.update(past_event.posterior_graph.nodes.keys())

    if not human_concepts:
        # No human concepts yet — everything AI introduces is novel
        return 1.0

    ai_concepts = set(event.posterior_graph.nodes.keys())
    if not ai_concepts:
        return 0.0

    new_to_ensemble = ai_concepts - human_concepts
    return float(len(new_to_ensemble) / len(ai_concepts))


# ─── VERDICT SCORER ───────────────────────────────────────────────────────────

class VerdictScorer:
    """
    Main VERDICT scoring engine.

    Instantiated once per session. Scores every AI EAT event
    across all four dimensions as it enters the PULSE stream.

    Maintains calibration state across the session.
    Populates event.groundedness, event.calibration_score,
    event.influence_survival, event.novelty_delta in-place.
    """

    def __init__(self, session: WorkflowSession):
        self.session = session
        self.calibration_tracker = CalibrationTracker()
        self.session_knowledge: Dict[str, float] = {}

    def update_session_knowledge(
        self,
        human_event: EATEvent
    ) -> None:
        """
        Update session knowledge base from confirmed human belief events.
        Only human-confirmed beliefs contribute to the knowledge base.
        """
        if (human_event.agent_type == AgentType.HUMAN
                and human_event.posterior_belief is not None):
            for h, confidence in human_event.posterior_belief.hypotheses.items():
                # Running average — weight recent human beliefs more
                existing = self.session_knowledge.get(h, confidence)
                self.session_knowledge[h] = 0.7 * confidence + 0.3 * existing

    def score_event(self, event: EATEvent) -> EATEvent:
        """
        Score a single EAT event across all four VERDICT dimensions.
        Populates the event's score fields in-place and returns it.
        """
        # Update knowledge base if human event
        if event.agent_type == AgentType.HUMAN:
            self.update_session_knowledge(event)
            return event  # Humans not scored by VERDICT

        # ── Score all four dimensions ──────────────────────────────────────
        event.groundedness = score_groundedness(
            event, self.session_knowledge
        )

        event.novelty_delta = score_novelty_delta(
            event, self.session
        )

        event.influence_survival = score_influence_survival(
            event, self.session
        )

        # Calibration uses tracker state — updated per session
        event.calibration_score = self.calibration_tracker.calibration_score(
            event.agent_id
        )

        return event

    def session_verdict_summary(self) -> Dict:
        """
        Aggregate VERDICT scores across the full session.
        Used by CHRONICLE for client reports.
        """
        ai_events = self.session.get_ai_events()

        if not ai_events:
            return {
                "total_ai_events": 0,
                "mean_groundedness": None,
                "mean_novelty_delta": None,
                "mean_influence_survival": None,
                "calibration_by_agent": {},
                "verdict_grade": "NO_DATA"
            }

        groundedness_scores = [
            e.groundedness for e in ai_events
            if e.groundedness is not None
        ]
        novelty_scores = [
            e.novelty_delta for e in ai_events
            if e.novelty_delta is not None
        ]
        survival_scores = [
            e.influence_survival for e in ai_events
            if e.influence_survival is not None
        ]

        mean_g = float(np.mean(groundedness_scores)) if groundedness_scores else None
        mean_n = float(np.mean(novelty_scores)) if novelty_scores else None
        mean_s = float(np.mean(survival_scores)) if survival_scores else None

        # Composite grade
        scores = [s for s in [mean_g, mean_n, mean_s] if s is not None]
        composite = float(np.mean(scores)) if scores else 0.0

        if composite >= 0.8:
            grade = "EXCELLENT"
        elif composite >= 0.6:
            grade = "GOOD"
        elif composite >= 0.4:
            grade = "MODERATE"
        else:
            grade = "POOR"

        return {
            "total_ai_events":       len(ai_events),
            "mean_groundedness":     round(mean_g, 4) if mean_g else None,
            "mean_novelty_delta":    round(mean_n, 4) if mean_n else None,
            "mean_influence_survival": round(mean_s, 4) if mean_s else None,
            "calibration_by_agent":  self.calibration_tracker.all_agent_scores(),
            "composite_score":       round(composite, 4),
            "verdict_grade":         grade
        }