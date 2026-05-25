"""
PRISM — decay/detector.py
Epistemic Degradation Detector

The silent killer of enterprise AI deployments:
humans stop thinking critically, start rubber-stamping AI outputs,
belief diversity collapses, novelty plateaus.

DECAY catches this before the client notices.

Three degradation signals tracked over time:
  1. CRITICAL ENGAGEMENT RATE  — how often humans revise/reject AI beliefs
  2. BELIEF DIVERSITY INDEX    — are humans maintaining independent reasoning?
  3. NOVELTY DECAY CURVE       — is the ensemble still producing new insights?

When all three decline simultaneously → DECAY alert fires.
The FDE gets a specific, data-backed recommendation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession, BeliefState
)
from core.eat.delta import (
    kl_divergence, belief_entropy,
    ensemble_entropy, rsa_similarity
)


# ─── DECAY ALERT ──────────────────────────────────────────────────────────────

@dataclass
class DecayAlert:
    """
    Fired when DECAY detects epistemic degradation in a session.
    Carries the specific signal that triggered it and a recommendation.
    """
    session_id:        str
    alert_type:        str          # ENGAGEMENT | DIVERSITY | NOVELTY | COMPOSITE
    severity:          str          # LOW | MEDIUM | HIGH | CRITICAL
    detected_at_t:     int
    signal_value:      float        # The metric value that triggered the alert
    threshold:         float        # The threshold it crossed
    trend:             List[float]  # Recent signal values leading to alert
    recommendation:    str
    timestamp:         datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"DecayAlert({self.alert_type} | {self.severity} | "
            f"t={self.detected_at_t} | value={self.signal_value:.3f})"
        )


# ─── SIGNAL 1: CRITICAL ENGAGEMENT RATE ──────────────────────────────────────

def compute_critical_engagement_rate(
    session: WorkflowSession,
    window: int = 10
) -> float:
    """
    How often do humans critically engage with AI outputs?

    Critical engagement = human B_UPDATE event that follows an
    AI B_TRIGGER event AND shows a belief revision AWAY from
    the AI's suggested posterior (not just rubber-stamping).

    Measured over the last `window` AI trigger events.

    Returns [0.0, 1.0]:
      1.0 → humans actively revising/challenging AI outputs
      0.0 → humans rubber-stamping every AI suggestion
    """
    # Find recent AI trigger events
    ai_triggers = [
        e for e in session.events
        if e.agent_type == AgentType.AI
        and e.event_type == EATEventType.B_TRIGGER
        and e.posterior_belief is not None
    ][-window:]

    if not ai_triggers:
        return 1.0  # No triggers yet — no degradation signal

    critical_count = 0

    for trigger in ai_triggers:
        triggered_agent = trigger.trigger_agent
        if triggered_agent is None:
            continue

        # Find the human's response event
        response_events = [
            e for e in session.events
            if e.agent_id == triggered_agent
            and e.event_type == EATEventType.B_UPDATE
            and e.t > trigger.t
            and e.t <= trigger.t + 3  # Response within 3 time steps
            and e.posterior_belief is not None
        ]

        if not response_events:
            continue

        # Check if human deviated from AI suggestion
        human_response = response_events[0]
        ai_suggestion  = trigger.posterior_belief.hypotheses
        human_posterior = human_response.posterior_belief.hypotheses

        divergence = kl_divergence(ai_suggestion, human_posterior)

        # If divergence > threshold, human critically engaged
        if divergence > 0.1:
            critical_count += 1

    return float(critical_count / len(ai_triggers)) if ai_triggers else 1.0


# ─── SIGNAL 2: BELIEF DIVERSITY INDEX ────────────────────────────────────────

def compute_belief_diversity_index(
    session: WorkflowSession,
    t: Optional[int] = None
) -> float:
    """
    Are human agents maintaining independent reasoning,
    or are their beliefs converging toward the AI's outputs?

    Diversity = mean pairwise KL divergence between human belief states.
    Declining diversity = dangerous groupthink / AI anchoring.

    Returns [0.0, 1.0]:
      1.0 → maximum diversity — agents thinking independently
      0.0 → complete convergence — all agents holding identical beliefs
    """
    # Get most recent belief state per human agent
    human_beliefs: Dict[str, BeliefState] = {}

    events_to_consider = [
        e for e in session.events
        if (t is None or e.t <= t)
        and e.agent_type == AgentType.HUMAN
        and e.posterior_belief is not None
    ]

    for event in events_to_consider:
        human_beliefs[event.agent_id] = event.posterior_belief

    if len(human_beliefs) < 2:
        return 1.0  # Can't measure diversity with < 2 agents

    agents = list(human_beliefs.keys())
    pairwise_divergences = []

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            b_i = human_beliefs[agents[i]].hypotheses
            b_j = human_beliefs[agents[j]].hypotheses
            div = kl_divergence(b_i, b_j)
            pairwise_divergences.append(div)

    if not pairwise_divergences:
        return 1.0

    # Normalize: map mean divergence to [0, 1]
    # Using tanh to bound unbounded KL divergence
    mean_div = float(np.mean(pairwise_divergences))
    return float(np.tanh(mean_div))


# ─── SIGNAL 3: NOVELTY DECAY CURVE ───────────────────────────────────────────

def compute_novelty_decay_curve(
    session: WorkflowSession,
    window_size: int = 5
) -> Tuple[float, List[float]]:
    """
    Is the ensemble still producing novel insights,
    or has it plateaued?

    Novelty rate = fraction of events per time window that introduce
    concepts not seen in any prior window.

    Returns:
      current_rate: novelty rate in the most recent window
      curve: list of novelty rates across all windows
    """
    if not session.events:
        return 1.0, []

    # Collect all C_UPDATE and C_TRIGGER events with graphs
    graph_events = [
        e for e in session.events
        if e.posterior_graph is not None
    ]

    if not graph_events:
        return 1.0, []

    # Sort by time step
    graph_events.sort(key=lambda e: e.t)
    max_t = graph_events[-1].t

    # Divide into windows
    windows = []
    t = 0
    while t <= max_t:
        window_events = [
            e for e in graph_events
            if t <= e.t < t + window_size
        ]
        windows.append(window_events)
        t += window_size

    if len(windows) < 2:
        return 1.0, [1.0]

    # Track cumulative concept vocabulary
    seen_concepts: set = set()
    novelty_rates = []

    for window_events in windows:
        if not window_events:
            novelty_rates.append(0.0)
            continue

        window_concepts: set = set()
        for event in window_events:
            window_concepts.update(event.posterior_graph.nodes.keys())

        new_concepts = window_concepts - seen_concepts
        novelty_rate = (
            len(new_concepts) / len(window_concepts)
            if window_concepts else 0.0
        )
        novelty_rates.append(float(novelty_rate))
        seen_concepts.update(window_concepts)

    current_rate = novelty_rates[-1] if novelty_rates else 1.0
    return current_rate, novelty_rates


# ─── DECAY DETECTOR ───────────────────────────────────────────────────────────

class DecayDetector:
    """
    Main DECAY detection engine.

    Instantiated once per session. Monitors all three signals
    continuously as events arrive. Fires DecayAlerts when
    thresholds are crossed.

    Thresholds (configurable per client):
      engagement_threshold: 0.3  → alert if < 30% critical engagement
      diversity_threshold:  0.2  → alert if belief diversity collapses
      novelty_threshold:    0.1  → alert if novelty rate near zero
    """

    def __init__(
        self,
        session: WorkflowSession,
        engagement_threshold: float = 0.3,
        diversity_threshold:  float = 0.2,
        novelty_threshold:    float = 0.1,
        check_every:          int   = 5    # Check every N events
    ):
        self.session              = session
        self.engagement_threshold = engagement_threshold
        self.diversity_threshold  = diversity_threshold
        self.novelty_threshold    = novelty_threshold
        self.check_every          = check_every

        self.alerts: List[DecayAlert]         = []
        self.engagement_history: List[float]  = []
        self.diversity_history:  List[float]  = []
        self.novelty_history:    List[float]  = []

    def _severity(self, value: float, threshold: float) -> str:
        gap = threshold - value
        if gap > 0.3:  return "CRITICAL"
        if gap > 0.2:  return "HIGH"
        if gap > 0.1:  return "MEDIUM"
        return "LOW"

    def check(self, current_t: int) -> List[DecayAlert]:
        """
        Run decay check at current time step.
        Returns list of new alerts fired this check.
        Called by PULSE after every `check_every` events.
        """
        new_alerts = []

        # ── Signal 1: Critical Engagement ─────────────────────────────────
        engagement = compute_critical_engagement_rate(self.session)
        self.engagement_history.append(engagement)

        if engagement < self.engagement_threshold:
            alert = DecayAlert(
                session_id     = self.session.session_id,
                alert_type     = "ENGAGEMENT",
                severity       = self._severity(engagement, self.engagement_threshold),
                detected_at_t  = current_t,
                signal_value   = engagement,
                threshold      = self.engagement_threshold,
                trend          = self.engagement_history[-5:],
                recommendation = (
                    "Human agents are rubber-stamping AI outputs. "
                    "Recommend: introduce structured adversarial review step "
                    "where humans must explicitly justify agreement with AI suggestions. "
                    "Consider reducing AI output frequency to force independent thinking."
                )
            )
            self.alerts.append(alert)
            new_alerts.append(alert)

        # ── Signal 2: Belief Diversity ─────────────────────────────────────
        diversity = compute_belief_diversity_index(self.session, t=current_t)
        self.diversity_history.append(diversity)

        if diversity < self.diversity_threshold:
            alert = DecayAlert(
                session_id     = self.session.session_id,
                alert_type     = "DIVERSITY",
                severity       = self._severity(diversity, self.diversity_threshold),
                detected_at_t  = current_t,
                signal_value   = diversity,
                threshold      = self.diversity_threshold,
                trend          = self.diversity_history[-5:],
                recommendation = (
                    "Human belief states are converging — possible AI anchoring effect. "
                    "Recommend: run a blind round where humans form beliefs "
                    "without AI input, then compare. "
                    "Consider assigning explicit devil's advocate role to one human agent."
                )
            )
            self.alerts.append(alert)
            new_alerts.append(alert)

        # ── Signal 3: Novelty Decay ────────────────────────────────────────
        novelty_rate, novelty_curve = compute_novelty_decay_curve(self.session)
        self.novelty_history.append(novelty_rate)

        if novelty_rate < self.novelty_threshold:
            alert = DecayAlert(
                session_id     = self.session.session_id,
                alert_type     = "NOVELTY",
                severity       = self._severity(novelty_rate, self.novelty_threshold),
                detected_at_t  = current_t,
                signal_value   = novelty_rate,
                threshold      = self.novelty_threshold,
                trend          = self.novelty_history[-5:],
                recommendation = (
                    "Ensemble novelty has plateaued — no new concepts introduced recently. "
                    "Recommend: inject external knowledge stimulus "
                    "(new documents, domain expert, alternative model). "
                    "Consider COMPASS prompt optimization to push AI toward "
                    "conceptually distant associations."
                )
            )
            self.alerts.append(alert)
            new_alerts.append(alert)

        # ── Composite: All three signals degrading simultaneously ──────────
        if (engagement < self.engagement_threshold
                and diversity < self.diversity_threshold
                and novelty_rate < self.novelty_threshold):
            alert = DecayAlert(
                session_id     = self.session.session_id,
                alert_type     = "COMPOSITE",
                severity       = "CRITICAL",
                detected_at_t  = current_t,
                signal_value   = float(np.mean([engagement, diversity, novelty_rate])),
                threshold      = float(np.mean([
                    self.engagement_threshold,
                    self.diversity_threshold,
                    self.novelty_threshold
                ])),
                trend          = self.engagement_history[-5:],
                recommendation = (
                    "CRITICAL: All three decay signals below threshold simultaneously. "
                    "This deployment has stopped producing epistemic value. "
                    "Immediate action required: pause session, restructure workflow, "
                    "reassign AI role from lead to support, "
                    "run Ghost Runner counterfactual to quantify value loss."
                )
            )
            self.alerts.append(alert)
            new_alerts.append(alert)

        return new_alerts

    def decay_summary(self) -> Dict:
        """
        Full decay analysis for CHRONICLE client reports.
        """
        novelty_rate, novelty_curve = compute_novelty_decay_curve(self.session)

        return {
            "total_alerts":            len(self.alerts),
            "alerts_by_type": {
                "ENGAGEMENT": len([a for a in self.alerts if a.alert_type == "ENGAGEMENT"]),
                "DIVERSITY":  len([a for a in self.alerts if a.alert_type == "DIVERSITY"]),
                "NOVELTY":    len([a for a in self.alerts if a.alert_type == "NOVELTY"]),
                "COMPOSITE":  len([a for a in self.alerts if a.alert_type == "COMPOSITE"]),
            },
            "current_engagement_rate":  round(self.engagement_history[-1], 4) if self.engagement_history else None,
            "current_diversity_index":  round(self.diversity_history[-1],  4) if self.diversity_history else None,
            "current_novelty_rate":     round(novelty_rate, 4),
            "novelty_curve":            [round(v, 4) for v in novelty_curve],
            "critical_alerts":          [
                {
                    "type":           a.alert_type,
                    "severity":       a.severity,
                    "at_t":           a.detected_at_t,
                    "recommendation": a.recommendation
                }
                for a in self.alerts if a.severity in ("HIGH", "CRITICAL")
            ]
        }