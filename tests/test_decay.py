"""
PRISM — tests/test_decay.py
Epistemic Degradation Detector Tests

Run: pytest tests/test_decay.py -v
"""

import pytest
from core.eat.models import (
    BeliefState, ConceptualGraph,
    EATEvent, EATEventType, AgentType,
    WorkflowSession, HumanAgent, AIAgent
)
from decay.detector import (
    DecayDetector, DecayAlert,
    compute_critical_engagement_rate,
    compute_belief_diversity_index,
    compute_novelty_decay_curve
)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

def make_session():
    s  = WorkflowSession(client_id="acme", workflow_id="wf-test")
    h1 = HumanAgent(name="Alice", role="analyst",       client_id="acme")
    h2 = HumanAgent(name="Bob",   role="domain-expert", client_id="acme")
    ai = AIAgent(
        model_name="claude-sonnet-4-6",
        provider="anthropic",
        client_id="acme"
    )
    s.human_agents[h1.agent_id] = h1
    s.human_agents[h2.agent_id] = h2
    s.ai_agents[ai.agent_id]    = ai
    return s, h1, h2, ai


def b_update(session_id, agent_id, agent_type, t, prior_h, post_h, **kwargs):
    allowed = {"confidence", "trigger_ref", "trigger_agent"}
    extra   = {k: v for k, v in kwargs.items() if k in allowed}
    return EATEvent(
        session_id       = session_id,
        agent_id         = agent_id,
        agent_type       = agent_type,
        event_type       = EATEventType.B_UPDATE,
        t                = t,
        prior_belief     = BeliefState(hypotheses=prior_h),
        posterior_belief = BeliefState(hypotheses=post_h),
        confidence       = kwargs.get("confidence", 0.9),
        **extra
    )


def b_trigger(session_id, agent_id, t, prior_h, post_h,
              trigger_ref, trigger_agent):
    return EATEvent(
        session_id       = session_id,
        agent_id         = agent_id,
        agent_type       = AgentType.HUMAN,
        event_type       = EATEventType.B_TRIGGER,
        t                = t,
        prior_belief     = BeliefState(hypotheses=prior_h),
        posterior_belief = BeliefState(hypotheses=post_h),
        trigger_ref      = trigger_ref,
        trigger_agent    = trigger_agent,
        confidence       = 0.85
    )


def c_update(session_id, agent_id, agent_type, t, nodes, edges):
    return EATEvent(
        session_id      = session_id,
        agent_id        = agent_id,
        agent_type      = agent_type,
        event_type      = EATEventType.C_UPDATE,
        t               = t,
        prior_graph     = ConceptualGraph(nodes={}, edges=[]),
        posterior_graph = ConceptualGraph(nodes=nodes, edges=edges),
        confidence      = 0.9
    )


# ─── CRITICAL ENGAGEMENT RATE ─────────────────────────────────────────────────

class TestCriticalEngagementRate:

    def test_no_triggers_returns_one(self):
        s, h1, h2, ai = make_session()
        assert compute_critical_engagement_rate(s) == 1.0

    def test_rate_in_range(self):
        s, h1, h2, ai = make_session()
        assert 0.0 <= compute_critical_engagement_rate(s) <= 1.0

    def test_human_diverges_from_ai(self):
        s, h1, h2, ai = make_session()
        ai_event = b_update(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {"expand": 0.5, "consolidate": 0.5},
            {"expand": 0.9, "consolidate": 0.1}
        )
        s.add_event(ai_event)
        trigger = b_trigger(
            s.session_id, h1.agent_id, 2,
            {"expand": 0.5, "consolidate": 0.5},
            {"expand": 0.2, "consolidate": 0.8},
            trigger_ref   = ai_event.event_id,
            trigger_agent = h1.agent_id
        )
        s.add_event(trigger)
        assert 0.0 <= compute_critical_engagement_rate(s) <= 1.0

    def test_empty_session_returns_one(self):
        s = WorkflowSession(client_id="acme", workflow_id="wf")
        assert compute_critical_engagement_rate(s) == 1.0


# ─── BELIEF DIVERSITY INDEX ───────────────────────────────────────────────────

class TestBeliefDiversityIndex:

    def test_single_agent_returns_one(self):
        s = WorkflowSession(client_id="acme", workflow_id="wf")
        h = HumanAgent(name="Alice", role="analyst", client_id="acme")
        s.human_agents[h.agent_id] = h
        s.add_event(b_update(
            s.session_id, h.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        assert compute_belief_diversity_index(s) == 1.0

    def test_no_events_returns_one(self):
        s, h1, h2, ai = make_session()
        assert compute_belief_diversity_index(s) == 1.0

    def test_divergent_beliefs_positive_diversity(self):
        s, h1, h2, ai = make_session()
        s.add_event(b_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.9, "H2": 0.1}
        ))
        s.add_event(b_update(
            s.session_id, h2.agent_id, AgentType.HUMAN, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.1, "H2": 0.9}
        ))
        assert compute_belief_diversity_index(s) > 0.0

    def test_identical_beliefs_low_diversity(self):
        s, h1, h2, ai = make_session()
        s.add_event(b_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        s.add_event(b_update(
            s.session_id, h2.agent_id, AgentType.HUMAN, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        assert compute_belief_diversity_index(s) < 0.1

    def test_diversity_in_range(self):
        s, h1, h2, ai = make_session()
        s.add_event(b_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.6, "H2": 0.4}
        ))
        s.add_event(b_update(
            s.session_id, h2.agent_id, AgentType.HUMAN, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.4, "H2": 0.6}
        ))
        assert 0.0 <= compute_belief_diversity_index(s) <= 1.0

    def test_time_filter_single_agent(self):
        s, h1, h2, ai = make_session()
        s.add_event(b_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.9, "H2": 0.1}
        ))
        s.add_event(b_update(
            s.session_id, h2.agent_id, AgentType.HUMAN, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.1, "H2": 0.9}
        ))
        assert compute_belief_diversity_index(s, t=1) == 1.0
        assert compute_belief_diversity_index(s, t=2) > 0.0


# ─── NOVELTY DECAY CURVE ─────────────────────────────────────────────────────

class TestNoveltyDecayCurve:

    def test_no_events_returns_one(self):
        s, h1, h2, ai = make_session()
        rate, curve = compute_novelty_decay_curve(s)
        assert rate == 1.0

    def test_all_new_concepts_high_rate(self):
        s, h1, h2, ai = make_session()
        s.add_event(c_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"risk": "Risk", "revenue": "Revenue"},
            [{"source": "risk", "target": "revenue", "weight": 0.7}]
        ))
        rate, curve = compute_novelty_decay_curve(s)
        assert rate > 0.0

    def test_repeated_concepts_decay(self):
        s, h1, h2, ai = make_session()
        for t in range(1, 12):
            s.add_event(c_update(
                s.session_id, h1.agent_id, AgentType.HUMAN, t,
                {"risk": "Risk"}, []
            ))
        rate, curve = compute_novelty_decay_curve(s)
        assert rate < 0.5

    def test_curve_values_in_range(self):
        s, h1, h2, ai = make_session()
        for i, concepts in enumerate([
            {"A": "A", "B": "B"},
            {"C": "C", "D": "D"},
            {"E": "E", "F": "F"},
        ]):
            s.add_event(c_update(
                s.session_id, h1.agent_id, AgentType.HUMAN,
                (i + 1) * 5, concepts, []
            ))
        rate, curve = compute_novelty_decay_curve(s)
        for v in curve:
            assert 0.0 <= v <= 1.0

    def test_returns_tuple(self):
        s, h1, h2, ai = make_session()
        result = compute_novelty_decay_curve(s)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_expanding_concepts_sustained_rate(self):
        s, h1, h2, ai = make_session()
        for i in range(6):
            s.add_event(c_update(
                s.session_id, h1.agent_id, AgentType.HUMAN,
                i + 1,
                {f"concept_{i}": f"Concept {i}"},
                []
            ))
        rate, curve = compute_novelty_decay_curve(s)
        assert rate >= 0.0


# ─── DECAY DETECTOR ──────────────────────────────────────────────────────────

class TestDecayDetector:

    def test_instantiation(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(s)
        assert detector.engagement_threshold == 0.3
        assert detector.diversity_threshold  == 0.2
        assert detector.novelty_threshold    == 0.1
        assert len(detector.alerts)          == 0

    def test_no_alerts_diverse_session(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(s)
        s.add_event(b_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.8, "H2": 0.2}
        ))
        s.add_event(b_update(
            s.session_id, h2.agent_id, AgentType.HUMAN, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.2, "H2": 0.8}
        ))
        alerts = detector.check(current_t=2)
        diversity_alerts = [a for a in alerts if a.alert_type == "DIVERSITY"]
        assert len(diversity_alerts) == 0

    def test_diversity_alert_fires_on_convergence(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(s, diversity_threshold=0.5)
        s.add_event(b_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        s.add_event(b_update(
            s.session_id, h2.agent_id, AgentType.HUMAN, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        alerts = detector.check(current_t=2)
        diversity_alerts = [a for a in alerts if a.alert_type == "DIVERSITY"]
        assert len(diversity_alerts) > 0

    def test_alert_has_recommendation(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(s, diversity_threshold=0.9)
        s.add_event(b_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        s.add_event(b_update(
            s.session_id, h2.agent_id, AgentType.HUMAN, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        alerts = detector.check(current_t=2)
        for alert in alerts:
            assert len(alert.recommendation) > 0
            assert alert.severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_severity_boundaries(self):
        detector = DecayDetector(
            WorkflowSession(client_id="x", workflow_id="y")
        )
        # gap > 0.3 → CRITICAL
        assert detector._severity(0.1,  0.5) == "CRITICAL"
        # gap > 0.2 → HIGH
        assert detector._severity(0.25, 0.5) == "HIGH"
        # gap > 0.1 → MEDIUM
        assert detector._severity(0.35, 0.5) == "MEDIUM"
        # gap <= 0.1 → LOW
        assert detector._severity(0.45, 0.5) == "LOW"

    def test_decay_summary_structure(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(s)
        detector.check(current_t=1)
        summary  = detector.decay_summary()
        assert "total_alerts"         in summary
        assert "alerts_by_type"       in summary
        assert "current_novelty_rate" in summary
        assert "critical_alerts"      in summary
        assert "ENGAGEMENT" in summary["alerts_by_type"]
        assert "DIVERSITY"  in summary["alerts_by_type"]
        assert "NOVELTY"    in summary["alerts_by_type"]
        assert "COMPOSITE"  in summary["alerts_by_type"]

    def test_history_tracks_across_checks(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(s)
        for t in range(1, 4):
            s.add_event(b_update(
                s.session_id, h1.agent_id, AgentType.HUMAN, t,
                {"H1": 0.5, "H2": 0.5},
                {"H1": 0.7, "H2": 0.3}
            ))
            detector.check(current_t=t)
        assert len(detector.engagement_history) == 3
        assert len(detector.diversity_history)  == 3

    def test_custom_thresholds(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(
            s,
            engagement_threshold = 0.8,
            diversity_threshold  = 0.8,
            novelty_threshold    = 0.8
        )
        assert detector.engagement_threshold == 0.8
        assert detector.diversity_threshold  == 0.8
        assert detector.novelty_threshold    == 0.8

    def test_decay_alert_repr(self):
        alert = DecayAlert(
            session_id     = "S-test",
            alert_type     = "DIVERSITY",
            severity       = "HIGH",
            detected_at_t  = 5,
            signal_value   = 0.1,
            threshold      = 0.2,
            trend          = [0.3, 0.2, 0.1],
            recommendation = "Fix it."
        )
        assert "DIVERSITY" in repr(alert)
        assert "HIGH"      in repr(alert)

    def test_returns_list_of_alerts(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(s)
        result = detector.check(current_t=1)
        assert isinstance(result, list)

    def test_diversity_alert_fires_high_threshold(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(
            s,
            engagement_threshold = 0.99,
            diversity_threshold  = 0.99,
            novelty_threshold    = 0.0
        )
        s.add_event(b_update(
            s.session_id, h1.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        s.add_event(b_update(
            s.session_id, h2.agent_id, AgentType.HUMAN, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        ))
        alerts      = detector.check(current_t=2)
        alert_types = [a.alert_type for a in alerts]
        assert "DIVERSITY" in alert_types

    def test_multiple_checks_accumulate_alerts(self):
        s, h1, h2, ai = make_session()
        detector = DecayDetector(s, diversity_threshold=0.5)
        for t in range(1, 4):
            s.add_event(b_update(
                s.session_id, h1.agent_id, AgentType.HUMAN, t,
                {"H1": 0.5, "H2": 0.5},
                {"H1": 0.7, "H2": 0.3}
            ))
            s.add_event(b_update(
                s.session_id, h2.agent_id, AgentType.HUMAN, t,
                {"H1": 0.5, "H2": 0.5},
                {"H1": 0.7, "H2": 0.3}
            ))
            detector.check(current_t=t)
        assert len(detector.alerts) > 0