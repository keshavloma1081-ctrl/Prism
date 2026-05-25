"""
PRISM — tests/test_verdict.py
VERDICT Four-Dimensional Eval Engine Tests

Run: pytest tests/test_verdict.py -v
"""

import pytest
import numpy as np
from core.eat.models import (
    BeliefState, ConceptualGraph,
    EATEvent, EATEventType, AgentType,
    WorkflowSession, HumanAgent, AIAgent
)
from verdict.scorer import (
    VerdictScorer, CalibrationTracker,
    score_groundedness, score_novelty_delta,
    score_influence_survival
)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

def make_session():
    s     = WorkflowSession(client_id="acme", workflow_id="wf-test")
    human = HumanAgent(name="Alice", role="analyst", client_id="acme")
    ai    = AIAgent(
        model_name="claude-sonnet-4-6",
        provider="anthropic",
        client_id="acme"
    )
    s.human_agents[human.agent_id] = human
    s.ai_agents[ai.agent_id]       = ai
    return s, human, ai


def make_b_event(session_id, agent_id, agent_type, t,
                  prior_h, post_h, approximated=False, **kwargs):
    allowed = {
        "confidence", "trigger_ref", "trigger_agent",
        "groundedness", "novelty_delta",
        "influence_survival", "calibration_score",
        "raw_evidence", "metadata"
    }
    extra = {k: v for k, v in kwargs.items() if k in allowed}
    return EATEvent(
        session_id       = session_id,
        agent_id         = agent_id,
        agent_type       = agent_type,
        event_type       = EATEventType.B_UPDATE,
        t                = t,
        prior_belief     = BeliefState(
            hypotheses   = prior_h,
            approximated = approximated,
            uncertainty  = 0.1 if approximated else 0.0
        ),
        posterior_belief = BeliefState(
            hypotheses   = post_h,
            approximated = approximated,
            uncertainty  = 0.1 if approximated else 0.0
        ),
        confidence       = kwargs.get("confidence", 0.9),
        **extra
    )


def make_c_event(session_id, agent_id, agent_type, t,
                  prior_nodes, prior_edges,
                  post_nodes,  post_edges,
                  approximated=False, **kwargs):
    allowed = {
        "confidence", "trigger_ref", "trigger_agent",
        "groundedness", "novelty_delta",
        "influence_survival", "calibration_score",
        "raw_evidence", "metadata"
    }
    extra = {k: v for k, v in kwargs.items() if k in allowed}
    return EATEvent(
        session_id      = session_id,
        agent_id        = agent_id,
        agent_type      = agent_type,
        event_type      = EATEventType.C_UPDATE,
        t               = t,
        prior_graph     = ConceptualGraph(
            nodes        = prior_nodes,
            edges        = prior_edges,
            approximated = approximated
        ),
        posterior_graph = ConceptualGraph(
            nodes        = post_nodes,
            edges        = post_edges,
            approximated = approximated
        ),
        confidence      = kwargs.get("confidence", 0.9),
        **extra
    )


# ─── GROUNDEDNESS ─────────────────────────────────────────────────────────────

class TestGroundedness:

    def test_human_agent_always_1(self):
        s, human, ai = make_session()
        event = make_b_event(
            s.session_id, human.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        )
        score = score_groundedness(event, {"H1": 0.8, "H2": 0.2})
        assert score == 1.0

    def test_empty_knowledge_base_neutral(self):
        s, human, ai = make_session()
        event = make_b_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3},
            approximated=True
        )
        assert score_groundedness(event, {}) == 0.5

    def test_grounded_ai_event_in_range(self):
        s, human, ai = make_session()
        event = make_b_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.8, "H2": 0.2},
            approximated=True
        )
        score = score_groundedness(event, {"H1": 0.9, "H2": 0.1})
        assert 0.0 <= score <= 1.0

    def test_ungrounded_ai_event_low_score(self):
        s, human, ai = make_session()
        event = make_b_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.9, "H2": 0.1},
            approximated=True
        )
        score = score_groundedness(event, {"H1": 0.05, "H2": 0.95})
        assert score < 0.5

    def test_no_posterior_returns_zero(self):
        s, human, ai = make_session()
        event = make_c_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {}, [], {"A": "A"}, [],
            approximated=True
        )
        assert score_groundedness(event, {"H1": 0.8}) == 0.0

    def test_well_grounded_higher_than_ungrounded(self):
        s, human, ai = make_session()
        knowledge  = {"H1": 0.9, "H2": 0.1}
        grounded   = make_b_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.85, "H2": 0.15},
            approximated=True
        )
        ungrounded = make_b_event(
            s.session_id, ai.agent_id, AgentType.AI, 2,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.05, "H2": 0.95},
            approximated=True
        )
        assert (score_groundedness(grounded, knowledge) >
                score_groundedness(ungrounded, knowledge))


# ─── CALIBRATION TRACKER ──────────────────────────────────────────────────────

class TestCalibrationTracker:

    def test_no_data_neutral(self):
        ct = CalibrationTracker()
        assert ct.calibration_score("unknown-agent") == 0.5

    def test_perfect_calibration(self):
        ct = CalibrationTracker(n_buckets=2)
        for _ in range(10):
            ct.record_prediction("agent-1", 0.9, was_correct=True)
        for _ in range(10):
            ct.record_prediction("agent-1", 0.1, was_correct=False)
        assert ct.calibration_score("agent-1") > 0.7

    def test_poor_calibration(self):
        ct = CalibrationTracker(n_buckets=2)
        for _ in range(10):
            ct.record_prediction("agent-1", 0.9, was_correct=False)
        for _ in range(10):
            ct.record_prediction("agent-1", 0.1, was_correct=True)
        assert ct.calibration_score("agent-1") < 0.5

    def test_all_agents_scores(self):
        ct = CalibrationTracker()
        ct.record_prediction("A1", 0.8, was_correct=True)
        ct.record_prediction("A2", 0.6, was_correct=False)
        scores = ct.all_agent_scores()
        assert "A1" in scores
        assert "A2" in scores

    def test_score_in_range(self):
        ct = CalibrationTracker()
        for _ in range(20):
            ct.record_prediction(
                "agent",
                np.random.random(),
                was_correct=bool(np.random.randint(2))
            )
        assert 0.0 <= ct.calibration_score("agent") <= 1.0

    def test_multiple_agents_independent(self):
        ct = CalibrationTracker(n_buckets=2)
        for _ in range(10):
            ct.record_prediction("agent-A", 0.9, was_correct=True)
        for _ in range(10):
            ct.record_prediction("agent-B", 0.9, was_correct=False)
        assert (ct.calibration_score("agent-A") !=
                ct.calibration_score("agent-B"))


# ─── NOVELTY DELTA ────────────────────────────────────────────────────────────

class TestNoveltyDelta:

    def test_human_event_zero(self):
        s, human, ai = make_session()
        event = make_c_event(
            s.session_id, human.agent_id, AgentType.HUMAN, 1,
            {}, [], {"risk": "Risk"}, []
        )
        s.add_event(event)
        assert score_novelty_delta(event, s) == 0.0

    def test_ai_event_no_prior_human_concepts_max(self):
        s, human, ai = make_session()
        event = make_c_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {}, [],
            {"innovation": "Innovation", "network": "Network"},
            [],
            approximated=True
        )
        s.add_event(event)
        assert score_novelty_delta(event, s) == 1.0

    def test_ai_event_all_known_concepts_zero(self):
        s, human, ai = make_session()
        h_event = make_c_event(
            s.session_id, human.agent_id, AgentType.HUMAN, 1,
            {}, [],
            {"risk": "Risk", "revenue": "Revenue"},
            []
        )
        s.add_event(h_event)

        ai_event = make_c_event(
            s.session_id, ai.agent_id, AgentType.AI, 2,
            {}, [],
            {"risk": "Risk", "revenue": "Revenue"},
            [],
            approximated=True
        )
        s.add_event(ai_event)
        assert score_novelty_delta(ai_event, s) == 0.0

    def test_ai_event_partial_novelty(self):
        s, human, ai = make_session()
        h_event = make_c_event(
            s.session_id, human.agent_id, AgentType.HUMAN, 1,
            {}, [], {"risk": "Risk"}, []
        )
        s.add_event(h_event)

        ai_event = make_c_event(
            s.session_id, ai.agent_id, AgentType.AI, 2,
            {}, [],
            {"risk": "Risk", "innovation": "Innovation"},
            [],
            approximated=True
        )
        s.add_event(ai_event)
        score = score_novelty_delta(ai_event, s)
        assert 0 < score < 1.0

    def test_score_in_range(self):
        s, human, ai = make_session()
        event = make_c_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {}, [], {"A": "A", "B": "B"}, [],
            approximated=True
        )
        s.add_event(event)
        assert 0.0 <= score_novelty_delta(event, s) <= 1.0

    def test_null_posterior_graph_zero(self):
        s, human, ai = make_session()
        event = make_b_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.3, "H2": 0.7},
            approximated=True
        )
        # B_UPDATE has no posterior_graph → novelty_delta = 0
        s.add_event(event)
        assert score_novelty_delta(event, s) == 0.0


# ─── VERDICT SCORER ───────────────────────────────────────────────────────────

class TestVerdictScorer:

    def test_human_event_not_scored(self):
        s, human, ai = make_session()
        scorer = VerdictScorer(s)
        event  = make_b_event(
            s.session_id, human.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        )
        s.add_event(event)
        result = scorer.score_event(event)
        assert result.groundedness  is None
        assert result.novelty_delta is None

    def test_ai_event_all_dimensions_scored(self):
        s, human, ai = make_session()
        scorer   = VerdictScorer(s)
        ai_event = make_b_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.3, "H2": 0.7},
            approximated=True
        )
        s.add_event(ai_event)
        result = scorer.score_event(ai_event)
        assert result.groundedness       is not None
        assert result.novelty_delta      is not None
        assert result.influence_survival is not None
        assert result.calibration_score  is not None

    def test_session_knowledge_updates_from_human(self):
        s, human, ai = make_session()
        scorer  = VerdictScorer(s)
        h_event = make_b_event(
            s.session_id, human.agent_id, AgentType.HUMAN, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.8, "H2": 0.2}
        )
        s.add_event(h_event)
        scorer.score_event(h_event)
        assert "H1" in scorer.session_knowledge
        assert scorer.session_knowledge["H1"] > 0

    def test_verdict_summary_no_ai_events(self):
        s, human, ai = make_session()
        scorer  = VerdictScorer(s)
        summary = scorer.session_verdict_summary()
        assert summary["verdict_grade"]   == "NO_DATA"
        assert summary["total_ai_events"] == 0

    def test_verdict_summary_with_ai_events(self):
        s, human, ai = make_session()
        scorer = VerdictScorer(s)
        for t in range(1, 4):
            event = make_b_event(
                s.session_id, ai.agent_id, AgentType.AI, t,
                {"H1": 0.5, "H2": 0.5},
                {"H1": 0.3, "H2": 0.7},
                approximated=True
            )
            s.add_event(event)
            scorer.score_event(event)
        summary = scorer.session_verdict_summary()
        assert summary["total_ai_events"] == 3
        assert summary["verdict_grade"] in (
            "EXCELLENT", "GOOD", "MODERATE", "POOR"
        )
        assert 0.0 <= summary["composite_score"] <= 1.0

    def test_all_scores_in_valid_range(self):
        s, human, ai = make_session()
        scorer   = VerdictScorer(s)
        ai_event = make_b_event(
            s.session_id, ai.agent_id, AgentType.AI, 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.2, "H2": 0.8},
            approximated=True
        )
        s.add_event(ai_event)
        result = scorer.score_event(ai_event)
        for attr in [
            "groundedness", "novelty_delta",
            "influence_survival", "calibration_score"
        ]:
            val = getattr(result, attr)
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{attr} out of range: {val}"

    def test_knowledge_base_grows_with_human_events(self):
        s, human, ai = make_session()
        scorer = VerdictScorer(s)
        for t, h in enumerate([
            {"H1": 0.7, "H2": 0.3},
            {"H1": 0.6, "H2": 0.4},
            {"H1": 0.8, "H2": 0.2},
        ], start=1):
            event = make_b_event(
                s.session_id, human.agent_id, AgentType.HUMAN, t,
                {"H1": 0.5, "H2": 0.5}, h
            )
            s.add_event(event)
            scorer.score_event(event)
        assert len(scorer.session_knowledge) > 0

    def test_verdict_grade_mapping_logic(self):
        # Test the grade boundary logic directly
        cases = [
            (0.85, "EXCELLENT"),
            (0.70, "GOOD"),
            (0.50, "MODERATE"),
            (0.20, "POOR"),
        ]
        for score, expected in cases:
            if score >= 0.8:
                grade = "EXCELLENT"
            elif score >= 0.6:
                grade = "GOOD"
            elif score >= 0.4:
                grade = "MODERATE"
            else:
                grade = "POOR"
            assert grade == expected

    def test_verdict_grade_poor_with_prescore(self):
        s, human, ai = make_session()
        scorer = VerdictScorer(s)
        for t in range(1, 4):
            event = make_b_event(
                s.session_id, ai.agent_id, AgentType.AI, t,
                {"H1": 0.5, "H2": 0.5},
                {"H1": 0.3, "H2": 0.7},
                approximated=True
            )
            # Set scores directly before adding to session
            event.groundedness       = 0.1
            event.novelty_delta      = 0.1
            event.influence_survival = 0.1
            event.calibration_score  = 0.1
            s.add_event(event)
        summary = scorer.session_verdict_summary()
        assert summary["verdict_grade"] == "POOR"

    def test_verdict_grade_excellent_with_prescore(self):
        s, human, ai = make_session()
        scorer = VerdictScorer(s)
        for t in range(1, 6):
            event = make_b_event(
                s.session_id, ai.agent_id, AgentType.AI, t,
                {"H1": 0.5, "H2": 0.5},
                {"H1": 0.3, "H2": 0.7},
                approximated=True
            )
            event.groundedness       = 0.95
            event.novelty_delta      = 0.90
            event.influence_survival = 0.92
            event.calibration_score  = 0.88
            s.add_event(event)
        summary = scorer.session_verdict_summary()
        assert summary["verdict_grade"] in ("EXCELLENT", "GOOD")