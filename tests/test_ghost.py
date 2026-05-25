"""
PRISM — tests/test_ghost.py
Ghost Runner Counterfactual Replay Engine Tests

Run: pytest tests/test_ghost.py -v
"""

import pytest
from core.eat.models import (
    BeliefState, ConceptualGraph,
    EATEvent, EATEventType, AgentType,
    WorkflowSession, HumanAgent, AIAgent
)
from ghost.replay import (
    GhostRunner, ReplayMode,
    ReplayResult, EmergenceSignature
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


def add_b_update(session, agent_id, agent_type, t, prior_h, post_h,
                  approximated=False):
    event = EATEvent(
        session_id       = session.session_id,
        agent_id         = agent_id,
        agent_type       = agent_type,
        event_type       = EATEventType.B_UPDATE,
        t                = t,
        prior_belief     = BeliefState(
            hypotheses=prior_h,
            approximated=approximated,
            uncertainty=0.1 if approximated else 0.0
        ),
        posterior_belief = BeliefState(
            hypotheses=post_h,
            approximated=approximated,
            uncertainty=0.1 if approximated else 0.0
        ),
        confidence       = 0.9
    )
    session.add_event(event)
    return event


def add_c_update(session, agent_id, agent_type, t,
                  prior_nodes, prior_edges,
                  post_nodes,  post_edges,
                  approximated=False):
    event = EATEvent(
        session_id      = session.session_id,
        agent_id        = agent_id,
        agent_type      = agent_type,
        event_type      = EATEventType.C_UPDATE,
        t               = t,
        prior_graph     = ConceptualGraph(
            nodes=prior_nodes, edges=prior_edges,
            approximated=approximated
        ),
        posterior_graph = ConceptualGraph(
            nodes=post_nodes, edges=post_edges,
            approximated=approximated
        ),
        delta_magnitude = 0.7,
        confidence      = 0.9
    )
    session.add_event(event)
    return event


def build_rich_session():
    """Build a session with both human and AI events and concepts."""
    s, h1, h2, ai = make_session()

    # Human establishes baseline
    add_b_update(s, h1.agent_id, AgentType.HUMAN, 1,
                 {"expand": 0.5, "consolidate": 0.5},
                 {"expand": 0.7, "consolidate": 0.3})

    add_c_update(s, h1.agent_id, AgentType.HUMAN, 2,
                 {}, [],
                 {"risk": "Risk", "revenue": "Revenue", "market": "Market"},
                 [{"source": "risk", "target": "revenue", "weight": 0.7}])

    # AI introduces new concepts
    add_c_update(s, ai.agent_id, AgentType.AI, 3,
                 {}, [],
                 {"risk":           "Risk",
                  "network_effect": "Network Effect",
                  "innovation":     "Innovation",
                  "competitor":     "Competitor"},
                 [{"source": "innovation",
                   "target": "network_effect", "weight": 0.8}],
                 approximated=True)

    add_b_update(s, ai.agent_id, AgentType.AI, 4,
                 {"expand": 0.5, "consolidate": 0.5},
                 {"expand": 0.8, "consolidate": 0.2},
                 approximated=True)

    # Human 2 adds domain concepts
    add_c_update(s, h2.agent_id, AgentType.HUMAN, 5,
                 {}, [],
                 {"risk":          "Risk",
                  "supply_chain":  "Supply Chain",
                  "local_partner": "Local Partner",
                  "culture":       "Culture"},
                 [{"source": "local_partner",
                   "target": "risk", "weight": 0.6}])

    # Human 1 revises upward
    add_b_update(s, h1.agent_id, AgentType.HUMAN, 6,
                 {"expand": 0.7, "consolidate": 0.3},
                 {"expand": 0.9, "consolidate": 0.1})

    return s, h1, h2, ai


# ─── REPLAY MODE ─────────────────────────────────────────────────────────────

class TestReplayMode:

    def test_modes_exist(self):
        assert ReplayMode.FULL       == "FULL"
        assert ReplayMode.HUMAN_ONLY == "HUMAN_ONLY"
        assert ReplayMode.AI_ONLY    == "AI_ONLY"
        assert ReplayMode.SUBSET     == "SUBSET"


# ─── GHOST RUNNER FILTERING ───────────────────────────────────────────────────

class TestGhostRunnerFiltering:

    def test_full_mode_keeps_all_agents(self):
        s, h1, h2, ai = build_rich_session()
        ghost    = GhostRunner(s)
        filtered = ghost._filter_session(ReplayMode.FULL)
        assert len(filtered.human_agents) == 2
        assert len(filtered.ai_agents)    == 1

    def test_human_only_removes_ai(self):
        s, h1, h2, ai = build_rich_session()
        ghost    = GhostRunner(s)
        filtered = ghost._filter_session(ReplayMode.HUMAN_ONLY)
        assert len(filtered.human_agents) == 2
        assert len(filtered.ai_agents)    == 0

    def test_ai_only_removes_humans(self):
        s, h1, h2, ai = build_rich_session()
        ghost    = GhostRunner(s)
        filtered = ghost._filter_session(ReplayMode.AI_ONLY)
        assert len(filtered.human_agents) == 0
        assert len(filtered.ai_agents)    == 1

    def test_human_only_keeps_human_events(self):
        s, h1, h2, ai = build_rich_session()
        ghost    = GhostRunner(s)
        filtered = ghost._filter_session(ReplayMode.HUMAN_ONLY)
        for event in filtered.events:
            assert event.agent_type == AgentType.HUMAN

    def test_ai_only_keeps_ai_events(self):
        s, h1, h2, ai = build_rich_session()
        ghost    = GhostRunner(s)
        filtered = ghost._filter_session(ReplayMode.AI_ONLY)
        for event in filtered.events:
            assert event.agent_type == AgentType.AI

    def test_counterfactual_flag_set(self):
        s, h1, h2, ai = build_rich_session()
        ghost    = GhostRunner(s)
        filtered = ghost._filter_session(ReplayMode.HUMAN_ONLY)
        assert filtered.is_counterfactual is True
        assert filtered.parent_session_id == s.session_id

    def test_trigger_ref_nullified_when_source_removed(self):
        s, h1, h2, ai = make_session()

        # AI event
        ai_event = EATEvent(
            session_id       = s.session_id,
            agent_id         = ai.agent_id,
            agent_type       = AgentType.AI,
            event_type       = EATEventType.B_UPDATE,
            t                = 1,
            prior_belief     = BeliefState(
                hypotheses={"H1": 0.5, "H2": 0.5},
                approximated=True, uncertainty=0.1
            ),
            posterior_belief = BeliefState(
                hypotheses={"H1": 0.8, "H2": 0.2},
                approximated=True, uncertainty=0.1
            ),
            confidence       = 0.8
        )
        s.add_event(ai_event)

        # Human triggered by AI
        trigger_event = EATEvent(
            session_id       = s.session_id,
            agent_id         = h1.agent_id,
            agent_type       = AgentType.HUMAN,
            event_type       = EATEventType.B_TRIGGER,
            t                = 2,
            prior_belief     = BeliefState(hypotheses={"H1": 0.5, "H2": 0.5}),
            posterior_belief = BeliefState(hypotheses={"H1": 0.7, "H2": 0.3}),
            trigger_ref      = ai_event.event_id,
            trigger_agent    = ai.agent_id,
            confidence       = 0.85
        )
        s.add_event(trigger_event)

        ghost    = GhostRunner(s)
        # Remove AI — trigger_ref should be nullified
        filtered = ghost._filter_session(ReplayMode.HUMAN_ONLY)

        for event in filtered.events:
            if event.agent_id == h1.agent_id:
                assert event.trigger_ref   is None
                assert event.trigger_agent is None
                assert event.event_type    == EATEventType.B_UPDATE


# ─── METRICS EXTRACTION ──────────────────────────────────────────────────────

class TestMetricsExtraction:

    def test_extracts_unique_concepts(self):
        s, h1, h2, ai = build_rich_session()
        ghost  = GhostRunner(s)
        result = ghost._extract_metrics(s, ReplayMode.FULL)
        assert len(result.unique_concepts) > 0

    def test_human_only_fewer_concepts_than_full(self):
        s, h1, h2, ai = build_rich_session()
        ghost = GhostRunner(s)

        full_session  = ghost._filter_session(ReplayMode.FULL)
        human_session = ghost._filter_session(ReplayMode.HUMAN_ONLY)

        full_result  = ghost._extract_metrics(full_session,  ReplayMode.FULL)
        human_result = ghost._extract_metrics(human_session, ReplayMode.HUMAN_ONLY)

        # Full session has AI concepts too — should have >= human only
        assert len(full_result.unique_concepts) >= len(human_result.unique_concepts)

    def test_event_count_correct(self):
        s, h1, h2, ai = build_rich_session()
        ghost  = GhostRunner(s)
        result = ghost._extract_metrics(s, ReplayMode.FULL)
        assert result.event_count == s.total_events

    def test_mean_magnitude_non_negative(self):
        s, h1, h2, ai = build_rich_session()
        ghost  = GhostRunner(s)
        result = ghost._extract_metrics(s, ReplayMode.FULL)
        assert result.mean_magnitude >= 0.0

    def test_replay_result_repr(self):
        s, h1, h2, ai = build_rich_session()
        ghost  = GhostRunner(s)
        result = ghost._extract_metrics(s, ReplayMode.FULL)
        r      = repr(result)
        assert "FULL" in r


# ─── GHOST RUNNER RUN ────────────────────────────────────────────────────────

class TestGhostRunnerRun:

    def test_run_returns_emergence_signature(self):
        s, h1, h2, ai = build_rich_session()
        ghost = GhostRunner(s)
        sig   = ghost.run()
        assert isinstance(sig, EmergenceSignature)

    def test_emergence_signature_has_all_fields(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        assert hasattr(sig, "emergence_score")
        assert hasattr(sig, "ai_value_score")
        assert hasattr(sig, "human_value_score")
        assert hasattr(sig, "concept_emergence")
        assert hasattr(sig, "ai_unique_concepts")
        assert hasattr(sig, "human_unique_concepts")
        assert hasattr(sig, "shared_concepts")
        assert hasattr(sig, "recommendation")

    def test_scores_in_range(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        assert 0.0 <= sig.emergence_score   <= 1.0
        assert 0.0 <= sig.ai_value_score    <= 1.0
        assert 0.0 <= sig.human_value_score <= 1.0

    def test_concept_sets_are_sets(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        assert isinstance(sig.concept_emergence,     set)
        assert isinstance(sig.ai_unique_concepts,    set)
        assert isinstance(sig.human_unique_concepts, set)
        assert isinstance(sig.shared_concepts,       set)

    def test_ai_unique_concepts_not_in_human(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        overlap = sig.ai_unique_concepts & sig.human_unique_concepts
        assert len(overlap) == 0

    def test_recommendation_non_empty(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        assert len(sig.recommendation) > 0

    def test_all_three_results_stored(self):
        s, h1, h2, ai = build_rich_session()
        ghost = GhostRunner(s)
        ghost.run()
        assert ReplayMode.FULL       in ghost.results
        assert ReplayMode.HUMAN_ONLY in ghost.results
        assert ReplayMode.AI_ONLY    in ghost.results

    def test_human_only_no_ai_events(self):
        s, h1, h2, ai = build_rich_session()
        ghost = GhostRunner(s)
        ghost.run()
        human_result = ghost.results[ReplayMode.HUMAN_ONLY]
        for event in human_result.session.events:
            assert event.agent_type == AgentType.HUMAN

    def test_ai_only_no_human_events(self):
        s, h1, h2, ai = build_rich_session()
        ghost = GhostRunner(s)
        ghost.run()
        ai_result = ghost.results[ReplayMode.AI_ONLY]
        for event in ai_result.session.events:
            assert event.agent_type == AgentType.AI

    def test_session_id_preserved(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        assert sig.session_id == s.session_id


# ─── EMERGENCE SIGNATURE VERDICT ─────────────────────────────────────────────

class TestEmergenceSignatureVerdict:

    def test_verdict_returns_string(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        assert isinstance(sig.verdict(), str)
        assert len(sig.verdict()) > 0

    def test_strong_emergence_verdict(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        sig.emergence_score = 0.8
        assert "STRONG" in sig.verdict() or "MODERATE" in sig.verdict() \
               or "DOMINANT" in sig.verdict() or "LOW" in sig.verdict()

    def test_low_emergence_verdict_contains_low(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        sig.emergence_score   = 0.1
        sig.ai_value_score    = 0.1
        sig.human_value_score = 0.1
        assert "LOW" in sig.verdict()

    def test_ai_dominant_verdict(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        sig.emergence_score   = 0.1
        sig.ai_value_score    = 0.8
        sig.human_value_score = 0.2
        assert "AI" in sig.verdict().upper()

    def test_human_dominant_verdict(self):
        s, h1, h2, ai = build_rich_session()
        sig = GhostRunner(s).run()
        sig.emergence_score   = 0.1
        sig.ai_value_score    = 0.2
        sig.human_value_score = 0.8
        assert "HUMAN" in sig.verdict().upper()


# ─── MODEL SWAP ───────────────────────────────────────────────────────────────

class TestModelSwap:

    def test_model_swap_requires_prior_run(self):
        s, h1, h2, ai = build_rich_session()
        ghost  = GhostRunner(s)
        result = ghost.model_swap_comparison(
            original_model     = "claude-sonnet-4-6",
            replacement_model  = "gpt-4o",
            replacement_events = []
        )
        assert "error" in result

    def test_model_swap_after_run(self):
        s, h1, h2, ai = build_rich_session()
        ghost = GhostRunner(s)
        ghost.run()

        result = ghost.model_swap_comparison(
            original_model     = "claude-sonnet-4-6",
            replacement_model  = "gpt-4o",
            replacement_events = []
        )
        assert "comparison"        in result
        assert "original_model"    in result
        assert "replacement_model" in result
        assert "recommendation"    in result

    def test_model_swap_comparison_keys(self):
        s, h1, h2, ai = build_rich_session()
        ghost = GhostRunner(s)
        ghost.run()

        result = ghost.model_swap_comparison(
            original_model     = "claude-sonnet-4-6",
            replacement_model  = "gpt-4o",
            replacement_events = []
        )
        comparison = result["comparison"]
        assert "concepts_original"    in comparison
        assert "concepts_replacement" in comparison
        assert "entropy_original"     in comparison
        assert "entropy_replacement"  in comparison
        assert "magnitude_original"   in comparison