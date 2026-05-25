"""
PRISM — tests/test_eat_models.py
EAT Schema Validation Tests

Tests every model, every validator, every edge case.
Run: pytest tests/test_eat_models.py -v
"""

import pytest
from datetime import datetime
from core.eat.models import (
    AgentType, EATEventType, AccessLevel, WorkflowStatus,
    HumanAgent, AIAgent, BeliefState, ConceptualGraph,
    EATEvent, WorkflowSession
)


# ─── BELIEF STATE ─────────────────────────────────────────────────────────────

class TestBeliefState:

    def test_valid_belief_state(self):
        b = BeliefState(hypotheses={"H1": 0.7, "H2": 0.3})
        assert b.hypotheses["H1"] == 0.7
        assert b.hypotheses["H2"] == 0.3

    def test_belief_confidence_out_of_range_raises(self):
        with pytest.raises(Exception):
            BeliefState(hypotheses={"H1": 1.5})

    def test_belief_negative_confidence_raises(self):
        with pytest.raises(Exception):
            BeliefState(hypotheses={"H1": -0.1})

    def test_belief_zero_confidence_valid(self):
        b = BeliefState(hypotheses={"H1": 0.0, "H2": 1.0})
        assert b.hypotheses["H1"] == 0.0

    def test_approximated_belief(self):
        b = BeliefState(
            hypotheses={"H1": 0.6},
            approximated=True,
            uncertainty=0.15
        )
        assert b.approximated is True
        assert b.uncertainty == 0.15

    def test_empty_hypotheses_valid(self):
        b = BeliefState(hypotheses={})
        assert b.hypotheses == {}


# ─── CONCEPTUAL GRAPH ─────────────────────────────────────────────────────────

class TestConceptualGraph:

    def test_valid_graph(self):
        g = ConceptualGraph(
            nodes={"risk": "Risk Factor", "revenue": "Revenue"},
            edges=[{"source": "risk", "target": "revenue", "weight": 0.7}]
        )
        assert "risk" in g.nodes
        assert len(g.edges) == 1

    def test_empty_graph(self):
        g = ConceptualGraph(nodes={}, edges=[])
        assert g.nodes == {}
        assert g.edges == []

    def test_approximated_graph(self):
        g = ConceptualGraph(
            nodes={"c1": "Concept 1"},
            edges=[],
            approximated=True,
            uncertainty=0.2
        )
        assert g.approximated is True


# ─── HUMAN AGENT ──────────────────────────────────────────────────────────────

class TestHumanAgent:

    def test_auto_agent_id(self):
        h = HumanAgent(name="Alice", role="analyst", client_id="acme")
        assert h.agent_id.startswith("H-")
        assert h.agent_type == AgentType.HUMAN

    def test_human_agent_fields(self):
        h = HumanAgent(name="Bob", role="expert", client_id="corp")
        assert h.name == "Bob"
        assert h.role == "expert"
        assert h.client_id == "corp"

    def test_unique_agent_ids(self):
        h1 = HumanAgent(name="A", role="r", client_id="c")
        h2 = HumanAgent(name="B", role="r", client_id="c")
        assert h1.agent_id != h2.agent_id


# ─── AI AGENT ─────────────────────────────────────────────────────────────────

class TestAIAgent:

    def test_auto_agent_id(self):
        a = AIAgent(
            model_name="claude-sonnet-4-6",
            provider="anthropic",
            client_id="acme"
        )
        assert a.agent_id.startswith("A-")
        assert a.agent_type == AgentType.AI

    def test_default_access_level(self):
        a = AIAgent(model_name="gpt-4o", provider="openai", client_id="acme")
        assert a.access_level == AccessLevel.GREY_BOX

    def test_white_box_access(self):
        a = AIAgent(
            model_name="claude-sonnet-4-6",
            provider="anthropic",
            client_id="acme",
            access_level=AccessLevel.WHITE_BOX
        )
        assert a.access_level == AccessLevel.WHITE_BOX


# ─── EAT EVENT ────────────────────────────────────────────────────────────────

class TestEATEvent:

    def _make_belief(self, h1=0.7, h2=0.3, approximated=False):
        return BeliefState(
            hypotheses={"H1": h1, "H2": h2},
            approximated=approximated,
            uncertainty=0.1 if approximated else 0.0
        )

    def _make_graph(self, nodes=None):
        nodes = nodes or {"risk": "Risk", "revenue": "Revenue"}
        return ConceptualGraph(
            nodes=nodes,
            edges=[{"source": k, "target": v, "weight": 0.5}
                   for k, v in list(nodes.items())[:1]]
            if len(nodes) > 1 else []
        )

    def test_valid_b_update_event(self):
        event = EATEvent(
            session_id       = "S-test",
            agent_id         = "H-001",
            agent_type       = AgentType.HUMAN,
            event_type       = EATEventType.B_UPDATE,
            t                = 1,
            prior_belief     = self._make_belief(0.5, 0.5),
            posterior_belief = self._make_belief(0.8, 0.2),
            confidence       = 0.9
        )
        assert event.event_id.startswith("EAT-")
        assert event.event_type == EATEventType.B_UPDATE

    def test_b_update_missing_belief_raises(self):
        with pytest.raises(Exception):
            EATEvent(
                session_id = "S-test",
                agent_id   = "H-001",
                agent_type = AgentType.HUMAN,
                event_type = EATEventType.B_UPDATE,
                t          = 1
                # Missing prior_belief and posterior_belief
            )

    def test_valid_c_update_event(self):
        event = EATEvent(
            session_id    = "S-test",
            agent_id      = "H-001",
            agent_type    = AgentType.HUMAN,
            event_type    = EATEventType.C_UPDATE,
            t             = 1,
            prior_graph   = ConceptualGraph(nodes={}, edges=[]),
            posterior_graph = self._make_graph(),
            confidence    = 0.9
        )
        assert event.event_type == EATEventType.C_UPDATE

    def test_c_update_missing_graph_raises(self):
        with pytest.raises(Exception):
            EATEvent(
                session_id = "S-test",
                agent_id   = "H-001",
                agent_type = AgentType.HUMAN,
                event_type = EATEventType.C_UPDATE,
                t          = 1
                # Missing prior_graph and posterior_graph
            )

    def test_b_trigger_requires_trigger_ref(self):
        with pytest.raises(Exception):
            EATEvent(
                session_id       = "S-test",
                agent_id         = "H-001",
                agent_type       = AgentType.HUMAN,
                event_type       = EATEventType.B_TRIGGER,
                t                = 2,
                prior_belief     = self._make_belief(0.5, 0.5),
                posterior_belief = self._make_belief(0.8, 0.2)
                # Missing trigger_ref
            )

    def test_b_trigger_with_trigger_ref(self):
        event = EATEvent(
            session_id       = "S-test",
            agent_id         = "H-001",
            agent_type       = AgentType.HUMAN,
            event_type       = EATEventType.B_TRIGGER,
            t                = 2,
            prior_belief     = self._make_belief(0.5, 0.5),
            posterior_belief = self._make_belief(0.8, 0.2),
            trigger_ref      = "EAT-abc123",
            trigger_agent    = "A-001",
            confidence       = 0.85
        )
        assert event.trigger_ref == "EAT-abc123"

    def test_unique_event_ids(self):
        kwargs = dict(
            session_id       = "S-test",
            agent_id         = "H-001",
            agent_type       = AgentType.HUMAN,
            event_type       = EATEventType.B_UPDATE,
            t                = 1,
            prior_belief     = self._make_belief(),
            posterior_belief = self._make_belief(0.3, 0.7)
        )
        e1 = EATEvent(**kwargs)
        e2 = EATEvent(**kwargs)
        assert e1.event_id != e2.event_id

    def test_verdict_scores_initially_none(self):
        event = EATEvent(
            session_id       = "S-test",
            agent_id         = "H-001",
            agent_type       = AgentType.HUMAN,
            event_type       = EATEventType.B_UPDATE,
            t                = 1,
            prior_belief     = self._make_belief(),
            posterior_belief = self._make_belief(0.3, 0.7)
        )
        assert event.groundedness is None
        assert event.novelty_delta is None
        assert event.calibration_score is None
        assert event.influence_survival is None


# ─── WORKFLOW SESSION ─────────────────────────────────────────────────────────

class TestWorkflowSession:

    def _make_session(self):
        return WorkflowSession(
            client_id="acme",
            workflow_id="wf-001"
        )

    def _make_b_update(self, session_id, agent_id, agent_type, t):
        return EATEvent(
            session_id       = session_id,
            agent_id         = agent_id,
            agent_type       = agent_type,
            event_type       = EATEventType.B_UPDATE,
            t                = t,
            prior_belief     = BeliefState(hypotheses={"H1": 0.5, "H2": 0.5}),
            posterior_belief = BeliefState(hypotheses={"H1": 0.7, "H2": 0.3}),
            confidence       = 0.9
        )

    def test_session_creation(self):
        s = self._make_session()
        assert s.session_id.startswith("S-")
        assert s.status == WorkflowStatus.ACTIVE
        assert s.total_events == 0

    def test_add_event(self):
        s = self._make_session()
        h = HumanAgent(name="Alice", role="analyst", client_id="acme")
        s.human_agents[h.agent_id] = h
        event = self._make_b_update(s.session_id, h.agent_id, AgentType.HUMAN, 1)
        s.add_event(event)
        assert s.total_events == 1

    def test_get_events_by_agent(self):
        s = self._make_session()
        h1 = HumanAgent(name="Alice", role="analyst", client_id="acme")
        h2 = HumanAgent(name="Bob",   role="expert",  client_id="acme")
        s.human_agents[h1.agent_id] = h1
        s.human_agents[h2.agent_id] = h2

        s.add_event(self._make_b_update(s.session_id, h1.agent_id, AgentType.HUMAN, 1))
        s.add_event(self._make_b_update(s.session_id, h1.agent_id, AgentType.HUMAN, 2))
        s.add_event(self._make_b_update(s.session_id, h2.agent_id, AgentType.HUMAN, 3))

        h1_events = s.get_events_by_agent(h1.agent_id)
        assert len(h1_events) == 2

    def test_get_human_events(self):
        s = self._make_session()
        h = HumanAgent(name="Alice", role="analyst", client_id="acme")
        a = AIAgent(model_name="claude-sonnet-4-6", provider="anthropic", client_id="acme")
        s.human_agents[h.agent_id] = h
        s.ai_agents[a.agent_id]    = a

        s.add_event(self._make_b_update(s.session_id, h.agent_id, AgentType.HUMAN, 1))
        s.add_event(self._make_b_update(s.session_id, a.agent_id, AgentType.AI,    2))

        assert len(s.get_human_events()) == 1
        assert len(s.get_ai_events())    == 1

    def test_all_agents(self):
        s = self._make_session()
        h = HumanAgent(name="Alice", role="analyst", client_id="acme")
        a = AIAgent(model_name="claude-sonnet-4-6", provider="anthropic", client_id="acme")
        s.human_agents[h.agent_id] = h
        s.ai_agents[a.agent_id]    = a
        assert len(s.all_agents) == 2

    def test_unique_session_ids(self):
        s1 = self._make_session()
        s2 = self._make_session()
        assert s1.session_id != s2.session_id

    def test_counterfactual_flag(self):
        s = WorkflowSession(
            client_id         = "acme",
            workflow_id       = "wf-001",
            is_counterfactual = True,
            parent_session_id = "S-parent"
        )
        assert s.is_counterfactual is True
        assert s.parent_session_id == "S-parent"