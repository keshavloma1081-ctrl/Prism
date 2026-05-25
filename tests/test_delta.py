"""
PRISM — tests/test_delta.py
Delta Magnitude + Scoring Engine Tests

Run: pytest tests/test_delta.py -v
"""

import pytest
import numpy as np
from core.eat.models import (
    BeliefState, ConceptualGraph,
    EATEvent, EATEventType, AgentType
)
from core.eat.delta import (
    kl_divergence, fractional_graph_edit,
    node_novelty_score, compute_delta_magnitude,
    belief_entropy, ensemble_entropy, rsa_similarity
)


# ─── KL DIVERGENCE ────────────────────────────────────────────────────────────

class TestKLDivergence:

    def test_identical_distributions_zero(self):
        p = {"H1": 0.7, "H2": 0.3}
        assert kl_divergence(p, p) < 1e-6

    def test_divergent_distributions_positive(self):
        prior     = {"H1": 0.9, "H2": 0.1}
        posterior = {"H1": 0.1, "H2": 0.9}
        assert kl_divergence(prior, posterior) > 0

    def test_asymmetry(self):
        # Genuinely asymmetric — different hypothesis spaces
        p = {"H1": 0.9, "H2": 0.1}
        q = {"H1": 0.4, "H2": 0.4, "H3": 0.2}
        assert kl_divergence(p, q) != kl_divergence(q, p)

    def test_disjoint_hypothesis_spaces(self):
        prior     = {"H1": 1.0}
        posterior = {"H2": 1.0}
        assert kl_divergence(prior, posterior) > 0

    def test_non_negative(self):
        for _ in range(20):
            vals  = np.random.dirichlet([1, 1, 1])
            vals2 = np.random.dirichlet([1, 1, 1])
            p = {"H1": vals[0],  "H2": vals[1],  "H3": vals[2]}
            q = {"H1": vals2[0], "H2": vals2[1], "H3": vals2[2]}
            assert kl_divergence(p, q) >= 0

    def test_single_hypothesis(self):
        assert kl_divergence({"H1": 1.0}, {"H1": 0.9}) >= 0

    def test_large_divergence_larger_than_small(self):
        small = kl_divergence({"H1": 0.6, "H2": 0.4}, {"H1": 0.5, "H2": 0.5})
        large = kl_divergence({"H1": 0.99, "H2": 0.01}, {"H1": 0.01, "H2": 0.99})
        assert large > small


# ─── GRAPH EDIT DISTANCE ──────────────────────────────────────────────────────

class TestFractionalGraphEdit:

    def _graph(self, edges):
        nodes = {}
        for e in edges:
            nodes[e["source"]] = e["source"]
            nodes[e["target"]] = e["target"]
        return ConceptualGraph(nodes=nodes, edges=edges)

    def test_identical_graphs_zero(self):
        g = self._graph([{"source": "A", "target": "B", "weight": 0.5}])
        assert fractional_graph_edit(g, g) == 0.0

    def test_empty_prior_to_nonempty(self):
        prior     = ConceptualGraph(nodes={}, edges=[])
        posterior = self._graph([{"source": "A", "target": "B", "weight": 0.5}])
        assert fractional_graph_edit(prior, posterior) == 1.0

    def test_nonempty_to_empty(self):
        prior     = self._graph([{"source": "A", "target": "B", "weight": 0.5}])
        posterior = ConceptualGraph(nodes={}, edges=[])
        assert fractional_graph_edit(prior, posterior) == 1.0

    def test_partial_change(self):
        prior = self._graph([
            {"source": "A", "target": "B", "weight": 0.5},
            {"source": "B", "target": "C", "weight": 0.5},
        ])
        posterior = self._graph([
            {"source": "A", "target": "B", "weight": 0.5},
            {"source": "C", "target": "D", "weight": 0.5},
        ])
        score = fractional_graph_edit(prior, posterior)
        assert 0 < score <= 1.0

    def test_non_negative(self):
        g1 = self._graph([{"source": "X", "target": "Y", "weight": 0.8}])
        g2 = self._graph([{"source": "Y", "target": "Z", "weight": 0.6}])
        assert fractional_graph_edit(g1, g2) >= 0

    def test_expansion_above_one(self):
        # Adding more edges than existed before → ratio > 1
        prior = self._graph([
            {"source": "A", "target": "B", "weight": 0.5}
        ])
        posterior = self._graph([
            {"source": "A", "target": "B", "weight": 0.5},
            {"source": "C", "target": "D", "weight": 0.5},
            {"source": "E", "target": "F", "weight": 0.5},
        ])
        score = fractional_graph_edit(prior, posterior)
        assert score > 0


# ─── NODE NOVELTY ─────────────────────────────────────────────────────────────

class TestNodeNoveltyScore:

    def test_all_new_nodes(self):
        prior     = ConceptualGraph(nodes={}, edges=[])
        posterior = ConceptualGraph(
            nodes={"A": "Concept A", "B": "Concept B"}, edges=[]
        )
        assert node_novelty_score(prior, posterior) == 1.0

    def test_no_new_nodes(self):
        g = ConceptualGraph(nodes={"A": "Concept A"}, edges=[])
        assert node_novelty_score(g, g) == 0.0

    def test_partial_novelty(self):
        prior     = ConceptualGraph(nodes={"A": "Concept A"}, edges=[])
        posterior = ConceptualGraph(
            nodes={"A": "Concept A", "B": "Concept B"}, edges=[]
        )
        assert node_novelty_score(prior, posterior) == 0.5

    def test_empty_posterior(self):
        prior     = ConceptualGraph(nodes={"A": "A"}, edges=[])
        posterior = ConceptualGraph(nodes={}, edges=[])
        assert node_novelty_score(prior, posterior) == 0.0

    def test_score_in_range(self):
        prior     = ConceptualGraph(nodes={"A": "A", "B": "B"}, edges=[])
        posterior = ConceptualGraph(
            nodes={"A": "A", "B": "B", "C": "C", "D": "D"}, edges=[]
        )
        score = node_novelty_score(prior, posterior)
        assert 0.0 <= score <= 1.0


# ─── COMPUTE DELTA MAGNITUDE ──────────────────────────────────────────────────

class TestComputeDeltaMagnitude:

    def _b_event(self, prior_h, post_h):
        return EATEvent(
            session_id       = "S-test",
            agent_id         = "H-001",
            agent_type       = AgentType.HUMAN,
            event_type       = EATEventType.B_UPDATE,
            t                = 1,
            prior_belief     = BeliefState(hypotheses=prior_h),
            posterior_belief = BeliefState(hypotheses=post_h),
            confidence       = 0.9
        )

    def _c_event(self, prior_edges, post_nodes, post_edges):
        prior_nodes = {}
        for e in prior_edges:
            prior_nodes[e["source"]] = e["source"]
            prior_nodes[e["target"]] = e["target"]
        return EATEvent(
            session_id      = "S-test",
            agent_id        = "H-001",
            agent_type      = AgentType.HUMAN,
            event_type      = EATEventType.C_UPDATE,
            t               = 1,
            prior_graph     = ConceptualGraph(
                nodes=prior_nodes, edges=prior_edges
            ),
            posterior_graph = ConceptualGraph(
                nodes=post_nodes, edges=post_edges
            ),
            confidence      = 0.9
        )

    def test_b_update_no_change(self):
        e = self._b_event({"H1": 0.5, "H2": 0.5}, {"H1": 0.5, "H2": 0.5})
        assert compute_delta_magnitude(e) < 1e-6

    def test_b_update_large_change(self):
        e = self._b_event(
            {"H1": 0.95, "H2": 0.05},
            {"H1": 0.05, "H2": 0.95}
        )
        assert compute_delta_magnitude(e) > 1.0

    def test_c_update_no_change(self):
        edges = [{"source": "A", "target": "B", "weight": 0.5}]
        nodes = {"A": "A", "B": "B"}
        e = self._c_event(edges, nodes, edges)
        assert compute_delta_magnitude(e) == 0.0

    def test_c_update_complete_change(self):
        prior_edges = [{"source": "A", "target": "B", "weight": 0.5}]
        post_nodes  = {"C": "C", "D": "D"}
        post_edges  = [{"source": "C", "target": "D", "weight": 0.5}]
        e = self._c_event(prior_edges, post_nodes, post_edges)
        assert compute_delta_magnitude(e) > 0

    def test_non_negative_magnitude(self):
        e = self._b_event({"H1": 0.3, "H2": 0.7}, {"H1": 0.8, "H2": 0.2})
        assert compute_delta_magnitude(e) >= 0

    def test_larger_change_larger_magnitude(self):
        small = compute_delta_magnitude(
            self._b_event({"H1": 0.55, "H2": 0.45}, {"H1": 0.5, "H2": 0.5})
        )
        large = compute_delta_magnitude(
            self._b_event({"H1": 0.95, "H2": 0.05}, {"H1": 0.1, "H2": 0.9})
        )
        assert large > small


# ─── BELIEF ENTROPY ───────────────────────────────────────────────────────────

class TestBeliefEntropy:

    def test_uniform_higher_entropy_than_skewed(self):
        uniform = BeliefState(hypotheses={"H1": 0.5,  "H2": 0.5})
        skewed  = BeliefState(hypotheses={"H1": 0.99, "H2": 0.01})
        assert belief_entropy(uniform) > belief_entropy(skewed)

    def test_certain_belief_low_entropy(self):
        certain = BeliefState(hypotheses={"H1": 1.0})
        assert belief_entropy(certain) < 0.01

    def test_entropy_non_negative(self):
        b = BeliefState(hypotheses={"H1": 0.3, "H2": 0.4, "H3": 0.3})
        assert belief_entropy(b) >= 0

    def test_more_hypotheses_higher_max_entropy(self):
        two   = BeliefState(hypotheses={"H1": 0.5, "H2": 0.5})
        four  = BeliefState(hypotheses={"H1": 0.25, "H2": 0.25,
                                         "H3": 0.25, "H4": 0.25})
        assert belief_entropy(four) > belief_entropy(two)

    def test_ensemble_entropy_between_extremes(self):
        high = BeliefState(hypotheses={"H1": 0.5,  "H2": 0.5})
        low  = BeliefState(hypotheses={"H1": 0.99, "H2": 0.01})
        ens  = ensemble_entropy([high, low])
        assert belief_entropy(low) < ens < belief_entropy(high)

    def test_empty_ensemble_zero(self):
        assert ensemble_entropy([]) == 0.0

    def test_single_belief_ensemble(self):
        b   = BeliefState(hypotheses={"H1": 0.7, "H2": 0.3})
        ens = ensemble_entropy([b])
        assert abs(ens - belief_entropy(b)) < 1e-6


# ─── RSA SIMILARITY ───────────────────────────────────────────────────────────

class TestRSASimilarity:

    def _graph(self, edges, extra_nodes=None):
        nodes = dict(extra_nodes or {})
        for e in edges:
            nodes[e["source"]] = e["source"]
            nodes[e["target"]] = e["target"]
        return ConceptualGraph(nodes=nodes, edges=edges)

    def test_identical_graphs_high_similarity(self):
        edges = [
            {"source": "A", "target": "B", "weight": 0.8},
            {"source": "B", "target": "C", "weight": 0.4},
        ]
        g     = self._graph(edges)
        score = rsa_similarity(g, g, shared_concepts=["A", "B", "C"])
        assert score > 0.9

    def test_no_shared_concepts_zero(self):
        g1 = self._graph([{"source": "A", "target": "B", "weight": 0.5}])
        g2 = self._graph([{"source": "C", "target": "D", "weight": 0.5}])
        assert rsa_similarity(g1, g2) == 0.0

    def test_similarity_in_range(self):
        g1 = self._graph([
            {"source": "X", "target": "Y", "weight": 0.9},
            {"source": "Y", "target": "Z", "weight": 0.2},
        ])
        g2 = self._graph([
            {"source": "X", "target": "Y", "weight": 0.3},
            {"source": "Y", "target": "Z", "weight": 0.8},
        ])
        score = rsa_similarity(g1, g2, shared_concepts=["X", "Y", "Z"])
        assert -1.0 <= score <= 1.0

    def test_single_shared_concept_zero(self):
        g1 = self._graph([{"source": "A", "target": "B", "weight": 0.5}])
        g2 = self._graph([{"source": "A", "target": "C", "weight": 0.5}])
        assert rsa_similarity(g1, g2, shared_concepts=["A"]) == 0.0

    def test_opposite_structures_negative_or_zero(self):
        g1 = self._graph([
            {"source": "A", "target": "B", "weight": 1.0},
            {"source": "B", "target": "C", "weight": 0.0},
        ])
        g2 = self._graph([
            {"source": "A", "target": "B", "weight": 0.0},
            {"source": "B", "target": "C", "weight": 1.0},
        ])
        score = rsa_similarity(g1, g2, shared_concepts=["A", "B", "C"])
        assert score <= 0.0