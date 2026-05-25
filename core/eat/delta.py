"""
PRISM — core/eat/delta.py
Delta Magnitude Computation Engine

Measures how much an epistemic act changed an agent's state.
Used by VERDICT for scoring and DECAY for degradation detection.

Four formulas — one per EAT event type:
  B_UPDATE  / B_TRIGGER → KL Divergence
  C_UPDATE  / C_TRIGGER → Fractional Graph Edit Distance
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from core.eat.models import (
    EATEvent, EATEventType,
    BeliefState, ConceptualGraph
)


# ─── KL DIVERGENCE ────────────────────────────────────────────────────────────

def kl_divergence(
    prior: Dict[str, float],
    posterior: Dict[str, float],
    epsilon: float = 1e-9
) -> float:
    """
    KL(posterior || prior)

    Measures information gained — how surprised the posterior
    would be relative to the prior. Asymmetric by design.

    epsilon: smoothing to handle zero-probability hypotheses.
    Returns 0.0 if distributions are identical.
    """
    # Align hypothesis spaces
    all_keys = set(prior.keys()) | set(posterior.keys())

    p = np.array([prior.get(k, epsilon)     for k in all_keys], dtype=np.float64)
    q = np.array([posterior.get(k, epsilon) for k in all_keys], dtype=np.float64)

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Smooth zeros
    p = np.where(p < epsilon, epsilon, p)
    q = np.where(q < epsilon, epsilon, q)

    # KL(q || p) — posterior relative to prior
    kl = np.sum(q * np.log(q / p))
    return float(np.clip(kl, 0.0, None))


# ─── GRAPH EDIT DISTANCE ──────────────────────────────────────────────────────

def fractional_graph_edit(
    prior_graph: ConceptualGraph,
    posterior_graph: ConceptualGraph
) -> float:
    """
    |ΔC| / |C_prior|

    Fractional change in conceptual graph structure.
    Normalised for agents with larger existing graphs.

    Returns value in [0.0, 1.0+]:
      0.0  → no change
      1.0  → complete replacement of prior graph
      >1.0 → more edges added than existed before (expansion)
    """
    def edge_set(graph: ConceptualGraph):
        return {
            (e["source"], e["target"])
            for e in graph.edges
            if "source" in e and "target" in e
        }

    prior_edges     = edge_set(prior_graph)
    posterior_edges = edge_set(posterior_graph)

    added   = posterior_edges - prior_edges
    removed = prior_edges - posterior_edges
    delta   = len(added) + len(removed)

    if len(prior_edges) == 0:
        # Prior graph was empty — any addition is maximal change
        return 1.0 if delta > 0 else 0.0

    return float(delta / len(prior_edges))


# ─── NODE NOVELTY ─────────────────────────────────────────────────────────────

def node_novelty_score(
    prior_graph: ConceptualGraph,
    posterior_graph: ConceptualGraph
) -> float:
    """
    Fraction of posterior nodes that are new — not present in prior.
    Used by VERDICT novelty_delta scoring.
    """
    prior_nodes     = set(prior_graph.nodes.keys())
    posterior_nodes = set(posterior_graph.nodes.keys())
    new_nodes       = posterior_nodes - prior_nodes

    if len(posterior_nodes) == 0:
        return 0.0

    return float(len(new_nodes) / len(posterior_nodes))


# ─── MAIN DELTA COMPUTER ──────────────────────────────────────────────────────

def compute_delta_magnitude(event: EATEvent) -> float:
    """
    Dispatch to the correct magnitude formula based on event type.

    B_UPDATE  → KL(posterior_belief || prior_belief)
    C_UPDATE  → fractional_graph_edit(prior_graph, posterior_graph)
    B_TRIGGER → KL on the TARGET agent's belief shift
    C_TRIGGER → fractional_graph_edit on the TARGET agent's graph shift

    Returns scalar magnitude ≥ 0.0
    """
    if event.event_type in (EATEventType.B_UPDATE, EATEventType.B_TRIGGER):
        if event.prior_belief is None or event.posterior_belief is None:
            return 0.0
        return kl_divergence(
            event.prior_belief.hypotheses,
            event.posterior_belief.hypotheses
        )

    if event.event_type in (EATEventType.C_UPDATE, EATEventType.C_TRIGGER):
        if event.prior_graph is None or event.posterior_graph is None:
            return 0.0
        return fractional_graph_edit(
            event.prior_graph,
            event.posterior_graph
        )

    return 0.0


# ─── BELIEF ENTROPY ───────────────────────────────────────────────────────────

def belief_entropy(belief: BeliefState) -> float:
    """
    Shannon entropy H(b) over a belief distribution.
    High entropy = high uncertainty across hypotheses.
    Used by DECAY to track epistemic engagement over time.
    """
    probs = np.array(list(belief.hypotheses.values()), dtype=np.float64)
    probs = probs / probs.sum()
    probs = np.where(probs < 1e-9, 1e-9, probs)
    return float(-np.sum(probs * np.log(probs)))


def ensemble_entropy(beliefs: List[BeliefState]) -> float:
    """
    Mean Shannon entropy across all agents in the ensemble.
    H_E(t) = (1/|agents|) Σ H(b_α(t))
    """
    if not beliefs:
        return 0.0
    return float(np.mean([belief_entropy(b) for b in beliefs]))


# ─── RSA SIMILARITY ───────────────────────────────────────────────────────────

def rsa_similarity(
    graph_a: ConceptualGraph,
    graph_b: ConceptualGraph,
    shared_concepts: Optional[List[str]] = None
) -> float:
    """
    Representational Similarity Analysis between two conceptual graphs.

    Compares second-order structure — not whether agents have
    the same concepts, but whether they represent concept
    relationships similarly.

    Returns Spearman correlation in [-1.0, 1.0]:
      1.0  → identical conceptual structure
      0.0  → no structural similarity
     -1.0  → perfectly opposing structure
    """
    from scipy.stats import spearmanr

    # Build shared concept vocabulary
    if shared_concepts is None:
        shared_concepts = list(
            set(graph_a.nodes.keys()) & set(graph_b.nodes.keys())
        )

    if len(shared_concepts) < 2:
        return 0.0

    def similarity_vector(graph: ConceptualGraph, concepts: List[str]) -> np.ndarray:
        """Pairwise edge weights between shared concepts."""
        edge_weights = {}
        for e in graph.edges:
            src, tgt = e.get("source"), e.get("target")
            w = float(e.get("weight", 1.0))
            if src in concepts and tgt in concepts:
                edge_weights[(src, tgt)] = w

        vec = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                src, tgt = concepts[i], concepts[j]
                vec.append(edge_weights.get((src, tgt),
                           edge_weights.get((tgt, src), 0.0)))
        return np.array(vec, dtype=np.float64)

    vec_a = similarity_vector(graph_a, shared_concepts)
    vec_b = similarity_vector(graph_b, shared_concepts)

    if vec_a.std() == 0 or vec_b.std() == 0:
        return 1.0 if np.array_equal(vec_a, vec_b) else 0.0

    corr, _ = spearmanr(vec_a, vec_b)
    return float(corr) if not np.isnan(corr) else 0.0