"""
PRISM — atlas/graph.py
Causal Discovery Fingerprint Engine

Every insight, decision, and discovery in a workflow
gets an ATLAS fingerprint — a directed acyclic graph
tracing the exact causal chain from first epistemic
event to final outcome.

Not "AI contributed 40%."
A timestamped, agent-attributed influence DAG showing:
  - Who introduced which concept
  - Which agent influenced which agent
  - The precise sequence that produced the discovery

This is the document that makes clients renew contracts.
"""

from __future__ import annotations
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field

from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession
)
from core.eat.delta import kl_divergence


# ─── FINGERPRINT NODE ─────────────────────────────────────────────────────────

@dataclass
class FingerprintNode:
    """
    A node in the ATLAS causal fingerprint graph.
    Represents a single epistemic event with its context.
    """
    event_id:    str
    agent_id:    str
    agent_type:  AgentType
    event_type:  EATEventType
    t:           int
    magnitude:   float
    label:       str
    concepts:    List[str]  = field(default_factory=list)
    metadata:    Dict       = field(default_factory=dict)


# ─── FINGERPRINT EDGE ─────────────────────────────────────────────────────────

@dataclass
class FingerprintEdge:
    """
    A directed edge in the ATLAS causal fingerprint graph.
    Represents a causal influence from one epistemic event to another.
    """
    source_event_id: str
    target_event_id: str
    influence_type:  str    # DIRECT | INDIRECT | CONCEPTUAL | BELIEF
    weight:          float  # Influence strength [0.0, 1.0]
    t_delta:         int    # Time steps between source and target
    label:           str


# ─── DISCOVERY NODE ───────────────────────────────────────────────────────────

@dataclass
class Discovery:
    """
    A detected novel outcome in the workflow.
    ATLAS traces the causal chain that produced it.
    """
    discovery_id:    str
    session_id:      str
    detected_at_t:   int
    description:     str
    novelty_score:   float
    origin_event_id: str          # First event in the causal chain
    apex_event_id:   str          # Event where discovery crystallised
    causal_chain:    List[str]    # Ordered list of event_ids
    agent_contributions: Dict[str, float]  # agent_id → contribution weight
    timestamp:       datetime = field(default_factory=datetime.utcnow)


# ─── ATLAS GRAPH ENGINE ───────────────────────────────────────────────────────

class AtlasGraph:
    """
    Main ATLAS causal fingerprint engine.

    Builds and maintains a directed graph over all EAT events
    in a session, with edges representing causal influence.

    Provides:
      - build_fingerprint()  → construct full influence DAG
      - trace_discovery()    → trace causal chain to a specific outcome
      - agent_contribution() → quantify each agent's contribution
      - export_dict()        → serializable format for dashboard + chronicle
    """

    def __init__(self, session: WorkflowSession):
        self.session = session
        self.graph   = nx.DiGraph()
        self.nodes:  Dict[str, FingerprintNode] = {}
        self.edges:  List[FingerprintEdge]      = []
        self.discoveries: List[Discovery]       = []

    # ── BUILD ──────────────────────────────────────────────────────────────

    def build_fingerprint(self) -> nx.DiGraph:
        """
        Construct the full causal influence DAG from session events.

        Three types of edges:
          1. DIRECT   — explicit trigger_ref links (B/C TRIGGER events)
          2. TEMPORAL — sequential events from same agent
          3. CONCEPTUAL — events sharing concepts across agents
        """
        self.graph.clear()
        self.nodes.clear()
        self.edges.clear()

        # ── Add all events as nodes ────────────────────────────────────────
        for event in self.session.events:
            concepts = []
            if event.posterior_graph is not None:
                concepts = list(event.posterior_graph.nodes.keys())

            node = FingerprintNode(
                event_id   = event.event_id,
                agent_id   = event.agent_id,
                agent_type = event.agent_type,
                event_type = event.event_type,
                t          = event.t,
                magnitude  = event.delta_magnitude,
                label      = self._event_label(event),
                concepts   = concepts,
                metadata   = {
                    "groundedness":       event.groundedness,
                    "novelty_delta":      event.novelty_delta,
                    "influence_survival": event.influence_survival,
                }
            )
            self.nodes[event.event_id] = node
            self.graph.add_node(
                event.event_id,
                **{
                    "agent_id":   event.agent_id,
                    "agent_type": event.agent_type.value,
                    "t":          event.t,
                    "magnitude":  event.delta_magnitude,
                    "label":      node.label
                }
            )

        # ── Add direct causal edges (TRIGGER events) ───────────────────────
        for event in self.session.events:
            if (event.trigger_ref is not None
                    and event.trigger_ref in self.nodes):
                edge = FingerprintEdge(
                    source_event_id = event.trigger_ref,
                    target_event_id = event.event_id,
                    influence_type  = "DIRECT",
                    weight          = min(event.delta_magnitude, 1.0),
                    t_delta         = event.t - self.nodes[event.trigger_ref].t,
                    label           = "triggered"
                )
                self.edges.append(edge)
                self.graph.add_edge(
                    event.trigger_ref,
                    event.event_id,
                    weight          = edge.weight,
                    influence_type  = "DIRECT",
                    label           = "triggered"
                )

        # ── Add temporal edges (same agent, sequential events) ─────────────
        agent_event_map: Dict[str, List[EATEvent]] = {}
        for event in self.session.events:
            agent_event_map.setdefault(event.agent_id, []).append(event)

        for agent_id, agent_events in agent_event_map.items():
            sorted_events = sorted(agent_events, key=lambda e: e.t)
            for i in range(len(sorted_events) - 1):
                src = sorted_events[i].event_id
                tgt = sorted_events[i + 1].event_id
                if not self.graph.has_edge(src, tgt):
                    edge = FingerprintEdge(
                        source_event_id = src,
                        target_event_id = tgt,
                        influence_type  = "TEMPORAL",
                        weight          = 0.3,
                        t_delta         = sorted_events[i+1].t - sorted_events[i].t,
                        label           = "precedes"
                    )
                    self.edges.append(edge)
                    self.graph.add_edge(
                        src, tgt,
                        weight         = 0.3,
                        influence_type = "TEMPORAL",
                        label          = "precedes"
                    )

        # ── Add conceptual edges (shared concepts across agents) ───────────
        self._add_conceptual_edges()

        return self.graph

    def _add_conceptual_edges(self) -> None:
        """
        Add edges between events from different agents that share concepts.
        Earlier event → later event if they share ≥ 1 concept.
        Weight proportional to concept overlap.
        """
        events_with_concepts = [
            (eid, node) for eid, node in self.nodes.items()
            if node.concepts
        ]

        for i in range(len(events_with_concepts)):
            eid_i, node_i = events_with_concepts[i]
            for j in range(i + 1, len(events_with_concepts)):
                eid_j, node_j = events_with_concepts[j]

                # Only cross-agent conceptual edges
                if node_i.agent_id == node_j.agent_id:
                    continue

                # Earlier → later only
                if node_i.t >= node_j.t:
                    continue

                shared = set(node_i.concepts) & set(node_j.concepts)
                if not shared:
                    continue

                overlap = len(shared) / max(
                    len(set(node_i.concepts)),
                    len(set(node_j.concepts))
                )

                if not self.graph.has_edge(eid_i, eid_j):
                    edge = FingerprintEdge(
                        source_event_id = eid_i,
                        target_event_id = eid_j,
                        influence_type  = "CONCEPTUAL",
                        weight          = float(overlap),
                        t_delta         = node_j.t - node_i.t,
                        label           = f"shares: {', '.join(list(shared)[:3])}"
                    )
                    self.edges.append(edge)
                    self.graph.add_edge(
                        eid_i, eid_j,
                        weight         = float(overlap),
                        influence_type = "CONCEPTUAL",
                        label          = edge.label
                    )

    # ── DISCOVERY TRACING ─────────────────────────────────────────────────

    def trace_discovery(
        self,
        apex_event_id: str,
        max_depth: int = 10
    ) -> Optional[Discovery]:
        """
        Trace the causal chain that led to a specific discovery event.

        Walks backwards through the influence DAG from the apex event,
        collecting all causally contributing events.

        Returns a Discovery with the full causal chain and
        agent contribution weights.
        """
        if apex_event_id not in self.graph:
            return None

        # BFS backwards from apex
        causal_chain: List[str] = []
        visited: Set[str] = set()
        queue = [apex_event_id]

        while queue and len(causal_chain) < max_depth:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            causal_chain.append(current)

            predecessors = list(self.graph.predecessors(current))
            # Sort by edge weight descending — follow strongest influences
            predecessors.sort(
                key=lambda p: self.graph[p][current].get("weight", 0),
                reverse=True
            )
            queue.extend(predecessors)

        causal_chain.reverse()  # Chronological order

        if not causal_chain:
            return None

        origin_event_id = causal_chain[0]

        # Compute agent contributions
        agent_contributions = self._compute_agent_contributions(causal_chain)

        # Compute novelty score for this discovery
        apex_node = self.nodes.get(apex_event_id)
        novelty_score = 0.0
        if apex_node and apex_node.metadata.get("novelty_delta"):
            novelty_score = float(apex_node.metadata["novelty_delta"])

        discovery = Discovery(
            discovery_id         = f"D-{apex_event_id[:8]}",
            session_id           = self.session.session_id,
            detected_at_t        = self.nodes[apex_event_id].t,
            description          = self._describe_discovery(causal_chain),
            novelty_score        = novelty_score,
            origin_event_id      = origin_event_id,
            apex_event_id        = apex_event_id,
            causal_chain         = causal_chain,
            agent_contributions  = agent_contributions
        )

        self.discoveries.append(discovery)
        return discovery

    def _compute_agent_contributions(
        self,
        causal_chain: List[str]
    ) -> Dict[str, float]:
        """
        Quantify each agent's contribution to a causal chain.
        Weight by event magnitude and position in chain
        (earlier contributions weighted slightly less —
        later synthesis weighted more).
        """
        contributions: Dict[str, float] = {}
        n = len(causal_chain)

        for i, event_id in enumerate(causal_chain):
            node = self.nodes.get(event_id)
            if node is None:
                continue

            # Position weight: later in chain = slightly higher weight
            position_weight = 0.5 + 0.5 * (i / max(n - 1, 1))
            magnitude_weight = max(node.magnitude, 0.01)
            contribution = position_weight * magnitude_weight

            if node.agent_id not in contributions:
                contributions[node.agent_id] = 0.0
            contributions[node.agent_id] += contribution

        # Normalize to sum to 1.0
        total = sum(contributions.values())
        if total > 0:
            contributions = {
                k: round(v / total, 4)
                for k, v in contributions.items()
            }

        return contributions

    def _describe_discovery(self, causal_chain: List[str]) -> str:
        """Generate a human-readable description of the discovery chain."""
        if not causal_chain:
            return "Unknown discovery"

        first_node = self.nodes.get(causal_chain[0])
        last_node  = self.nodes.get(causal_chain[-1])

        if first_node is None or last_node is None:
            return "Discovery chain could not be described"

        return (
            f"Discovery traced across {len(causal_chain)} epistemic events "
            f"from t={first_node.t} to t={last_node.t}. "
            f"Origin: {first_node.agent_type.value} agent "
            f"({first_node.agent_id}). "
            f"Apex: {last_node.agent_type.value} agent "
            f"({last_node.agent_id})."
        )

    # ── ANALYTICS ─────────────────────────────────────────────────────────

    def agent_contribution_summary(self) -> Dict[str, Dict]:
        """
        Per-agent contribution summary across all discoveries.
        Used by CHRONICLE for client reports.
        """
        summary: Dict[str, Dict] = {}

        for agent_id, agent in self.session.all_agents.items():
            agent_events = [
                node for node in self.nodes.values()
                if node.agent_id == agent_id
            ]

            total_magnitude = sum(n.magnitude for n in agent_events)
            event_count     = len(agent_events)

            # Contribution across all discoveries
            discovery_contributions = []
            for discovery in self.discoveries:
                if agent_id in discovery.agent_contributions:
                    discovery_contributions.append(
                        discovery.agent_contributions[agent_id]
                    )

            summary[agent_id] = {
                "agent_type":               agent.agent_type.value,
                "total_events":             event_count,
                "total_magnitude":          round(total_magnitude, 4),
                "mean_magnitude":           round(
                    total_magnitude / event_count, 4
                ) if event_count > 0 else 0.0,
                "discovery_contributions":  discovery_contributions,
                "mean_discovery_contribution": round(
                    float(np.mean(discovery_contributions)), 4
                ) if discovery_contributions else 0.0
            }

        return summary

    def coupling_index(self) -> float:
        """
        Measure of epistemic coupling between human and AI agents.
        High coupling = agents are genuinely influencing each other.
        Low coupling  = agents running in parallel, not collaborating.

        Returns [0.0, 1.0]
        """
        if not self.graph.edges:
            return 0.0

        cross_agent_edges = [
            (src, tgt) for src, tgt in self.graph.edges
            if (self.nodes.get(src) and self.nodes.get(tgt)
                and self.nodes[src].agent_id != self.nodes[tgt].agent_id)
        ]

        total_edges = self.graph.number_of_edges()
        if total_edges == 0:
            return 0.0

        # Weighted by edge weights
        cross_weight = sum(
            self.graph[src][tgt].get("weight", 0)
            for src, tgt in cross_agent_edges
        )
        total_weight = sum(
            d.get("weight", 0)
            for _, _, d in self.graph.edges(data=True)
        )

        return float(cross_weight / total_weight) if total_weight > 0 else 0.0

    # ── EXPORT ────────────────────────────────────────────────────────────

    def export_dict(self) -> Dict:
        """
        Serialize the full ATLAS graph for dashboard rendering
        and CHRONICLE report generation.
        """
        return {
            "session_id":    self.session.session_id,
            "node_count":    len(self.nodes),
            "edge_count":    len(self.edges),
            "coupling_index": round(self.coupling_index(), 4),
            "nodes": [
                {
                    "id":         n.event_id,
                    "agent_id":   n.agent_id,
                    "agent_type": n.agent_type.value,
                    "t":          n.t,
                    "magnitude":  round(n.magnitude, 4),
                    "label":      n.label,
                    "concepts":   n.concepts[:5],
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source":         e.source_event_id,
                    "target":         e.target_event_id,
                    "influence_type": e.influence_type,
                    "weight":         round(e.weight, 4),
                    "label":          e.label,
                }
                for e in self.edges
            ],
            "discoveries": [
                {
                    "id":             d.discovery_id,
                    "detected_at_t":  d.detected_at_t,
                    "novelty_score":  round(d.novelty_score, 4),
                    "causal_chain":   d.causal_chain,
                    "contributions":  d.agent_contributions,
                    "description":    d.description,
                }
                for d in self.discoveries
            ]
        }

    def _event_label(self, event: EATEvent) -> str:
        """Generate readable label for a node."""
        agent_prefix = "H" if event.agent_type == AgentType.HUMAN else "AI"
        return f"{agent_prefix}:{event.event_type.value}@t{event.t}"