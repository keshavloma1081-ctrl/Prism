"""
PRISM — ghost/replay.py
Counterfactual Workflow Replay Engine

The feature that exists nowhere else in the world.

Take any recorded workflow session.
Strip out the AI agents. Replay it.
What does the human team discover alone?

Strip out the humans.
What does the AI ensemble discover alone?

The delta between three runs:
  FULL     → humans + AI together
  HUMAN    → humans only
  AI       → AI only

Is the EMERGENCE SIGNATURE:
Mathematical proof of what the collaboration produced
that neither side could have alone.

Used by FDEs to answer the question every CTO asks:
"Did we actually need the AI for this?"
"""

from __future__ import annotations
import copy
import uuid
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession, HumanAgent, AIAgent,
    WorkflowStatus
)
from core.eat.delta import (
    kl_divergence, belief_entropy,
    ensemble_entropy, fractional_graph_edit
)
from core.eat.validators import validate_session_events
from atlas.graph import AtlasGraph


# ─── REPLAY MODE ──────────────────────────────────────────────────────────────

class ReplayMode(str, Enum):
    FULL        = "FULL"        # All agents — baseline
    HUMAN_ONLY  = "HUMAN_ONLY"  # Strip AI agents
    AI_ONLY     = "AI_ONLY"     # Strip human agents
    SUBSET      = "SUBSET"      # Custom agent subset


# ─── REPLAY RESULT ────────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    """
    Result of a single counterfactual replay run.
    Carries the replayed session and all derived metrics.
    """
    mode:              ReplayMode
    session:           WorkflowSession
    event_count:       int
    unique_concepts:   Set[str]
    final_entropy:     float
    mean_magnitude:    float
    discovery_count:   int
    coupling_index:    float
    timestamp:         datetime = field(default_factory=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"ReplayResult({self.mode.value} | "
            f"events={self.event_count} | "
            f"concepts={len(self.unique_concepts)} | "
            f"entropy={self.final_entropy:.3f})"
        )


# ─── EMERGENCE SIGNATURE ──────────────────────────────────────────────────────

@dataclass
class EmergenceSignature:
    """
    The core output of Ghost Runner.

    Quantifies what the full ensemble produced
    that neither humans nor AI could produce alone.

    This is the answer to: "Did we need the AI?"
    And more importantly: "Did we need the humans?"
    """
    session_id:            str

    # Raw results from all three runs
    full_result:           ReplayResult
    human_result:          ReplayResult
    ai_result:             ReplayResult

    # Emergence metrics
    concept_emergence:     Set[str]    # Concepts only in FULL, not in HUMAN or AI alone
    entropy_lift:          float       # Entropy reduction beyond what either alone achieved
    magnitude_lift:        float       # Mean event magnitude in FULL vs best of solo runs
    discovery_lift:        float       # Discoveries in FULL vs best of solo runs

    # Attribution
    human_unique_concepts: Set[str]    # Concepts only humans contributed
    ai_unique_concepts:    Set[str]    # Concepts only AI contributed
    shared_concepts:       Set[str]    # Concepts both contributed

    # Verdict
    emergence_score:       float       # [0.0, 1.0] — how much did ensemble > sum of parts?
    ai_value_score:        float       # [0.0, 1.0] — how much did AI add beyond humans?
    human_value_score:     float       # [0.0, 1.0] — how much did humans add beyond AI?

    recommendation:        str
    timestamp:             datetime = field(default_factory=datetime.utcnow)

    def verdict(self) -> str:
        """
        Plain English verdict for the client report.
        """
        if self.emergence_score > 0.7:
            return (
                "STRONG EMERGENCE: The human-AI ensemble produced outcomes "
                "substantially beyond what either could achieve alone. "
                "The collaboration is generating genuine collective intelligence."
            )
        elif self.emergence_score > 0.4:
            return (
                "MODERATE EMERGENCE: The ensemble shows meaningful collaborative "
                "advantage over solo performance. "
                "Optimising workflow structure could increase emergence further."
            )
        elif self.ai_value_score > self.human_value_score:
            return (
                "AI-DOMINANT: The AI is contributing more unique value than the humans. "
                "Consider whether human involvement is optimally structured, "
                "or whether humans are under-leveraged in this workflow."
            )
        elif self.human_value_score > self.ai_value_score:
            return (
                "HUMAN-DOMINANT: Humans are contributing more unique value than the AI. "
                "The AI may not be well-suited to this workflow. "
                "Consider model substitution via Ghost Runner model swap."
            )
        else:
            return (
                "LOW EMERGENCE: The ensemble is not producing meaningful collaborative "
                "advantage. Neither side is substantially influencing the other. "
                "Immediate workflow restructuring recommended."
            )


# ─── GHOST RUNNER ─────────────────────────────────────────────────────────────

class GhostRunner:
    """
    Main Ghost Runner engine.

    Takes a completed WorkflowSession and runs three
    counterfactual replays to compute the emergence signature.

    Also supports model swap: replay the same session with
    a different AI backend to compare epistemic contribution
    quality across models.
    """

    def __init__(self, session: WorkflowSession):
        self.session  = session
        self.results: Dict[ReplayMode, ReplayResult] = {}

    # ── SESSION FILTERING ─────────────────────────────────────────────────

    def _filter_session(
        self,
        mode: ReplayMode,
        keep_agent_ids: Optional[Set[str]] = None
    ) -> WorkflowSession:
        """
        Create a filtered copy of the session for replay.
        Removes agents and their events based on mode.
        """
        filtered = WorkflowSession(
            session_id        = f"S-ghost-{uuid.uuid4().hex[:8]}",
            client_id         = self.session.client_id,
            workflow_id       = self.session.workflow_id,
            status            = WorkflowStatus.REPLAYING,
            is_counterfactual = True,
            parent_session_id = self.session.session_id
        )

        # Determine which agents to keep
        if mode == ReplayMode.FULL:
            keep_ids = set(self.session.all_agents.keys())
        elif mode == ReplayMode.HUMAN_ONLY:
            keep_ids = set(self.session.human_agents.keys())
        elif mode == ReplayMode.AI_ONLY:
            keep_ids = set(self.session.ai_agents.keys())
        elif mode == ReplayMode.SUBSET:
            keep_ids = keep_agent_ids or set()
        else:
            keep_ids = set(self.session.all_agents.keys())

        # Copy kept agents
        for agent_id in keep_ids:
            if agent_id in self.session.human_agents:
                filtered.human_agents[agent_id] = copy.deepcopy(
                    self.session.human_agents[agent_id]
                )
            elif agent_id in self.session.ai_agents:
                filtered.ai_agents[agent_id] = copy.deepcopy(
                    self.session.ai_agents[agent_id]
                )

        # Copy only events from kept agents
        # Also remove TRIGGER events whose trigger_ref
        # points to a removed agent's event
        removed_event_ids: Set[str] = {
            e.event_id for e in self.session.events
            if e.agent_id not in keep_ids
        }

        for event in self.session.events:
            if event.agent_id not in keep_ids:
                continue

            # Deep copy event
            event_copy = copy.deepcopy(event)

            # Nullify broken trigger references
            if (event_copy.trigger_ref is not None
                    and event_copy.trigger_ref in removed_event_ids):
                event_copy.trigger_ref   = None
                event_copy.trigger_agent = None
                # Downgrade TRIGGER to UPDATE
                if event_copy.event_type == EATEventType.B_TRIGGER:
                    event_copy.event_type = EATEventType.B_UPDATE
                elif event_copy.event_type == EATEventType.C_TRIGGER:
                    event_copy.event_type = EATEventType.C_UPDATE

            filtered.add_event(event_copy)

        return filtered

    # ── METRICS EXTRACTION ────────────────────────────────────────────────

    def _extract_metrics(
        self,
        session: WorkflowSession,
        mode: ReplayMode
    ) -> ReplayResult:
        """Extract measurement metrics from a replayed session."""

        # Collect all concepts introduced in this session
        unique_concepts: Set[str] = set()
        for event in session.events:
            if event.posterior_graph is not None:
                unique_concepts.update(event.posterior_graph.nodes.keys())

        # Final ensemble entropy
        final_beliefs = []
        latest_belief: Dict[str, object] = {}
        for event in session.events:
            if event.posterior_belief is not None:
                latest_belief[event.agent_id] = event.posterior_belief
        final_beliefs = list(latest_belief.values())
        final_entropy = ensemble_entropy(final_beliefs) if final_beliefs else 0.0

        # Mean event magnitude
        magnitudes = [e.delta_magnitude for e in session.events if e.delta_magnitude > 0]
        mean_magnitude = float(np.mean(magnitudes)) if magnitudes else 0.0

        # Discovery count via ATLAS
        atlas = AtlasGraph(session)
        atlas.build_fingerprint()
        high_magnitude_events = [
            e.event_id for e in session.events
            if e.delta_magnitude > 0.5
        ]
        for event_id in high_magnitude_events:
            atlas.trace_discovery(event_id)
        discovery_count = len(atlas.discoveries)

        return ReplayResult(
            mode            = mode,
            session         = session,
            event_count     = session.total_events,
            unique_concepts = unique_concepts,
            final_entropy   = final_entropy,
            mean_magnitude  = mean_magnitude,
            discovery_count = discovery_count,
            coupling_index  = atlas.coupling_index()
        )

    # ── RUN ───────────────────────────────────────────────────────────────

    def run(self) -> EmergenceSignature:
        """
        Run all three counterfactual configurations and
        compute the emergence signature.

        This is the main Ghost Runner method.
        Call once on a completed session.
        """
        # ── Three replay runs ──────────────────────────────────────────────
        full_session   = self._filter_session(ReplayMode.FULL)
        human_session  = self._filter_session(ReplayMode.HUMAN_ONLY)
        ai_session     = self._filter_session(ReplayMode.AI_ONLY)

        full_result   = self._extract_metrics(full_session,  ReplayMode.FULL)
        human_result  = self._extract_metrics(human_session, ReplayMode.HUMAN_ONLY)
        ai_result     = self._extract_metrics(ai_session,    ReplayMode.AI_ONLY)

        self.results[ReplayMode.FULL]       = full_result
        self.results[ReplayMode.HUMAN_ONLY] = human_result
        self.results[ReplayMode.AI_ONLY]    = ai_result

        # ── Concept emergence analysis ─────────────────────────────────────
        full_concepts  = full_result.unique_concepts
        human_concepts = human_result.unique_concepts
        ai_concepts    = ai_result.unique_concepts

        # Emerged only in full collaboration
        concept_emergence     = full_concepts - (human_concepts | ai_concepts)
        human_unique_concepts = human_concepts - ai_concepts
        ai_unique_concepts    = ai_concepts - human_concepts
        shared_concepts       = human_concepts & ai_concepts

        # ── Emergence scores ───────────────────────────────────────────────

        # Concept emergence score
        concept_emergence_score = (
            len(concept_emergence) / max(len(full_concepts), 1)
        )

        # Entropy lift: did full ensemble reduce uncertainty more?
        best_solo_entropy = min(human_result.final_entropy, ai_result.final_entropy)
        entropy_lift = max(best_solo_entropy - full_result.final_entropy, 0.0)

        # Magnitude lift: did full ensemble produce stronger epistemic acts?
        best_solo_magnitude = max(human_result.mean_magnitude, ai_result.mean_magnitude)
        magnitude_lift = max(full_result.mean_magnitude - best_solo_magnitude, 0.0)

        # Discovery lift
        best_solo_discoveries = max(human_result.discovery_count, ai_result.discovery_count)
        discovery_lift = max(full_result.discovery_count - best_solo_discoveries, 0.0)

        # Composite emergence score
        emergence_components = [concept_emergence_score]
        if best_solo_entropy > 0:
            emergence_components.append(min(entropy_lift / best_solo_entropy, 1.0))
        if best_solo_magnitude > 0:
            emergence_components.append(min(magnitude_lift / best_solo_magnitude, 1.0))

        emergence_score = float(np.mean(emergence_components))

        # AI value: what did AI add beyond humans alone?
        ai_concept_contribution = (
            len(ai_unique_concepts) / max(len(full_concepts), 1)
        )
        ai_value_score = float(np.mean([
            ai_concept_contribution,
            min(ai_result.mean_magnitude / max(human_result.mean_magnitude, 0.01), 1.0)
        ]))

        # Human value: what did humans add beyond AI alone?
        human_concept_contribution = (
            len(human_unique_concepts) / max(len(full_concepts), 1)
        )
        human_value_score = float(np.mean([
            human_concept_contribution,
            min(human_result.mean_magnitude / max(ai_result.mean_magnitude, 0.01), 1.0)
        ]))

        # ── Recommendation ─────────────────────────────────────────────────
        recommendation = self._generate_recommendation(
            emergence_score, ai_value_score, human_value_score,
            concept_emergence, ai_unique_concepts, human_unique_concepts
        )

        return EmergenceSignature(
            session_id            = self.session.session_id,
            full_result           = full_result,
            human_result          = human_result,
            ai_result             = ai_result,
            concept_emergence     = concept_emergence,
            entropy_lift          = round(entropy_lift, 4),
            magnitude_lift        = round(magnitude_lift, 4),
            discovery_lift        = float(discovery_lift),
            human_unique_concepts = human_unique_concepts,
            ai_unique_concepts    = ai_unique_concepts,
            shared_concepts       = shared_concepts,
            emergence_score       = round(emergence_score, 4),
            ai_value_score        = round(ai_value_score, 4),
            human_value_score     = round(human_value_score, 4),
            recommendation        = recommendation
        )

    def _generate_recommendation(
        self,
        emergence_score:   float,
        ai_value_score:    float,
        human_value_score: float,
        emerged_concepts:  Set[str],
        ai_unique:         Set[str],
        human_unique:      Set[str]
    ) -> str:
        parts = []

        if emergence_score > 0.6:
            parts.append(
                f"Collaboration is working well "
                f"(emergence score: {emergence_score:.2f}). "
                f"Maintain current workflow structure."
            )
        else:
            parts.append(
                f"Collaboration is underperforming "
                f"(emergence score: {emergence_score:.2f}). "
                f"Workflow restructuring recommended."
            )

        if ai_unique:
            parts.append(
                f"AI contributed {len(ai_unique)} unique concepts "
                f"not reached by humans alone: "
                f"{', '.join(list(ai_unique)[:3])}{'...' if len(ai_unique) > 3 else ''}."
            )

        if human_unique:
            parts.append(
                f"Humans contributed {len(human_unique)} unique concepts "
                f"not reached by AI alone: "
                f"{', '.join(list(human_unique)[:3])}{'...' if len(human_unique) > 3 else ''}."
            )

        if emerged_concepts:
            parts.append(
                f"{len(emerged_concepts)} concepts emerged only in full collaboration: "
                f"{', '.join(list(emerged_concepts)[:3])}{'...' if len(emerged_concepts) > 3 else ''}. "
                f"These represent genuine collective intelligence."
            )

        return " ".join(parts)

    # ── MODEL SWAP ────────────────────────────────────────────────────────

    def model_swap_comparison(
        self,
        original_model:     str,
        replacement_model:  str,
        replacement_events: List[EATEvent]
    ) -> Dict:
        """
        Compare epistemic contribution quality between two AI models
        on the same session.

        original_model:     model name of the original AI in the session
        replacement_model:  model name of the replacement AI
        replacement_events: EAT events generated by replacement model
                           on the same session context

        Returns comparison dict for CHRONICLE client reports.
        """
        # Build replacement session
        replacement_session = self._filter_session(ReplayMode.HUMAN_ONLY)
        replacement_session.status = WorkflowStatus.REPLAYING

        # Add replacement AI agent
        replacement_agent = AIAgent(
            model_name = replacement_model,
            provider   = replacement_model.split("-")[0],
            client_id  = self.session.client_id
        )
        replacement_session.ai_agents[replacement_agent.agent_id] = replacement_agent

        for event in replacement_events:
            replacement_session.add_event(copy.deepcopy(event))

        # Extract metrics for both
        original_result     = self.results.get(ReplayMode.FULL)
        replacement_result  = self._extract_metrics(
            replacement_session, ReplayMode.FULL
        )

        if original_result is None:
            return {"error": "Run Ghost Runner first before model swap comparison"}

        return {
            "original_model":    original_model,
            "replacement_model": replacement_model,
            "comparison": {
                "concepts_original":    len(original_result.unique_concepts),
                "concepts_replacement": len(replacement_result.unique_concepts),
                "concept_overlap":      len(
                    original_result.unique_concepts &
                    replacement_result.unique_concepts
                ),
                "concepts_only_original":    list(
                    original_result.unique_concepts -
                    replacement_result.unique_concepts
                )[:10],
                "concepts_only_replacement": list(
                    replacement_result.unique_concepts -
                    original_result.unique_concepts
                )[:10],
                "entropy_original":    round(original_result.final_entropy, 4),
                "entropy_replacement": round(replacement_result.final_entropy, 4),
                "magnitude_original":  round(original_result.mean_magnitude, 4),
                "magnitude_replacement": round(replacement_result.mean_magnitude, 4),
            },
            "recommendation": (
                f"{replacement_model} introduces "
                f"{len(replacement_result.unique_concepts - original_result.unique_concepts)} "
                f"new concepts vs {original_model}. "
                f"{'Switch recommended.' if replacement_result.mean_magnitude > original_result.mean_magnitude else 'Keep original model.'}"
            )
        }