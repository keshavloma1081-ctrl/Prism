"""
PRISM — chronicle/reporter.py
Client Intelligence Report Engine

One command. 30 seconds.
A structured, signed PDF that answers every question
a CTO has after writing a $1M AI deployment check.

What did the ensemble discover?
Who contributed what?
Would removing the AI have gotten us there?
Is the collaboration getting better or worse?
What should we change?

The document that makes clients renew contracts.
The report that separates FDEs who deploy and disappear
from FDEs who leave behind proof.
"""

from __future__ import annotations
import json
import uuid
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession
)
from verdict.scorer   import VerdictScorer
from decay.detector   import DecayDetector
from atlas.graph      import AtlasGraph
from ghost.replay     import GhostRunner, EmergenceSignature
from compass.optimizer import CompassOptimizer


# ─── REPORT SECTION ───────────────────────────────────────────────────────────

@dataclass
class ReportSection:
    title:    str
    content:  Dict[str, Any]
    grade:    Optional[str] = None
    warnings: List[str]     = field(default_factory=list)


# ─── PRISM REPORT ─────────────────────────────────────────────────────────────

@dataclass
class PrismReport:
    """
    Full PRISM client intelligence report.
    Generated from a completed session with all
    seven systems contributing their analysis.
    """
    report_id:   str
    client_id:   str
    workflow_id: str
    session_id:  str
    generated_at: datetime = field(default_factory=datetime.utcnow)

    # Report sections
    executive_summary:     Optional[ReportSection] = None
    verdict_section:       Optional[ReportSection] = None
    decay_section:         Optional[ReportSection] = None
    atlas_section:         Optional[ReportSection] = None
    ghost_section:         Optional[ReportSection] = None
    compass_section:       Optional[ReportSection] = None
    recommendations:       Optional[ReportSection] = None

    # Overall grade
    overall_grade:  str = "PENDING"
    overall_score:  float = 0.0

    def to_dict(self) -> Dict:
        return {
            "report_id":    self.report_id,
            "client_id":    self.client_id,
            "workflow_id":  self.workflow_id,
            "session_id":   self.session_id,
            "generated_at": self.generated_at.isoformat(),
            "overall_grade": self.overall_grade,
            "overall_score": self.overall_score,
            "sections": {
                "executive_summary": self._section_dict(self.executive_summary),
                "verdict":           self._section_dict(self.verdict_section),
                "decay":             self._section_dict(self.decay_section),
                "atlas":             self._section_dict(self.atlas_section),
                "ghost":             self._section_dict(self.ghost_section),
                "compass":           self._section_dict(self.compass_section),
                "recommendations":   self._section_dict(self.recommendations),
            }
        }

    def _section_dict(self, section: Optional[ReportSection]) -> Optional[Dict]:
        if section is None:
            return None
        return {
            "title":    section.title,
            "grade":    section.grade,
            "warnings": section.warnings,
            "content":  section.content
        }

    def print_report(self) -> None:
        """Print formatted report to terminal."""
        sep = "=" * 70

        print(f"\n{sep}")
        print(f"  PRISM CLIENT INTELLIGENCE REPORT")
        print(f"  Client:    {self.client_id}")
        print(f"  Workflow:  {self.workflow_id}")
        print(f"  Session:   {self.session_id}")
        print(f"  Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"  Grade:     {self.overall_grade}  ({self.overall_score:.2f})")
        print(f"{sep}\n")

        sections = [
            self.executive_summary,
            self.verdict_section,
            self.decay_section,
            self.atlas_section,
            self.ghost_section,
            self.compass_section,
            self.recommendations
        ]

        for section in sections:
            if section is None:
                continue
            grade_str = f"  [{section.grade}]" if section.grade else ""
            print(f"── {section.title}{grade_str}")
            print(f"{'─' * 60}")
            self._print_dict(section.content, indent=2)
            if section.warnings:
                print(f"  ⚠ Warnings:")
                for w in section.warnings:
                    print(f"    • {w}")
            print()

    def _print_dict(self, d: Dict, indent: int = 0) -> None:
        prefix = " " * indent
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                self._print_dict(v, indent + 2)
            elif isinstance(v, list):
                print(f"{prefix}{k}:")
                for item in v[:5]:
                    print(f"{prefix}  • {item}")
                if len(v) > 5:
                    print(f"{prefix}  ... and {len(v) - 5} more")
            else:
                print(f"{prefix}{k}: {v}")


# ─── CHRONICLE REPORTER ───────────────────────────────────────────────────────

class ChronicleReporter:
    """
    Main CHRONICLE report generation engine.

    Orchestrates all seven PRISM systems to produce
    a complete client intelligence report from a
    completed workflow session.

    Usage:
        reporter = ChronicleReporter(session)
        reporter.attach_compass(compass_optimizer)
        report   = reporter.generate()
        report.print_report()
    """

    def __init__(self, session: WorkflowSession):
        self.session  = session
        self.compass: Optional[CompassOptimizer] = None

    def attach_compass(self, compass: CompassOptimizer) -> None:
        """Attach a CompassOptimizer for optimization history."""
        self.compass = compass

    # ── SECTION GENERATORS ────────────────────────────────────────────────

    def _build_verdict_section(self) -> ReportSection:
        """Run VERDICT scoring and summarize."""
        scorer  = VerdictScorer(self.session)
        warnings = []

        # Score all events
        for event in self.session.events:
            scorer.score_event(event)

        summary = scorer.session_verdict_summary()
        grade   = summary.get("verdict_grade", "NO_DATA")

        if grade == "POOR":
            warnings.append(
                "AI epistemic contribution quality is below acceptable threshold. "
                "Consider model substitution via Ghost Runner."
            )
        if summary.get("mean_groundedness") is not None:
            if summary["mean_groundedness"] < 0.4:
                warnings.append(
                    "Groundedness score critically low — "
                    "AI beliefs not anchored in session evidence."
                )

        return ReportSection(
            title    = "AI Epistemic Quality — VERDICT",
            content  = {
                "total_ai_events":          summary.get("total_ai_events"),
                "mean_groundedness":         summary.get("mean_groundedness"),
                "mean_novelty_delta":        summary.get("mean_novelty_delta"),
                "mean_influence_survival":   summary.get("mean_influence_survival"),
                "composite_score":           summary.get("composite_score"),
                "calibration_by_agent":      summary.get("calibration_by_agent"),
            },
            grade    = grade,
            warnings = warnings
        )

    def _build_decay_section(self) -> ReportSection:
        """Run DECAY detection and summarize."""
        detector = DecayDetector(self.session)
        warnings = []

        # Run checks at each time step
        time_steps = sorted(set(e.t for e in self.session.events))
        for t in time_steps:
            alerts = detector.check(current_t=t)

        summary  = detector.decay_summary()
        grade    = "HEALTHY"

        if summary["total_alerts"] > 0:
            grade = "DEGRADED"
            warnings.append(
                f"{summary['total_alerts']} decay alert(s) detected in this session."
            )

        critical = summary.get("critical_alerts", [])
        for alert in critical:
            warnings.append(
                f"[{alert['severity']}] {alert['type']}: {alert['recommendation'][:80]}..."
            )

        return ReportSection(
            title   = "Epistemic Health — DECAY",
            content = {
                "total_alerts":              summary["total_alerts"],
                "alerts_by_type":            summary["alerts_by_type"],
                "current_engagement_rate":   summary["current_engagement_rate"],
                "current_diversity_index":   summary["current_diversity_index"],
                "current_novelty_rate":      summary["current_novelty_rate"],
                "novelty_curve":             summary["novelty_curve"],
            },
            grade    = grade,
            warnings = warnings
        )

    def _build_atlas_section(self) -> ReportSection:
        """Build ATLAS causal fingerprint and summarize."""
        atlas    = AtlasGraph(self.session)
        atlas.build_fingerprint()
        warnings = []

        # Trace discoveries for high-magnitude events
        high_mag_events = sorted(
            [e for e in self.session.events if e.delta_magnitude > 0.3],
            key=lambda e: e.delta_magnitude,
            reverse=True
        )[:3]  # Top 3 discoveries

        for event in high_mag_events:
            atlas.trace_discovery(event.event_id)

        exported     = atlas.export_dict()
        contribution = atlas.agent_contribution_summary()

        if exported["coupling_index"] < 0.2:
            warnings.append(
                "Low epistemic coupling detected — agents are not meaningfully "
                "influencing each other. Collaboration structure needs review."
            )

        return ReportSection(
            title   = "Causal Discovery Fingerprint — ATLAS",
            content = {
                "total_nodes":          exported["node_count"],
                "total_edges":          exported["edge_count"],
                "coupling_index":       exported["coupling_index"],
                "discoveries_traced":   len(exported["discoveries"]),
                "agent_contributions":  {
                    aid: {
                        "type":              c["agent_type"],
                        "total_events":      c["total_events"],
                        "mean_magnitude":    c["mean_magnitude"],
                        "mean_contribution": c["mean_discovery_contribution"]
                    }
                    for aid, c in contribution.items()
                },
                "top_discoveries": [
                    {
                        "id":            d["id"],
                        "novelty_score": d["novelty_score"],
                        "chain_length":  len(d["causal_chain"]),
                        "description":   d["description"][:100] + "..."
                        if len(d["description"]) > 100 else d["description"]
                    }
                    for d in exported["discoveries"][:3]
                ]
            },
            grade    = "STRONG" if exported["coupling_index"] > 0.5 else "MODERATE",
            warnings = warnings
        )

    def _build_ghost_section(self) -> ReportSection:
        """Run Ghost Runner and summarize emergence signature."""
        ghost    = GhostRunner(self.session)
        sig      = ghost.run()
        warnings = []

        if sig.emergence_score < 0.3:
            warnings.append(
                "Low emergence score — the ensemble is not producing "
                "meaningful value beyond solo performance. "
                "Immediate workflow restructuring recommended."
            )

        if sig.ai_value_score < 0.2:
            warnings.append(
                "AI is contributing minimal unique value. "
                "Consider model substitution or role restructuring."
            )

        grade = (
            "STRONG"   if sig.emergence_score > 0.6 else
            "MODERATE" if sig.emergence_score > 0.3 else
            "WEAK"
        )

        return ReportSection(
            title   = "Counterfactual Analysis — GHOST",
            content = {
                "emergence_score":        sig.emergence_score,
                "ai_value_score":         sig.ai_value_score,
                "human_value_score":      sig.human_value_score,
                "emerged_concepts":       list(sig.concept_emergence)[:10],
                "ai_unique_concepts":     list(sig.ai_unique_concepts)[:10],
                "human_unique_concepts":  list(sig.human_unique_concepts)[:10],
                "entropy_lift":           sig.entropy_lift,
                "magnitude_lift":         sig.magnitude_lift,
                "discovery_lift":         sig.discovery_lift,
                "full_events":            sig.full_result.event_count,
                "human_only_events":      sig.human_result.event_count,
                "ai_only_events":         sig.ai_result.event_count,
                "verdict":                sig.verdict(),
                "recommendation":         sig.recommendation
            },
            grade    = grade,
            warnings = warnings
        )

    def _build_compass_section(self) -> ReportSection:
        """Summarize COMPASS optimization history."""
        if self.compass is None:
            return ReportSection(
                title   = "Prompt Optimization — COMPASS",
                content = {"status": "No CompassOptimizer attached to this report."},
                grade   = "N/A"
            )

        summary  = self.compass.optimization_summary()
        warnings = []

        if summary["total_records"] > 5:
            warnings.append(
                f"{summary['total_records']} optimizations triggered — "
                "this workflow requires significant prompt engineering attention."
            )

        return ReportSection(
            title   = "Prompt Optimization — COMPASS",
            content = {
                "total_sessions":      summary["total_sessions"],
                "optimization_cycles": summary["optimization_cycles"],
                "total_variants":      summary["total_variants"],
                "signals_optimized":   summary["signals_optimized"],
                "agents_optimized":    summary["agents_optimized"],
                "recent_optimizations": [
                    {
                        "signal":        r["signal"],
                        "trigger_score": r["trigger_score"],
                        "generation":    r["generation"]
                    }
                    for r in summary["recent_records"]
                ]
            },
            grade    = "ACTIVE" if summary["total_records"] > 0 else "STABLE",
            warnings = warnings
        )

    def _build_executive_summary(
        self,
        verdict_section: ReportSection,
        decay_section:   ReportSection,
        ghost_section:   ReportSection,
        atlas_section:   ReportSection
    ) -> ReportSection:
        """
        Executive summary — the first page a CTO reads.
        """
        # Compute overall score
        scores = []
        grade_map = {
            "EXCELLENT": 1.0, "STRONG": 0.85, "GOOD": 0.7,
            "MODERATE": 0.55, "WEAK": 0.35, "POOR": 0.2,
            "HEALTHY": 0.8, "DEGRADED": 0.3,
            "ACTIVE": 0.7, "STABLE": 0.8,
            "NO_DATA": 0.5, "N/A": 0.5, "PENDING": 0.5
        }

        for section in [verdict_section, decay_section, ghost_section, atlas_section]:
            if section.grade and section.grade in grade_map:
                scores.append(grade_map[section.grade])

        overall_score = float(np.mean(scores)) if scores else 0.5

        overall_grade = (
            "EXCELLENT" if overall_score >= 0.85 else
            "GOOD"      if overall_score >= 0.70 else
            "MODERATE"  if overall_score >= 0.55 else
            "POOR"
        )

        # Key findings
        findings = []

        ghost_content = ghost_section.content
        if ghost_content.get("emergence_score", 0) > 0.6:
            findings.append(
                f"Strong collaborative emergence detected "
                f"(score: {ghost_content.get('emergence_score', 0):.2f}) — "
                f"the ensemble is producing genuine collective intelligence."
            )
        else:
            findings.append(
                f"Limited collaborative emergence "
                f"(score: {ghost_content.get('emergence_score', 0):.2f}) — "
                f"workflow optimization recommended."
            )

        decay_content = decay_section.content
        if decay_content.get("total_alerts", 0) > 0:
            findings.append(
                f"{decay_content['total_alerts']} epistemic decay alert(s) — "
                f"human engagement with AI outputs is declining."
            )
        else:
            findings.append("No epistemic decay detected — collaboration health is good.")

        verdict_content = verdict_section.content
        if verdict_content.get("composite_score") is not None:
            findings.append(
                f"AI epistemic quality score: "
                f"{verdict_content['composite_score']:.2f} "
                f"({verdict_section.grade})"
            )

        return ReportSection(
            title   = "Executive Summary",
            content = {
                "overall_grade":     overall_grade,
                "overall_score":     round(overall_score, 3),
                "session_id":        self.session.session_id,
                "total_events":      self.session.total_events,
                "human_agents":      len(self.session.human_agents),
                "ai_agents":         len(self.session.ai_agents),
                "key_findings":      findings,
                "section_grades": {
                    "AI Quality (VERDICT)":  verdict_section.grade,
                    "Health (DECAY)":        decay_section.grade,
                    "Attribution (ATLAS)":   atlas_section.grade,
                    "Emergence (GHOST)":     ghost_section.grade,
                }
            },
            grade = overall_grade
        )

    def _build_recommendations(
        self,
        verdict_section: ReportSection,
        decay_section:   ReportSection,
        ghost_section:   ReportSection,
        atlas_section:   ReportSection,
        compass_section: ReportSection
    ) -> ReportSection:
        """
        Consolidated, prioritized recommendations for the FDE.
        """
        recs = []
        priority = 1

        # Collect all warnings across sections
        all_warnings = (
            verdict_section.warnings +
            decay_section.warnings   +
            ghost_section.warnings   +
            atlas_section.warnings   +
            compass_section.warnings
        )

        for warning in all_warnings:
            recs.append({
                "priority":       priority,
                "recommendation": warning
            })
            priority += 1

        # Ghost recommendation
        ghost_rec = ghost_section.content.get("recommendation")
        if ghost_rec:
            recs.append({
                "priority":       priority,
                "recommendation": ghost_rec
            })

        if not recs:
            recs.append({
                "priority": 1,
                "recommendation": (
                    "No critical issues detected. "
                    "Continue monitoring with PRISM. "
                    "Consider running COMPASS optimization "
                    "to explore further performance improvements."
                )
            })

        return ReportSection(
            title   = "Recommendations",
            content = {
                "total_recommendations": len(recs),
                "items":                 recs
            }
        )

    # ── MAIN GENERATE ─────────────────────────────────────────────────────

    def generate(self) -> PrismReport:
        """
        Generate the complete PRISM client intelligence report.
        Orchestrates all seven systems.
        Returns a PrismReport ready for printing or PDF export.
        """
        report = PrismReport(
            report_id   = f"RPT-{uuid.uuid4().hex[:12]}",
            client_id   = self.session.client_id,
            workflow_id = self.session.workflow_id,
            session_id  = self.session.session_id
        )

        # Run all sections
        verdict_section  = self._build_verdict_section()
        decay_section    = self._build_decay_section()
        atlas_section    = self._build_atlas_section()
        ghost_section    = self._build_ghost_section()
        compass_section  = self._build_compass_section()

        executive_summary = self._build_executive_summary(
            verdict_section, decay_section,
            ghost_section,   atlas_section
        )

        recommendations = self._build_recommendations(
            verdict_section, decay_section,
            ghost_section,   atlas_section,
            compass_section
        )

        # Assemble report
        report.executive_summary  = executive_summary
        report.verdict_section    = verdict_section
        report.decay_section      = decay_section
        report.atlas_section      = atlas_section
        report.ghost_section      = ghost_section
        report.compass_section    = compass_section
        report.recommendations    = recommendations
        report.overall_grade      = executive_summary.content["overall_grade"]
        report.overall_score      = executive_summary.content["overall_score"]

        return report