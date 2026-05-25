"""
PRISM — demo.py
Live Interview Demo — Rich Epistemic Data

Shows all seven systems producing real scores.
Run this in any interview.

Usage:
    # Terminal 1
    uvicorn api.main:app --reload --port 8000

    # Terminal 2
    python demo.py
"""

import threading
import time
import requests
import uvicorn
from api.main import app


def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")


BASE = "http://localhost:8000"


def post(path, payload=None):
    r = requests.post(f"{BASE}{path}", json=payload or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def get(path):
    r = requests.get(f"{BASE}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def run_demo():
    print("\n" + "=" * 60)
    print("  PRISM — Live Demo")
    print("  Epistemic Observability for Human-AI Workflows")
    print("=" * 60 + "\n")

    # ── 1. Create session ─────────────────────────────────────────
    session = post("/sessions/", {
        "client_id":   "demo-corp",
        "workflow_id": "strategic-analysis-q3"
    })
    sid = session["session_id"]
    print(f"[PRISM] Session: {sid}\n")

    # ── 2. Register agents ────────────────────────────────────────
    sarah = post(f"/sessions/{sid}/agents/human", {
        "name": "Sarah Chen",
        "role": "analyst"
    })
    james = post(f"/sessions/{sid}/agents/human", {
        "name": "James Okafor",
        "role": "domain-expert"
    })
    claude = post(f"/sessions/{sid}/agents/ai", {
        "model_name":     "claude-sonnet-4-6",
        "provider":       "anthropic",
        "initial_prompt": "You are a strategic analysis AI."
    })

    sarah_id  = sarah["agent_id"]
    james_id  = james["agent_id"]
    claude_id = claude["agent_id"]

    print(f"[PRISM] Agents: {sarah_id} | {james_id} | {claude_id}\n")
    print("── Workflow Session ──\n")

    # ── 3. Submit rich EAT events ─────────────────────────────────

    # t=1: Sarah forms initial hypothesis
    print("  [t=1] Sarah forms initial hypothesis")
    post(f"/sessions/{sid}/events", {
        "agent_id":    sarah_id,
        "agent_type":  "HUMAN",
        "event_type":  "B_UPDATE",
        "t":           1,
        "prior_belief": {
            "hypotheses":   {"expand": 0.5, "consolidate": 0.5},
            "approximated": False,
            "uncertainty":  0.0
        },
        "posterior_belief": {
            "hypotheses":   {"expand": 0.65, "consolidate": 0.35},
            "approximated": False,
            "uncertainty":  0.0
        },
        "confidence": 0.85
    })

    # t=2: Sarah builds concept graph
    print("  [t=2] Sarah maps conceptual relationships")
    post(f"/sessions/{sid}/events", {
        "agent_id":   sarah_id,
        "agent_type": "HUMAN",
        "event_type": "C_UPDATE",
        "t":          2,
        "prior_graph": {
            "nodes": {},
            "edges": [],
            "approximated": False,
            "uncertainty":  0.0
        },
        "posterior_graph": {
            "nodes": {
                "risk":     "Market Risk",
                "revenue":  "Revenue Potential",
                "market":   "Market Size",
                "timing":   "Entry Timing"
            },
            "edges": [
                {"source": "risk",   "target": "revenue", "weight": 0.7},
                {"source": "market", "target": "revenue", "weight": 0.8},
                {"source": "timing", "target": "risk",    "weight": 0.6}
            ],
            "approximated": False,
            "uncertainty":  0.0
        },
        "confidence": 0.9
    })

    # t=3: Claude introduces novel concepts
    print("  [t=3] Claude AI introduces novel concepts")
    e3 = post(f"/sessions/{sid}/events", {
        "agent_id":   claude_id,
        "agent_type": "AI",
        "event_type": "C_UPDATE",
        "t":          3,
        "prior_graph": {
            "nodes": {},
            "edges": [],
            "approximated": True,
            "uncertainty":  0.1
        },
        "posterior_graph": {
            "nodes": {
                "risk":           "Market Risk",
                "network_effect": "Network Effect",
                "competitor":     "Competitor Response",
                "innovation":     "Innovation Velocity",
                "timing":         "Entry Timing",
                "regulation":     "Regulatory Environment"
            },
            "edges": [
                {"source": "network_effect", "target": "risk",        "weight": 0.5},
                {"source": "competitor",     "target": "risk",        "weight": 0.8},
                {"source": "innovation",     "target": "network_effect", "weight": 0.7},
                {"source": "regulation",     "target": "risk",        "weight": 0.6},
                {"source": "timing",         "target": "competitor",  "weight": 0.9}
            ],
            "approximated": True,
            "uncertainty":  0.1
        },
        "confidence":         0.82,
        "groundedness":       0.72,
        "novelty_delta":      0.75,
        "influence_survival": 0.81,
        "raw_evidence":       "CoT: network effects create compounding advantage..."
    })
    claude_event_id = e3["event_id"]
    print(f"     Groundedness: {e3['verdict_scores']['groundedness']} | "
          f"Novelty: {e3['verdict_scores']['novelty_delta']}")

    # t=4: Claude updates belief
    print("  [t=4] Claude revises strategic belief")
    e4 = post(f"/sessions/{sid}/events", {
        "agent_id":   claude_id,
        "agent_type": "AI",
        "event_type": "B_UPDATE",
        "t":          4,
        "prior_belief": {
            "hypotheses":   {"expand": 0.5, "consolidate": 0.5},
            "approximated": True,
            "uncertainty":  0.15
        },
        "posterior_belief": {
            "hypotheses":   {"expand": 0.78, "consolidate": 0.22},
            "approximated": True,
            "uncertainty":  0.1
        },
        "confidence":         0.80,
        "groundedness":       0.68,
        "calibration_score":  0.74,
        "novelty_delta":      0.60,
        "influence_survival": 0.77
    })
    print(f"     Calibration: {e4['verdict_scores']['calibration_score']} | "
          f"Influence: {e4['verdict_scores']['influence_survival']}")

    # t=5: James triggered by Claude — B_TRIGGER
    print("  [t=5] James updates belief influenced by Claude")
    post(f"/sessions/{sid}/events", {
        "agent_id":     james_id,
        "agent_type":   "HUMAN",
        "event_type":   "B_TRIGGER",
        "t":            5,
        "prior_belief": {
            "hypotheses":   {"expand": 0.55, "consolidate": 0.45},
            "approximated": False,
            "uncertainty":  0.0
        },
        "posterior_belief": {
            "hypotheses":   {"expand": 0.80, "consolidate": 0.20},
            "approximated": False,
            "uncertainty":  0.0
        },
        "trigger_ref":   claude_event_id,
        "trigger_agent": claude_id,
        "confidence":    0.88
    })

    # t=6: James adds domain concepts
    print("  [t=6] James adds domain-expert concepts")
    post(f"/sessions/{sid}/events", {
        "agent_id":   james_id,
        "agent_type": "HUMAN",
        "event_type": "C_UPDATE",
        "t":          6,
        "prior_graph": {
            "nodes": {},
            "edges": [],
            "approximated": False,
            "uncertainty":  0.0
        },
        "posterior_graph": {
            "nodes": {
                "risk":          "Market Risk",
                "supply_chain":  "Supply Chain Depth",
                "local_partner": "Local Partnership",
                "regulation":    "Regulatory Environment",
                "culture":       "Cultural Fit"
            },
            "edges": [
                {"source": "local_partner", "target": "regulation",  "weight": 0.8},
                {"source": "supply_chain",  "target": "risk",        "weight": 0.7},
                {"source": "culture",       "target": "local_partner", "weight": 0.6}
            ],
            "approximated": False,
            "uncertainty":  0.0
        },
        "confidence": 0.92
    })

    # t=7: Sarah revises upward after full ensemble
    print("  [t=7] Sarah revises confidence after ensemble input")
    post(f"/sessions/{sid}/events", {
        "agent_id":   sarah_id,
        "agent_type": "HUMAN",
        "event_type": "B_UPDATE",
        "t":          7,
        "prior_belief": {
            "hypotheses":   {"expand": 0.65, "consolidate": 0.35},
            "approximated": False,
            "uncertainty":  0.0
        },
        "posterior_belief": {
            "hypotheses":   {"expand": 0.87, "consolidate": 0.13},
            "approximated": False,
            "uncertainty":  0.0
        },
        "confidence": 0.91
    })

    # t=8: Claude synthesises final recommendation
    print("  [t=8] Claude synthesises final recommendation\n")
    post(f"/sessions/{sid}/events", {
        "agent_id":   claude_id,
        "agent_type": "AI",
        "event_type": "B_UPDATE",
        "t":          8,
        "prior_belief": {
            "hypotheses":   {"expand": 0.78, "consolidate": 0.22},
            "approximated": True,
            "uncertainty":  0.1
        },
        "posterior_belief": {
            "hypotheses":   {"expand": 0.91, "consolidate": 0.09},
            "approximated": True,
            "uncertainty":  0.08
        },
        "confidence":         0.87,
        "groundedness":       0.79,
        "calibration_score":  0.81,
        "novelty_delta":      0.55,
        "influence_survival": 0.84
    })

    # ── 4. Complete session ───────────────────────────────────────
    post(f"/sessions/{sid}/complete")
    time.sleep(0.5)

    # ── 5. Pull all results ───────────────────────────────────────
    print("=" * 60)
    print("  PRISM ANALYSIS RESULTS")
    print("=" * 60)

    # VERDICT
    verdict = get(f"/sessions/{sid}/verdict")
    print(f"\n── VERDICT  [AI Epistemic Quality]")
    print(f"   Grade:               {verdict.get('verdict_grade')}")
    print(f"   Composite score:     {verdict.get('composite_score')}")
    print(f"   Groundedness:        {verdict.get('mean_groundedness')}")
    print(f"   Novelty delta:       {verdict.get('mean_novelty_delta')}")
    print(f"   Influence survival:  {verdict.get('mean_influence_survival')}")
    print(f"   Total AI events:     {verdict.get('total_ai_events')}")

    # DECAY
    decay = get(f"/sessions/{sid}/decay")
    print(f"\n── DECAY  [Epistemic Health]")
    print(f"   Total alerts:        {decay.get('total_alerts')}")
    print(f"   Engagement rate:     {decay.get('current_engagement_rate')}")
    print(f"   Diversity index:     {decay.get('current_diversity_index')}")
    print(f"   Novelty rate:        {decay.get('current_novelty_rate')}")

    # ATLAS
    atlas = get(f"/sessions/{sid}/atlas")
    print(f"\n── ATLAS  [Causal Fingerprint]")
    print(f"   Nodes in graph:      {atlas.get('node_count')}")
    print(f"   Edges in graph:      {atlas.get('edge_count')}")
    print(f"   Coupling index:      {atlas.get('coupling_index')}")
    print(f"   Discoveries traced:  {len(atlas.get('discoveries', []))}")

    # GHOST
    ghost = post(f"/sessions/{sid}/ghost")
    print(f"\n── GHOST  [Counterfactual Analysis]")
    print(f"   Emergence score:     {ghost.get('emergence_score')}")
    print(f"   AI value score:      {ghost.get('ai_value_score')}")
    print(f"   Human value score:   {ghost.get('human_value_score')}")
    emerged = ghost.get('concept_emergence', [])
    print(f"   Emerged concepts:    {emerged[:5]}")
    print(f"   AI unique:           {list(ghost.get('ai_unique', []))[:4]}")
    print(f"   Human unique:        {list(ghost.get('human_unique', []))[:4]}")
    print(f"   Verdict:             {ghost.get('verdict', '')[:70]}")

    # COMPASS
    compass = get(f"/sessions/{sid}/compass")
    print(f"\n── COMPASS  [Prompt Optimization]")
    print(f"   Optimization cycles: {compass.get('optimization_cycles')}")
    print(f"   Total records:       {compass.get('total_records')}")

    # CHRONICLE
    report = get(f"/sessions/{sid}/report")
    print(f"\n── CHRONICLE  [Client Intelligence Report]")
    print(f"   Overall grade:       {report.get('overall_grade')}")
    print(f"   Overall score:       {report.get('overall_score')}")

    findings = (
        report.get("sections", {})
              .get("executive_summary", {})
              .get("content", {})
              .get("key_findings", [])
    )
    if findings:
        print(f"   Key findings:")
        for f in findings:
            print(f"     • {f[:70]}")

    recs = (
        report.get("sections", {})
              .get("recommendations", {})
              .get("content", {})
              .get("items", [])
    )
    if recs:
        print(f"   Top recommendation:")
        print(f"     → {recs[0]['recommendation'][:80]}")

    print(f"\n{'=' * 60}")
    print(f"  API docs:   http://localhost:8000/docs")
    print(f"  Session:    {sid}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    run_demo()