"""
PRISM — demo_live.py
Live Demo with Real Groq API

Uses Groq's ultra-fast inference for real AI responses.
Get your free API key at: https://console.groq.com

Setup:
    pip install groq
    $env:GROQ_API_KEY="your-key-here"
    python demo_live.py
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import threading
import time
import requests
import uvicorn
from api.main import app
from adapters.groq.adapter import GroqAdapter


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


def run_live_demo():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY not set.")
        print("Run: $env:GROQ_API_KEY=\"your-key-here\"")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  PRISM — Live Demo with Groq API")
    print("  Model: llama-3.1-8b-instant")
    print("=" * 60 + "\n")

    adapter = GroqAdapter(
        api_key = api_key,
        model   = "llama-3.1-8b-instant"  # High rate limit tier
    )
    print(f"[PRISM] Adapter: {adapter}\n")

    # ── Create session ────────────────────────────────────────────
    session = post("/sessions/", {
        "client_id":   "live-demo-corp",
        "workflow_id": "strategic-analysis-groq"
    })
    sid = session["session_id"]
    print(f"[PRISM] Session: {sid}\n")

    # ── Register agents ───────────────────────────────────────────
    sarah = post(f"/sessions/{sid}/agents/human",
                 {"name": "Sarah Chen",   "role": "analyst"})
    james = post(f"/sessions/{sid}/agents/human",
                 {"name": "James Okafor", "role": "domain-expert"})
    llama = post(f"/sessions/{sid}/agents/ai", {
        "model_name":     "llama-3.1-8b-instant",
        "provider":       "groq",
        "initial_prompt": adapter.system
    })

    h1_id = sarah["agent_id"]
    h2_id = james["agent_id"]
    ai_id = llama["agent_id"]

    print(f"[PRISM] Agents: {h1_id} | {h2_id} | {ai_id}\n")
    print("── Workflow Session ──\n")

    problem = (
        "Should our fintech startup expand to Southeast Asia "
        "in the next 12 months given our 18-month runway?"
    )

    # ── t=1: Sarah forms initial hypothesis ──────────────────────
    print("  [t=1] Sarah forms initial hypothesis")
    post(f"/sessions/{sid}/events", {
        "agent_id":   h1_id,
        "agent_type": "HUMAN",
        "event_type": "B_UPDATE",
        "t":          1,
        "prior_belief": {
            "hypotheses":   {
                "expand_now":  0.5,
                "wait":        0.3,
                "dont_expand": 0.2
            },
            "approximated": False,
            "uncertainty":  0.0
        },
        "posterior_belief": {
            "hypotheses":   {
                "expand_now":  0.6,
                "wait":        0.3,
                "dont_expand": 0.1
            },
            "approximated": False,
            "uncertainty":  0.0
        },
        "confidence": 0.75
    })
    time.sleep(1)

    # ── t=2: Groq analyses strategy ───────────────────────────────
    print("  [t=2] Llama-3.1-8b analyses strategy (real API call)...")
    ai_response = adapter.complete(
        f"You are advising a fintech startup. {problem} "
        f"Analyse the key strategic factors in 3-4 sentences."
    )
    print(f"     Llama: {ai_response[:90]}...")
    time.sleep(2)

    # Extract concepts
    concepts = adapter.extract_concepts(ai_response, max_concepts=6)
    print(f"     Concepts: {list(concepts.keys())}")
    time.sleep(2)

    # Get distribution
    distribution = adapter.get_output_distribution(
        prompt     = f"{problem}\n\nAnalysis: {ai_response}",
        hypotheses = ["expand_now", "wait", "dont_expand"]
    )
    print(f"     Distribution: {distribution}")
    time.sleep(2)

    # Submit AI concept update
    post(f"/sessions/{sid}/events", {
        "agent_id":   ai_id,
        "agent_type": "AI",
        "event_type": "C_UPDATE",
        "t":          2,
        "prior_graph": {
            "nodes": {}, "edges": [],
            "approximated": True, "uncertainty": 0.1
        },
        "posterior_graph": {
            "nodes": concepts if concepts else {
                "market_risk":    "Market Risk",
                "regulatory":     "Regulatory",
                "network_effect": "Network Effect",
                "timing":         "Entry Timing"
            },
            "edges": [
                {
                    "source": list(concepts.keys())[i],
                    "target": list(concepts.keys())[i+1],
                    "weight": 0.6
                }
                for i in range(min(len(concepts)-1, 3))
            ] if len(concepts) > 1 else [
                {"source": "market_risk", "target": "timing",    "weight": 0.7},
                {"source": "regulatory",  "target": "market_risk", "weight": 0.6},
            ],
            "approximated": True, "uncertainty": 0.1
        },
        "confidence":    0.82,
        "novelty_delta": 0.75
    })

    # Submit AI belief update
    e3 = post(f"/sessions/{sid}/events", {
        "agent_id":   ai_id,
        "agent_type": "AI",
        "event_type": "B_UPDATE",
        "t":          3,
        "prior_belief": {
            "hypotheses":   {
                "expand_now":  0.5,
                "wait":        0.3,
                "dont_expand": 0.2
            },
            "approximated": True,
            "uncertainty":  0.15
        },
        "posterior_belief": {
            "hypotheses":   distribution if distribution else {
                "expand_now":  0.6,
                "wait":        0.3,
                "dont_expand": 0.1
            },
            "approximated": True,
            "uncertainty":  0.1
        },
        "confidence":         0.82,
        "groundedness":       0.71,
        "calibration_score":  0.74,
        "novelty_delta":      0.68,
        "influence_survival": 0.79
    })
    claude_event_id = e3["event_id"]
    print(f"     VERDICT — Groundedness: {e3['verdict_scores']['groundedness']}")
    time.sleep(2)

    # ── t=4: Chain of thought ─────────────────────────────────────
    print(f"\n  [t=4] Llama generates chain-of-thought (real API call)...")
    cot = adapter.get_chain_of_thought(
        f"What are the top 3 risks of expanding to Southeast Asia "
        f"for a fintech startup with 18-month runway?"
    )
    if cot:
        print(f"     CoT: {cot[:100]}...")
    else:
        print(f"     CoT: [rate limited — skipped]")
    time.sleep(2)

    # ── t=5: James triggered by AI ───────────────────────────────
    print(f"\n  [t=5] James updates belief after Llama's analysis")
    post(f"/sessions/{sid}/events", {
        "agent_id":     h2_id,
        "agent_type":   "HUMAN",
        "event_type":   "B_TRIGGER",
        "t":            5,
        "prior_belief": {
            "hypotheses":   {
                "expand_now":  0.4,
                "wait":        0.4,
                "dont_expand": 0.2
            },
            "approximated": False,
            "uncertainty":  0.0
        },
        "posterior_belief": {
            "hypotheses":   {
                "expand_now":  0.65,
                "wait":        0.25,
                "dont_expand": 0.1
            },
            "approximated": False,
            "uncertainty":  0.0
        },
        "trigger_ref":   claude_event_id,
        "trigger_agent": ai_id,
        "confidence":    0.88
    })
    time.sleep(2)

    # ── t=6: Final calibrated recommendation ─────────────────────
    print(f"\n  [t=6] Llama final recommendation (real API call)...")
    final = adapter.complete_with_confidence(
        f"Given team analysis of: '{problem}' "
        f"give a final strategic recommendation in 2 sentences."
    )
    print(f"     Response:   {final['response'][:90]}...")
    print(f"     Confidence: {final['confidence']:.0%}")
    time.sleep(1)

    post(f"/sessions/{sid}/events", {
        "agent_id":   ai_id,
        "agent_type": "AI",
        "event_type": "B_UPDATE",
        "t":          6,
        "prior_belief": {
            "hypotheses":   distribution if distribution else {
                "expand_now":  0.6,
                "wait":        0.3,
                "dont_expand": 0.1
            },
            "approximated": True,
            "uncertainty":  0.1
        },
        "posterior_belief": {
            "hypotheses": {
                "expand_now":  final["confidence"],
                "wait":        round((1 - final["confidence"]) * 0.6, 3),
                "dont_expand": round((1 - final["confidence"]) * 0.4, 3)
            },
            "approximated": True,
            "uncertainty":  0.08
        },
        "confidence":         final["confidence"],
        "groundedness":       0.78,
        "calibration_score":  0.81,
        "novelty_delta":      0.55,
        "influence_survival": 0.83
    })

    # ── t=7: Sarah final revision ─────────────────────────────────
    print(f"\n  [t=7] Sarah final belief revision")
    post(f"/sessions/{sid}/events", {
        "agent_id":   h1_id,
        "agent_type": "HUMAN",
        "event_type": "B_UPDATE",
        "t":          7,
        "prior_belief": {
            "hypotheses":   {
                "expand_now":  0.6,
                "wait":        0.3,
                "dont_expand": 0.1
            },
            "approximated": False,
            "uncertainty":  0.0
        },
        "posterior_belief": {
            "hypotheses":   {
                "expand_now":  0.82,
                "wait":        0.13,
                "dont_expand": 0.05
            },
            "approximated": False,
            "uncertainty":  0.0
        },
        "confidence": 0.91
    })

    post(f"/sessions/{sid}/complete")
    time.sleep(0.5)

    # ── Results ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PRISM ANALYSIS — LIVE GROQ SESSION")
    print("=" * 60)

    verdict = get(f"/sessions/{sid}/verdict")
    print(f"\n── VERDICT  [AI Epistemic Quality]")
    print(f"   Grade:              {verdict.get('verdict_grade')}")
    print(f"   Composite:          {verdict.get('composite_score')}")
    print(f"   Groundedness:       {verdict.get('mean_groundedness')}")
    print(f"   Novelty delta:      {verdict.get('mean_novelty_delta')}")
    print(f"   Influence survival: {verdict.get('mean_influence_survival')}")

    decay = get(f"/sessions/{sid}/decay")
    print(f"\n── DECAY  [Epistemic Health]")
    print(f"   Total alerts:       {decay.get('total_alerts')}")
    print(f"   Engagement rate:    {decay.get('current_engagement_rate')}")
    print(f"   Diversity index:    {decay.get('current_diversity_index')}")
    print(f"   Novelty rate:       {decay.get('current_novelty_rate')}")

    atlas = get(f"/sessions/{sid}/atlas")
    print(f"\n── ATLAS  [Causal Fingerprint]")
    print(f"   Nodes:              {atlas.get('node_count')}")
    print(f"   Edges:              {atlas.get('edge_count')}")
    print(f"   Coupling index:     {atlas.get('coupling_index')}")
    print(f"   Discoveries:        {len(atlas.get('discoveries', []))}")

    ghost = post(f"/sessions/{sid}/ghost")
    print(f"\n── GHOST  [Counterfactual Analysis]")
    print(f"   Emergence score:    {ghost.get('emergence_score')}")
    print(f"   AI value:           {ghost.get('ai_value_score')}")
    print(f"   Human value:        {ghost.get('human_value_score')}")
    print(f"   AI unique:          {list(ghost.get('ai_unique', []))[:4]}")
    print(f"   Human unique:       {list(ghost.get('human_unique', []))[:4]}")
    print(f"   Verdict:            {ghost.get('verdict', '')[:70]}")

    compass = get(f"/sessions/{sid}/compass")
    print(f"\n── COMPASS  [Optimization]")
    print(f"   Cycles:             {compass.get('optimization_cycles')}")
    print(f"   Records:            {compass.get('total_records')}")

    report = get(f"/sessions/{sid}/report")
    print(f"\n── CHRONICLE  [Client Report]")
    print(f"   Overall grade:      {report.get('overall_grade')}")
    print(f"   Overall score:      {report.get('overall_score')}")

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

    print(f"\n{'=' * 60}")
    print(f"  API docs: http://localhost:8000/docs")
    print(f"  Session:  {sid}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    run_live_demo()