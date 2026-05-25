"""
PRISM — tests/test_api.py
FastAPI REST Layer Integration Tests

Run: pytest tests/test_api.py -v
"""

import pytest
import threading
import time
import uvicorn
from fastapi.testclient import TestClient
from api.main import app


# ─── TEST CLIENT ──────────────────────────────────────────────────────────────

client = TestClient(app)


# ─── FIXTURES ─────────────────────────────────────────────────────────────────

def create_session(client_id="test-corp", workflow_id="wf-test"):
    resp = client.post("/sessions/", json={
        "client_id":   client_id,
        "workflow_id": workflow_id
    })
    assert resp.status_code == 200
    return resp.json()


def register_human(session_id, name="Alice", role="analyst"):
    resp = client.post(f"/sessions/{session_id}/agents/human", json={
        "name": name,
        "role": role
    })
    assert resp.status_code == 200
    return resp.json()


def register_ai(session_id, model="claude-sonnet-4-6", provider="anthropic"):
    resp = client.post(f"/sessions/{session_id}/agents/ai", json={
        "model_name": model,
        "provider":   provider,
        "initial_prompt": "You are a helpful AI assistant."
    })
    assert resp.status_code == 200
    return resp.json()


def submit_b_update(session_id, agent_id, agent_type, t,
                     prior_h, post_h, approximated=False, **kwargs):
    payload = {
        "agent_id":    agent_id,
        "agent_type":  agent_type,
        "event_type":  "B_UPDATE",
        "t":           t,
        "prior_belief": {
            "hypotheses":   prior_h,
            "approximated": approximated,
            "uncertainty":  0.1 if approximated else 0.0
        },
        "posterior_belief": {
            "hypotheses":   post_h,
            "approximated": approximated,
            "uncertainty":  0.1 if approximated else 0.0
        },
        "confidence": kwargs.get("confidence", 0.9)
    }
    for k in ["groundedness", "novelty_delta",
              "influence_survival", "calibration_score"]:
        if k in kwargs:
            payload[k] = kwargs[k]
    resp = client.post(f"/sessions/{session_id}/events", json=payload)
    return resp


def submit_c_update(session_id, agent_id, agent_type, t,
                     prior_nodes, prior_edges,
                     post_nodes,  post_edges,
                     approximated=False):
    resp = client.post(f"/sessions/{session_id}/events", json={
        "agent_id":   agent_id,
        "agent_type": agent_type,
        "event_type": "C_UPDATE",
        "t":          t,
        "prior_graph": {
            "nodes": prior_nodes, "edges": prior_edges,
            "approximated": approximated, "uncertainty": 0.0
        },
        "posterior_graph": {
            "nodes": post_nodes, "edges": post_edges,
            "approximated": approximated, "uncertainty": 0.0
        },
        "confidence": 0.9
    })
    return resp


def build_full_session():
    """Create a complete session with agents and events."""
    session  = create_session()
    sid      = session["session_id"]
    human    = register_human(sid, "Sarah Chen",   "analyst")
    human2   = register_human(sid, "James Okafor", "domain-expert")
    ai_agent = register_ai(sid)

    h1_id = human["agent_id"]
    h2_id = human2["agent_id"]
    ai_id = ai_agent["agent_id"]

    # t=1: Human belief update
    submit_b_update(sid, h1_id, "HUMAN", 1,
                    {"expand": 0.5, "consolidate": 0.5},
                    {"expand": 0.7, "consolidate": 0.3})

    # t=2: Human concept graph
    submit_c_update(sid, h1_id, "HUMAN", 2,
                    {}, [],
                    {"risk": "Risk", "revenue": "Revenue", "market": "Market"},
                    [{"source": "risk", "target": "revenue", "weight": 0.7}])

    # t=3: AI concept graph with novel concepts
    submit_c_update(sid, ai_id, "AI", 3,
                    {}, [],
                    {"risk":           "Risk",
                     "network_effect": "Network Effect",
                     "innovation":     "Innovation"},
                    [{"source": "innovation",
                      "target": "network_effect", "weight": 0.8}],
                    approximated=True)

    # t=4: AI belief update
    r = submit_b_update(
        sid, ai_id, "AI", 4,
        {"expand": 0.5, "consolidate": 0.5},
        {"expand": 0.8, "consolidate": 0.2},
        approximated     = True,
        groundedness     = 0.72,
        novelty_delta    = 0.65,
        influence_survival = 0.80,
        calibration_score  = 0.74
    )

    # t=5: Human 2 concept graph
    submit_c_update(sid, h2_id, "HUMAN", 5,
                    {}, [],
                    {"risk":         "Risk",
                     "supply_chain": "Supply Chain",
                     "culture":      "Culture"},
                    [{"source": "supply_chain",
                      "target": "risk", "weight": 0.6}])

    # t=6: Human 1 revises upward
    submit_b_update(sid, h1_id, "HUMAN", 6,
                    {"expand": 0.7, "consolidate": 0.3},
                    {"expand": 0.9, "consolidate": 0.1})

    return sid, h1_id, h2_id, ai_id


# ─── HEALTH ───────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_check(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"]  == "healthy"
        assert data["version"] == "0.1.0"
        assert "sessions"      in data
        assert "timestamp"     in data


# ─── SESSION MANAGEMENT ───────────────────────────────────────────────────────

class TestSessionManagement:

    def test_create_session(self):
        resp = client.post("/sessions/", json={
            "client_id":   "acme",
            "workflow_id": "wf-001"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"].startswith("S-")
        assert data["client_id"]   == "acme"
        assert data["workflow_id"] == "wf-001"
        assert data["status"]      == "ACTIVE"

    def test_get_session(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.get(f"/sessions/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid

    def test_get_nonexistent_session(self):
        resp = client.get("/sessions/S-nonexistent")
        assert resp.status_code == 404

    def test_list_sessions(self):
        create_session("list-test-corp", "wf-list")
        resp = client.get("/sessions/")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "total"    in data
        assert data["total"] >= 1

    def test_list_sessions_filter_by_client(self):
        unique_client = "unique-client-xyz-123"
        create_session(unique_client, "wf-filter")
        resp = client.get(f"/sessions/?client_id={unique_client}")
        assert resp.status_code == 200
        data = resp.json()
        for s in data["sessions"]:
            assert s["client_id"] == unique_client

    def test_complete_session(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.post(f"/sessions/{sid}/complete")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "COMPLETED"


# ─── AGENT REGISTRATION ───────────────────────────────────────────────────────

class TestAgentRegistration:

    def test_register_human_agent(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.post(f"/sessions/{sid}/agents/human", json={
            "name": "Alice",
            "role": "analyst"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"].startswith("H-")
        assert data["agent_type"] == "HUMAN"
        assert data["name"]       == "Alice"
        assert data["role"]       == "analyst"

    def test_register_ai_agent(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.post(f"/sessions/{sid}/agents/ai", json={
            "model_name": "claude-sonnet-4-6",
            "provider":   "anthropic"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"].startswith("A-")
        assert data["agent_type"]  == "AI"
        assert data["model_name"]  == "claude-sonnet-4-6"
        assert data["provider"]    == "anthropic"

    def test_register_multiple_humans(self):
        session = create_session()
        sid     = session["session_id"]
        h1 = register_human(sid, "Alice", "analyst")
        h2 = register_human(sid, "Bob",   "expert")
        assert h1["agent_id"] != h2["agent_id"]

    def test_register_agent_nonexistent_session(self):
        resp = client.post("/sessions/S-nonexistent/agents/human", json={
            "name": "Alice",
            "role": "analyst"
        })
        assert resp.status_code == 404


# ─── EVENT SUBMISSION ─────────────────────────────────────────────────────────

class TestEventSubmission:

    def test_submit_b_update_event(self):
        session = create_session()
        sid     = session["session_id"]
        human   = register_human(sid)
        resp    = submit_b_update(
            sid, human["agent_id"], "HUMAN", 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "event_id"        in data
        assert "delta_magnitude" in data
        assert "verdict_scores"  in data
        assert data["delta_magnitude"] >= 0

    def test_submit_c_update_event(self):
        session = create_session()
        sid     = session["session_id"]
        human   = register_human(sid)
        resp    = submit_c_update(
            sid, human["agent_id"], "HUMAN", 1,
            {}, [],
            {"risk": "Risk", "revenue": "Revenue"},
            [{"source": "risk", "target": "revenue", "weight": 0.7}]
        )
        assert resp.status_code == 200

    def test_submit_ai_event_returns_verdict_scores(self):
        session  = create_session()
        sid      = session["session_id"]
        ai_agent = register_ai(sid)
        resp     = submit_b_update(
            sid, ai_agent["agent_id"], "AI", 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.3, "H2": 0.7},
            approximated     = True,
            groundedness     = 0.7,
            novelty_delta    = 0.6,
            influence_survival = 0.8,
            calibration_score  = 0.75
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "verdict_scores" in data

    def test_submit_invalid_event_type(self):
        session = create_session()
        sid     = session["session_id"]
        human   = register_human(sid)
        resp    = client.post(f"/sessions/{sid}/events", json={
            "agent_id":   human["agent_id"],
            "agent_type": "HUMAN",
            "event_type": "INVALID_TYPE",
            "t":          1,
            "confidence": 0.9
        })
        assert resp.status_code == 422

    def test_submit_event_nonexistent_session(self):
        resp = client.post("/sessions/S-nonexistent/events", json={
            "agent_id":   "H-001",
            "agent_type": "HUMAN",
            "event_type": "B_UPDATE",
            "t":          1
        })
        assert resp.status_code == 404

    def test_get_events(self):
        session = create_session()
        sid     = session["session_id"]
        human   = register_human(sid)
        submit_b_update(
            sid, human["agent_id"], "HUMAN", 1,
            {"H1": 0.5, "H2": 0.5},
            {"H1": 0.7, "H2": 0.3}
        )
        resp = client.get(f"/sessions/{sid}/events")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["events"]) == 1

    def test_get_events_filter_by_agent(self):
        session = create_session()
        sid     = session["session_id"]
        h1      = register_human(sid, "Alice", "analyst")
        h2      = register_human(sid, "Bob",   "expert")
        submit_b_update(sid, h1["agent_id"], "HUMAN", 1,
                        {"H1": 0.5, "H2": 0.5}, {"H1": 0.7, "H2": 0.3})
        submit_b_update(sid, h2["agent_id"], "HUMAN", 2,
                        {"H1": 0.5, "H2": 0.5}, {"H1": 0.3, "H2": 0.7})
        resp = client.get(
            f"/sessions/{sid}/events?agent_id={h1['agent_id']}"
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["events"][0]["agent_id"] == h1["agent_id"]


# ─── VERDICT ENDPOINT ─────────────────────────────────────────────────────────

class TestVerdictEndpoint:

    def test_verdict_empty_session(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.get(f"/sessions/{sid}/verdict")
        assert resp.status_code == 200
        data = resp.json()
        assert data["verdict_grade"]   == "NO_DATA"
        assert data["total_ai_events"] == 0

    def test_verdict_with_ai_events(self):
        sid, h1_id, h2_id, ai_id = build_full_session()
        resp = client.get(f"/sessions/{sid}/verdict")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_ai_events"] > 0
        assert data["verdict_grade"] in (
            "EXCELLENT", "GOOD", "MODERATE", "POOR", "NO_DATA"
        )


# ─── DECAY ENDPOINT ───────────────────────────────────────────────────────────

class TestDecayEndpoint:

    def test_decay_structure(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.get(f"/sessions/{sid}/decay")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_alerts"         in data
        assert "current_novelty_rate" in data

    def test_decay_with_events(self):
        sid, h1_id, h2_id, ai_id = build_full_session()
        resp = client.get(f"/sessions/{sid}/decay")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_alerts" in data
        assert isinstance(data["total_alerts"], int)


# ─── ATLAS ENDPOINT ───────────────────────────────────────────────────────────

class TestAtlasEndpoint:

    def test_atlas_empty_session(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.get(f"/sessions/{sid}/atlas")
        assert resp.status_code == 200
        data = resp.json()
        assert "node_count"    in data
        assert "edge_count"    in data
        assert "coupling_index" in data

    def test_atlas_with_events(self):
        sid, h1_id, h2_id, ai_id = build_full_session()
        resp = client.get(f"/sessions/{sid}/atlas")
        assert resp.status_code == 200
        data = resp.json()
        assert data["node_count"] > 0
        assert "discoveries"    in data
        assert "nodes"          in data
        assert "edges"          in data


# ─── GHOST ENDPOINT ───────────────────────────────────────────────────────────

class TestGhostEndpoint:

    def test_ghost_requires_events(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.post(f"/sessions/{sid}/ghost")
        assert resp.status_code == 400

    def test_ghost_with_full_session(self):
        sid, h1_id, h2_id, ai_id = build_full_session()
        resp = client.post(f"/sessions/{sid}/ghost")
        assert resp.status_code == 200
        data = resp.json()
        assert "emergence_score"   in data
        assert "ai_value_score"    in data
        assert "human_value_score" in data
        assert "verdict"           in data
        assert "recommendation"    in data

    def test_ghost_scores_in_range(self):
        sid, h1_id, h2_id, ai_id = build_full_session()
        resp = client.post(f"/sessions/{sid}/ghost")
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["emergence_score"]   <= 1.0
        assert 0.0 <= data["ai_value_score"]    <= 1.0
        assert 0.0 <= data["human_value_score"] <= 1.0


# ─── COMPASS ENDPOINT ─────────────────────────────────────────────────────────

class TestCompassEndpoint:

    def test_compass_structure(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.get(f"/sessions/{sid}/compass")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_sessions"      in data
        assert "optimization_cycles" in data
        assert "total_records"       in data

    def test_compass_nonexistent_session(self):
        resp = client.get("/sessions/S-nonexistent/compass")
        assert resp.status_code == 404


# ─── CHRONICLE ENDPOINT ───────────────────────────────────────────────────────

class TestChronicleEndpoint:

    def test_report_structure(self):
        session = create_session()
        sid     = session["session_id"]
        resp    = client.get(f"/sessions/{sid}/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "report_id"      in data
        assert "overall_grade"  in data
        assert "overall_score"  in data
        assert "sections"       in data

    def test_report_sections_present(self):
        sid, h1_id, h2_id, ai_id = build_full_session()
        resp = client.get(f"/sessions/{sid}/report")
        assert resp.status_code == 200
        data     = resp.json()
        sections = data["sections"]
        assert "executive_summary" in sections
        assert "verdict"           in sections
        assert "decay"             in sections
        assert "atlas"             in sections
        assert "ghost"             in sections
        assert "compass"           in sections
        assert "recommendations"   in sections

    def test_report_grade_valid(self):
        sid, h1_id, h2_id, ai_id = build_full_session()
        resp = client.get(f"/sessions/{sid}/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_grade"] in (
            "EXCELLENT", "GOOD", "MODERATE", "POOR"
        )

    def test_report_score_in_range(self):
        sid, h1_id, h2_id, ai_id = build_full_session()
        resp = client.get(f"/sessions/{sid}/report")
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["overall_score"] <= 1.0

    def test_report_nonexistent_session(self):
        resp = client.get("/sessions/S-nonexistent/report")
        assert resp.status_code == 404