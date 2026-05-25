# PRISM

**Epistemic Observability Platform for Enterprise Human-AI Workflows**

95% of enterprise AI deployments produce zero measurable ROI.  
Not because the models are wrong.  
Because nobody built the instrument to measure whether  
humans and AI are thinking better together.

PRISM measures the quality of human-AI cognition in production —  
not outputs, not latency, not cost. **The thinking process itself.**

Built for Forward Deployed Engineers at Anthropic, OpenAI,  
Palantir, Scale AI, and anyone embedding AI into organizations  
that need proof it's working.

---

## The Problem

Every observability tool in existence measures what AI *outputs*.  
None of them answer the question every CTO actually asks:

> "Is our organization making better decisions because of this AI?"

That question has no rigorous answer today. PRISM is the first tool that provides one.

---

## Seven Systems

| System | What it does | Why it matters |
|---|---|---|
| **PULSE** | Live epistemic telemetry — captures every belief update and conceptual shift from every agent in real time | Flight recorder for human-AI workflows |
| **VERDICT** | Scores every AI event across 4 dimensions: groundedness, calibration, influence survival, novelty delta | No other eval tool measures these in production |
| **DECAY** | Detects epistemic degradation — when humans stop thinking critically and start rubber-stamping AI | Catches silent deployment failure before the client notices |
| **ATLAS** | Causal discovery fingerprint — directed graph tracing exactly where every insight came from | Proof of value for client renewals |
| **GHOST** | Counterfactual replay — removes AI or humans from recorded sessions to isolate what each contributed | Answers "did we actually need the AI?" |
| **COMPASS** | DSPy closed-loop prompt optimizer — improves AI prompts automatically based on VERDICT scores | Self-improving deployment system |
| **CHRONICLE** | One-command client intelligence report — 20-page PDF from every session | The document that makes clients renew |

---

## Quickstart

```bash
git clone https://github.com/YOUR_GITHUB/prism
cd prism
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Instrument any workflow in 3 decorators:**

```python
from sdk.session import WorkflowSession

with WorkflowSession(client='acme', workflow='q3-analysis') as session:

    @session.human('analyst-1', role='domain-expert')
    def analyst_hypothesis(claim: str, confidence: float):
        return {claim: confidence}

    @session.ai(model='claude-sonnet-4-6', provider='anthropic')
    def ai_inference(prompt: str) -> str:
        return claude_client.complete(prompt)

    # Run your workflow — PRISM captures everything automatically
    analyst_hypothesis(claim='expand-market', confidence=0.7)
    ai_inference('What does the data suggest?')

# Reports available instantly
report = session.report()   # Full Chronicle PDF
graph  = session.atlas()    # Causal fingerprint
ghost  = session.ghost()    # Counterfactual analysis
```

---

## VERDICT — Four-Dimensional AI Evaluation

Every AI epistemic event scored live across four dimensions  
no other tool measures in production:

Groundedness      [0.0 → 1.0]  Is this belief anchored in session evidence?
Calibration       [0.0 → 1.0]  When confident, is it right more often?
Influence Survival [0.0 → 1.0]  Do AI-triggered human beliefs survive scrutiny?
Novelty Delta     [0.0 → 1.0]  Did AI introduce concepts humans didn't have?

---

## GHOST — Counterfactual Replay Engine

The feature that exists nowhere else.

FULL SESSION   → humans + AI     → 47 concepts, emergence score: 0.74
HUMANS ONLY    → no AI           → 31 concepts
AI ONLY        → no humans       → 28 concepts
EMERGED        → neither alone   → 12 concepts (25% of full session)

**12 concepts only existed because of the collaboration.**  
That is the emergence signature.  
That is what you show a CTO when they ask if the AI was worth it.

---

## DECAY — Epistemic Degradation Detection

Three signals tracked continuously:

- **Critical Engagement Rate** — are humans still challenging AI outputs?
- **Belief Diversity Index** — are human beliefs converging toward AI outputs?
- **Novelty Decay Curve** — is the ensemble still producing new insights?

When all three decline → `COMPOSITE CRITICAL` alert fires.  
The FDE gets a specific, data-backed workflow recommendation  
before the client notices the AI stopped adding value.

---

## API

```bash
# Full interactive docs
http://localhost:8000/docs

POST /sessions/                     Create session
POST /sessions/{id}/agents/human    Register human agent
POST /sessions/{id}/agents/ai       Register AI agent
POST /sessions/{id}/events          Stream EAT events
GET  /sessions/{id}/verdict         Real-time VERDICT scores
GET  /sessions/{id}/decay           Epistemic health
GET  /sessions/{id}/atlas           Causal fingerprint
POST /sessions/{id}/ghost           Counterfactual analysis
GET  /sessions/{id}/report          Chronicle report
GET  /sessions/{id}/compass         Optimization history
```

---

## Architecture
prism/
├── core/
│   └── eat/          ← Epistemic Action Trace schema (Pydantic, zero deps)
├── pulse/            ← Live telemetry engine
├── verdict/          ← 4-dimensional eval scoring
├── decay/            ← Epistemic degradation detection
├── atlas/            ← Causal discovery fingerprint (NetworkX)
├── ghost/            ← Counterfactual replay engine
├── compass/          ← DSPy closed-loop optimizer
├── chronicle/        ← Client report generator
├── api/              ← FastAPI REST layer
├── sdk/              ← 3-decorator instrumentation SDK
└── deploy/           ← Docker + Kubernetes manifests

---

## Performance Targets

| Metric | Target |
|---|---|
| EAT event capture latency | < 80ms |
| Ghost Runner (3 configurations) | < 10 seconds |
| Chronicle report generation | < 30 seconds |
| New model adapter integration | < 2 hours |
| New client environment provisioning | < 5 minutes |

---

## Stack

Python 3.11+ · FastAPI · Pydantic v2 · NetworkX · NumPy · SciPy  
Kafka · PostgreSQL · MLflow · Docker · Kubernetes · DSPy

---

## Context

Built as part of research into measurement frameworks for  
human-AI collective intelligence. Companion specification:  
[HACI-M Stage 1 — Translation Substrate](docs/HACI-M_Stage1.md)

The measurement problem this solves:  
[The Collaboration Gap (arxiv 2511.02687)](https://arxiv.org/abs/2511.02687)

---

*PRISM is under active development. Contributions welcome.*
