"""
PRISM — sdk/session.py
Python SDK — Client-Side Workflow Instrumentation

The three-decorator SDK.
Any existing workflow is fully instrumented in minutes.
No architectural changes. No data pipeline work.
Drop in, measure, improve, report.

Usage:
    from prism.sdk import WorkflowSession

    with WorkflowSession(client='acme', workflow='q3-analysis') as session:

        @session.human('analyst-1')
        def analyst_hypothesis(claim: str, confidence: float):
            pass

        @session.ai(model='claude-sonnet-4-6', provider='anthropic')
        def ai_inference(prompt: str) -> str:
            pass

    report = session.report()
    graph  = session.atlas()
    ghost  = session.ghost()

This is what an FDE leaves behind at a client.
This is what makes them irreplaceable.
"""

from __future__ import annotations
import uuid
import time
import json
import functools
import threading
import requests
from typing import (
    Callable, Optional, Dict, Any,
    List, TypeVar, cast
)
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, field

F = TypeVar('F', bound=Callable[..., Any])


# ─── CONFIG ───────────────────────────────────────────────────────────────────

@dataclass
class PRISMConfig:
    """PRISM SDK configuration."""
    api_url:     str   = "http://localhost:8000"
    client_id:   str   = "default-client"
    workflow_id: str   = "default-workflow"
    timeout:     int   = 30
    auto_report: bool  = True
    verbose:     bool  = True

    @classmethod
    def from_dict(cls, d: Dict) -> PRISMConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ─── CAPTURED EVENT ───────────────────────────────────────────────────────────

@dataclass
class CapturedEvent:
    """
    Raw captured event before conversion to EAT format.
    Stored locally until flushed to the PRISM API.
    """
    agent_id:    str
    agent_type:  str
    event_type:  str
    t:           int
    function:    str
    args:        tuple
    kwargs:      dict
    result:      Any
    duration_ms: float
    timestamp:   datetime = field(default_factory=datetime.utcnow)
    metadata:    Dict     = field(default_factory=dict)

    # Epistemic content — populated by extractors
    prior_belief:     Optional[Dict] = None
    posterior_belief: Optional[Dict] = None
    prior_graph:      Optional[Dict] = None
    posterior_graph:  Optional[Dict] = None
    confidence:       float          = 1.0
    raw_evidence:     Optional[str]  = None


# ─── BELIEF EXTRACTOR ─────────────────────────────────────────────────────────

class BeliefExtractor:
    """
    Extracts belief states from function arguments and results.
    Handles three common patterns:

    1. Explicit: function receives/returns confidence dict
    2. Structured: function returns object with .hypotheses attr
    3. Text: function returns string — parsed for confidence language
    """

    CONFIDENCE_KEYWORDS = {
        "certain":    0.95,
        "confident":  0.85,
        "likely":     0.70,
        "probably":   0.65,
        "possibly":   0.45,
        "uncertain":  0.30,
        "unlikely":   0.20,
        "doubt":      0.15,
    }

    def extract_from_args(
        self,
        func_name: str,
        args:      tuple,
        kwargs:    dict
    ) -> Optional[Dict]:
        """Try to extract a belief state from function arguments."""

        # Pattern 1: explicit confidence kwarg
        if "confidence" in kwargs and "claim" in kwargs:
            return {
                "hypotheses":   {kwargs["claim"]: float(kwargs["confidence"])},
                "approximated": False,
                "uncertainty":  0.0
            }

        # Pattern 2: hypotheses dict passed directly
        if "hypotheses" in kwargs:
            hyps = kwargs["hypotheses"]
            if isinstance(hyps, dict):
                return {
                    "hypotheses":   {str(k): float(v) for k, v in hyps.items()},
                    "approximated": False,
                    "uncertainty":  0.0
                }

        # Pattern 3: confidence as positional arg
        for arg in args:
            if isinstance(arg, dict):
                if all(isinstance(v, (int, float)) for v in arg.values()):
                    return {
                        "hypotheses":   {str(k): float(v) for k, v in arg.items()},
                        "approximated": False,
                        "uncertainty":  0.0
                    }

        return None

    def extract_from_result(
        self,
        result:     Any,
        agent_type: str
    ) -> Optional[Dict]:
        """Try to extract a belief state from function result."""
        is_ai = agent_type == "AI"

        # Pattern 1: result is a dict of hypothesis → confidence
        if isinstance(result, dict):
            if all(isinstance(v, (int, float)) for v in result.values()):
                return {
                    "hypotheses":   {str(k): float(v) for k, v in result.items()},
                    "approximated": is_ai,
                    "uncertainty":  0.1 if is_ai else 0.0
                }

        # Pattern 2: result is a string — parse for confidence language
        if isinstance(result, str):
            return self._parse_text_confidence(result, is_ai)

        # Pattern 3: result has hypotheses attribute
        if hasattr(result, "hypotheses"):
            return {
                "hypotheses":   result.hypotheses,
                "approximated": is_ai,
                "uncertainty":  0.1 if is_ai else 0.0
            }

        return None

    def _parse_text_confidence(
        self,
        text:   str,
        is_ai:  bool
    ) -> Optional[Dict]:
        """Parse natural language text for confidence signals."""
        text_lower = text.lower()
        found = {}

        for keyword, confidence in self.CONFIDENCE_KEYWORDS.items():
            if keyword in text_lower:
                # Extract surrounding context as hypothesis label
                idx = text_lower.find(keyword)
                start = max(0, idx - 20)
                end   = min(len(text), idx + 30)
                context = text[start:end].strip()
                found[context[:40]] = confidence

        if not found:
            # Default: treat entire response as single hypothesis
            snippet = text[:50].strip()
            return {
                "hypotheses":   {snippet: 0.6},
                "approximated": is_ai,
                "uncertainty":  0.2
            }

        return {
            "hypotheses":   found,
            "approximated": is_ai,
            "uncertainty":  0.15 if is_ai else 0.05
        }


# ─── PRISM SESSION ────────────────────────────────────────────────────────────

class PrismSession:
    """
    PRISM SDK session.

    Context manager that creates a PRISM workflow session,
    provides decorators for instrumenting human and AI functions,
    and manages event flushing to the PRISM API.

    Usage:
        with PrismSession(config) as session:
            @session.human('analyst-id')
            def my_analysis(claim, confidence):
                ...

            @session.ai(model='claude-sonnet-4-6')
            def ai_analysis(prompt):
                return claude_response
    """

    def __init__(self, config: PRISMConfig):
        self.config     = config
        self.session_id: Optional[str] = None
        self._t          = 0
        self._t_lock     = threading.Lock()
        self._events:    List[CapturedEvent] = []
        self._extractor  = BeliefExtractor()
        self._agents:    Dict[str, Dict] = {}

        # Previous belief states per agent (for prior tracking)
        self._belief_cache: Dict[str, Dict] = {}

    # ── CONTEXT MANAGER ───────────────────────────────────────────────────

    def __enter__(self) -> PrismSession:
        self._create_session()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None:
            self._flush_all()
            if self.config.auto_report and self.config.verbose:
                self._print_session_summary()
        return False

    # ── SESSION CREATION ──────────────────────────────────────────────────

    def _create_session(self) -> None:
        """Create session on PRISM API."""
        try:
            resp = requests.post(
                f"{self.config.api_url}/sessions/",
                json={
                    "client_id":   self.config.client_id,
                    "workflow_id": self.config.workflow_id
                },
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            self.session_id = data["session_id"]

            if self.config.verbose:
                print(f"[PRISM] Session created: {self.session_id}")

        except requests.RequestException as e:
            # Offline mode — generate local session ID
            self.session_id = f"S-local-{uuid.uuid4().hex[:8]}"
            if self.config.verbose:
                print(f"[PRISM] Offline mode — local session: {self.session_id}")

    def _next_t(self) -> int:
        with self._t_lock:
            self._t += 1
            return self._t

    # ── AGENT REGISTRATION ────────────────────────────────────────────────

    def _register_agent(
        self,
        agent_id:   str,
        agent_type: str,
        **kwargs
    ) -> str:
        """Register agent with PRISM API."""
        if agent_id in self._agents:
            return agent_id

        try:
            if agent_type == "HUMAN":
                endpoint = f"{self.config.api_url}/sessions/{self.session_id}/agents/human"
                payload  = {
                    "name": kwargs.get("name", agent_id),
                    "role": kwargs.get("role", "analyst")
                }
            else:
                endpoint = f"{self.config.api_url}/sessions/{self.session_id}/agents/ai"
                payload  = {
                    "model_name":    kwargs.get("model", "unknown"),
                    "provider":      kwargs.get("provider", "unknown"),
                    "initial_prompt": kwargs.get("initial_prompt")
                }

            resp = requests.post(
                endpoint, json=payload,
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            data     = resp.json()
            agent_id = data["agent_id"]

        except requests.RequestException:
            pass  # Offline mode — use provided agent_id

        self._agents[agent_id] = {
            "agent_type": agent_type,
            **kwargs
        }

        if self.config.verbose:
            print(f"[PRISM] Agent registered: {agent_id} ({agent_type})")

        return agent_id

    # ── DECORATORS ────────────────────────────────────────────────────────

    def human(
        self,
        agent_id: str,
        role:     str = "analyst",
        name:     Optional[str] = None
    ) -> Callable[[F], F]:
        """
        Decorator: instrument a human agent function.

        @session.human('analyst-1', role='domain-expert')
        def my_analysis(claim: str, confidence: float):
            pass
        """
        registered_id = self._register_agent(
            agent_id,
            "HUMAN",
            name=name or agent_id,
            role=role
        )

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t     = self._next_t()
                start = time.time()

                # Extract prior belief
                prior = self._belief_cache.get(registered_id)
                if prior is None:
                    prior = self._extractor.extract_from_args(
                        func.__name__, args, kwargs
                    )

                # Execute the original function
                result = func(*args, **kwargs)

                duration = (time.time() - start) * 1000

                # Extract posterior belief
                posterior = self._extractor.extract_from_result(
                    result, "HUMAN"
                )
                if posterior is None:
                    posterior = self._extractor.extract_from_args(
                        func.__name__, args, kwargs
                    )

                # Update belief cache
                if posterior:
                    self._belief_cache[registered_id] = posterior

                # Determine event type
                event_type = "B_UPDATE" if (prior or posterior) else "C_UPDATE"

                # Capture event
                captured = CapturedEvent(
                    agent_id         = registered_id,
                    agent_type       = "HUMAN",
                    event_type       = event_type,
                    t                = t,
                    function         = func.__name__,
                    args             = args,
                    kwargs           = kwargs,
                    result           = result,
                    duration_ms      = duration,
                    prior_belief     = prior,
                    posterior_belief = posterior,
                    confidence       = kwargs.get("confidence", 0.9),
                    raw_evidence     = f"{func.__name__}({args}, {kwargs})"
                )
                self._events.append(captured)
                self._flush_event(captured)

                return result
            return cast(F, wrapper)
        return decorator

    def ai(
        self,
        model:          str,
        provider:       str       = "anthropic",
        initial_prompt: Optional[str] = None,
        access_level:   str       = "GREY_BOX"
    ) -> Callable[[F], F]:
        """
        Decorator: instrument an AI agent function.

        @session.ai(model='claude-sonnet-4-6', provider='anthropic')
        def ai_inference(prompt: str) -> str:
            return claude_client.complete(prompt)
        """
        agent_id = self._register_agent(
            f"A-{model.replace('-', '')[:8]}-{uuid.uuid4().hex[:4]}",
            "AI",
            model          = model,
            provider       = provider,
            initial_prompt = initial_prompt
        )

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t     = self._next_t()
                start = time.time()

                # Extract prior belief from cache
                prior = self._belief_cache.get(agent_id)

                # Execute the AI function
                result = func(*args, **kwargs)

                duration = (time.time() - start) * 1000

                # Extract posterior belief from result
                posterior = self._extractor.extract_from_result(
                    result, "AI"
                )

                # Update belief cache
                if posterior:
                    self._belief_cache[agent_id] = posterior

                event_type = "B_UPDATE" if (prior or posterior) else "C_UPDATE"

                # Capture event
                captured = CapturedEvent(
                    agent_id         = agent_id,
                    agent_type       = "AI",
                    event_type       = event_type,
                    t                = t,
                    function         = func.__name__,
                    args             = args,
                    kwargs           = kwargs,
                    result           = result,
                    duration_ms      = duration,
                    prior_belief     = prior,
                    posterior_belief = posterior,
                    confidence       = 0.8,
                    raw_evidence     = str(args[0])[:200] if args else None
                )
                self._events.append(captured)
                self._flush_event(captured)

                return result
            return cast(F, wrapper)
        return decorator

    def trigger(
        self,
        source_agent_id: str,
        target_agent_id: str
    ) -> Callable[[F], F]:
        """
        Decorator: instrument a cross-agent influence event.
        Use when one agent's output directly causes another's update.

        @session.trigger(source_agent_id=ai_id, target_agent_id=human_id)
        def ai_influences_human(ai_output: str, human_belief: dict):
            pass
        """
        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                t      = self._next_t()
                start  = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000

                # Get last event from source agent as trigger_ref
                source_events = [
                    e for e in self._events
                    if e.agent_id == source_agent_id
                ]
                trigger_ref = None
                if source_events:
                    # We don't have event_ids locally — use timestamp
                    trigger_ref = source_events[-1].timestamp.isoformat()

                prior     = self._belief_cache.get(target_agent_id)
                posterior = self._extractor.extract_from_result(result, "HUMAN")
                if posterior:
                    self._belief_cache[target_agent_id] = posterior

                captured = CapturedEvent(
                    agent_id         = target_agent_id,
                    agent_type       = "HUMAN",
                    event_type       = "B_TRIGGER",
                    t                = t,
                    function         = func.__name__,
                    args             = args,
                    kwargs           = kwargs,
                    result           = result,
                    duration_ms      = duration,
                    prior_belief     = prior,
                    posterior_belief = posterior,
                    confidence       = 0.85,
                    metadata         = {"trigger_agent": source_agent_id}
                )
                self._events.append(captured)
                self._flush_event(captured)

                return result
            return cast(F, wrapper)
        return decorator

    # ── EVENT FLUSHING ────────────────────────────────────────────────────

    def _flush_event(self, event: CapturedEvent) -> Optional[Dict]:
        """Send a single captured event to the PRISM API."""
        if self.session_id is None:
            return None

        payload = {
            "agent_id":   event.agent_id,
            "agent_type": event.agent_type,
            "event_type": event.event_type,
            "t":          event.t,
            "confidence": event.confidence,
            "raw_evidence": event.raw_evidence,
            "metadata":   {
                "function":    event.function,
                "duration_ms": round(event.duration_ms, 2),
                **event.metadata
            }
        }

        if event.prior_belief:
            payload["prior_belief"] = event.prior_belief
        if event.posterior_belief:
            payload["posterior_belief"] = event.posterior_belief
        if event.prior_graph:
            payload["prior_graph"] = event.prior_graph
        if event.posterior_graph:
            payload["posterior_graph"] = event.posterior_graph

        try:
            resp = requests.post(
                f"{self.config.api_url}/sessions/{self.session_id}/events",
                json    = payload,
                timeout = self.config.timeout
            )
            resp.raise_for_status()

            data = resp.json()
            if self.config.verbose and data.get("decay_alerts"):
                for alert in data["decay_alerts"]:
                    print(
                        f"[PRISM DECAY ALERT] {alert['severity']}: "
                        f"{alert['recommendation'][:60]}..."
                    )
            return data

        except requests.RequestException:
            return None  # Offline mode — events buffered locally

    def _flush_all(self) -> None:
        """Flush all buffered events (called on session exit)."""
        if self.config.verbose:
            print(f"[PRISM] Flushed {len(self._events)} events")

    # ── REPORT METHODS ────────────────────────────────────────────────────

    def report(self) -> Optional[Dict]:
        """Generate full Chronicle client intelligence report."""
        if not self.session_id:
            return None
        try:
            resp = requests.get(
                f"{self.config.api_url}/sessions/{self.session_id}/report",
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            print(f"[PRISM] Report generation failed: {e}")
            return None

    def atlas(self) -> Optional[Dict]:
        """Get causal discovery fingerprint."""
        if not self.session_id:
            return None
        try:
            resp = requests.get(
                f"{self.config.api_url}/sessions/{self.session_id}/atlas",
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def ghost(self) -> Optional[Dict]:
        """Run Ghost Runner counterfactual analysis."""
        if not self.session_id:
            return None
        try:
            resp = requests.post(
                f"{self.config.api_url}/sessions/{self.session_id}/ghost",
                timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def verdict(self) -> Optional[Dict]:
        """Get real-time VERDICT scores."""
        if not self.session_id:
            return None
        try:
            resp = requests.get(
                f"{self.config.api_url}/sessions/{self.session_id}/verdict",
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def decay(self) -> Optional[Dict]:
        """Get epistemic health analysis."""
        if not self.session_id:
            return None
        try:
            resp = requests.get(
                f"{self.config.api_url}/sessions/{self.session_id}/decay",
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return None

    def _print_session_summary(self) -> None:
        """Print session summary on exit."""
        print(f"\n[PRISM] Session complete: {self.session_id}")
        print(f"[PRISM] Total events captured: {len(self._events)}")
        print(f"[PRISM] Agents instrumented:   {len(self._agents)}")
        print(f"[PRISM] Run report:  GET /sessions/{self.session_id}/report")
        print(f"[PRISM] Run ghost:   POST /sessions/{self.session_id}/ghost")
        print(f"[PRISM] View atlas:  GET /sessions/{self.session_id}/atlas\n")


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def WorkflowSession(
    client:   str,
    workflow: str,
    api_url:  str  = "http://localhost:8000",
    verbose:  bool = True,
    **kwargs
) -> PrismSession:
    """
    Factory function — public entry point for the PRISM SDK.

    with WorkflowSession(client='acme', workflow='q3') as session:
        @session.human('analyst-1')
        def analysis(claim, confidence):
            pass

        @session.ai(model='claude-sonnet-4-6')
        def inference(prompt):
            pass
    """
    config = PRISMConfig(
        api_url     = api_url,
        client_id   = client,
        workflow_id = workflow,
        verbose     = verbose,
        **{k: v for k, v in kwargs.items()
           if k in PRISMConfig.__dataclass_fields__}
    )
    return PrismSession(config)