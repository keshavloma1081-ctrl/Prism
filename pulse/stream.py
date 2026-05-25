"""
PRISM — pulse/stream.py
Live Epistemic Telemetry Engine

PULSE is the heartbeat of PRISM.
Every epistemic act from every agent — human or AI —
flows through PULSE before entering the session.

Responsibilities:
  1. Event buffering — batch EAT events before API flush
  2. Real-time scoring — trigger VERDICT on every AI event
  3. Decay monitoring — check DECAY every N events
  4. Stream statistics — track throughput and latency
  5. Replay support — emit events for Ghost Runner

Think of PULSE as the flight recorder for a thinking organization.
Sub-100ms capture latency. Zero workflow interruption.
"""

from __future__ import annotations
import time
import threading
import queue
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum

from core.eat.models import (
    EATEvent, EATEventType, AgentType,
    WorkflowSession
)
from core.eat.validators import validate_eat_event
from core.eat.delta import compute_delta_magnitude


# ─── STREAM EVENT ─────────────────────────────────────────────────────────────

class StreamEventStatus(str, Enum):
    PENDING   = "PENDING"
    VALIDATED = "VALIDATED"
    SCORED    = "SCORED"
    FLUSHED   = "FLUSHED"
    FAILED    = "FAILED"


@dataclass
class StreamEvent:
    """
    A single event in the PULSE stream.
    Wraps an EATEvent with stream metadata.
    """
    eat_event:    EATEvent
    status:       StreamEventStatus = StreamEventStatus.PENDING
    captured_at:  datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    validated_at: Optional[datetime] = None
    flushed_at:   Optional[datetime] = None
    latency_ms:   Optional[float]    = None
    errors:       List[str]          = field(default_factory=list)
    warnings:     List[str]          = field(default_factory=list)

    @property
    def capture_latency_ms(self) -> Optional[float]:
        if self.flushed_at and self.captured_at:
            delta = self.flushed_at - self.captured_at
            return round(delta.total_seconds() * 1000, 2)
        return None


# ─── STREAM STATISTICS ────────────────────────────────────────────────────────

@dataclass
class StreamStats:
    """
    Real-time statistics for a PULSE stream session.
    Updated continuously as events flow through.
    """
    total_captured:   int   = 0
    total_validated:  int   = 0
    total_flushed:    int   = 0
    total_failed:     int   = 0
    total_human:      int   = 0
    total_ai:         int   = 0
    mean_latency_ms:  float = 0.0
    max_latency_ms:   float = 0.0
    decay_alerts:     int   = 0
    started_at:       datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def uptime_seconds(self) -> float:
        delta = datetime.now(timezone.utc) - self.started_at
        return round(delta.total_seconds(), 2)

    @property
    def events_per_second(self) -> float:
        uptime = self.uptime_seconds
        if uptime == 0:
            return 0.0
        return round(self.total_captured / uptime, 2)

    @property
    def success_rate(self) -> float:
        if self.total_captured == 0:
            return 1.0
        return round(
            (self.total_flushed / self.total_captured), 4
        )

    def to_dict(self) -> Dict:
        return {
            "total_captured":  self.total_captured,
            "total_validated": self.total_validated,
            "total_flushed":   self.total_flushed,
            "total_failed":    self.total_failed,
            "total_human":     self.total_human,
            "total_ai":        self.total_ai,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "max_latency_ms":  round(self.max_latency_ms, 2),
            "decay_alerts":    self.decay_alerts,
            "uptime_seconds":  self.uptime_seconds,
            "events_per_second": self.events_per_second,
            "success_rate":    self.success_rate
        }


# ─── PULSE STREAM ─────────────────────────────────────────────────────────────

class PulseStream:
    """
    Main PULSE telemetry engine.

    Captures EAT events in real time, validates them,
    computes delta magnitudes, and flushes to the session.

    Supports:
      - Synchronous capture (immediate)
      - Asynchronous buffered capture (batched)
      - Event callbacks (for real-time dashboard updates)
      - Replay mode (for Ghost Runner)

    Usage:
        stream = PulseStream(session)
        stream.start()

        event = EATEvent(...)
        result = stream.capture(event)

        stats = stream.stats
        stream.stop()
    """

    def __init__(
        self,
        session:       WorkflowSession,
        buffer_size:   int  = 100,
        flush_interval: float = 1.0,   # seconds
        decay_check_every: int = 5,    # events
        verbose:       bool = False
    ):
        self.session            = session
        self.buffer_size        = buffer_size
        self.flush_interval     = flush_interval
        self.decay_check_every  = decay_check_every
        self.verbose            = verbose

        self._buffer:    queue.Queue = queue.Queue(maxsize=buffer_size)
        self._flushed:   List[StreamEvent] = []
        self._failed:    List[StreamEvent] = []
        self._callbacks: List[Callable] = []
        self._stats      = StreamStats()
        self._lock       = threading.Lock()
        self._running    = False
        self._thread:    Optional[threading.Thread] = None
        self._event_count = 0

    # ── LIFECYCLE ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the async flush thread."""
        self._running = True
        self._thread  = threading.Thread(
            target   = self._flush_loop,
            daemon   = True,
            name     = f"pulse-{self.session.session_id[:8]}"
        )
        self._thread.start()
        if self.verbose:
            print(f"[PULSE] Stream started — session {self.session.session_id}")

    def stop(self) -> None:
        """Stop the stream and flush remaining events."""
        self._running = False
        self._flush_buffer()
        if self.verbose:
            print(f"[PULSE] Stream stopped — {self._stats.total_flushed} events flushed")

    def __enter__(self) -> PulseStream:
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ── EVENT CAPTURE ─────────────────────────────────────────────────────

    def capture(self, event: EATEvent) -> StreamEvent:
        """
        Capture a single EAT event synchronously.

        Validates the event, computes delta magnitude,
        adds to session, and updates stats.

        Returns StreamEvent with status and any warnings.
        This is the primary ingestion method — sub-100ms target.
        """
        start_ms = time.time() * 1000
        stream_event = StreamEvent(eat_event=event)

        # ── Validate ──────────────────────────────────────────────
        validation = validate_eat_event(event, self.session)
        if not validation.passed:
            stream_event.status = StreamEventStatus.FAILED
            stream_event.errors = validation.errors
            with self._lock:
                self._failed.append(stream_event)
                self._stats.total_captured += 1
                self._stats.total_failed   += 1
            if self.verbose:
                print(f"[PULSE] Event failed validation: {validation.errors}")
            return stream_event

        stream_event.status      = StreamEventStatus.VALIDATED
        stream_event.warnings    = validation.warnings
        stream_event.validated_at = datetime.now(timezone.utc)

        # ── Compute delta magnitude ───────────────────────────────
        event.delta_magnitude = compute_delta_magnitude(event)

        # ── Add to session ────────────────────────────────────────
        self.session.add_event(event)

        # ── Update stats ──────────────────────────────────────────
        latency = time.time() * 1000 - start_ms
        stream_event.flushed_at = datetime.now(timezone.utc)
        stream_event.status     = StreamEventStatus.FLUSHED
        stream_event.latency_ms = latency

        with self._lock:
            self._flushed.append(stream_event)
            self._stats.total_captured  += 1
            self._stats.total_validated += 1
            self._stats.total_flushed   += 1
            self._event_count           += 1

            if event.agent_type == AgentType.HUMAN:
                self._stats.total_human += 1
            else:
                self._stats.total_ai += 1

            # Update latency stats
            n = self._stats.total_flushed
            self._stats.mean_latency_ms = (
                (self._stats.mean_latency_ms * (n - 1) + latency) / n
            )
            self._stats.max_latency_ms = max(
                self._stats.max_latency_ms, latency
            )

        # ── Decay check ───────────────────────────────────────────
        if self._event_count % self.decay_check_every == 0:
            self._run_decay_check(event.t)

        # ── Fire callbacks ────────────────────────────────────────
        self._fire_callbacks(stream_event)

        if self.verbose:
            print(
                f"[PULSE] Captured {event.event_type.value} "
                f"from {event.agent_id[:8]} "
                f"in {latency:.1f}ms"
            )

        return stream_event

    def capture_batch(
        self,
        events: List[EATEvent]
    ) -> List[StreamEvent]:
        """
        Capture multiple EAT events in sequence.
        Returns list of StreamEvents in same order.
        """
        return [self.capture(event) for event in events]

    # ── ASYNC BUFFER ──────────────────────────────────────────────────────

    def capture_async(self, event: EATEvent) -> bool:
        """
        Non-blocking capture — adds event to buffer queue.
        Returns True if successfully queued, False if buffer full.
        Flushed by background thread at flush_interval.
        """
        try:
            self._buffer.put_nowait(event)
            return True
        except queue.Full:
            if self.verbose:
                print("[PULSE] Buffer full — dropping event")
            return False

    def _flush_loop(self) -> None:
        """Background thread — flushes buffer at regular intervals."""
        while self._running:
            time.sleep(self.flush_interval)
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Drain buffer queue and capture all pending events."""
        events = []
        while not self._buffer.empty():
            try:
                events.append(self._buffer.get_nowait())
            except queue.Empty:
                break
        for event in events:
            self.capture(event)

    # ── DECAY CHECK ───────────────────────────────────────────────────────

    def _run_decay_check(self, current_t: int) -> None:
        """
        Run DECAY detector check at current time step.
        Fires when event count is a multiple of decay_check_every.
        """
        try:
            from decay.detector import DecayDetector
            detector = DecayDetector(self.session)
            alerts   = detector.check(current_t=current_t)
            if alerts:
                with self._lock:
                    self._stats.decay_alerts += len(alerts)
                if self.verbose:
                    for alert in alerts:
                        print(
                            f"[PULSE DECAY] {alert.severity} — "
                            f"{alert.alert_type}: "
                            f"{alert.recommendation[:60]}..."
                        )
        except Exception:
            pass

    # ── CALLBACKS ─────────────────────────────────────────────────────────

    def on_event(self, callback: Callable[[StreamEvent], None]) -> None:
        """
        Register a callback fired on every successfully captured event.
        Use for real-time dashboard updates, alerting, logging.

        callback signature: fn(stream_event: StreamEvent) -> None
        """
        self._callbacks.append(callback)

    def _fire_callbacks(self, stream_event: StreamEvent) -> None:
        for callback in self._callbacks:
            try:
                callback(stream_event)
            except Exception:
                pass

    # ── REPLAY SUPPORT ────────────────────────────────────────────────────

    def replay(
        self,
        events:     List[EATEvent],
        speed:      float = 1.0,
        on_event:   Optional[Callable] = None
    ) -> List[StreamEvent]:
        """
        Replay a sequence of EAT events through PULSE.
        Used by Ghost Runner for counterfactual sessions.

        speed: replay multiplier (1.0 = real time, 2.0 = 2x faster)
        on_event: optional callback per replayed event
        """
        results = []
        prev_t  = None

        for event in sorted(events, key=lambda e: e.t):
            # Simulate timing between events
            if prev_t is not None and speed > 0:
                delay = max(0, (event.t - prev_t) * 0.1 / speed)
                time.sleep(delay)

            result = self.capture(event)
            results.append(result)

            if on_event:
                try:
                    on_event(result)
                except Exception:
                    pass

            prev_t = event.t

        return results

    # ── PROPERTIES ────────────────────────────────────────────────────────

    @property
    def stats(self) -> StreamStats:
        return self._stats

    @property
    def flushed_events(self) -> List[StreamEvent]:
        return list(self._flushed)

    @property
    def failed_events(self) -> List[StreamEvent]:
        return list(self._failed)

    @property
    def is_running(self) -> bool:
        return self._running

    def summary(self) -> Dict:
        """
        Full stream summary for API and dashboard display.
        """
        return {
            "session_id":    self.session.session_id,
            "is_running":    self._running,
            "stats":         self._stats.to_dict(),
            "recent_events": [
                {
                    "event_id":   e.eat_event.event_id,
                    "agent_type": e.eat_event.agent_type.value,
                    "event_type": e.eat_event.event_type.value,
                    "status":     e.status.value,
                    "latency_ms": e.latency_ms,
                    "t":          e.eat_event.t
                }
                for e in self._flushed[-10:]  # Last 10 events
            ]
        }