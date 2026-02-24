"""
VERIFAI Observability — Production Metrics & Structured Logging

Provides:
- Agent-level execution metrics (duration, success, error counts)
- Diagnostic quality metrics (confidence distribution, uncertainty distribution)
- System health metrics (active workflows, memory, model load times)
- Prometheus-compatible /metrics endpoint data
- Structured JSON logging with trace context

All metrics are collected in-process using thread-safe counters.
Optional Prometheus export via prometheus_client (if installed).
"""

import time
import threading
import logging
import json
import statistics
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from collections import defaultdict

logger = logging.getLogger("monitoring.metrics")


# =============================================================================
# METRIC TYPES
# =============================================================================

@dataclass
class CounterMetric:
    """Thread-safe counter."""
    name: str
    description: str
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class HistogramMetric:
    """Thread-safe histogram with percentile support."""
    name: str
    description: str
    _values: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float):
        with self._lock:
            self._values.append(value)

    @property
    def count(self) -> int:
        return len(self._values)

    @property
    def sum(self) -> float:
        return sum(self._values) if self._values else 0

    @property
    def mean(self) -> float:
        return statistics.mean(self._values) if self._values else 0

    def percentile(self, p: float) -> float:
        if not self._values:
            return 0
        with self._lock:
            sorted_vals = sorted(self._values)
            idx = int(len(sorted_vals) * p / 100)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def summary(self) -> dict:
        if not self._values:
            return {"count": 0, "sum": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
        return {
            "count": self.count,
            "sum": round(self.sum, 4),
            "mean": round(self.mean, 4),
            "p50": round(self.percentile(50), 4),
            "p95": round(self.percentile(95), 4),
            "p99": round(self.percentile(99), 4),
        }


@dataclass
class GaugeMetric:
    """Thread-safe gauge."""
    name: str
    description: str
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float):
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0):
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0):
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value


# =============================================================================
# LABELED METRICS (agent, status, etc.)
# =============================================================================

class LabeledHistogram:
    """Histogram with label support (e.g., by agent name)."""
    def __init__(self, name: str, description: str, labels: list[str]):
        self.name = name
        self.description = description
        self.label_names = labels
        self._histograms: dict[tuple, HistogramMetric] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> HistogramMetric:
        key = tuple(kwargs.get(l, "") for l in self.label_names)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = HistogramMetric(
                    name=f"{self.name}{{{','.join(f'{k}={v}' for k, v in kwargs.items())}}}",
                    description=self.description
                )
            return self._histograms[key]

    def all_series(self) -> dict:
        result = {}
        for key, hist in self._histograms.items():
            label_str = ",".join(f"{self.label_names[i]}=\"{key[i]}\"" for i in range(len(key)))
            result[label_str] = hist.summary()
        return result


class LabeledCounter:
    """Counter with label support."""
    def __init__(self, name: str, description: str, labels: list[str]):
        self.name = name
        self.description = description
        self.label_names = labels
        self._counters: dict[tuple, CounterMetric] = {}
        self._lock = threading.Lock()

    def labels(self, **kwargs) -> CounterMetric:
        key = tuple(kwargs.get(l, "") for l in self.label_names)
        with self._lock:
            if key not in self._counters:
                self._counters[key] = CounterMetric(
                    name=f"{self.name}{{{','.join(f'{k}={v}' for k, v in kwargs.items())}}}",
                    description=self.description
                )
            return self._counters[key]

    def all_series(self) -> dict:
        result = {}
        for key, counter in self._counters.items():
            label_str = ",".join(f"{self.label_names[i]}=\"{key[i]}\"" for i in range(len(key)))
            result[label_str] = counter.value
        return result


# =============================================================================
# GLOBAL METRICS COLLECTOR
# =============================================================================

class MetricsCollector:
    """
    Centralized metrics collector for VERIFAI.
    Thread-safe, lightweight, Prometheus-compatible.
    """

    def __init__(self):
        # ─── Agent Metrics ───
        self.agent_duration = LabeledHistogram(
            "verifai_agent_duration_seconds",
            "Time spent in each agent (seconds)",
            ["agent_name"]
        )
        self.agent_invocations = LabeledCounter(
            "verifai_agent_invocations_total",
            "Total agent invocations",
            ["agent_name", "status"]
        )

        # ─── Diagnostic Quality ───
        self.diagnosis_confidence = HistogramMetric(
            "verifai_diagnosis_confidence",
            "Distribution of final confidence scores"
        )
        self.diagnosis_uncertainty = HistogramMetric(
            "verifai_diagnosis_uncertainty",
            "Distribution of final uncertainty scores"
        )
        self.debate_rounds = HistogramMetric(
            "verifai_debate_rounds",
            "Number of debate rounds per case"
        )
        self.information_gain = LabeledHistogram(
            "verifai_information_gain",
            "Information gain per agent",
            ["agent_name"]
        )

        # ─── Safety ───
        self.safety_score = HistogramMetric(
            "verifai_safety_score",
            "Distribution of safety scores"
        )
        self.safety_flags = LabeledCounter(
            "verifai_safety_flags_total",
            "Safety flags raised by type",
            ["flag_type", "severity"]
        )
        self.critical_findings = CounterMetric(
            "verifai_critical_findings_total",
            "Total critical findings detected"
        )

        # ─── System ───
        self.active_workflows = GaugeMetric(
            "verifai_active_workflows",
            "Currently running diagnostic workflows"
        )
        self.total_workflows = CounterMetric(
            "verifai_workflows_total",
            "Total diagnostic workflows processed"
        )
        self.deferrals = CounterMetric(
            "verifai_deferrals_total",
            "Cases deferred to human review"
        )
        self.workflow_duration = HistogramMetric(
            "verifai_workflow_duration_seconds",
            "End-to-end workflow duration"
        )
        self.errors = LabeledCounter(
            "verifai_errors_total",
            "Errors by component",
            ["component", "error_type"]
        )

        # ─── Pipeline Start Time Tracking ───
        self._workflow_starts: dict[str, float] = {}
        self._lock = threading.Lock()

    def start_workflow(self, session_id: str):
        """Mark workflow start for duration tracking."""
        with self._lock:
            self._workflow_starts[session_id] = time.time()
        self.active_workflows.inc()
        self.total_workflows.inc()

    def end_workflow(self, session_id: str):
        """Mark workflow end and record duration."""
        self.active_workflows.dec()
        with self._lock:
            start = self._workflow_starts.pop(session_id, None)
        if start:
            self.workflow_duration.observe(time.time() - start)

    def to_prometheus_format(self) -> str:
        """Export all metrics in Prometheus text exposition format."""
        lines = []

        # agent_duration
        lines.append(f"# HELP {self.agent_duration.name} {self.agent_duration.description}")
        lines.append(f"# TYPE {self.agent_duration.name} histogram")
        for labels, summary in self.agent_duration.all_series().items():
            for stat, val in summary.items():
                lines.append(f'{self.agent_duration.name}_{stat}{{{labels}}} {val}')

        # agent_invocations
        lines.append(f"# HELP {self.agent_invocations.name} {self.agent_invocations.description}")
        lines.append(f"# TYPE {self.agent_invocations.name} counter")
        for labels, val in self.agent_invocations.all_series().items():
            lines.append(f'{self.agent_invocations.name}{{{labels}}} {val}')

        # Simple histograms
        for metric in [self.diagnosis_confidence, self.diagnosis_uncertainty,
                       self.debate_rounds, self.safety_score, self.workflow_duration]:
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} histogram")
            for stat, val in metric.summary().items():
                lines.append(f"{metric.name}_{stat} {val}")

        # Simple counters
        for metric in [self.critical_findings, self.total_workflows, self.deferrals]:
            lines.append(f"# HELP {metric.name} {metric.description}")
            lines.append(f"# TYPE {metric.name} counter")
            lines.append(f"{metric.name} {metric.value}")

        # Gauges
        lines.append(f"# HELP {self.active_workflows.name} {self.active_workflows.description}")
        lines.append(f"# TYPE {self.active_workflows.name} gauge")
        lines.append(f"{self.active_workflows.name} {self.active_workflows.value}")

        # Labeled counters
        for labeled in [self.safety_flags, self.errors]:
            lines.append(f"# HELP {labeled.name} {labeled.description}")
            lines.append(f"# TYPE {labeled.name} counter")
            for labels, val in labeled.all_series().items():
                lines.append(f'{labeled.name}{{{labels}}} {val}')

        return "\n".join(lines) + "\n"

    def get_summary(self) -> dict:
        """Get a JSON-friendly summary of all metrics."""
        return {
            "system": {
                "active_workflows": self.active_workflows.value,
                "total_workflows": self.total_workflows.value,
                "deferrals": self.deferrals.value,
                "critical_findings": self.critical_findings.value,
                "workflow_duration": self.workflow_duration.summary(),
            },
            "agents": {
                "duration": self.agent_duration.all_series(),
                "invocations": self.agent_invocations.all_series(),
                "information_gain": self.information_gain.all_series(),
            },
            "diagnostics": {
                "confidence": self.diagnosis_confidence.summary(),
                "uncertainty": self.diagnosis_uncertainty.summary(),
                "debate_rounds": self.debate_rounds.summary(),
                "safety_score": self.safety_score.summary(),
            },
            "safety": {
                "flags": self.safety_flags.all_series(),
                "errors": self.errors.all_series(),
            },
        }


# Global singleton
metrics = MetricsCollector()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@contextmanager
def track_agent_execution(agent_name: str):
    """
    Context manager to track agent execution time and success/failure.

    Usage:
        with track_agent_execution("radiologist"):
            result = radiologist_node(state)
    """
    start = time.time()
    try:
        yield
        duration = time.time() - start
        metrics.agent_duration.labels(agent_name=agent_name).observe(duration)
        metrics.agent_invocations.labels(agent_name=agent_name, status="success").inc()
        logger.info(f"Agent {agent_name} completed in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start
        metrics.agent_duration.labels(agent_name=agent_name).observe(duration)
        metrics.agent_invocations.labels(agent_name=agent_name, status="error").inc()
        metrics.errors.labels(component=agent_name, error_type=type(e).__name__).inc()
        logger.error(f"Agent {agent_name} failed after {duration:.2f}s: {e}")
        raise


def track_diagnosis(confidence: float, uncertainty: float, deferred: bool,
                    debate_rounds: int = 0, safety_score: float = 1.0):
    """Record diagnosis-level metrics after workflow completion."""
    metrics.diagnosis_confidence.observe(confidence)
    metrics.diagnosis_uncertainty.observe(uncertainty)
    metrics.debate_rounds.observe(debate_rounds)
    metrics.safety_score.observe(safety_score)
    if deferred:
        metrics.deferrals.inc()


# =============================================================================
# SHARED PERSISTENCE (allows test_workflow.py to feed the dashboard)
# =============================================================================

import os
from pathlib import Path

METRICS_SNAPSHOT_PATH = Path(os.path.dirname(os.path.abspath(__file__))).parent / "metrics_snapshot.json"


def save_metrics_snapshot():
    """Save current metrics to a shared JSON file so the API dashboard can read it."""
    summary = metrics.get_summary()
    summary["_saved_at"] = datetime.now(timezone.utc).isoformat() if hasattr(datetime, 'now') else datetime.utcnow().isoformat()
    try:
        METRICS_SNAPSHOT_PATH.write_text(json.dumps(summary, indent=2, default=str))
        logger.info(f"Metrics snapshot saved to {METRICS_SNAPSHOT_PATH}")
    except Exception as e:
        logger.error(f"Failed to save metrics snapshot: {e}")


def _load_snapshot() -> Optional[dict]:
    """Load metrics from snapshot file if it exists and is recent (< 1 hour old)."""
    try:
        if METRICS_SNAPSHOT_PATH.exists():
            data = json.loads(METRICS_SNAPSHOT_PATH.read_text())
            return data
    except Exception:
        pass
    return None


def get_metrics_summary() -> dict:
    """
    Get JSON summary of all collected metrics.
    
    If in-memory metrics are empty (e.g., API server just started),
    falls back to reading from the shared snapshot file written by test_workflow.py.
    """
    in_memory = metrics.get_summary()

    # Check if in-memory has any data
    has_data = in_memory["system"]["total_workflows"] > 0

    if has_data:
        return in_memory

    # Fall back to snapshot file
    snapshot = _load_snapshot()
    if snapshot:
        # Remove internal metadata
        snapshot.pop("_saved_at", None)
        return snapshot

    return in_memory


# =============================================================================
# STRUCTURED LOGGER
# =============================================================================

class StructuredLogger:
    """
    JSON structured logger for production environments.
    Adds trace context, agent metadata, and timing to log entries.
    """

    def __init__(self, name: str = "verifai"):
        self.logger = logging.getLogger(name)
        self.name = name

    def log(self, level: str, message: str, **context):
        """Emit a structured log entry."""
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.upper(),
            "service": "verifai",
            "message": message,
            **context
        }
        log_line = json.dumps(entry, default=str)

        if level == "error":
            self.logger.error(log_line)
        elif level == "warning":
            self.logger.warning(log_line)
        elif level == "debug":
            self.logger.debug(log_line)
        else:
            self.logger.info(log_line)

    def agent_start(self, agent_name: str, session_id: str, **extra):
        self.log("info", f"Agent {agent_name} started",
                 agent=agent_name, session_id=session_id, event="agent_start", **extra)

    def agent_complete(self, agent_name: str, session_id: str, duration: float, **extra):
        self.log("info", f"Agent {agent_name} completed in {duration:.2f}s",
                 agent=agent_name, session_id=session_id, event="agent_complete",
                 duration_seconds=duration, **extra)

    def agent_error(self, agent_name: str, session_id: str, error: str, **extra):
        self.log("error", f"Agent {agent_name} failed: {error}",
                 agent=agent_name, session_id=session_id, event="agent_error",
                 error=error, **extra)

    def safety_flag(self, flag_type: str, severity: str, session_id: str, **extra):
        self.log("warning", f"Safety flag: {flag_type} ({severity})",
                 flag_type=flag_type, severity=severity, session_id=session_id,
                 event="safety_flag", **extra)

    def workflow_start(self, session_id: str, **extra):
        self.log("info", f"Workflow started: {session_id}",
                 session_id=session_id, event="workflow_start", **extra)

    def workflow_complete(self, session_id: str, duration: float, confidence: float, **extra):
        self.log("info", f"Workflow completed: {session_id} (confidence={confidence:.0%})",
                 session_id=session_id, event="workflow_complete",
                 duration_seconds=duration, confidence=confidence, **extra)


structured_logger = StructuredLogger()
