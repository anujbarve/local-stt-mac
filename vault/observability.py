"""
Vault Observability
===================

Structured logging and metrics collection for vault operations.
Provides JSON-formatted logs, operation tracing, and performance metrics.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional


class StructuredLogger:
    """JSON structured logger for vault operations."""

    def __init__(self, name: str = "vault", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    def _emit(self, level: str, event: str, **kwargs: Any) -> None:
        """Emit a structured log entry."""
        entry: Dict[str, Any] = {
            "timestamp": time.time(),
            "level": level,
            "event": event,
        }
        entry.update(kwargs)
        getattr(self.logger, level.lower(), self.logger.info)(json.dumps(entry))

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit("ERROR", event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit("DEBUG", event, **kwargs)

    @contextmanager
    def trace(self, operation: str, **kwargs: Any) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
            elapsed = time.time() - start
            self.info(operation, duration_ms=round(elapsed * 1000, 2), **kwargs)
        except Exception as e:
            elapsed = time.time() - start
            self.error(
                f"{operation}_failed",
                duration_ms=round(elapsed * 1000, 2),
                error=str(e),
                **kwargs,
            )
            raise


class MetricsCollector:
    """Collect and report operational metrics."""

    def __init__(self) -> None:
        self._counters: Dict[str, int] = {}
        self._timers: Dict[str, list] = {}
        self._gauges: Dict[str, float] = {}

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self._counters[name] = self._counters.get(name, 0) + value

    def timer(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement."""
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(duration_ms)

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self._gauges[name] = value

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """Context manager that automatically times a block."""
        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self.timer(name, round(elapsed_ms, 2))

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        metrics: Dict[str, Any] = {"counters": dict(self._counters)}

        for name, durations in self._timers.items():
            if durations:
                sorted_d = sorted(durations)
                metrics[f"{name}_p50"] = sorted_d[len(sorted_d) // 2]
                metrics[f"{name}_p99"] = sorted_d[int(len(sorted_d) * 0.99)]
                metrics[f"{name}_count"] = len(durations)
                metrics[f"{name}_avg"] = round(sum(sorted_d) / len(sorted_d), 2)
            else:
                metrics[f"{name}_p50"] = 0
                metrics[f"{name}_p99"] = 0
                metrics[f"{name}_count"] = 0

        metrics["gauges"] = dict(self._gauges)
        return metrics

    def reset(self) -> None:
        """Clear all metrics."""
        self._counters.clear()
        self._timers.clear()
        self._gauges.clear()


# Module-level singleton
logger = StructuredLogger()
metrics = MetricsCollector()
