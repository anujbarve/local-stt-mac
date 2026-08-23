"""
Vault Resilience
================

Circuit breaker, retry policy, and health aggregator for fault tolerance.
Protects the vault from cascading failures and external service issues.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


class CircuitBreaker:
    """Circuit breaker for external calls.

    States:
      - closed:   normal operation, calls pass through
      - open:     failures exceeded threshold, calls are blocked
      - half-open: recovery probe — one call allowed to test
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "default",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self.state = "closed"
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._success_count = 0

    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* with circuit breaker protection.

        Raises ``CircuitOpenError`` when the circuit is open.
        """
        if self.state == "open":
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self.state = "half-open"
            else:
                raise CircuitOpenError(f"Circuit '{self.name}' is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        if self.state == "half-open":
            # Recovery probe succeeded — close the circuit
            self.state = "closed"
            self._failure_count = 0
        elif self.state == "closed":
            self._failure_count = max(0, self._failure_count - 1)

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self.state = "open"

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = "closed"
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }


class CircuitOpenError(Exception):
    """Raised when a circuit breaker is open."""
    pass


class RetryPolicy:
    """Exponential backoff retry policy."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        max_backoff: float = 30.0,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff

    def execute(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute *func* with retry on failure.

        Raises the last exception after all retries are exhausted.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    delay = min(
                        self.backoff_factor * (2 ** attempt),
                        self.max_backoff,
                    )
                    time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    def stats(self) -> Dict[str, Any]:
        return {
            "max_retries": self.max_retries,
            "backoff_factor": self.backoff_factor,
            "max_backoff": self.max_backoff,
        }


class HealthAggregator:
    """Aggregate health checks from multiple components."""

    def __init__(self) -> None:
        self._checks: Dict[str, Callable[[], Dict[str, Any]]] = {}

    def register(self, name: str, check_fn: Callable[[], Dict[str, Any]]) -> None:
        """Register a health check function."""
        self._checks[name] = check_fn

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks and aggregate results."""
        results: Dict[str, Any] = {}
        overall_healthy = True

        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                results[name] = result
                if not result.get("healthy", True):
                    overall_healthy = False
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
                overall_healthy = False

        return {
            "healthy": overall_healthy,
            "components": results,
            "checked_at": datetime.utcnow().isoformat() + "Z",
        }

    def check_component(self, name: str) -> Dict[str, Any]:
        """Run a single health check."""
        check_fn = self._checks.get(name)
        if check_fn is None:
            return {"healthy": False, "error": f"Unknown component: {name}"}
        try:
            return check_fn()
        except Exception as e:
            return {"healthy": False, "error": str(e)}

