"""Tests for vault.resilience — CircuitBreaker, RetryPolicy, HealthAggregator."""

import time

import pytest

from vault.resilience import CircuitBreaker, CircuitOpenError, HealthAggregator, RetryPolicy


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_successful_call(self):
        cb = CircuitBreaker(failure_threshold=3)
        result = cb.call(lambda: 42)
        assert result == 42
        assert cb.state == "closed"

    def test_failure_increments_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.failure_count == 1
        assert cb.state == "closed"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=2)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("e")))
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("e")))
        assert cb.state == "open"

    def test_open_circuit_blocks_calls(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("e")))
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: 42)

    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("e")))
        time.sleep(0.1)
        # Should be in half-open, allowing one probe call
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("e")))
        time.sleep(0.1)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("e")))
        assert cb.state == "open"

    def test_success_decrements_failure_count(self):
        cb = CircuitBreaker(failure_threshold=5)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("e")))
        assert cb.failure_count == 1
        cb.call(lambda: "ok")
        assert cb.failure_count == 0

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("e")))
        assert cb.state == "open"
        cb.reset()
        assert cb.state == "closed"
        assert cb.failure_count == 0

    def test_stats(self):
        cb = CircuitBreaker(failure_threshold=5, name="test")
        s = cb.stats()
        assert s["name"] == "test"
        assert s["state"] == "closed"
        assert s["failure_threshold"] == 5

    def test_passes_args_kwargs(self):
        cb = CircuitBreaker()
        result = cb.call(lambda a, b: a + b, 3, b=4)
        assert result == 7


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_succeeds_first_try(self):
        rp = RetryPolicy(max_retries=3, backoff_factor=0)
        call_count = 0
        def fn():
            nonlocal call_count
            call_count += 1
            return "ok"
        result = rp.execute(fn)
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_failure(self):
        rp = RetryPolicy(max_retries=2, backoff_factor=0)
        call_count = 0
        def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "done"
        result = rp.execute(fn)
        assert result == "done"
        assert call_count == 3

    def test_exhausts_retries(self):
        rp = RetryPolicy(max_retries=2, backoff_factor=0)
        def fn():
            raise ValueError("always fail")
        with pytest.raises(ValueError, match="always fail"):
            rp.execute(fn)

    def test_passes_args_kwargs(self):
        rp = RetryPolicy(max_retries=0)
        result = rp.execute(lambda a, b, c=0: a + b + c, 1, 2, c=3)
        assert result == 6

    def test_stats(self):
        rp = RetryPolicy(max_retries=5, backoff_factor=2.0, max_backoff=60.0)
        s = rp.stats()
        assert s["max_retries"] == 5
        assert s["backoff_factor"] == 2.0
        assert s["max_backoff"] == 60.0


# ---------------------------------------------------------------------------
# HealthAggregator
# ---------------------------------------------------------------------------


class TestHealthAggregator:
    def test_all_healthy(self):
        ha = HealthAggregator()
        ha.register("db", lambda: {"healthy": True})
        ha.register("cache", lambda: {"healthy": True})
        result = ha.check_all()
        assert result["healthy"] is True
        assert len(result["components"]) == 2
        assert "checked_at" in result

    def test_one_unhealthy(self):
        ha = HealthAggregator()
        ha.register("good", lambda: {"healthy": True})
        ha.register("bad", lambda: {"healthy": False, "reason": "down"})
        result = ha.check_all()
        assert result["healthy"] is False

    def test_exception_in_check(self):
        ha = HealthAggregator()
        ha.register("crash", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        result = ha.check_all()
        assert result["healthy"] is False
        assert "error" in result["components"]["crash"]

    def test_check_component(self):
        ha = HealthAggregator()
        ha.register("svc", lambda: {"healthy": True, "latency": 5})
        result = ha.check_component("svc")
        assert result["healthy"] is True
        assert result["latency"] == 5

    def test_check_unknown_component(self):
        ha = HealthAggregator()
        result = ha.check_component("nonexistent")
        assert result["healthy"] is False
        assert "Unknown" in result["error"]

    def test_empty_checks(self):
        ha = HealthAggregator()
        result = ha.check_all()
        assert result["healthy"] is True
        assert result["components"] == {}
