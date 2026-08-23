"""Tests for vault/observability.py"""

import pytest

from vault.observability import StructuredLogger, MetricsCollector


class TestStructuredLogger:
    def test_info_emits(self):
        logger = StructuredLogger("test_logger_info")
        # Should not raise
        logger.info("test_event", key="value")

    def test_error_includes_exception(self):
        logger = StructuredLogger("test_logger_error")
        logger.error("test_error", error="something broke")

    def test_warning(self):
        logger = StructuredLogger("test_logger_warning")
        logger.warning("test_warning", detail="check this")

    def test_debug(self):
        logger = StructuredLogger("test_logger_debug")
        logger.debug("test_debug", data=42)

    def test_trace_success(self):
        logger = StructuredLogger("test_logger_trace")
        with logger.trace("my_operation"):
            x = 1 + 1
        assert x == 2

    def test_trace_failure(self):
        logger = StructuredLogger("test_logger_trace_fail")
        with pytest.raises(ValueError):
            with logger.trace("failing_op"):
                raise ValueError("boom")


class TestMetricsCollector:
    def test_increment(self):
        m = MetricsCollector()
        m.increment("writes")
        m.increment("writes")
        m.increment("writes", 3)
        assert m.get_metrics()["counters"]["writes"] == 5

    def test_timer(self):
        m = MetricsCollector()
        m.timer("search", 10.5)
        m.timer("search", 20.3)
        metrics = m.get_metrics()
        assert metrics["search_count"] == 2
        # p50 of [10.5, 20.3] = sorted[1] = 20.3
        assert metrics["search_p50"] == 20.3

    def test_gauge(self):
        m = MetricsCollector()
        m.gauge("vault_size", 42.0)
        assert m.get_metrics()["gauges"]["vault_size"] == 42.0

    def test_measure_context(self):
        m = MetricsCollector()
        with m.measure("operation"):
            pass
        assert m.get_metrics()["operation_count"] == 1

    def test_reset(self):
        m = MetricsCollector()
        m.increment("test")
        m.timer("test", 1.0)
        m.gauge("test", 1.0)
        m.reset()
        metrics = m.get_metrics()
        assert metrics["counters"] == {}
        assert metrics["gauges"] == {}

    def test_empty_timer_stats(self):
        m = MetricsCollector()
        metrics = m.get_metrics()
        assert "search_p50" not in metrics  # no timers yet
