"""Tests for vault/analytics.py"""

import os

import pytest

from vault.analytics import UsageAnalytics


@pytest.fixture
def analytics(tmp_path):
    vault_dir = str(tmp_path / "vault")
    os.makedirs(vault_dir, exist_ok=True)
    return UsageAnalytics(vault_root=vault_dir)


class TestUsageAnalytics:
    def test_track_event(self, analytics):
        analytics.track("note_created", type="task", note_id="n1")
        events = analytics._read_events()
        assert len(events) == 1
        assert events[0]["event"] == "note_created"

    def test_track_multiple_events(self, analytics):
        analytics.track("note_created", type="task")
        analytics.track("search", query="hello")
        analytics.track("note_created", type="idea")
        events = analytics._read_events()
        assert len(events) == 3

    def test_note_creation_stats(self, analytics):
        analytics.track("note_created", type="task")
        analytics.track("note_created", type="task")
        analytics.track("note_created", type="idea")
        stats = analytics.note_creation_stats(days=30)
        assert stats["total_created"] == 3
        assert stats["by_type"]["task"] == 2
        assert stats["by_type"]["idea"] == 1

    def test_note_creation_stats_empty(self, analytics):
        stats = analytics.note_creation_stats()
        assert stats["total_created"] == 0

    def test_search_stats(self, analytics):
        analytics.track("search", query="hello", result_count=5)
        analytics.track("search", query="world", result_count=0)
        analytics.track("search", query="hello", result_count=3)
        stats = analytics.search_stats(days=30)
        assert stats["total_searches"] == 3
        assert stats["successful"] == 2
        assert stats["success_rate"] > 0

    def test_search_stats_empty(self, analytics):
        stats = analytics.search_stats()
        assert stats["total_searches"] == 0
        assert stats["success_rate"] == 0

    def test_generate_report(self, analytics):
        analytics.track("note_created", type="task")
        analytics.track("search", query="test", result_count=1)
        report = analytics.generate_report(period="weekly")
        assert "period" in report
        assert "notes" in report
        assert "search" in report

    def test_get_recent_events(self, analytics):
        for i in range(5):
            analytics.track("event", index=i)
        recent = analytics.get_recent_events(limit=3)
        assert len(recent) == 3

    def test_clear(self, analytics):
        analytics.track("event")
        analytics.track("event")
        count = analytics.clear()
        assert count == 2
        assert analytics._read_events() == []

    def test_accuracy_trend_empty(self, analytics):
        trend = analytics.model_accuracy_trend()
        assert trend == []
