"""Tests for brain/insights.py — InsightGenerator."""

from datetime import datetime, timezone, timedelta
import pytest

from brain.insights import InsightGenerator, Insight


@pytest.fixture
def gen():
    return InsightGenerator()


def _make_note(**overrides):
    defaults = {
        "id": "1",
        "text": "test note",
        "type": "observation",
        "priority": 0.5,
        "topics": [],
        "entities": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    defaults.update(overrides)
    return defaults


class TestInsightGenerator:
    def test_daily_insights_returns_list(self, gen):
        notes = [_make_note()]
        result = gen.daily_insights(notes)
        assert isinstance(result, list)
        assert all(isinstance(i, Insight) for i in result)

    def test_daily_insights_empty_notes(self, gen):
        result = gen.daily_insights([])
        assert len(result) >= 1  # should suggest capturing notes
        assert result[0].type == "alert"

    def test_daily_insights_focused_session(self, gen):
        notes = [_make_note(id=str(i), type="task") for i in range(5)]
        result = gen.daily_insights(notes)
        messages = [i.message for i in result]
        assert any("focused" in m.lower() or "5" in m for m in messages)

    def test_weekly_report(self, gen):
        notes = [
            _make_note(id=str(i), type="task", topics=["tech"])
            for i in range(10)
        ]
        report = gen.weekly_report(notes)
        assert "total_notes" in report
        assert "by_type" in report
        assert "top_topics" in report
        assert report["total_notes"] == 10

    def test_weekly_report_empty(self, gen):
        report = gen.weekly_report([])
        assert report["total_notes"] == 0

    def test_detect_patterns(self, gen):
        notes = [
            _make_note(id=str(i), topics=["python", "python", "python"])
            for i in range(5)
        ]
        patterns = gen.detect_patterns(notes)
        assert isinstance(patterns, list)

    def test_detect_patterns_empty(self, gen):
        patterns = gen.detect_patterns([])
        assert patterns == []

    def test_cross_references(self, gen):
        notes = [
            _make_note(id="1", topics=["python"], entities=[{"name": "Alice", "entity_type": "person"}]),
            _make_note(id="2", topics=["python"], entities=[{"name": "Alice", "entity_type": "person"}]),
            _make_note(id="3", topics=["rust"]),
        ]
        refs = gen.cross_references("1", notes)
        assert len(refs) >= 1
        assert refs[0]["note_id"] == "2"

    def test_cross_references_not_found(self, gen):
        refs = gen.cross_references("nonexistent", [_make_note()])
        assert refs == []

    def test_cross_references_self_excluded(self, gen):
        notes = [_make_note(id="1", topics=["tech"])]
        refs = gen.cross_references("1", notes)
        assert len(refs) == 0

    def test_insight_structure(self, gen):
        notes = [_make_note()]
        result = gen.daily_insights(notes)
        for insight in result:
            assert hasattr(insight, "type")
            assert hasattr(insight, "message")
            assert hasattr(insight, "severity")
            assert hasattr(insight, "related_notes")
