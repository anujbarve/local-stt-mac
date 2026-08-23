"""Tests for brain/temporal.py — TemporalReasoner."""

from datetime import datetime, timedelta, timezone
import pytest

from brain.temporal import TemporalReasoner, TemporalResult


@pytest.fixture
def reasoner():
    now = datetime(2026, 8, 24, 10, 0, 0, tzinfo=timezone.utc)
    return TemporalReasoner(now=now)


class TestTemporalReasoner:
    def test_parse_by_friday(self, reasoner):
        r = reasoner.parse("I need to fix this by Friday")
        assert r.has_time is True
        assert r.deadline is not None
        assert r.deadline.weekday() == 4  # Friday
        assert r.confidence > 0.7

    def test_parse_by_next_friday(self, reasoner):
        r = reasoner.parse("complete by next Tuesday")
        assert r.has_time is True
        assert r.deadline.weekday() == 1  # Tuesday

    def test_parse_in_hours(self, reasoner):
        r = reasoner.parse("I need this in 3 hours")
        assert r.has_time is True
        assert r.deadline is not None
        expected = reasoner.now + timedelta(hours=3)
        assert abs((r.deadline - expected).total_seconds()) < 60

    def test_parse_in_days(self, reasoner):
        r = reasoner.parse("finish in 2 days")
        assert r.has_time is True
        expected = reasoner.now + timedelta(days=2)
        assert r.deadline.date() == expected.date()

    def test_parse_tomorrow(self, reasoner):
        r = reasoner.parse("do it tomorrow")
        assert r.has_time is True
        assert r.deadline.date() == (reasoner.now + timedelta(days=1)).date()

    def test_parse_end_of_week(self, reasoner):
        r = reasoner.parse("deadline end of week")
        assert r.has_time is True
        assert r.deadline.weekday() == 4  # Friday

    def test_parse_end_of_month(self, reasoner):
        r = reasoner.parse("due end of month")
        assert r.has_time is True
        assert r.deadline.day >= 28  # near end of month

    def test_parse_asap(self, reasoner):
        r = reasoner.parse("need this ASAP")
        assert r.has_time is True
        assert r.raw_expression == "ASAP"

    def test_parse_month_day(self, reasoner):
        r = reasoner.parse("meeting on September 15th")
        assert r.has_time is True
        assert r.deadline.month == 9
        assert r.deadline.day == 15

    def test_parse_no_time(self, reasoner):
        r = reasoner.parse("I like python programming")
        assert r.has_time is False
        assert r.deadline is None

    def test_parse_recurring_daily(self, reasoner):
        r = reasoner.parse("daily standup at 9am")
        assert r.has_time is True
        assert r.recurring == "daily"

    def test_parse_recurring_weekly(self, reasoner):
        r = reasoner.parse("weekly review")
        assert r.has_time is True
        assert r.recurring == "weekly"

    def test_time_until_hours(self, reasoner):
        deadline = reasoner.now + timedelta(hours=3)
        result = reasoner.time_until(deadline)
        assert "3" in result and "hour" in result

    def test_time_until_days(self, reasoner):
        deadline = reasoner.now + timedelta(days=5)
        result = reasoner.time_until(deadline)
        assert "5 days" in result

    def test_time_until_tomorrow(self, reasoner):
        deadline = reasoner.now + timedelta(days=1, hours=2)
        result = reasoner.time_until(deadline)
        assert "tomorrow" in result

    def test_time_until_overdue(self, reasoner):
        deadline = reasoner.now - timedelta(hours=1)
        result = reasoner.time_until(deadline)
        assert result == "overdue"

    def test_suggest_schedule(self, reasoner):
        notes = [
            {"id": "1", "text": "Fix bug", "priority": 0.9, "urgency": "immediate",
             "deadline": (reasoner.now + timedelta(days=1)).isoformat()},
            {"id": "2", "text": "Write docs", "priority": 0.5, "urgency": "later",
             "deadline": (reasoner.now + timedelta(days=3)).isoformat()},
        ]
        schedule = reasoner.suggest_schedule(notes)
        assert len(schedule) == 2
        assert schedule[0].priority >= schedule[1].priority  # sorted by priority

    def test_detect_recurring(self, reasoner):
        notes = [
            {"text": "daily standup notes"},
            {"text": "weekly review"},
            {"text": "random note"},
        ]
        patterns = reasoner.detect_recurring(notes)
        assert len(patterns) >= 2

    def test_relative_future(self, reasoner):
        r = reasoner.parse("by Friday")
        assert r.relative == "future"

    def test_relative_today(self, reasoner):
        r = reasoner.parse("do it today")
        assert r.relative == "today"
