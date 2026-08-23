"""Tests for brain/recommend.py — RecommendationEngine."""

from datetime import datetime, timezone, timedelta
import pytest

from brain.recommend import RecommendationEngine


@pytest.fixture
def engine():
    return RecommendationEngine()


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


class TestRecommendationEngine:
    def test_recommend_related_returns_list(self, engine):
        note = _make_note(topics=["python"])
        all_notes = [_make_note(id=str(i), topics=["python"]) for i in range(5)]
        result = engine.recommend_related(note, all_notes)
        assert isinstance(result, list)

    def test_recommend_related_same_topic(self, engine):
        note = _make_note(id="0", topics=["python"])
        all_notes = [
            _make_note(id="1", topics=["python"]),
            _make_note(id="2", topics=["rust"]),
        ]
        result = engine.recommend_related(note, all_notes)
        assert len(result) >= 1
        assert result[0]["note_id"] == "1"

    def test_recommend_related_same_entity(self, engine):
        note = _make_note(id="0", entities=[{"name": "Alice", "entity_type": "person"}])
        all_notes = [
            _make_note(id="1", entities=[{"name": "Alice", "entity_type": "person"}]),
            _make_note(id="2", entities=[{"name": "Bob", "entity_type": "person"}]),
        ]
        result = engine.recommend_related(note, all_notes)
        assert len(result) >= 1
        assert result[0]["note_id"] == "1"
        assert result[0]["reason"] == "same_entity"

    def test_recommend_related_excludes_self(self, engine):
        note = _make_note(id="0", topics=["tech"])
        all_notes = [_make_note(id="0", topics=["tech"])]
        result = engine.recommend_related(note, all_notes)
        assert len(result) == 0

    def test_recommend_related_top_k(self, engine):
        note = _make_note(id="0", topics=["tech"])
        all_notes = [_make_note(id=str(i), topics=["tech"]) for i in range(20)]
        result = engine.recommend_related(note, all_notes, top_k=3)
        assert len(result) <= 3

    def test_suggest_actions_task(self, engine):
        note = _make_note(type="task", priority=0.9)
        suggestions = engine.suggest_actions(note)
        assert len(suggestions) >= 1
        assert any(s["action"] == "prioritize" for s in suggestions)

    def test_suggest_actions_meeting(self, engine):
        note = _make_note(type="meeting")
        suggestions = engine.suggest_actions(note)
        assert any(s["action"] == "create_action_items" for s in suggestions)

    def test_suggest_actions_idea(self, engine):
        note = _make_note(type="idea")
        suggestions = engine.suggest_actions(note)
        assert any(s["action"] == "develop_idea" for s in suggestions)

    def test_suggest_actions_bug_keyword(self, engine):
        note = _make_note(text="fix the broken VAD threshold bug")
        suggestions = engine.suggest_actions(note)
        assert any(s["action"] == "create_bug_ticket" for s in suggestions)

    def test_find_stale_projects(self, engine):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        notes = [
            _make_note(id="1", entities=[{"name": "old-project", "entity_type": "project"}], timestamp=old_ts),
            _make_note(id="2", entities=[{"name": "active-project", "entity_type": "project"}]),
        ]
        stale = engine.find_stale_projects(notes, stale_days=14)
        assert len(stale) >= 1
        assert stale[0]["project"] == "old-project"

    def test_find_stale_projects_none(self, engine):
        notes = [_make_note(entities=[{"name": "fresh", "entity_type": "project"}])]
        stale = engine.find_stale_projects(notes, stale_days=100)
        assert len(stale) == 0

    def test_priority_trends(self, engine):
        now = datetime.now(timezone.utc)
        notes = [
            _make_note(id="1", topics=["ai"], priority=0.9, timestamp=now.isoformat()),
            _make_note(id="2", topics=["ai"], priority=0.3, timestamp=(now - timedelta(days=20)).isoformat()),
        ]
        trends = engine.priority_trends(notes, window_days=30)
        assert "increasing" in trends
        assert "decreasing" in trends
        assert "stable" in trends
