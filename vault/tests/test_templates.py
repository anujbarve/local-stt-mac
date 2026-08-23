"""Tests for vault/templates.py"""

import pytest

from vault.templates import NoteTemplate, _priority_label, _entity_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_u(**overrides):
    """Create a minimal Understanding-like dict for testing."""
    base = {
        "text": "I need to fix the VAD threshold by Friday",
        "timestamp": "2026-08-23T14:30:00Z",
        "type": "task",
        "priority": 0.82,
        "urgency": "soon",
        "sentiment": "neutral",
        "confidence": 0.87,
        "entities": [
            {"name": "VAD", "entity_type": "tool", "confidence": 1.0, "attributes": {}},
            {"name": "Friday", "entity_type": "date", "confidence": 1.0, "attributes": {}},
        ],
        "action": "fix VAD threshold",
        "summary": "The Silero VAD threshold is too sensitive.",
        "topics": ["technology", "engineering"],
        "tags": ["bug", "vad"],
        "embedding": None,
        "temporal": None,
        "related_notes": ["note_2026_08_20_003"],
        "note_id": "note_2026_08_23_001",
        "file_path": None,
        "metadata": {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoteTemplate:
    def test_render_contains_frontmatter(self):
        content = NoteTemplate.render(_make_u())
        assert content.startswith("---")
        assert "id: note_2026_08_23_001" in content
        assert "type: task" in content

    def test_render_contains_body(self):
        content = NoteTemplate.render(_make_u())
        assert "Fix VAD Threshold" in content or "fix VAD threshold" in content

    def test_render_frontmatter_classification(self):
        fm = NoteTemplate.render_frontmatter(_make_u())
        assert "priority: high" in fm
        assert "urgency: soon" in fm
        assert "confidence: 0.87" in fm

    def test_render_frontmatter_entities(self):
        fm = NoteTemplate.render_frontmatter(_make_u())
        assert "VAD" in fm
        assert "Friday" in fm

    def test_render_frontmatter_tags(self):
        fm = NoteTemplate.render_frontmatter(_make_u())
        assert "bug" in fm
        assert "vad" in fm
        assert "technology" in fm

    def test_task_body_has_details(self):
        content = NoteTemplate.render(_make_u())
        assert "**Status:** todo" in content
        assert "**Priority:** high" in content

    def test_observation_type(self):
        u = _make_u(type="observation", action=None, summary="Noticed something.")
        content = NoteTemplate.render(u)
        assert "# Observation" in content
        assert "Noticed something." in content

    def test_idea_type(self):
        u = _make_u(type="idea", action=None, summary="What if we used a local LLM?")
        content = NoteTemplate.render(u)
        assert "# Idea" in content
        assert "local LLM" in content

    def test_meeting_type(self):
        u = _make_u(
            type="meeting",
            action=None,
            entities=[
                {"name": "Alice", "entity_type": "person", "confidence": 1.0, "attributes": {}},
                {"name": "Bob", "entity_type": "person", "confidence": 1.0, "attributes": {}},
            ],
        )
        content = NoteTemplate.render(u)
        assert "Meeting" in content
        assert "Alice" in content
        assert "Bob" in content

    def test_learning_type(self):
        u = _make_u(type="learning", action=None, summary="Learned about transformers.")
        content = NoteTemplate.render(u)
        assert "# Learning" in content

    def test_decision_type(self):
        u = _make_u(type="decision", action=None, summary="Chose FAISS over Annoy.")
        content = NoteTemplate.render(u)
        assert "# Decision" in content

    def test_reminder_type(self):
        u = _make_u(type="reminder", action=None, summary="Call dentist on Monday.")
        content = NoteTemplate.render(u)
        assert "# Reminder" in content

    def test_person_type(self):
        u = _make_u(
            type="person_info",
            action=None,
            entities=[
                {"name": "Sarah Chen", "entity_type": "person", "confidence": 1.0,
                 "attributes": {"birthday": "1990-05-12"}},
            ],
        )
        content = NoteTemplate.render(u)
        assert "# Sarah Chen" in content

    def test_preference_type(self):
        u = _make_u(type="preference", action=None, summary="Dark mode everywhere.")
        content = NoteTemplate.render(u)
        assert "# Preference" in content

    def test_goal_type(self):
        u = _make_u(type="goal", action=None, summary="Run a marathon.")
        content = NoteTemplate.render(u)
        assert "# Goal" in content or "# Plan" in content
        assert "Milestones" in content or "Progress" in content

    def test_journal_type(self):
        u = _make_u(type="journal", action=None, summary="Productive day.")
        content = NoteTemplate.render(u)
        assert "# Journal Entry" in content

    def test_celebration_type(self):
        u = _make_u(type="celebration", action=None, summary="Shipped v1.0!")
        content = NoteTemplate.render(u)
        assert "Celebration" in content

    def test_health_type(self):
        u = _make_u(type="health", action=None, summary="Ran 5K today.")
        content = NoteTemplate.render(u)
        assert "# Health" in content

    def test_finance_type(self):
        u = _make_u(type="finance", action=None, summary="Monthly budget review.")
        content = NoteTemplate.render(u)
        assert "# Finance" in content

    def test_travel_type(self):
        u = _make_u(type="travel", action=None, summary="Trip to Tokyo planned.")
        content = NoteTemplate.render(u)
        assert "# Travel" in content

    def test_habit_type(self):
        u = _make_u(type="habit", action=None, summary="Meditate daily.")
        content = NoteTemplate.render(u)
        assert "# Habit" in content
        assert "Tracking" in content

    def test_wish_type(self):
        u = _make_u(type="wish", action=None, summary="I wish I had more time.")
        content = NoteTemplate.render(u)
        assert "# Wish" in content

    def test_media_type(self):
        u = _make_u(type="media", action=None, summary="Watched Inception again.")
        content = NoteTemplate.render(u)
        assert "# Media" in content

    def test_tool_info_type(self):
        u = _make_u(type="tool_info", action=None, summary="Using Silero VAD v5.")
        content = NoteTemplate.render(u)
        assert "Silero" in content


class TestPriorityLabel:
    def test_critical(self):
        assert _priority_label(0.95) == "critical"

    def test_high(self):
        assert _priority_label(0.8) == "high"

    def test_medium(self):
        assert _priority_label(0.5) == "medium"

    def test_low(self):
        assert _priority_label(0.3) == "low"

    def test_someday(self):
        assert _priority_label(0.1) == "someday"


class TestEntityList:
    def test_filters_by_type(self):
        entities = [
            {"name": "Alice", "entity_type": "person"},
            {"name": "VAD", "entity_type": "tool"},
        ]
        assert _entity_list(entities, "person") == ["Alice"]
        assert _entity_list(entities, "tool") == ["VAD"]

    def test_empty(self):
        assert _entity_list([], "person") == []
