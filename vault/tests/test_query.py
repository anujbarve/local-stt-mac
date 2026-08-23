"""Tests for vault/query.py"""

import pytest

from vault.query import QueryParser


@pytest.fixture
def parser():
    return QueryParser()


class TestQueryParser:
    def test_parse_list_query(self, parser):
        result = parser.parse("What tasks are pending?")
        assert result["intent"] == "list"
        assert result["entity_type"] == "task"
        assert result["filters"].get("status") == "todo"

    def test_parse_search_query(self, parser):
        result = parser.parse("When is the meeting with Sarah?")
        assert result["intent"] == "search"
        assert result["entity_type"] == "meeting"

    def test_parse_explain_query(self, parser):
        result = parser.parse("Why did we choose FAISS?")
        assert result["intent"] == "explain"

    def test_parse_summarize_query(self, parser):
        result = parser.parse("Summarize this week's progress")
        assert result["intent"] == "summarize"
        assert result["temporal"]["relative"] == "this_week"

    def test_extract_entity_type_task(self, parser):
        assert parser._extract_entity_type("Show me my tasks") == "task"

    def test_extract_entity_type_project(self, parser):
        assert parser._extract_entity_type("Which projects are active?") == "project"

    def test_extract_entity_type_person(self, parser):
        assert parser._extract_entity_type("Who is Sarah?") == "person"

    def test_extract_entity_type_idea(self, parser):
        assert parser._extract_entity_type("What ideas do I have?") == "idea"

    def test_extract_entity_type_learning(self, parser):
        assert parser._extract_entity_type("What learnings from today?") == "learning"

    def test_extract_priority_high(self, parser):
        filters = parser._extract_filters("high priority tasks")
        assert filters.get("priority") == "high"

    def test_extract_priority_critical(self, parser):
        filters = parser._extract_filters("critical issues")
        assert filters.get("priority") == "critical"

    def test_extract_status_blocked(self, parser):
        filters = parser._extract_filters("blocked tasks")
        assert filters.get("status") == "blocked"

    def test_extract_status_done(self, parser):
        filters = parser._extract_filters("completed tasks")
        assert filters.get("status") == "done"

    def test_extract_temporal_today(self, parser):
        temporal = parser._extract_temporal("What happened today?")
        assert temporal["relative"] == "today"

    def test_extract_temporal_this_week(self, parser):
        temporal = parser._extract_temporal("Show this week's notes")
        assert temporal["relative"] == "this_week"

    def test_extract_temporal_absolute(self, parser):
        temporal = parser._extract_temporal("Notes from 2026-08-23")
        assert temporal["absolute"] == "2026-08-23"

    def test_extract_temporal_none(self, parser):
        temporal = parser._extract_temporal("Show me everything")
        assert temporal["relative"] is None
        assert temporal["absolute"] is None

    def test_is_vague_short(self, parser):
        assert parser._is_vague("things")

    def test_is_vague_specific(self, parser):
        assert not parser._is_vague("What tasks are high priority?")

    def test_parse_preserves_text(self, parser):
        query = "Show me blocked tasks involving Python"
        result = parser.parse(query)
        assert result["text"] == query

    def test_parse_default_expand(self, parser):
        result = parser.parse("detailed task list")
        assert isinstance(result["expand"], bool)
