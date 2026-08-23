"""Tests for vault/synthesizer.py"""

import os

import pytest

from vault.synthesizer import KnowledgeSynthesizer
from vault.writer import VaultWriter
from brain_contract import Understanding, NoteType


@pytest.fixture
def synthesizer_with_notes(tmp_path):
    vault_dir = str(tmp_path / "vault")
    writer = VaultWriter(vault_dir)

    writer.write(Understanding(
        text="Fix the VAD threshold",
        type=NoteType.TASK, priority=0.8, topics=["technology"],
        tags=["bug", "vad"],
    ))
    writer.write(Understanding(
        text="Circular buffer idea for audio",
        type=NoteType.IDEA, priority=0.5, topics=["engineering"],
        tags=["architecture", "audio"],
    ))
    writer.write(Understanding(
        text="Met with Sarah about project",
        type=NoteType.MEETING, priority=0.6, topics=["technology"],
        tags=["meeting", "planning"],
    ))

    return KnowledgeSynthesizer(vault_root=vault_dir)


class TestKnowledgeSynthesizer:
    def test_daily_digest(self, synthesizer_with_notes):
        digest = synthesizer_with_notes.daily_digest()
        assert "date" in digest
        assert "notes_created" in digest
        assert digest["notes_created"] >= 3
        assert "by_type" in digest

    def test_daily_digest_specific_date(self, synthesizer_with_notes):
        digest = synthesizer_with_notes.daily_digest(date="2026-08-23")
        assert digest["date"] == "2026-08-23"

    def test_weekly_report(self, synthesizer_with_notes):
        report = synthesizer_with_notes.weekly_report()
        assert "week" in report
        assert "total_notes" in report
        assert report["total_notes"] >= 3
        assert "top_topics" in report

    def test_topic_summary(self, synthesizer_with_notes):
        summary = synthesizer_with_notes.topic_summary("technology")
        assert summary["topic"] == "technology"
        assert summary["note_count"] >= 1

    def test_topic_summary_no_match(self, synthesizer_with_notes):
        summary = synthesizer_with_notes.topic_summary("nonexistent")
        assert summary["note_count"] == 0

    def test_insight_report(self, synthesizer_with_notes):
        insights = synthesizer_with_notes.insight_report()
        assert isinstance(insights, list)
        assert len(insights) >= 1
        # Should have topic activity insight
        types = [i["type"] for i in insights]
        assert "topic_activity" in types or "productivity" in types
