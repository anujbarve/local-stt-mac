"""Tests for brain/interactive.py — InteractiveCorrector."""

import pytest

from brain.interactive import InteractiveCorrector
from brain.correct import CorrectionLogger
from brain_contract import Understanding, NoteType, Urgency, Sentiment


@pytest.fixture
def logger(tmp_path):
    path = str(tmp_path / "corrections.jsonl")
    return CorrectionLogger(buffer_path=path)


@pytest.fixture
def corrector(logger):
    return InteractiveCorrector(logger)


def _make_understanding(**overrides):
    defaults = dict(
        text="I need to fix the VAD threshold by Friday",
        type=NoteType.TASK,
        priority=0.82,
        urgency=Urgency.SOON,
        sentiment=Sentiment.NEUTRAL,
        confidence=0.87,
        action="fix VAD threshold",
        topics=["technology"],
    )
    defaults.update(overrides)
    return Understanding(**defaults)


class TestInteractiveCorrector:
    def test_format_summary(self, corrector):
        """_format_summary returns a readable string."""
        u = _make_understanding()
        summary = corrector._format_summary(u)
        assert "task" in summary
        assert "0.82" in summary
        assert "soon" in summary

    def test_format_summary_with_entities(self, corrector):
        """_format_summary includes entities when present."""
        from brain_contract import Entity, EntityType
        u = _make_understanding(
            entities=[Entity(name="VAD", entity_type=EntityType.TOOL)]
        )
        summary = corrector._format_summary(u)
        assert "VAD" in summary
        assert "tool" in summary

    def test_format_summary_with_topics(self, corrector):
        """_format_summary includes topics."""
        u = _make_understanding(topics=["technology", "engineering"])
        summary = corrector._format_summary(u)
        assert "technology" in summary
        assert "engineering" in summary

    def test_present_and_correct_accept(self, corrector, monkeypatch):
        """Accepting returns original with no modification."""
        u = _make_understanding()
        monkeypatch.setattr("builtins.input", lambda _: "")
        result, modified = corrector.present_and_correct("test text", u)
        assert result is u
        assert modified is False

    def test_present_and_correct_skip(self, corrector, monkeypatch):
        """Skipping returns original with no modification."""
        u = _make_understanding()
        monkeypatch.setattr("builtins.input", lambda _: "s")
        result, modified = corrector.present_and_correct("test text", u)
        assert result is u
        assert modified is False

    def test_collect_corrections_no_changes(self, corrector, monkeypatch):
        """Editing with no input returns None."""
        u = _make_understanding()
        # Simulate: type=Enter, priority=Enter, urgency=Enter, sentiment=Enter, action=Enter
        responses = iter(["", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = corrector._collect_corrections(u)
        assert result is None

    def test_collect_corrections_type_change(self, corrector, monkeypatch):
        """Changing type returns corrected understanding."""
        u = _make_understanding()
        # type=new_value, rest=Enter
        responses = iter(["idea", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = corrector._collect_corrections(u)
        assert result is not None
        assert result.type == NoteType.IDEA

    def test_collect_corrections_priority_change(self, corrector, monkeypatch):
        """Changing priority returns corrected understanding."""
        u = _make_understanding()
        # type=Enter, priority=0.3, rest=Enter
        responses = iter(["", "0.3", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = corrector._collect_corrections(u)
        assert result is not None
        assert result.priority == 0.3

    def test_collect_corrections_invalid_type_keeps_original(self, corrector, monkeypatch):
        """Invalid type keeps original."""
        u = _make_understanding()
        responses = iter(["badtype", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(responses))
        result = corrector._collect_corrections(u)
        assert result is None  # no valid changes made

    def test_log_correction(self, corrector, logger):
        """_log_correction adds to the buffer."""
        u_pred = _make_understanding(type=NoteType.OBSERVATION)
        u_corr = _make_understanding(type=NoteType.TASK)
        corrector._log_correction("test text", u_pred, u_corr)
        assert len(logger.buffer) == 1
        assert logger.buffer[0].predicted.type == NoteType.OBSERVATION
        assert logger.buffer[0].corrected.type == NoteType.TASK

    def test_confidence_bar(self, corrector):
        """_confidence_bar returns a string."""
        bar = corrector._confidence_bar(0.85)
        assert isinstance(bar, str)
        assert len(bar) > 0

    def test_priority_label(self, corrector):
        """_priority_label maps scores to labels."""
        assert corrector._priority_label(0.95) == "critical"
        assert corrector._priority_label(0.8) == "high"
        assert corrector._priority_label(0.5) == "medium"
        assert corrector._priority_label(0.3) == "low"
        assert corrector._priority_label(0.1) == "someday"
