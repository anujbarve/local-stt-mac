"""Tests for brain/distill.py — DistillationPipeline."""

import pytest

from brain.distill import DistillationPipeline
from brain.correct import CorrectionLogger
from brain_contract import (
    Understanding, NoteType, Urgency, Sentiment, Correction,
)


@pytest.fixture
def logger(tmp_path):
    path = str(tmp_path / "corrections.jsonl")
    return CorrectionLogger(buffer_path=path)


def _make_u(**overrides):
    defaults = dict(
        text="test text",
        type=NoteType.OBSERVATION,
        priority=0.5,
        urgency=Urgency.LATER,
        sentiment=Sentiment.NEUTRAL,
        confidence=0.6,
    )
    defaults.update(overrides)
    return Understanding(**defaults)


class StubBrain:
    """Stub brain engine for testing."""

    def __init__(self, return_u=None):
        self._return_u = return_u or _make_u()

    def understand(self, text):
        return self._return_u


class StubLLM:
    """Stub LLM for testing."""

    def __init__(self, return_u=None, available=True):
        self._return_u = return_u or _make_u()
        self.is_available = available

    def analyze(self, text):
        return self._return_u


class TestDistillationPipeline:
    def test_compare_outputs_same(self, logger):
        """Identical outputs should match."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)

        u1 = _make_u(type=NoteType.TASK, priority=0.8)
        u2 = _make_u(type=NoteType.TASK, priority=0.8)

        result = distiller.compare_outputs(u1, u2)
        assert result["match"] is True
        assert len(result["differences"]) == 0

    def test_compare_outputs_different_type(self, logger):
        """Different types should not match."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)

        u1 = _make_u(type=NoteType.OBSERVATION)
        u2 = _make_u(type=NoteType.TASK)

        result = distiller.compare_outputs(u1, u2)
        assert result["match"] is False
        assert any(d["field"] == "type" for d in result["differences"])

    def test_compare_outputs_different_priority(self, logger):
        """Different priorities should not match."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)

        u1 = _make_u(priority=0.3)
        u2 = _make_u(priority=0.9)

        result = distiller.compare_outputs(u1, u2)
        assert result["match"] is False
        assert any(d["field"] == "priority" for d in result["differences"])

    def test_compare_outputs_close_priority(self, logger):
        """Close priorities should match (within 0.1 tolerance)."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)

        u1 = _make_u(priority=0.50)
        u2 = _make_u(priority=0.55)

        result = distiller.compare_outputs(u1, u2)
        assert result["match"] is True

    def test_compare_outputs_different_urgency(self, logger):
        """Different urgencies should not match."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)

        u1 = _make_u(urgency=Urgency.SOON)
        u2 = _make_u(urgency=Urgency.LATER)

        result = distiller.compare_outputs(u1, u2)
        assert result["match"] is False
        assert any(d["field"] == "urgency" for d in result["differences"])

    def test_compare_outputs_different_entities(self, logger):
        """Different entities should not match."""
        from brain_contract import Entity, EntityType
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)

        u1 = _make_u(entities=[Entity(name="Alice", entity_type=EntityType.PERSON)])
        u2 = _make_u(entities=[Entity(name="Bob", entity_type=EntityType.PERSON)])

        result = distiller.compare_outputs(u1, u2)
        assert result["match"] is False
        assert any(d["field"] == "entities" for d in result["differences"])

    def test_process_transcription_no_llm(self, logger):
        """Without LLM, returns NN output."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)

        result = distiller.process_transcription("hello")
        assert isinstance(result, Understanding)
        assert len(logger.buffer) == 0  # no correction logged

    def test_process_transcription_with_matching_llm(self, logger):
        """With matching LLM, no correction logged."""
        u = _make_u(type=NoteType.TASK)
        brain = StubBrain(return_u=u)
        llm = StubLLM(return_u=u)
        distiller = DistillationPipeline(brain, llm=llm, correction_logger=logger)

        result = distiller.process_transcription("hello")
        assert result.type == NoteType.TASK
        assert len(logger.buffer) == 0

    def test_process_transcription_with_mismatching_llm(self, logger):
        """With mismatching LLM, correction is logged."""
        nn_u = _make_u(type=NoteType.OBSERVATION)
        llm_u = _make_u(type=NoteType.TASK)
        brain = StubBrain(return_u=nn_u)
        llm = StubLLM(return_u=llm_u)
        distiller = DistillationPipeline(brain, llm=llm, correction_logger=logger)

        result = distiller.process_transcription("hello")
        assert result.type == NoteType.TASK  # LLM output wins
        assert len(logger.buffer) == 1

    def test_should_retrain_false(self, logger):
        """should_retrain returns False when not enough corrections."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)
        assert distiller.should_retrain() is False

    def test_retrain_no_data(self, logger):
        """retrain with no data returns error."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)
        result = distiller.retrain()
        assert "error" in result

    def test_run_distillation_loop(self, logger):
        """run_distillation_loop returns stats."""
        brain = StubBrain()
        distiller = DistillationPipeline(brain, llm=None, correction_logger=logger)

        result = distiller.run_distillation_loop(["text1", "text2", "text3"])
        assert result["total"] == 3
        assert result["nn_correct"] == 3
        assert result["llm_corrections"] == 0
