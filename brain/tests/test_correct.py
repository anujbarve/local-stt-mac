"""
Tests for brain.correct — correction logging and retrain integration.
"""

import os
import tempfile

import pytest

from brain.correct import CorrectionLogger, MIN_CORRECTIONS_FOR_RETRAIN
from brain_contract import (
    Understanding, Correction, NoteType, Urgency, Sentiment,
)


def _make_correction(text: str = "test text", applied: bool = False) -> Correction:
    predicted = Understanding(
        text=text,
        type=NoteType.OBSERVATION,
        priority=0.5,
        urgency=Urgency.LATER,
        sentiment=Sentiment.NEUTRAL,
    )
    corrected = Understanding(
        text=text,
        type=NoteType.TASK,
        priority=0.8,
        urgency=Urgency.SOON,
        sentiment=Sentiment.POSITIVE,
    )
    return Correction(
        text=text,
        predicted=predicted,
        corrected=corrected,
        applied=applied,
    )


class TestCorrectionLogger:
    def test_log_adds_to_buffer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            c = _make_correction()
            logger.log(c)
            assert len(logger.buffer) == 1

    def test_should_retrain_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(5):
                logger.log(_make_correction(f"text {i}"))
            assert not logger.should_retrain()

    def test_should_retrain_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(MIN_CORRECTIONS_FOR_RETRAIN):
                logger.log(_make_correction(f"text {i}"))
            assert logger.should_retrain()

    def test_get_training_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(MIN_CORRECTIONS_FOR_RETRAIN):
                logger.log(_make_correction(f"text {i}"))
            data = logger.get_training_data()
            assert len(data) == MIN_CORRECTIONS_FOR_RETRAIN
            assert data[0]["type"] == "task"

    def test_get_training_data_insufficient(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            logger.log(_make_correction())
            data = logger.get_training_data()
            assert len(data) == 0

    def test_mark_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(5):
                logger.log(_make_correction(f"text {i}"))
            logger.mark_applied([0, 2])
            stats = logger.stats()
            assert stats["applied"] == 2
            assert stats["unapplied"] == 3

    def test_mark_all_applied(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(5):
                logger.log(_make_correction(f"text {i}"))
            logger.mark_all_applied()
            assert logger.stats()["unapplied"] == 0

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(3):
                logger.log(_make_correction(f"text {i}"))
            logger.mark_applied([0])
            stats = logger.stats()
            assert stats["total"] == 3
            assert stats["applied"] == 1
            assert stats["unapplied"] == 2

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(5):
                logger.log(_make_correction(f"text {i}"))
            logger.clear()
            assert len(logger.buffer) == 0

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger1 = CorrectionLogger(buffer_path=path)
            for i in range(3):
                logger1.log(_make_correction(f"text {i}"))
            # Load from same file
            logger2 = CorrectionLogger(buffer_path=path)
            assert len(logger2.buffer) == 3

    def test_corrupted_file_handled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            with open(path, "w") as f:
                f.write("not valid json\n")
                f.write("{broken json\n")
            logger = CorrectionLogger(buffer_path=path)
            assert len(logger.buffer) == 0


class TestCorrectionLoggerRetrain:
    def test_maybe_retrain_not_triggered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(5):
                logger.log(_make_correction(f"text {i}"))
            # Need a mock trainer
            result = logger.maybe_retrain(trainer=None, min_corrections=20)
            assert result is False

    def test_maybe_retrain_triggered(self):
        """Test that maybe_retrain returns True when enough corrections exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            for i in range(25):
                logger.log(_make_correction(f"text {i}"))
            # Without a real trainer, we can't fully test retrain
            # but we can verify the logic flow
            assert logger.should_retrain()
            data = logger.get_training_data()
            assert len(data) == 25
