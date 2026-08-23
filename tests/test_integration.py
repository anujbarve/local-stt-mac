"""
Tests for brain integration — end-to-end pipeline.
"""

import os
import tempfile

import pytest

from brain.understand import BrainEngine, _StubEmbedder
from brain.correct import CorrectionLogger
from brain.model import BrainModel
from brain.tokenizer import Tokenizer
from brain_contract import Understanding, NoteType, Correction


def _make_engine():
    """Create a minimal engine for testing."""
    model = BrainModel(vocab_size=1000)
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.fit([
        "I need to fix the VAD threshold by Friday",
        "meeting with Sarah about the project",
        "TODO: update the README",
    ])
    embedder = _StubEmbedder()
    return BrainEngine(model, tokenizer, embedder)


class TestBrainExplain:
    def test_explain_returns_string(self):
        engine = _make_engine()
        u = engine.understand("I need to fix the VAD threshold")
        result = engine.explain(u)
        assert isinstance(result, str)
        assert "Type:" in result
        assert "Priority:" in result

    def test_explain_contains_entities(self):
        engine = _make_engine()
        u = engine.understand("meeting with Sarah")
        result = engine.explain(u)
        assert "Entities:" in result or "Type:" in result


class TestBrainSuggestCorrection:
    def test_suggest_correction_accepted(self, monkeypatch):
        engine = _make_engine()
        u = engine.understand("I need to fix the VAD threshold")
        monkeypatch.setattr("builtins.input", lambda _: "y")
        result = engine.suggest_correction("test text", u)
        assert result["accepted"] is True

    def test_suggest_correction_rejected(self, monkeypatch):
        engine = _make_engine()
        u = engine.understand("I need to fix the VAD threshold")
        monkeypatch.setattr("builtins.input", lambda _: "n")
        result = engine.suggest_correction("test text", u)
        assert result["accepted"] is False


class TestBrainBatchInference:
    def test_batch_returns_list(self):
        engine = _make_engine()
        texts = ["I need to fix the VAD threshold", "meeting with Sarah"]
        results = engine.understand_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, Understanding) for r in results)

    def test_batch_empty(self):
        engine = _make_engine()
        results = engine.understand_batch([])
        assert len(results) == 0


class TestCorrectionLoggerInteractive:
    def test_interactive_correct_accepted(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            u = Understanding(
                text="test text",
                type=NoteType.OBSERVATION,
                priority=0.5,
            )
            # Accept without editing
            monkeypatch.setattr("builtins.input", lambda _: "y")
            result = logger.interactive_correct("test text", u)
            # result should be a Understanding (accepted)
            assert result is not None
            assert isinstance(result, Understanding)

    def test_interactive_correct_rejected(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            u = Understanding(
                text="test text",
                type=NoteType.OBSERVATION,
                priority=0.5,
            )
            monkeypatch.setattr("builtins.input", lambda _: "n")
            result = logger.interactive_correct("test text", u)
            assert result is None

    def test_batch_correct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "corrections.jsonl")
            logger = CorrectionLogger(buffer_path=path)
            corrections = [
                {"text": "text1", "type": "task", "priority": "0.8", "urgency": "soon", "sentiment": "neutral"},
                {"text": "text2", "type": "idea", "priority": "0.5", "urgency": "later", "sentiment": "positive"},
            ]
            count = logger.batch_correct(corrections)
            assert count == 2
            assert len(logger.buffer) == 2


class TestEndToEndPipeline:
    def test_understand_then_explain(self):
        """Full pipeline: text → understanding → explanation."""
        engine = _make_engine()
        text = "I need to fix the VAD threshold before Friday"
        understanding = engine.understand(text)
        explanation = engine.explain(understanding)

        # Untrained model — just verify structure, not specific type
        assert isinstance(understanding.type, NoteType)
        assert 0.0 <= understanding.priority <= 1.0
        assert "Type:" in explanation
        assert "Priority:" in explanation

    def test_should_call_llm(self):
        engine = _make_engine()
        u = engine.understand("some random text")
        decision = engine.should_call_llm(u)
        assert isinstance(decision, bool)
