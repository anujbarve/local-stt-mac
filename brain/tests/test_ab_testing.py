"""Tests for brain/understand.py — A/B testing with ModelRegistry."""

import pytest

from brain.understand import BrainEngine, ModelRegistry, _StubEmbedder
from brain.model import BrainModel
from brain.tokenizer import Tokenizer


def _make_engine():
    model = BrainModel(vocab_size=1000)
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.fit(["placeholder text for vocab"])
    embedder = _StubEmbedder()
    return BrainEngine(model, tokenizer, embedder)


class TestModelRegistry:
    def test_register_model(self):
        """register() adds a model to the registry."""
        registry = ModelRegistry()
        engine = _make_engine()
        registry.register("model_a", engine)
        assert "model_a" in registry.models

    def test_unregister_model(self):
        """unregister() removes a model from the registry."""
        registry = ModelRegistry()
        engine = _make_engine()
        registry.register("model_a", engine)
        registry.unregister("model_a")
        assert "model_a" not in registry.models

    def test_unregister_nonexistent(self):
        """unregister() with nonexistent name doesn't crash."""
        registry = ModelRegistry()
        registry.unregister("nonexistent")  # should not raise

    def test_compare_two_models(self):
        """compare() with two models returns agreement rate."""
        registry = ModelRegistry()
        registry.register("a", _make_engine())
        registry.register("b", _make_engine())

        texts = ["hello world", "fix the bug", "meeting today"]
        result = registry.compare(texts)

        assert "agreement_rate" in result
        assert "total_texts" in result
        assert result["total_texts"] == 3
        assert "a" in result
        assert "b" in result

    def test_compare_single_model(self):
        """compare() with one model has agreement_rate of 1.0."""
        registry = ModelRegistry()
        registry.register("a", _make_engine())

        result = registry.compare(["hello"])
        assert result["agreement_rate"] == 1.0

    def test_compare_empty_texts(self):
        """compare() with empty texts returns 0 total_texts."""
        registry = ModelRegistry()
        registry.register("a", _make_engine())
        result = registry.compare([])
        assert result["total_texts"] == 0

    def test_compare_per_model_metrics(self):
        """compare() returns per-model avg_confidence and type_distribution."""
        registry = ModelRegistry()
        registry.register("a", _make_engine())

        result = registry.compare(["hello world"])
        model_result = result["a"]
        assert "avg_confidence" in model_result
        assert "type_distribution" in model_result
        assert "num_samples" in model_result
        assert model_result["num_samples"] == 1

    def test_predict_all(self):
        """predict_all() runs all models on a single text."""
        registry = ModelRegistry()
        registry.register("a", _make_engine())
        registry.register("b", _make_engine())

        result = registry.predict_all("hello world")
        assert "a" in result
        assert "b" in result
        for name in ["a", "b"]:
            assert "type" in result[name]
            assert "confidence" in result[name]
            assert "priority" in result[name]
            assert "urgency" in result[name]

    def test_predict_all_single_model(self):
        """predict_all() works with a single model."""
        registry = ModelRegistry()
        registry.register("a", _make_engine())
        result = registry.predict_all("hello")
        assert "a" in result
        assert len(result) == 1

    def test_compare_model_metrics_structure(self):
        """compare() model results have correct structure."""
        registry = ModelRegistry()
        registry.register("a", _make_engine())
        result = registry.compare(["test"])
        model_a = result["a"]
        assert isinstance(model_a["avg_confidence"], float)
        assert isinstance(model_a["avg_priority"], float)
        assert isinstance(model_a["type_distribution"], dict)
