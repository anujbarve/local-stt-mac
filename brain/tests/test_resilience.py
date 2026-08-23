"""Tests for brain/understand.py and brain/model.py — resilience."""

import pytest

from brain.understand import BrainEngine, _StubEmbedder
from brain.model import BrainModel
from brain.tokenizer import Tokenizer


@pytest.fixture
def engine():
    model = BrainModel(vocab_size=1000)
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.fit(["placeholder text for vocab"])
    embedder = _StubEmbedder()
    return BrainEngine(model, tokenizer, embedder)


class TestHealthCheck:
    def test_health_returns_dict(self, engine):
        """health() returns a dict."""
        result = engine.health()
        assert isinstance(result, dict)

    def test_health_status_healthy(self, engine):
        """health() returns healthy when model is loaded."""
        result = engine.health()
        assert result["status"] == "healthy"
        assert result["model_loaded"] is True
        assert result["tokenizer_ready"] is True

    def test_health_has_temperature(self, engine):
        """health() includes temperature."""
        result = engine.health()
        assert "temperature" in result
        assert isinstance(result["temperature"], float)

    def test_health_has_cache_stats(self, engine):
        """health() includes cache stats."""
        result = engine.health()
        assert "cache_stats" in result

    def test_health_has_device(self, engine):
        """health() includes device info."""
        result = engine.health()
        assert "device" in result

    def test_health_has_parameters(self, engine):
        """health() includes parameter count."""
        result = engine.health()
        assert "parameters" in result
        assert result["parameters"] > 0


class TestLoadWithRetry:
    def test_load_with_retry_success(self, tmp_path):
        """load_with_retry loads successfully on first attempt."""
        model = BrainModel(vocab_size=100)
        path = str(tmp_path / "model.pt")
        model.save(path, version="1.0.0")

        loaded = BrainModel.load_with_retry(path)
        assert loaded is not None

    def test_load_with_retry_nonexistent(self):
        """load_with_retry raises RuntimeError after retries."""
        with pytest.raises(RuntimeError, match="Failed to load model"):
            BrainModel.load_with_retry(
                "/nonexistent/model.pt",
                max_retries=2,
                backoff_factor=0.01,
            )

    def test_load_with_retry_preserves_model(self, tmp_path):
        """load_with_retry returns valid model."""
        model = BrainModel(vocab_size=100)
        path = str(tmp_path / "model.pt")
        model.save(path)

        loaded = BrainModel.load_with_retry(path)
        assert loaded.count_parameters() > 0
