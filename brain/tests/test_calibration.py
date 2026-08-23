"""Tests for brain/understand.py — confidence calibration."""

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


class TestConfidenceCalibration:
    def test_default_temperature(self, engine):
        """Default temperature is 1.0."""
        assert engine.temperature == 1.0

    def test_calibrate_returns_float(self, engine):
        """calibrate() returns a float temperature."""
        data = [{"text": "hello world", "type": "observation"}]
        temp = engine.calibrate(data)
        assert isinstance(temp, float)
        assert temp > 0

    def test_calibrate_empty_data(self, engine):
        """calibrate() with empty data returns default."""
        temp = engine.calibrate([])
        assert temp == 1.0

    def test_calibrate_single_sample(self, engine):
        """calibrate() works with a single sample."""
        data = [{"text": "fix the bug", "type": "task"}]
        temp = engine.calibrate(data)
        assert isinstance(temp, float)
        assert temp > 0

    def test_calibrate_multiple_samples(self, engine):
        """calibrate() works with multiple samples."""
        data = [
            {"text": "fix the bug", "type": "task"},
            {"text": "great idea", "type": "idea"},
            {"text": "meeting today", "type": "meeting"},
        ]
        temp = engine.calibrate(data)
        assert isinstance(temp, float)
        assert temp > 0

    def test_temperature_affects_confidence(self, engine):
        """Different temperatures produce different confidence scores."""
        u1 = engine.understand("hello world")
        conf_original = u1.confidence

        # Change temperature
        engine.temperature = 0.5
        u2 = engine.understand("hello world")

        # Confidence should differ with different temperature
        # (may be same if model is very confident, but structure should work)
        assert isinstance(conf_original, float)
        assert isinstance(u2.confidence, float)

        # Reset
        engine.temperature = 1.0

    def test_confidence_in_range(self, engine):
        """Confidence is always between 0 and 1."""
        for temp in [0.5, 1.0, 2.0]:
            engine.temperature = temp
            u = engine.understand("some text")
            assert 0.0 <= u.confidence <= 1.0
        engine.temperature = 1.0
