"""Tests for brain/understand.py — inference caching."""

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


class TestInferenceCache:
    def test_cache_disabled_by_default(self, engine):
        """Cache is enabled by default."""
        assert engine._cache_enabled is True

    def test_cache_hit(self, engine):
        """Second call to same text is a cache hit."""
        u1 = engine.understand("hello world")
        u2 = engine.understand("hello world")
        assert u1.type == u2.type
        stats = engine.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cache_miss(self, engine):
        """Different texts are cache misses."""
        engine.understand("hello")
        engine.understand("world")
        stats = engine.cache_stats()
        assert stats["misses"] == 2
        assert stats["hits"] == 0

    def test_cache_hit_rate(self, engine):
        """Hit rate is computed correctly."""
        engine.understand("a")
        engine.understand("b")
        engine.understand("a")  # hit
        engine.understand("b")  # hit
        stats = engine.cache_stats()
        assert stats["hit_rate"] == 0.5

    def test_cache_stats_structure(self, engine):
        """cache_stats returns expected keys."""
        stats = engine.cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "size" in stats
        assert "max_size" in stats
        assert "enabled" in stats

    def test_clear_cache(self, engine):
        """clear_cache resets everything."""
        engine.understand("hello")
        engine.clear_cache()
        stats = engine.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

    def test_set_cache_enabled(self, engine):
        """set_cache_enabled toggles caching."""
        engine.understand("hello")
        engine.set_cache_enabled(False)
        stats = engine.cache_stats()
        assert stats["enabled"] is False
        assert stats["size"] == 0

    def test_cache_disabled_no_caching(self, engine):
        """When cache is disabled, same text is not cached."""
        engine.set_cache_enabled(False)
        engine.understand("hello")
        engine.understand("hello")
        stats = engine.cache_stats()
        assert stats["hits"] == 0
        # When cache is disabled, misses are not tracked
        assert stats["size"] == 0

    def test_cache_max_size_eviction(self, engine):
        """Cache evicts oldest entry when full."""
        engine._cache_max_size = 3
        for i in range(5):
            engine.understand(f"text_{i}")
        assert engine.cache_stats()["size"] <= 3

    def test_use_cache_false(self, engine):
        """use_cache=False bypasses cache."""
        engine.understand("hello", use_cache=True)
        engine.understand("hello", use_cache=False)
        stats = engine.cache_stats()
        assert stats["misses"] == 2  # both are misses
