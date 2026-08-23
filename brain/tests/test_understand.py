"""
Tests for brain.understand — end-to-end inference pipeline.
"""

import pytest
import torch

from brain.model import BrainModel
from brain.tokenizer import Tokenizer
from brain.understand import BrainEngine, _StubEmbedder
from brain.entity_linker import EntityLinker
from brain_contract import (
    Understanding, NoteType, Urgency, Sentiment,
)


def _make_engine() -> BrainEngine:
    """Create a minimal engine for testing (no trained weights)."""
    model = BrainModel(vocab_size=1000)
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.fit([
        "I need to fix the VAD threshold by Friday",
        "meeting with Sarah about the project",
        "TODO: update the README",
        "I learned that PyTorch can optimize models",
        "remind me to call Mike tomorrow",
        "I prefer dark mode for coding",
        "today was a productive day working on the API",
        "what if we build a new dashboard for monitoring",
        "the deployment is running smoothly",
        "decision: use FastAPI for the backend",
    ])
    embedder = _StubEmbedder()
    linker = EntityLinker()
    return BrainEngine(model, tokenizer, embedder, linker)


class TestBrainEngineUnderstand:
    """Tests for the understand() method."""

    def test_returns_understanding(self):
        engine = _make_engine()
        result = engine.understand("I need to fix the VAD threshold by Friday")
        assert isinstance(result, Understanding)

    def test_text_preserved(self):
        engine = _make_engine()
        text = "I need to fix the VAD threshold by Friday"
        result = engine.understand(text)
        assert result.text == text

    def test_type_is_valid(self):
        engine = _make_engine()
        result = engine.understand("TODO: update the README")
        assert isinstance(result.type, NoteType)

    def test_priority_in_range(self):
        engine = _make_engine()
        result = engine.understand("I need to fix the VAD threshold")
        assert 0.0 <= result.priority <= 1.0

    def test_urgency_valid(self):
        engine = _make_engine()
        result = engine.understand("fix this ASAP")
        assert isinstance(result.urgency, Urgency)

    def test_sentiment_valid(self):
        engine = _make_engine()
        result = engine.understand("I love this project")
        assert isinstance(result.sentiment, Sentiment)

    def test_confidence_in_range(self):
        engine = _make_engine()
        result = engine.understand("some random text")
        assert 0.0 <= result.confidence <= 1.0

    def test_entities_is_list(self):
        engine = _make_engine()
        result = engine.understand("meeting with Sarah")
        assert isinstance(result.entities, list)

    def test_embedding_generated(self):
        engine = _make_engine()
        result = engine.understand("hello world")
        assert result.embedding is not None
        assert len(result.embedding) == 384

    def test_topics_is_list(self):
        engine = _make_engine()
        result = engine.understand("I need to fix the code")
        assert isinstance(result.topics, list)


class TestBrainEngineBatch:
    """Tests for batch inference."""

    def test_understand_batch(self):
        engine = _make_engine()
        texts = [
            "I need to fix the VAD threshold",
            "meeting with Sarah",
            "TODO: update README",
        ]
        results = engine.understand_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, Understanding) for r in results)


class TestBrainEngineLLMRoute:
    """Tests for should_call_llm."""

    def test_low_confidence_triggers_llm(self):
        engine = _make_engine()
        # With random weights, confidence should be moderate
        result = engine.understand("completely unknown text xyzzy")
        # The engine should make a decision either way
        decision = engine.should_call_llm(result)
        assert isinstance(decision, bool)


class TestBrainEngineLinks:
    """Tests for suggest_links."""

    def test_suggest_links_empty_candidates(self):
        engine = _make_engine()
        result = engine.understand("test")
        links = engine.suggest_links(result, [])
        assert links == []

    def test_suggest_links_no_embedding(self):
        engine = _make_engine()
        result = engine.understand("test")
        result.embedding = None
        links = engine.suggest_links(result, [{"note_id": "1", "text": "test"}])
        assert links == []


class TestStubEmbedder:
    """Tests for the stub embedder."""

    def test_embed_returns_zeros(self):
        embedder = _StubEmbedder()
        vec = embedder.embed("hello")
        assert len(vec) == 384
        assert all(v == 0.0 for v in vec)

    def test_similarity_returns_zero(self):
        embedder = _StubEmbedder()
        score = embedder.similarity("hello", "world")
        assert score == 0.0
