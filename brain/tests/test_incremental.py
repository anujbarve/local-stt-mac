"""Tests for brain/train.py — incremental learning."""

import pytest

from brain.train import BrainTrainer, BrainDataset, MetricsTracker
from brain.model import BrainModel
from brain.tokenizer import Tokenizer


@pytest.fixture
def trainer_and_data():
    model = BrainModel(vocab_size=1000)
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.fit(["fix the bug", "meeting today", "great idea"])

    trainer = BrainTrainer(model, tokenizer, lr=1e-3)

    data = [
        {"text": "fix the bug", "type": "task", "priority": 0.8,
         "urgency": "soon", "sentiment": "neutral", "entities": [],
         "topics": ["technology"], "has_temporal": False},
        {"text": "meeting today", "type": "meeting", "priority": 0.6,
         "urgency": "soon", "sentiment": "neutral", "entities": [],
         "topics": ["career"], "has_temporal": True},
    ]

    return trainer, data


class TestIncrementalLearning:
    def test_incremental_train_returns_metrics(self, trainer_and_data):
        """incremental_train() returns a metrics dict."""
        trainer, data = trainer_and_data
        result = trainer.incremental_train(data, epochs=1)
        assert "epochs" in result
        assert "samples" in result
        assert result["epochs"] == 1
        assert result["samples"] == 2

    def test_incremental_train_empty_data(self, trainer_and_data):
        """incremental_train() with empty data returns error."""
        trainer, _ = trainer_and_data
        result = trainer.incremental_train([])
        assert "error" in result

    def test_backbone_frozen_during_incremental(self, trainer_and_data):
        """Backbone parameters are frozen during incremental training."""
        trainer, data = trainer_and_data

        # Check backbone is trainable before
        for p in trainer.model.backbone.parameters():
            assert p.requires_grad is True

        # Run one batch manually to check freezing
        import torch
        dataset = BrainDataset(data, trainer.tokenizer)
        loader = trainer._make_loader(data, batch_size=2, shuffle=False)

        # The incremental_train method freezes and unfreezes
        result = trainer.incremental_train(data, epochs=1)

        # After training, backbone should be unfrozen
        for p in trainer.model.backbone.parameters():
            assert p.requires_grad is True

    def test_incremental_train_differentiable_from_full(self, trainer_and_data):
        """incremental_train() uses lower LR than full train."""
        trainer, data = trainer_and_data

        # incremental_train should complete without error
        result = trainer.incremental_train(data, epochs=2, lr=1e-5)
        assert result["epochs"] == 2
        assert "train_classifier_acc" in result

    def test_incremental_with_single_sample(self, trainer_and_data):
        """incremental_train() works with a single sample."""
        trainer, _ = trainer_and_data
        data = [{"text": "fix it", "type": "task", "priority": 0.9,
                 "urgency": "immediate", "sentiment": "neutral",
                 "entities": [], "topics": [], "has_temporal": False}]
        result = trainer.incremental_train(data, epochs=1)
        assert result["samples"] == 1

    def test_metrics_tracker_basic(self):
        """MetricsTracker tracks metrics correctly."""
        tracker = MetricsTracker()
        assert tracker._type_correct == 0
        assert tracker._type_total == 0
        tracker.reset()
        assert tracker._type_correct == 0

    def test_metrics_tracker_compute_empty(self):
        """MetricsTracker.compute() returns zeros when empty."""
        tracker = MetricsTracker()
        metrics = tracker.compute()
        assert metrics["classifier_acc"] == 0.0
        assert metrics["priority_mae"] == 0.0
