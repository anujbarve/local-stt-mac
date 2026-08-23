"""
Tests for brain.train — MetricsTracker, LR scheduling, early stopping, augmentation.
"""

import os
import tempfile

import pytest
import torch

from brain.train import (
    MetricsTracker, DataAugmenter, BrainTrainer,
    compute_loss, load_vault_data, save_history, load_history,
    BrainDataset,
)
from brain.model import BrainModel
from brain.tokenizer import Tokenizer
from brain.config import VOCAB_SIZE, MAX_SEQ_LEN


def _make_sample(type_str: str = "task") -> dict:
    return {
        "text": "I need to fix the VAD threshold before Friday",
        "type": type_str,
        "priority": 0.8,
        "urgency": "soon",
        "sentiment": "neutral",
        "entities": [{"name": "VAD", "type": "tool"}],
        "topics": ["technology"],
        "has_temporal": True,
    }


def _make_batch(n: int = 4):
    return [_make_sample() for _ in range(n)]


def _make_trainer():
    model = BrainModel(vocab_size=1000)
    tok = Tokenizer(vocab_size=1000)
    tok.fit(["I need to fix the VAD threshold before Friday"])
    return BrainTrainer(model, tok, lr=1e-4, scheduler_type="none", patience=5)


# =============================================================================
# MetricsTracker
# =============================================================================

class TestMetricsTracker:
    def test_reset(self):
        m = MetricsTracker()
        m._type_correct = 10
        m.reset()
        result = m.compute()
        assert result["classifier_acc"] == 0.0

    def test_compute_empty(self):
        m = MetricsTracker()
        result = m.compute()
        assert result["classifier_acc"] == 0.0
        assert result["priority_mae"] == 0.0

    def test_update_and_compute(self):
        m = MetricsTracker()
        model = BrainModel(vocab_size=1000)
        model.eval()
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["test"])
        ds = BrainDataset(_make_batch(4), tok)
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        batch = next(iter(loader))
        with torch.no_grad():
            outputs = model(batch["token_ids"])
        m.update(outputs, batch)
        result = m.compute()
        assert "classifier_acc" in result
        assert "priority_mae" in result
        assert "entity_f1" in result
        assert "topic_f1" in result

    def test_classifier_accuracy_perfect(self):
        m = MetricsTracker()
        m._type_correct = 10
        m._type_total = 10
        result = m.compute()
        assert result["classifier_acc"] == 1.0

    def test_priority_mae(self):
        m = MetricsTracker()
        m._priority_errors = [0.1, 0.2, 0.3]
        result = m.compute()
        assert abs(result["priority_mae"] - 0.2) < 1e-6


# =============================================================================
# DataAugmenter
# =============================================================================

class TestDataAugmenter:
    def test_augment_increases_size(self):
        aug = DataAugmenter(seed=42)
        data = [_make_sample() for _ in range(10)]
        augmented = aug.augment(data, factor=3)
        assert len(augmented) == 30

    def test_original_data_preserved(self):
        aug = DataAugmenter(seed=42)
        data = [_make_sample()]
        original_text = data[0]["text"]
        augmented = aug.augment(data, factor=2)
        # First copy should be original
        assert augmented[0]["text"] == original_text

    def test_synonym_replace(self):
        aug = DataAgumenter_safe(seed=42)
        result = aug._synonym_replace("I need to fix the bug")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_random_delete(self):
        aug = DataAugmenter(seed=42)
        result = aug._random_delete("I need to fix the bug before Friday", p=0.3)
        assert isinstance(result, str)

    def test_random_swap(self):
        aug = DataAugmenter(seed=42)
        result = aug._random_swap("I need to fix the bug")
        assert isinstance(result, str)
        words = result.split()
        assert len(words) == 6  # original has 6 words


# Helper to avoid name collision
class DataAgumenter_safe(DataAugmenter):
    pass


# =============================================================================
# BrainTrainer enhancements
# =============================================================================

class TestBrainTrainerScheduler:
    def test_cosine_scheduler(self):
        model = BrainModel(vocab_size=1000)
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["test"])
        trainer = BrainTrainer(model, tok, scheduler_type="cosine")
        assert trainer.scheduler is not None
        assert trainer.scheduler_type == "cosine"

    def test_plateau_scheduler(self):
        trainer = _make_trainer()
        trainer.scheduler_type = "plateau"
        model = BrainModel(vocab_size=1000)
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["test"])
        t = BrainTrainer(model, tok, scheduler_type="plateau")
        assert t.scheduler is not None

    def test_none_scheduler(self):
        trainer = _make_trainer()
        assert trainer.scheduler is None

    def test_early_stopping_patience(self):
        trainer = _make_trainer()
        assert trainer.patience == 5
        assert trainer.wait == 0


class TestBrainTrainerTrain:
    def test_train_runs(self):
        trainer = _make_trainer()
        data = _make_batch(8)
        history = trainer.train(data[:6], data[6:], epochs=2, verbose=False)
        assert len(history) == 2
        assert "train_total" in history[0]
        assert "val_total" in history[0]

    def test_train_with_scheduler(self):
        trainer = _make_trainer()
        trainer.scheduler_type = "cosine"
        model = BrainModel(vocab_size=1000)
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["test"])
        t = BrainTrainer(model, tok, scheduler_type="cosine")
        data = _make_batch(8)
        history = t.train(data[:6], data[6:], epochs=2, verbose=False)
        assert len(history) >= 1

    def test_checkpoint_saves(self):
        trainer = _make_trainer()
        data = _make_batch(8)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.train(data[:6], data[6:], epochs=2, checkpoint_dir=tmpdir, verbose=False)
            files = os.listdir(tmpdir)
            assert any("model" in f for f in files)


# =============================================================================
# History save/load
# =============================================================================

class TestHistory:
    def test_save_and_load(self):
        history = [{"epoch": 1, "train_total": 5.0}, {"epoch": 2, "train_total": 4.0}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "history.json")
            save_history(history, path)
            loaded = load_history(path)
            assert len(loaded) == 2
            assert loaded[0]["epoch"] == 1


# =============================================================================
# BrainDataset
# =============================================================================

class TestBrainDataset:
    def test_len(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["test"])
        ds = BrainDataset(_make_batch(5), tok)
        assert len(ds) == 5

    def test_getitem_shapes(self):
        tok = Tokenizer(vocab_size=1000)
        tok.fit(["test"])
        ds = BrainDataset(_make_batch(2), tok)
        sample = ds[0]
        assert sample["token_ids"].shape == (MAX_SEQ_LEN,)
        assert sample["type_label"].shape == ()
        assert sample["priority_score"].shape == ()
        assert sample["urgency_label"].shape == ()
        assert sample["sentiment_label"].shape == ()
        assert sample["bio_labels"].shape == (MAX_SEQ_LEN,)
        assert sample["has_date"].shape == ()
