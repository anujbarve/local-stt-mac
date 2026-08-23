"""
Tests for brain.model — forward pass, output shapes, save/load.
"""

import os
import tempfile

import pytest
import torch

from brain.model import BrainModel
from brain.config import VOCAB_SIZE, MAX_SEQ_LEN, EMBEDDING_DIM
from brain.heads import NUM_BIO_TAGS, NUM_TYPES


class TestBrainModelInit:
    """Tests for model initialization."""

    def test_creates_model(self):
        model = BrainModel()
        assert model is not None

    def test_parameter_count_positive(self):
        model = BrainModel()
        params = model.count_parameters()
        assert params > 0

    def test_custom_vocab_size(self):
        model = BrainModel(vocab_size=1000)
        assert model.backbone.token_embedding.num_embeddings == 1000


class TestBrainModelForward:
    """Tests for forward pass."""

    def test_forward_returns_dict(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        assert isinstance(outputs, dict)

    def test_output_keys(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)

        expected_keys = [
            "classifier", "entities", "priority", "urgency",
            "sentiment", "action", "summary_scores", "temporal", "topics",
        ]
        for key in expected_keys:
            assert key in outputs, f"Missing key: {key}"

    def test_classifier_shape(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        assert outputs["classifier"].shape == (4, NUM_TYPES)

    def test_entity_shape(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        assert outputs["entities"].shape == (4, MAX_SEQ_LEN, NUM_BIO_TAGS)

    def test_priority_shape(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        assert outputs["priority"].shape == (4, 1)

    def test_priority_in_range(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        # Priority uses Sigmoid, so should be in [0, 1]
        assert outputs["priority"].min() >= 0.0
        assert outputs["priority"].max() <= 1.0

    def test_urgency_shape(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        assert outputs["urgency"].shape == (4, 4)

    def test_sentiment_shape(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        assert outputs["sentiment"].shape == (4, 5)

    def test_summary_scores_shape(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        assert outputs["summary_scores"].shape == (4, MAX_SEQ_LEN)

    def test_topics_shape(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        from brain_contract import TOPICS
        assert outputs["topics"].shape == (4, len(TOPICS))

    def test_temporal_has_date_shape(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (4, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        assert outputs["temporal"]["has_date"].shape == (4, 1)
        assert outputs["temporal"]["relative"].shape == (4, 3)
        assert outputs["temporal"]["month"].shape == (4, 12)


class TestBrainModelGradients:
    """Tests for gradient flow."""

    def test_gradients_flow(self):
        model = BrainModel()
        model.train()
        token_ids = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ_LEN))
        outputs = model(token_ids)

        # Compute a dummy loss
        loss = outputs["classifier"].sum() + outputs["priority"].sum()
        loss.backward()

        # Check that at least some parameters have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed through the model"

    def test_no_nan_in_output(self):
        model = BrainModel()
        model.eval()
        token_ids = torch.randint(0, VOCAB_SIZE, (2, MAX_SEQ_LEN))
        with torch.no_grad():
            outputs = model(token_ids)
        for key, val in outputs.items():
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    assert not torch.isnan(v2).any(), f"NaN in {key}.{k2}"
            else:
                assert not torch.isnan(val).any(), f"NaN in {key}"


class TestBrainModelSaveLoad:
    """Tests for save/load."""

    def test_save_and_load(self):
        model = BrainModel(vocab_size=1000)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            model.save(path)
            assert os.path.exists(path)

            loaded = BrainModel.load(path)
            loaded.eval()

            # Verify same outputs
            token_ids = torch.randint(0, 1000, (1, MAX_SEQ_LEN))
            with torch.no_grad():
                out1 = model(token_ids)
                out2 = loaded(token_ids)

            for key in out1:
                if isinstance(out1[key], dict):
                    for k2 in out1[key]:
                        assert torch.allclose(
                            out1[key][k2], out2[key][k2], atol=1e-6
                        ), f"Mismatch in {key}.{k2}"
                else:
                    assert torch.allclose(
                        out1[key], out2[key], atol=1e-6
                    ), f"Mismatch in {key}"

    def test_predict_relation(self):
        model = BrainModel()
        model.eval()
        emb_a = torch.randn(2, EMBEDDING_DIM)
        emb_b = torch.randn(2, EMBEDDING_DIM)
        with torch.no_grad():
            result = model.predict_relation(emb_a, emb_b)
        from brain.heads import NUM_RELATIONS
        assert result.shape == (2, NUM_RELATIONS)
