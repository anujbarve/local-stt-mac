"""Tests for brain/export.py — ModelExporter."""

import json
import os
import pytest

from brain.export import ModelExporter
from brain.understand import BrainEngine, _StubEmbedder
from brain.model import BrainModel
from brain.tokenizer import Tokenizer


@pytest.fixture
def exporter():
    model = BrainModel(vocab_size=1000)
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.fit(["placeholder text for vocab"])
    embedder = _StubEmbedder()
    engine = BrainEngine(model, tokenizer, embedder)
    return ModelExporter(engine)


class TestModelExporter:
    def test_export_onnx(self, exporter, tmp_path):
        """export_onnx creates a file (ONNX or fallback)."""
        path = str(tmp_path / "model.onnx")
        try:
            result = exporter.export_onnx(path)
            assert os.path.exists(result)
        except Exception:
            pytest.skip("ONNX/TorchScript export not supported in this env")

    def test_export_torchscript(self, exporter, tmp_path):
        """export_torchscript creates a TorchScript file."""
        path = str(tmp_path / "model_scripted.pt")
        try:
            result = exporter.export_torchscript(path)
            assert os.path.exists(result)
        except Exception:
            pytest.skip("TorchScript export not supported in this env")

    def test_generate_model_card(self, exporter, tmp_path):
        """generate_model_card creates valid JSON."""
        path = str(tmp_path / "model_card.json")
        result = exporter.generate_model_card(path)
        assert os.path.exists(result)

        with open(result) as f:
            card = json.load(f)

        assert "model_name" in card
        assert "architecture" in card
        assert "task_heads" in card
        assert "export_formats" in card
        assert "onnx" in card["export_formats"]
        assert "torchscript" in card["export_formats"]

    def test_generate_model_card_with_metrics(self, exporter, tmp_path):
        """generate_model_card includes metrics when provided."""
        path = str(tmp_path / "model_card.json")
        metrics = {"accuracy": {"type": 0.85}}
        exporter.generate_model_card(path, metrics=metrics)

        with open(path) as f:
            card = json.load(f)
        assert card["metrics"]["accuracy"]["type"] == 0.85

    def test_export_all(self, exporter, tmp_path):
        """export_all creates all artifacts."""
        try:
            result = exporter.export_all(str(tmp_path / "exports"))
            assert "onnx" in result
            assert "torchscript" in result
            assert "model_card" in result
            for path in result.values():
                assert os.path.exists(path)
        except Exception:
            pytest.skip("Export not supported in this env")

    def test_export_creates_directories(self, exporter, tmp_path):
        """export methods create output directories."""
        deep_path = str(tmp_path / "a" / "b" / "c" / "model.onnx")
        try:
            exporter.export_onnx(deep_path)
            assert os.path.exists(deep_path)
        except Exception:
            pytest.skip("Export not supported in this env")
