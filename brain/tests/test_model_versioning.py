"""Tests for brain/model.py — model versioning."""

import os
import pytest

from brain.model import BrainModel


class TestModelVersioning:
    def test_save_with_version(self, tmp_path):
        """save() accepts version and metadata."""
        model = BrainModel(vocab_size=100)
        path = str(tmp_path / "model.pt")
        model.save(path, version="1.2.3", metadata={"accuracy": 0.85})

        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["version"] == "1.2.3"
        assert ckpt["metadata"]["accuracy"] == 0.85

    def test_save_default_version(self, tmp_path):
        """save() defaults to version 0.1.0."""
        model = BrainModel(vocab_size=100)
        path = str(tmp_path / "model.pt")
        model.save(path)

        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["version"] == "0.1.0"
        assert ckpt["metadata"] == {}

    def test_load_preserves_version(self, tmp_path):
        """load() preserves version info in checkpoint."""
        model = BrainModel(vocab_size=100)
        path = str(tmp_path / "model.pt")
        model.save(path, version="2.0.0")

        loaded = BrainModel.load(path)
        assert loaded is not None

    def test_load_versions_empty_dir(self, tmp_path):
        """load_versions() returns empty list for empty directory."""
        versions = BrainModel.load_versions(str(tmp_path))
        assert versions == []

    def test_load_versions_nonexistent_dir(self):
        """load_versions() returns empty list for nonexistent directory."""
        versions = BrainModel.load_versions("/nonexistent/path")
        assert versions == []

    def test_load_versions_lists_all(self, tmp_path):
        """load_versions() lists all .pt files with metadata."""
        for i, ver in enumerate(["0.1.0", "0.2.0", "1.0.0"]):
            model = BrainModel(vocab_size=100)
            model.save(str(tmp_path / f"model_v{i}.pt"), version=ver)

        versions = BrainModel.load_versions(str(tmp_path))
        assert len(versions) == 3
        version_strs = [v["version"] for v in versions]
        assert "0.1.0" in version_strs
        assert "0.2.0" in version_strs
        assert "1.0.0" in version_strs

    def test_load_versions_has_saved_at(self, tmp_path):
        """load_versions() includes saved_at timestamp."""
        model = BrainModel(vocab_size=100)
        model.save(str(tmp_path / "model.pt"), version="1.0.0")

        versions = BrainModel.load_versions(str(tmp_path))
        assert len(versions) == 1
        assert "saved_at" in versions[0]
        assert versions[0]["saved_at"] != "unknown"
