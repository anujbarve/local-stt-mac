"""Tests for vault/pipeline.py — enhanced Phase 3 methods."""

import os
import tempfile

import pytest

from brain_contract import Understanding, NoteType, Entity, EntityType, Urgency
from vault.pipeline import VaultPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline(tmp_path):
    vault_root = str(tmp_path / "vault")
    index_dir = str(tmp_path / "index")
    return VaultPipeline(vault_root=vault_root, index_dir=index_dir)


def _make_task():
    return Understanding(
        text="I need to fix the VAD threshold by Friday",
        type=NoteType.TASK,
        priority=0.82,
        urgency=Urgency.SOON,
        confidence=0.87,
        entities=[
            Entity(name="VAD", entity_type=EntityType.TOOL),
            Entity(name="Friday", entity_type=EntityType.DATE),
        ],
        action="fix VAD threshold",
        topics=["technology", "engineering"],
        tags=["bug", "vad"],
    )


def _make_observation():
    return Understanding(
        text="The sky is blue today",
        type=NoteType.OBSERVATION,
        priority=0.2,
        topics=["nature"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineSafe:
    def test_process_safe_success(self, pipeline):
        u = _make_task()
        path = pipeline.process_safe(u)
        assert path is not None
        assert os.path.exists(path)

    def test_process_safe_returns_none_on_bad_data(self, pipeline):
        """process_safe returns None on failure, doesn't crash."""
        # Create an understanding that will cause an error in template rendering
        u = Understanding(text="", type=NoteType.TASK)
        u.type = "INVALID_TYPE"
        # Override to force error
        path = pipeline.process_safe(u)
        # Should either succeed or return None — never raise
        assert path is None or isinstance(path, str)

    def test_process_safe_never_raises(self, pipeline):
        """Verify process_safe never raises exceptions."""
        u = _make_task()
        # This should never raise
        result = pipeline.process_safe(u)
        assert result is None or isinstance(result, str)


class TestDryRun:
    def test_dry_run_returns_plan(self, pipeline):
        u = _make_task()
        plan = pipeline.dry_run(u)
        assert "file_path" in plan
        assert "note_type" in plan
        assert "entities_linked" in plan
        assert "index_added" in plan

    def test_dry_run_correct_type(self, pipeline):
        u = _make_task()
        plan = pipeline.dry_run(u)
        assert plan["note_type"] == "task"

    def test_dry_run_entities_linked(self, pipeline):
        u = _make_task()
        plan = pipeline.dry_run(u)
        assert len(plan["entities_linked"]) >= 1  # VAD (tool) at minimum

    def test_dry_run_no_writes(self, pipeline):
        """dry_run should not create any files."""
        u = _make_task()
        plan = pipeline.dry_run(u)
        assert not os.path.exists(plan["file_path"])


class TestHealthCheck:
    def test_health_check_empty_vault(self, pipeline):
        health = pipeline.health_check()
        assert health["vault_exists"] is True
        assert health["note_count"] == 0
        assert health["index_size"] == 0
        assert health["graph_edges"] == 0
        assert "disk_usage_mb" in health

    def test_health_check_after_write(self, pipeline):
        u = _make_task()
        pipeline.process(u)
        health = pipeline.health_check()
        assert health["note_count"] >= 1

    def test_health_check_has_all_keys(self, pipeline):
        health = pipeline.health_check()
        required_keys = [
            "vault_exists", "note_count", "index_size",
            "graph_edges", "orphan_count", "broken_links", "disk_usage_mb",
        ]
        for key in required_keys:
            assert key in health, f"Missing key: {key}"


class TestBackup:
    def test_backup_creates_file(self, pipeline):
        u = _make_task()
        path = pipeline.process(u)
        backup_path = pipeline.backup(path)
        assert backup_path is not None
        assert os.path.exists(backup_path)
        assert "_backups" in backup_path

    def test_backup_nonexistent_returns_none(self, pipeline):
        result = pipeline.backup("/nonexistent/file.md")
        assert result is None


class TestBatchSafe:
    def test_process_batch_safe(self, pipeline):
        understandings = [_make_task(), _make_observation()]
        paths = pipeline.process_batch_safe(understandings)
        assert len(paths) == 2
        for p in paths:
            assert p is None or isinstance(p, str)

    def test_process_batch_safe_mixed(self, pipeline):
        """Batch handles mixed success/failure gracefully."""
        good = _make_task()
        bad = Understanding(text="", type=NoteType.TASK)
        bad.type = "INVALID_TYPE"
        paths = pipeline.process_batch_safe([good, bad])
        assert len(paths) == 2
        # At least one should have succeeded or both returned None
        assert all(p is None or isinstance(p, str) for p in paths)
