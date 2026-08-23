"""End-to-end vault integration tests."""

import os
import tempfile

import pytest

from brain_contract import (
    Understanding, NoteType, Entity, EntityType, Urgency,
)
from vault.pipeline import VaultPipeline
from vault.reader import VaultReader
from vault.writer import VaultWriter
from vault.index import VaultIndex
from vault.graph import VaultGraph
from vault.search import VaultSearch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault_env(tmp_path):
    """Create a full vault environment."""
    vault_root = str(tmp_path / "vault")
    index_dir = str(tmp_path / "index")
    pipeline = VaultPipeline(vault_root=vault_root, index_dir=index_dir)
    return pipeline


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

class TestReadWriteRoundtrip:
    def test_write_then_read(self, vault_env):
        """Write a note, read it back, verify contents."""
        u = _make_task()
        path = vault_env.writer.write(u)

        # Read it back
        note = vault_env.reader.read(path)
        assert note is not None
        assert note["type"] == "task"
        assert "VAD" in note["body"] or "fix" in note["body"].lower()

    def test_write_then_read_by_id(self, vault_env):
        """Write a note, find it by ID."""
        u = _make_task()
        vault_env.writer.write(u)

        note = vault_env.reader.read_by_id(u.note_id)
        assert note is not None
        assert note["id"] == u.note_id

    def test_write_then_read_by_type(self, vault_env):
        """Write notes, filter by type."""
        vault_env.writer.write(_make_task())
        vault_env.writer.write(_make_observation())

        tasks = vault_env.reader.read_by_type(NoteType.TASK)
        obs = vault_env.reader.read_by_type(NoteType.OBSERVATION)
        assert len(tasks) >= 1
        assert len(obs) >= 1


class TestSearchAfterWrite:
    def test_structural_search_after_write(self, vault_env):
        """Write a note, find it with structural search."""
        vault_env.writer.write(_make_task())

        results = vault_env.search.structural_search({"type": "task"})
        assert len(results) >= 1
        assert results[0]["note"]["type"] == "task"

    def test_tag_search_after_write(self, vault_env):
        """Write a note, find it by tag."""
        vault_env.writer.write(_make_task())

        results = vault_env.search.structural_search({"tags": "vad"})
        assert len(results) >= 1


class TestGraphAfterWrite:
    def test_graph_updated_after_write(self, vault_env):
        """Write a note, verify graph has nodes."""
        vault_env.writer.write(_make_task())
        vault_env.graph.build()

        assert vault_env.graph.num_nodes >= 1


class TestPipelineEndToEnd:
    def test_full_pipeline(self, vault_env):
        """Full pipeline: Understanding → write → index → search."""
        u = _make_task()
        u.embedding = [0.1] * 384  # simulate embedding
        path = vault_env.process(u)

        # Verify note exists
        assert os.path.exists(path)

        # Verify can be read back
        note = vault_env.reader.read(path)
        assert note is not None

        # Verify health check
        health = vault_env.health_check()
        assert health["note_count"] >= 1

    def test_pipeline_safe_end_to_end(self, vault_env):
        """process_safe works end-to-end."""
        u = _make_task()
        path = vault_env.process_safe(u)
        assert path is not None
        assert os.path.exists(path)

    def test_multiple_notes_pipeline(self, vault_env):
        """Process multiple notes through the pipeline."""
        notes = [_make_task(), _make_observation()]
        paths = vault_env.process_batch_safe(notes)
        assert len(paths) == 2
        for p in paths:
            if p:
                assert os.path.exists(p)


class TestErrorHandling:
    def test_process_safe_never_crashes(self, vault_env):
        """Pipeline handles errors gracefully."""
        # Try processing with various bad inputs
        bad = Understanding(text="", type=NoteType.TASK)
        bad.type = "INVALID"
        result = vault_env.process_safe(bad)
        # Should return None, not crash
        assert result is None or isinstance(result, str)

    def test_health_check_always_works(self, vault_env):
        """Health check should always return a dict."""
        health = vault_env.health_check()
        assert isinstance(health, dict)
        assert "vault_exists" in health
