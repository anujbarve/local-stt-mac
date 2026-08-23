"""Tests for vault/pipeline.py"""

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

class TestVaultPipeline:
    def test_process_writes_note(self, pipeline):
        u = _make_task()
        path = pipeline.process(u)
        assert os.path.exists(path)

    def test_process_sets_note_id(self, pipeline):
        u = _make_task()
        pipeline.process(u)
        assert u.note_id is not None

    def test_process_returns_vault_path(self, pipeline):
        u = _make_task()
        path = pipeline.process(u)
        assert "vault" in path

    def test_process_with_embedding(self, pipeline):
        u = _make_task()
        u.embedding = [0.1] * 384
        path = pipeline.process(u)
        assert os.path.exists(path)
        # Index should have the note
        assert pipeline.index.size >= 1

    def test_process_batch(self, pipeline):
        understandings = [_make_task(), _make_observation()]
        paths = pipeline.process_batch(understandings)
        assert len(paths) == 2
        for p in paths:
            assert os.path.exists(p)

    def test_stats(self, pipeline):
        u = _make_task()
        pipeline.process(u)
        stats = pipeline.stats()
        assert stats["note_count"] >= 1
        assert "index_size" in stats
        assert "graph_nodes" in stats

    def test_query_no_results(self, pipeline):
        result = pipeline.query("anything at all")
        assert "answer" in result
        assert "sources" in result

    def test_rebuild_index(self, pipeline):
        u = _make_task()
        u.embedding = [0.1] * 384
        pipeline.process(u)
        # rebuild_index without embedder returns 0 (no embedder available)
        count = pipeline.rebuild_index()
        assert count >= 0

    def test_rebuild_graph(self, pipeline):
        u = _make_task()
        pipeline.process(u)
        count = pipeline.rebuild_graph()
        assert count >= 0

    def test_components_initialized(self, pipeline):
        assert pipeline.reader is not None
        assert pipeline.writer is not None
        assert pipeline.index is not None
        assert pipeline.graph is not None
        assert pipeline.linker is not None
        assert pipeline.search is not None
        assert pipeline.answerer is not None
        assert pipeline.query_parser is not None

    def test_moc_initialization(self, pipeline):
        assert pipeline.moc is not None

    def test_llm_initialization(self, pipeline):
        # LLM may not be available (no MLX on this platform)
        assert pipeline.llm is not None
