"""Tests for vault/index.py"""

import os
import shutil
import tempfile

import pytest

from vault.index import VaultIndex, _cosine_similarity, HAS_FAISS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def index_dir(tmp_path):
    return str(tmp_path / "vault_index")


@pytest.fixture
def index(index_dir):
    return VaultIndex(index_dir=index_dir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVaultIndex:
    def test_initial_state(self, index):
        assert index.size == 0

    def test_add_single(self, index):
        index.add("note_1", "/path/to/note1.md", [1.0] * 384)
        assert index.size == 1

    def test_add_multiple(self, index):
        for i in range(5):
            index.add(f"note_{i}", f"/path/note{i}.md", [float(i)] * 384)
        assert index.size == 5

    def test_search_returns_results(self, index):
        # Add some notes with distinct embeddings
        index.add("note_1", "/p1.md", [1.0] + [0.0] * 383)
        index.add("note_2", "/p2.md", [0.0, 1.0] + [0.0] * 382)
        index.add("note_3", "/p3.md", [0.0, 0.0, 1.0] + [0.0] * 381)

        results = index.search([1.0] + [0.0] * 383, top_k=3)
        assert len(results) > 0
        assert results[0]["note_id"] == "note_1"

    def test_search_scores_descending(self, index):
        index.add("far", "/far.md", [0.0, 1.0] + [0.0] * 382)
        index.add("near", "/near.md", [1.0, 0.0] + [0.0] * 382)
        index.add("mid", "/mid.md", [0.5, 0.5] + [0.0] * 382)

        results = index.search([1.0] + [0.0] * 383, top_k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_remove(self, index):
        index.add("note_1", "/p1.md", [1.0] * 384)
        index.add("note_2", "/p2.md", [0.0] * 384)

        assert index.remove("note_1")
        assert index.size == 1  # note_2 still active

    def test_remove_nonexistent(self, index):
        assert not index.remove("nonexistent")

    def test_save_and_load(self, index):
        index.add("note_1", "/p1.md", [1.0] + [0.0] * 383)
        index.add("note_2", "/p2.md", [0.0, 1.0] + [0.0] * 382)
        index.save()

        # Create new index from same dir
        index2 = VaultIndex(index_dir=index.index_dir)
        assert index2.size == 2

    def test_load_no_index_returns_false(self, index_dir):
        idx = VaultIndex(index_dir=index_dir)
        assert idx.size == 0

    def test_clear(self, index):
        index.add("note_1", "/p1.md", [1.0] * 384)
        index.clear()
        assert index.size == 0

    def test_stats(self, index):
        index.add("note_1", "/p1.md", [1.0] * 384)
        stats = index.stats()
        assert stats["total_entries"] == 1
        assert stats["active_entries"] == 1
        assert stats["embedding_dim"] == 384

    def test_empty_search(self, index):
        results = index.search([1.0] * 384, top_k=10)
        assert results == []

    def test_build_from_reader(self, tmp_path):
        """Test build with a mock reader."""
        vault_dir = str(tmp_path / "vault")
        os.makedirs(os.path.join(vault_dir, "tasks"), exist_ok=True)
        with open(os.path.join(vault_dir, "tasks", "n1.md"), "w") as f:
            f.write("---\nid: n1\ntype: task\ncreated: x\nupdated: x\nsource: audio\n---\n\nSome body text\n")

        from vault.reader import VaultReader
        reader = VaultReader(vault_dir)

        # Build with a mock embedder
        class MockEmbedder:
            def encode(self, texts):
                return [[0.1] * 384 for _ in texts]

        idx = VaultIndex(index_dir=str(tmp_path / "idx"))
        count = idx.build(reader, MockEmbedder())
        assert count == 1
        assert idx.size == 1


class TestCosineSim:
    def test_identical(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_different_lengths(self):
        assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0
