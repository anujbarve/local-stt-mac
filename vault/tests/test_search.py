"""Tests for vault/search.py"""

import os
import tempfile

import pytest

from vault.reader import VaultReader
from vault.index import VaultIndex
from vault.graph import VaultGraph
from vault.search import VaultSearch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_NOTES = {
    "tasks/fix-vad.md": """\
---
id: note_2026_08_23_001
type: task
created: 2026-08-23T14:30:00Z
updated: 2026-08-23T14:30:00Z
source: audio
priority: high
tags:
  - bug
  - vad
  - technology
category: task
---

# Fix VAD Threshold

The Silero VAD threshold is too sensitive.

See [[design-meeting]] for context.
""",
    "daily/sky.md": """\
---
id: note_2026_08_23_002
type: observation
created: 2026-08-23T15:00:00Z
updated: 2026-08-23T15:00:00Z
source: audio
tags:
  - nature
category: observation
---

# The Sky is Blue

Beautiful day today.
""",
    "ideas/design-meeting.md": """\
---
id: note_2026_08_23_003
type: idea
created: 2026-08-23T16:00:00Z
updated: 2026-08-23T16:00:00Z
source: audio
tags:
  - technology
  - engineering
category: idea
---

# Circular Buffer

What if we use a circular buffer?

See [[fix-vad]] for the related task.
""",
}


@pytest.fixture
def vault_dir(tmp_path):
    vault_dir = str(tmp_path / "vault")
    for rel_path, content in SAMPLE_NOTES.items():
        fpath = os.path.join(vault_dir, rel_path)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w") as f:
            f.write(content)
    return vault_dir


@pytest.fixture
def search_components(vault_dir):
    reader = VaultReader(vault_dir)
    index = VaultIndex(index_dir=os.path.join(vault_dir, "_index"))
    graph = VaultGraph(vault_dir)
    graph.build()
    return reader, index, graph


@pytest.fixture
def search(search_components):
    reader, index, graph = search_components
    return VaultSearch(reader, index, graph)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVaultSearch:
    def test_structural_search_by_type(self, search):
        results = search.structural_search({"type": "task"})
        assert len(results) == 1
        assert results[0]["note"]["type"] == "task"

    def test_structural_search_by_tag(self, search):
        results = search.structural_search({"tags": "vad"})
        assert len(results) == 1
        assert "vad" in results[0]["note"]["tags"]

    def test_structural_search_by_priority(self, search):
        results = search.structural_search({"priority": "high"})
        assert len(results) == 1

    def test_structural_search_no_match(self, search):
        results = search.structural_search({"type": "meeting"})
        assert len(results) == 0

    def test_hybrid_search(self, search):
        results = search.search("tasks", strategy="hybrid")
        assert results.total >= 1
        assert results.strategy == "hybrid"

    def test_hybrid_search_returns_query_result(self, search):
        from brain_contract import QueryResult
        result = search.search("technology ideas")
        assert isinstance(result, QueryResult)
        assert len(result.notes) > 0

    def test_empty_vault_search(self, tmp_path):
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir, exist_ok=True)
        reader = VaultReader(empty_dir)
        index = VaultIndex(index_dir=str(tmp_path / "idx"))
        graph = VaultGraph(empty_dir)
        s = VaultSearch(reader, index, graph)
        result = s.search("anything")
        assert result.total == 0

    def test_merge_results(self, search):
        set1 = [{"note": {"id": "a", "type": "task", "body": "a", "tags": []}, "score": 0.8}]
        set2 = [{"note": {"id": "a", "type": "task", "body": "a", "tags": []}, "score": 0.6}]
        merged = search._merge_results([set1, set2], [0.5, 0.5], top_k=5)
        assert len(merged) == 1
        assert merged[0]["score"] == 0.5 * 0.8 + 0.5 * 0.6

    def test_matches_filters(self, search):
        note = {"type": "task", "priority": "high", "tags": ["bug"], "people": [], "projects": []}
        assert search._matches_filters(note, {"type": "task"})
        assert search._matches_filters(note, {"priority": "high"})
        assert search._matches_filters(note, {"tags": "bug"})
        assert not search._matches_filters(note, {"type": "idea"})
