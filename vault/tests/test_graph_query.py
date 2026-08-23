"""Tests for vault/graph_query.py"""

import os

import pytest

from vault.graph import VaultGraph
from vault.graph_query import GraphQueryEngine


@pytest.fixture
def graph_with_notes(tmp_path):
    vault_dir = str(tmp_path / "vault")
    os.makedirs(os.path.join(vault_dir, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(vault_dir, "ideas"), exist_ok=True)

    notes = {
        "tasks/n1.md": "---\nid: n1\n---\n\nSee [[n2]].\n",
        "tasks/n2.md": "---\nid: n2\n---\n\nSee [[n3]].\n",
        "ideas/n3.md": "---\nid: n3\n---\n\nSee [[n1]].\n",
    }
    for path, content in notes.items():
        full = os.path.join(vault_dir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)

    g = VaultGraph(vault_dir)
    g.build()
    return GraphQueryEngine(g)


class TestGraphQueryEngine:
    def test_find_path(self, graph_with_notes):
        result = graph_with_notes.find_path("n1", "n3")
        assert result["distance"] >= 1
        assert "n1" in result["path"]
        assert "n3" in result["path"]

    def test_find_path_no_path(self, graph_with_notes):
        result = graph_with_notes.find_path("n1", "nonexistent")
        assert result["distance"] == -1

    def test_find_cluster(self, graph_with_notes):
        result = graph_with_notes.find_cluster("n")
        assert result["count"] >= 3

    def test_find_central(self, graph_with_notes):
        result = graph_with_notes.find_central(top_k=3)
        assert "notes" in result
        assert len(result["notes"]) <= 3

    def test_find_communities(self, graph_with_notes):
        result = graph_with_notes.find_communities()
        assert "communities" in result
        assert len(result["communities"]) >= 1

    def test_execute_path_query(self, graph_with_notes):
        result = graph_with_notes.execute("path from n1 to n3")
        assert "path" in result

    def test_execute_cluster_query(self, graph_with_notes):
        result = graph_with_notes.execute("cluster n")
        assert "notes" in result

    def test_execute_central_query(self, graph_with_notes):
        result = graph_with_notes.execute("central notes")
        assert "notes" in result

    def test_execute_communities_query(self, graph_with_notes):
        result = graph_with_notes.execute("communities")
        assert "communities" in result

    def test_execute_stats_query(self, graph_with_notes):
        result = graph_with_notes.execute("stats")
        assert "num_nodes" in result

    def test_execute_unknown_query(self, graph_with_notes):
        result = graph_with_notes.execute("something random")
        assert "error" in result
