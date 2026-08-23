"""Tests for vault/graph.py — enhanced intelligence features."""

import os

import pytest

from vault.graph import VaultGraph


@pytest.fixture
def graph_with_links(tmp_path):
    """Create a graph with multiple connected notes."""
    vault_dir = str(tmp_path / "vault")
    os.makedirs(os.path.join(vault_dir, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(vault_dir, "ideas"), exist_ok=True)
    os.makedirs(os.path.join(vault_dir, "daily"), exist_ok=True)

    # Chain: n1 → n2 → n3, n1 → n4, n2 → n5
    notes = {
        "tasks/n1.md": "---\nid: n1\n---\n\nSee [[n2]] and [[n4]].\n",
        "tasks/n2.md": "---\nid: n2\n---\n\nSee [[n3]] and [[n5]].\n",
        "ideas/n3.md": "---\nid: n3\n---\n\nLinks to [[n1]].\n",
        "ideas/n4.md": "---\nid: n4\n---\n\nNo links.\n",
        "daily/n5.md": "---\nid: n5\n---\n\nLinks to [[n2]].\n",
    }
    for path, content in notes.items():
        full = os.path.join(vault_dir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)

    g = VaultGraph(vault_dir)
    g.build()
    return g


class TestGraphIntelligence:
    def test_centrality(self, graph_with_links):
        cent = graph_with_links.centrality(top_k=3)
        assert len(cent) <= 3
        assert all("note_id" in c for c in cent)
        assert all("centrality_score" in c for c in cent)
        # n2 should be central (connected to n1, n3, n5)
        top_ids = [c["note_id"] for c in cent]
        assert "n2" in top_ids

    def test_centrality_empty_graph(self, tmp_path):
        g = VaultGraph(str(tmp_path / "empty"))
        assert g.centrality() == []

    def test_graph_stats(self, graph_with_links):
        stats = graph_with_links.graph_stats()
        assert stats["num_nodes"] == 5
        assert stats["num_edges"] >= 5
        assert stats["density"] > 0
        assert stats["avg_degree"] > 0
        assert stats["num_communities"] >= 1
        assert stats["largest_community"] >= 3

    def test_graph_stats_empty(self, tmp_path):
        g = VaultGraph(str(tmp_path / "empty"))
        stats = g.graph_stats()
        assert stats["num_nodes"] == 0
        assert stats["num_edges"] == 0

    def test_communities(self, graph_with_links):
        comms = graph_with_links.communities()
        assert len(comms) >= 1
        # All notes should be in one community (connected)
        all_in = set()
        for c in comms:
            all_in.update(c)
        assert len(all_in) == 5

    def test_stats_delegates_to_graph_stats(self, graph_with_links):
        s1 = graph_with_links.stats()
        s2 = graph_with_links.graph_stats()
        assert s1 == s2
