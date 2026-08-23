"""Tests for vault/graph.py"""

import os
import shutil
import tempfile

import pytest

from vault.graph import VaultGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_vault_with_links(tmp_path):
    """Create a vault with wikilinked notes."""
    vault_dir = str(tmp_path / "vault")
    os.makedirs(os.path.join(vault_dir, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(vault_dir, "ideas"), exist_ok=True)

    with open(os.path.join(vault_dir, "tasks", "note1.md"), "w") as f:
        f.write("---\nid: n1\ntype: task\ncreated: x\nupdated: x\nsource: audio\n---\n\nSee [[note2]] and [[note3]].\n")
    with open(os.path.join(vault_dir, "ideas", "note2.md"), "w") as f:
        f.write("---\nid: n2\ntype: idea\ncreated: x\nupdated: x\nsource: audio\n---\n\nLinks to [[note1]].\n")
    with open(os.path.join(vault_dir, "ideas", "note3.md"), "w") as f:
        f.write("---\nid: n3\ntype: idea\ncreated: x\nupdated: x\nsource: audio\n---\n\nNo outbound links.\n")

    return vault_dir


@pytest.fixture
def vault_dir(tmp_path):
    return _make_vault_with_links(tmp_path)


@pytest.fixture
def graph(vault_dir):
    g = VaultGraph(vault_dir)
    g.build()
    return g


@pytest.fixture
def empty_vault(tmp_path):
    d = str(tmp_path / "empty_vault")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVaultGraph:
    def test_build_finds_edges(self, graph):
        assert graph.num_edges >= 2  # note1→note2, note1→note3

    def test_build_finds_nodes(self, graph):
        assert graph.num_nodes == 3

    def test_add_and_remove_link(self, empty_vault):
        g = VaultGraph(empty_vault)
        g.add_link("a", "b")
        assert g.num_edges == 1
        assert "b" in g.forward_links("a")
        assert "a" in g.backlinks("b")

        g.remove_link("a", "b")
        assert g.num_edges == 0
        assert "b" not in g.forward_links("a")

    def test_backlinks(self, graph):
        bl = graph.backlinks("note1")
        assert "note2" in bl  # note2 links to note1

    def test_forward_links(self, graph):
        fl = graph.forward_links("note1")
        assert "note2" in fl
        assert "note3" in fl

    def test_neighbors_depth_1(self, graph):
        neighbors = graph.neighbors("note1", depth=1)
        assert "note2" in neighbors
        assert "note3" in neighbors

    def test_neighbors_depth_2(self, vault_dir):
        # Add a chain: note1→note2→note4
        os.makedirs(os.path.join(vault_dir, "daily"), exist_ok=True)
        with open(os.path.join(vault_dir, "daily", "note4.md"), "w") as f:
            f.write("---\nid: n4\ntype: observation\ncreated: x\nupdated: x\nsource: audio\n---\n\nLinks to [[note2]].\n")
        g = VaultGraph(vault_dir)
        g.build()

        neighbors_d1 = g.neighbors("note1", depth=1)
        neighbors_d2 = g.neighbors("note1", depth=2)
        assert "note4" in neighbors_d2
        # note4 is 2 hops away via note2

    def test_orphans(self, empty_vault):
        with open(os.path.join(empty_vault, "lonely.md"), "w") as f:
            f.write("---\nid: lonely\ntype: note\ncreated: x\nupdated: x\nsource: audio\n---\n\nNo links here.\n")
        g = VaultGraph(empty_vault)
        g.build()
        orphans = g.orphans()
        assert "lonely" in orphans

    def test_shortest_path(self, graph):
        path = graph.shortest_path("note1", "note2")
        assert path is not None
        assert path[0] == "note1"
        assert path[-1] == "note2"

    def test_shortest_path_same_node(self, graph):
        path = graph.shortest_path("note1", "note1")
        assert path == ["note1"]

    def test_shortest_path_no_path(self, empty_vault):
        g = VaultGraph(empty_vault)
        g.add_link("a", "b")
        g.add_link("c", "d")
        assert g.shortest_path("a", "d") is None

    def test_communities(self, graph):
        communities = graph.communities()
        assert len(communities) >= 1
        # All three notes should be in one community (connected)
        all_in = set()
        for c in communities:
            all_in.update(c)
        assert "note1" in all_in
        assert "note2" in all_in
        assert "note3" in all_in

    def test_save_and_load(self, vault_dir, graph):
        save_path = os.path.join(vault_dir, "_graph.json")
        graph.save()

        g2 = VaultGraph(vault_dir)
        g2.load()
        assert g2.num_edges == graph.num_edges
        assert g2.num_nodes == graph.num_nodes

    def test_stats(self, graph):
        stats = graph.stats()
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert stats["num_nodes"] == 3

    def test_build_empty_vault(self, empty_vault):
        g = VaultGraph(empty_vault)
        count = g.build()
        assert count == 0
        assert g.num_nodes == 0
