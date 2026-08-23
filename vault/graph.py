"""
Vault Graph
===========

Maintains a directed graph of ``[[wikilinks]]`` between notes.  Provides
traversal, shortest-path, community detection, and orphan detection.
"""

from __future__ import annotations

import json
import os
from collections import deque
from typing import Any, Dict, List, Optional, Set

from vault.config import VAULT_ROOT
from vault.schema import _extract_wikilinks  # reuse


class VaultGraph:
    """Knowledge graph of note connections."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self._outgoing: Dict[str, Set[str]] = {}  # source → {targets}
        self._incoming: Dict[str, Set[str]] = {}  # target → {sources}
        self._all_notes: Set[str] = set()  # all known note basenames

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, reader: Any = None) -> int:
        """Scan vault for ``[[wikilinks]]`` and build the graph.

        Returns number of edges found.
        """
        self._outgoing.clear()
        self._incoming.clear()
        self._all_notes.clear()

        # Discover all note basenames
        for dirpath, _dn, fnames in os.walk(self.vault_root):
            for fname in fnames:
                if fname.endswith(".md"):
                    basename = fname[:-3]
                    self._all_notes.add(basename)
                    self._outgoing.setdefault(basename, set())
                    self._incoming.setdefault(basename, set())

        # Parse wikilinks
        edge_count = 0
        for dirpath, _dn, fnames in os.walk(self.vault_root):
            for fname in fnames:
                if not fname.endswith(".md"):
                    continue
                source = fname[:-3]
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        body = fh.read()
                except OSError:
                    continue

                # Skip frontmatter when extracting links
                if body.startswith("---"):
                    end = body.find("\n---", 3)
                    if end != -1:
                        body = body[end + 4:]

                targets = _extract_wikilinks(body)
                for target in targets:
                    self.add_link(source, target)
                    edge_count += 1

        return edge_count

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def add_link(self, source: str, target: str) -> None:
        """Add a directed edge: ``source → target``."""
        self._all_notes.add(source)
        self._all_notes.add(target)
        self._outgoing.setdefault(source, set()).add(target)
        self._incoming.setdefault(target, set()).add(source)
        self._outgoing.setdefault(target, set())

    def remove_link(self, source: str, target: str) -> None:
        """Remove a directed edge."""
        if source in self._outgoing:
            self._outgoing[source].discard(target)
        if target in self._incoming:
            self._incoming[target].discard(source)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def backlinks(self, note: str) -> List[str]:
        """Notes that link TO *note*."""
        return sorted(self._incoming.get(note, set()))

    def forward_links(self, note: str) -> List[str]:
        """Notes that *note* links TO."""
        return sorted(self._outgoing.get(note, set()))

    def neighbors(self, note: str, depth: int = 1) -> List[str]:
        """All notes reachable within *depth* hops (BFS)."""
        visited: Set[str] = {note}
        frontier: deque = deque([(note, 0)])
        result: List[str] = []

        while frontier:
            current, d = frontier.popleft()
            if d >= depth:
                continue
            for nxt in self._outgoing.get(current, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    result.append(nxt)
                    frontier.append((nxt, d + 1))
            for nxt in self._incoming.get(current, set()):
                if nxt not in visited:
                    visited.add(nxt)
                    result.append(nxt)
                    frontier.append((nxt, d + 1))

        return sorted(result)

    def orphans(self) -> List[str]:
        """Find notes with no links in or out."""
        orphans: List[str] = []
        for note in sorted(self._all_notes):
            out = self._outgoing.get(note, set())
            inc = self._incoming.get(note, set())
            if not out and not inc:
                orphans.append(note)
        return orphans

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two notes (BFS).

        Returns the path as ``[source, ..., target]`` or ``None``.
        """
        if source == target:
            return [source]

        visited: Set[str] = {source}
        # Each queue entry: (current_node, path_so_far)
        queue: deque = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            all_next = self._outgoing.get(current, set()) | self._incoming.get(current, set())
            for nxt in all_next:
                if nxt == target:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))

        return None

    def communities(self) -> List[List[str]]:
        """Detect connected components (clusters) via BFS undirected traversal."""
        visited: Set[str] = set()
        clusters: List[List[str]] = []

        for note in sorted(self._all_notes):
            if note in visited:
                continue
            component: List[str] = []
            queue: deque = deque([note])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                all_next = self._outgoing.get(current, set()) | self._incoming.get(current, set())
                for nxt in all_next:
                    if nxt not in visited:
                        queue.append(nxt)
            clusters.append(sorted(component))

        return clusters

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Save graph to JSON."""
        path = path or os.path.join(self.vault_root, "_graph.json")
        data = {
            "outgoing": {k: sorted(v) for k, v in self._outgoing.items()},
            "incoming": {k: sorted(v) for k, v in self._incoming.items()},
            "all_notes": sorted(self._all_notes),
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load(self, path: Optional[str] = None) -> None:
        """Load graph from JSON."""
        path = path or os.path.join(self.vault_root, "_graph.json")
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._outgoing = {k: set(v) for k, v in data.get("outgoing", {}).items()}
        self._incoming = {k: set(v) for k, v in data.get("incoming", {}).items()}
        self._all_notes = set(data.get("all_notes", []))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self._all_notes)

    @property
    def num_edges(self) -> int:
        return sum(len(v) for v in self._outgoing.values())

    def stats(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_orphans": len(self.orphans()),
            "num_communities": len(self.communities()),
        }
