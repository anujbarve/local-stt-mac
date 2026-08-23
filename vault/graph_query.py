"""
Vault Graph Query
=================

Simple query interface for graph traversal, path finding,
community detection, and centrality analysis.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class GraphQueryEngine:
    """Execute graph queries on the vault knowledge graph."""

    def __init__(self, graph: Any):
        self.graph = graph

    def find_path(self, from_id: str, to_id: str) -> Dict[str, Any]:
        """Find shortest path between two notes."""
        path = self.graph.shortest_path(from_id, to_id)
        if path:
            return {"path": path, "distance": len(path) - 1}
        return {"path": [], "distance": -1}

    def find_cluster(self, topic: str) -> Dict[str, Any]:
        """Find all notes related to a topic via graph traversal."""
        # Find notes that contain the topic in their name
        matches = [n for n in self.graph._all_notes if topic.lower() in n.lower()]

        # Expand via neighbors
        expanded: set = set(matches)
        for m in matches:
            neighbors = self.graph.neighbors(m, depth=2)
            expanded.update(neighbors)

        return {
            "notes": sorted(expanded),
            "count": len(expanded),
            "seed_matches": sorted(matches),
        }

    def find_central(self, top_k: int = 10) -> Dict[str, Any]:
        """Find most central/important notes."""
        return {"notes": self.graph.centrality(top_k)}

    def find_communities(self) -> Dict[str, Any]:
        """List all communities."""
        communities = self.graph.communities()
        result = []
        for i, community in enumerate(communities):
            # Try to infer a label from common substrings
            label = self._infer_community_label(community)
            result.append({
                "id": i,
                "size": len(community),
                "label": label,
                "notes": community[:5],  # first 5 for preview
            })
        return {"communities": result}

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute a natural language graph query."""
        lower = query.lower().strip()

        # "path from A to B"
        path_match = re.search(r"path\s+(?:from\s+)?(.+?)\s+to\s+(.+)", lower)
        if path_match:
            from_id = path_match.group(1).strip()
            to_id = path_match.group(2).strip()
            return self.find_path(from_id, to_id)

        # "cluster <topic>"
        cluster_match = re.search(r"cluster\s+(.+)", lower)
        if cluster_match:
            topic = cluster_match.group(1).strip()
            return self.find_cluster(topic)

        # "central" or "important"
        if "central" in lower or "important" in lower:
            return self.find_central()

        # "communities"
        if "communit" in lower:
            return self.find_communities()

        # "orphans" or "isolated"
        if "orphan" in lower or "isolat" in lower:
            return {"notes": self.graph.orphans(), "count": len(self.graph.orphans())}

        # "stats" or "overview"
        if "stat" in lower or "overview" in lower:
            return self.graph.graph_stats()

        return {"error": "Unrecognized query", "query": query}

    def _infer_community_label(self, community: List[str]) -> str:
        """Infer a label for a community from note names."""
        if not community:
            return "unknown"
        # Simple: use the most common word across note names
        words: Dict[str, int] = {}
        for note in community:
            for word in note.replace("-", " ").split():
                if len(word) > 2:
                    words[word] = words.get(word, 0) + 1
        if words:
            return max(words, key=words.get)
        return community[0]
