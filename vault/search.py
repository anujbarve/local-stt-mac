"""
Vault Search
============

Multi-strategy search combining structural filtering, semantic similarity,
and graph traversal with weighted hybrid scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from brain_contract import QueryResult
from vault.config import SEARCH_TOP_K

# Default hybrid weights
STRUCTURAL_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.5
GRAPH_WEIGHT = 0.2


class VaultSearch:
    """Hybrid search across the vault."""

    def __init__(
        self,
        reader: Any,
        index: Any,
        graph: Any,
    ):
        self.reader = reader
        self.index = index
        self.graph = graph

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = SEARCH_TOP_K,
        embedder: Any = None,
    ) -> QueryResult:
        """Search the vault using the specified strategy.

        Strategies: ``"structural"``, ``"semantic"``, ``"graph"``, ``"hybrid"``.
        """
        if strategy == "structural":
            results = self._structural_from_text(query, top_k)
        elif strategy == "semantic":
            results = self.semantic_search(query, embedder, top_k)
        elif strategy == "graph":
            results = self._graph_from_text(query, top_k)
        else:
            results = self._hybrid(query, embedder, top_k)

        notes = [r["note"] for r in results]
        scores = [r["score"] for r in results]

        return QueryResult(
            notes=notes,
            scores=scores,
            strategy=strategy,
            total=len(notes),
            query_parsed={"query": query, "strategy": strategy},
        )

    # ------------------------------------------------------------------
    # Structural search
    # ------------------------------------------------------------------

    def structural_search(
        self,
        filters: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Filter notes by frontmatter fields."""
        all_notes = self.reader.read_all()
        matched: List[Dict[str, Any]] = []

        for note in all_notes:
            if self._matches_filters(note, filters):
                matched.append({"note": note, "score": 1.0})

        return matched[:top_k]

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query_text: str,
        embedder: Any = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Embed query and search the FAISS index."""
        if embedder is None:
            return []
        raw_results = self.index.search_by_text(query_text, embedder, top_k)
        out: List[Dict[str, Any]] = []
        for r in raw_results:
            note = self.reader.read_by_id(r["note_id"])
            if note is None:
                # Fallback: try by file_path
                note = self.reader.read(r.get("file_path", ""))
            if note is not None:
                out.append({"note": note, "score": r["score"]})
        return out

    # ------------------------------------------------------------------
    # Graph search
    # ------------------------------------------------------------------

    def graph_search(
        self,
        seed_note: str,
        depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Find notes connected to *seed_note* via wikilinks."""
        neighbor_ids = self.graph.neighbors(seed_note, depth)
        results: List[Dict[str, Any]] = []
        # Closer notes get higher scores
        for i, nid in enumerate(neighbor_ids):
            note = self.reader.read_by_id(nid)
            if note is None:
                continue
            score = 1.0 - (i / max(len(neighbor_ids), 1)) * 0.5
            results.append({"note": note, "score": score})
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hybrid(
        self, query: str, embedder: Any, top_k: int
    ) -> List[Dict[str, Any]]:
        """Combine structural, semantic, and graph results."""
        # Structural: infer filters from text
        filters = self._text_to_filters(query)
        struct_results = self.structural_search(filters, top_k * 2) if filters else []

        # Semantic
        sem_results = self.semantic_search(query, embedder, top_k * 2) if embedder else []

        # Graph: use first matching note as seed
        graph_results: List[Dict[str, Any]] = []
        if struct_results or sem_results:
            seed = (struct_results or sem_results)[0]["note"]
            seed_id = seed.get("id", "")
            if seed_id:
                graph_results = self.graph_search(seed_id, depth=2)

        return self._merge_results(
            [struct_results, sem_results, graph_results],
            [STRUCTURAL_WEIGHT, SEMANTIC_WEIGHT, GRAPH_WEIGHT],
            top_k,
        )

    def _merge_results(
        self,
        result_sets: List[List[Dict[str, Any]]],
        weights: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Merge multiple result sets with weighted scores."""
        scores: Dict[str, float] = {}  # note_id → combined score
        note_map: Dict[str, Dict[str, Any]] = {}  # note_id → note dict

        for results, weight in zip(result_sets, weights):
            for r in results:
                nid = r["note"].get("id", r["note"].get("file_path", ""))
                if not nid:
                    continue
                scores[nid] = scores.get(nid, 0.0) + weight * r["score"]
                note_map[nid] = r["note"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for nid, score in ranked[:top_k]:
            out.append({"note": note_map[nid], "score": score})
        return out

    def _text_to_filters(self, text: str) -> Dict[str, Any]:
        """Best-effort extraction of filters from query text."""
        import re

        filters: Dict[str, Any] = {}
        lower = text.lower()

        # Type detection
        type_map = {
            "task": r"\b(tasks?|to-?do|action items?)\b",
            "meeting": r"\b(meetings?|calls?|discussions?)\b",
            "idea": r"\b(ideas?|thoughts?)\b",
            "learning": r"\b(learnings?|insights?)\b",
            "project": r"\b(projects?)\b",
            "person": r"\b(people|person|contacts?)\b",
        }
        for ntype, pat in type_map.items():
            if re.search(pat, lower):
                filters["type"] = ntype
                break

        # Priority
        for p in ["critical", "high", "medium", "low"]:
            if p in lower:
                filters["priority"] = p
                break

        # Status
        for s in ["todo", "in_progress", "done", "blocked", "cancelled"]:
            if s in lower:
                filters["status"] = s
                break

        return filters

    def _matches_filters(self, note: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a note matches all filter conditions."""
        for key, value in filters.items():
            if key == "tags":
                if isinstance(value, str):
                    value = [value]
                note_tags = [t.lower() for t in note.get("tags", [])]
                if not any(v.lower() in note_tags for v in value):
                    return False
            elif key == "people":
                if isinstance(value, str):
                    value = [value]
                note_people = [p.lower() for p in note.get("people", [])]
                if not any(v.lower() in note_people for v in value):
                    return False
            elif key == "projects":
                if isinstance(value, str):
                    value = [value]
                note_projects = [p.lower() for p in note.get("projects", [])]
                if not any(v.lower() in note_projects for v in value):
                    return False
            else:
                note_val = str(note.get(key, "")).lower()
                filter_val = str(value).lower()
                if note_val != filter_val:
                    return False
        return True

    def _structural_from_text(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """Run structural search inferred from text."""
        filters = self._text_to_filters(text)
        if not filters:
            # Fall back to returning all notes
            return [{"note": n, "score": 0.5} for n in self.reader.read_all()[:top_k]]
        return self.structural_search(filters, top_k)

    def _graph_from_text(self, text: str, top_k: int) -> List[Dict[str, Any]]:
        """Run graph search by finding the best seed note."""
        all_notes = self.reader.read_all()
        if not all_notes:
            return []
        # Use first note as seed (in real usage, query parser picks the best)
        seed_id = all_notes[0].get("id", "")
        return self.graph_search(seed_id, depth=2)[:top_k]

    def _note_to_result(self, note: Dict[str, Any], score: float, rank: int) -> Dict[str, Any]:
        """Convert note dict to search result format."""
        return {
            "note_id": note.get("id", ""),
            "file_path": note.get("file_path", ""),
            "score": score,
            "rank": rank,
            "note": note,
        }
