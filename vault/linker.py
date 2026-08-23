"""
Vault Linker
============

Automatically link new notes to existing ones via entity resolution,
shared tags, and graph proximity.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from brain_contract import Understanding, EntityType, ENTITY_PAGE_TYPES
from vault.config import LINK_CONFIDENCE_THRESHOLD
from vault.schema import NoteSchema


class VaultLinker:
    """Auto-link notes via entity resolution."""

    def __init__(
        self,
        reader: Any,
        graph: Any,
        confidence_threshold: float = LINK_CONFIDENCE_THRESHOLD,
    ):
        self.reader = reader
        self.graph = graph
        self.confidence_threshold = confidence_threshold
        self._entity_index: Dict[str, List[str]] = {}  # entity_name_lower → [note_ids]
        self._build_entity_index()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def link(
        self,
        understanding: Understanding,
        existing_notes: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Find and add links for a new note.

        Returns list of note_ids that were linked.
        """
        if existing_notes is None:
            existing_notes = self.reader.read_all()

        related = self.find_related(
            self._understanding_to_dict(understanding),
            existing_notes,
        )

        linked_ids: List[str] = []
        new_note_id = understanding.note_id or ""

        for rel in related:
            rel_id = rel.get("note_id", "")
            if rel_id and rel_id != new_note_id:
                self.graph.add_link(new_note_id, rel_id)
                linked_ids.append(rel_id)

        return linked_ids

    def find_related(
        self,
        note: Dict[str, Any],
        candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Find notes related to *note*.

        Returns ``[{note_id, score, reason}, ...]`` sorted by score.
        """
        if candidates is None:
            candidates = self.reader.read_all()

        note_id = note.get("id", "")
        scored: List[Dict[str, Any]] = []

        for candidate in candidates:
            cid = candidate.get("id", "")
            if cid == note_id:
                continue

            score, reasons = self._score_related(note, candidate)
            if score >= self.confidence_threshold:
                scored.append({
                    "note_id": cid,
                    "score": score,
                    "reason": ", ".join(reasons),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    def merge_entity_pages(
        self,
        entity_name: str,
        pages: List[str],
    ) -> str:
        """Merge multiple entity pages into one.

        Returns path of merged page.
        """
        slug = NoteSchema.slugify(entity_name)
        # Find the page directory (e.g., people/, tools/)
        target_dir = None
        for dirpath, _dn, fnames in os.walk(self.reader.vault_root):
            for fname in fnames:
                if fname == f"{slug}.md":
                    target_dir = dirpath
                    break
            if target_dir:
                break

        if target_dir is None:
            target_dir = os.path.join(self.reader.vault_root, "people")

        merged_path = os.path.join(target_dir, f"{slug}.md")

        # Collect all backlinks from all pages
        all_backlinks: set = set()
        for page_path in pages:
            if os.path.exists(page_path):
                with open(page_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
                links = re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]", content)
                all_backlinks.update(links)

        # Write merged page
        content = (
            f"---\nid: entity_{slug}\nname: {entity_name}\n---\n\n"
            f"# {entity_name}\n\n"
            f"# Backlinks\n"
        )
        for link in sorted(all_backlinks):
            content += f"- [[{link}]]\n"

        os.makedirs(target_dir, exist_ok=True)
        with open(merged_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        return merged_path

    def suggest_links(
        self,
        note: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Suggest links without adding them.

        Returns ``[{note_id, score, reason}, ...]``.
        """
        return self.find_related(note)[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score_related(
        self, note: Dict[str, Any], candidate: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """Score how related two notes are. Returns (score, reasons)."""
        score = 0.0
        reasons: List[str] = []

        # 1. Shared entities (person, project, tool)
        for field in ("people", "projects", "tools"):
            a = set(note.get(field, []))
            b = set(candidate.get(field, []))
            overlap = a & b
            if overlap:
                score += 0.3 * min(len(overlap), 3)
                reasons.append(f"shared_{field}={','.join(overlap)}")

        # 2. Shared tags
        tags_a = set(t.lower() for t in note.get("tags", []))
        tags_b = set(t.lower() for t in candidate.get("tags", []))
        tag_overlap = tags_a & tags_b
        if tag_overlap:
            score += 0.25 * min(len(tag_overlap), 3)
            reasons.append(f"shared_tags={','.join(tag_overlap)}")

        # 3. Same type
        if note.get("type") == candidate.get("type"):
            score += 0.1
            reasons.append("same_type")

        # 4. Graph proximity
        note_id = note.get("id", "")
        cand_id = candidate.get("id", "")
        if note_id and cand_id:
            path = self.graph.shortest_path(note_id, cand_id)
            if path and len(path) <= 4:
                proximity_score = 0.2 / max(len(path) - 1, 1)
                score += proximity_score
                reasons.append(f"graph_distance={len(path) - 1}")

        return min(score, 1.0), reasons

    def _build_entity_index(self) -> None:
        """Scan vault and index all entities mentioned in notes."""
        self._entity_index = defaultdict(list)
        for note in self.reader.read_all():
            nid = note.get("id", "")
            for field in ("people", "projects", "tools"):
                for entity in note.get(field, []):
                    self._entity_index[entity.lower()].append(nid)

    def _understanding_to_dict(self, u: Understanding) -> Dict[str, Any]:
        """Convert an Understanding to a note-like dict for find_related."""
        return {
            "id": u.note_id or "",
            "type": u.type.value if hasattr(u.type, "value") else str(u.type),
            "tags": u.tags + u.topics,
            "people": [e.name for e in u.entities if e.entity_type == EntityType.PERSON],
            "projects": [e.name for e in u.entities if e.entity_type == EntityType.PROJECT],
            "tools": [e.name for e in u.entities if e.entity_type == EntityType.TOOL],
            "body": u.text,
        }
