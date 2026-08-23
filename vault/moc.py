"""
Vault MOC — Map of Content
===========================

Auto-generate and maintain topic index pages (MOCs) that group notes
by topic with categorized links.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from vault.config import VAULT_ROOT
from vault.schema import NoteSchema


class VaultMOC:
    """Map of Content generator."""

    def __init__(
        self,
        vault_root: str = VAULT_ROOT,
        reader: Any = None,
        graph: Any = None,
    ):
        self.vault_root = vault_root
        self.moc_dir = os.path.join(vault_root, "topics")
        self._reader = reader
        self._graph = graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_moc(self, topic: str) -> str:
        """Generate a MOC markdown page for *topic*.

        Returns the markdown content (does NOT write to disk).
        """
        notes = self._notes_for_topic(topic)
        grouped = self._categorize(notes)
        slug = NoteSchema.slugify(topic)
        count = len(notes)

        lines = [
            "---",
            f"id: moc_{slug}",
            "type: moc",
            f"created: {datetime.utcnow().strftime('%Y-%m-%d')}",
            f"updated: {datetime.utcnow().strftime('%Y-%m-%d')}",
            f"topic: {topic}",
            f"note_count: {count}",
            "---",
            "",
            f"# {topic.title()}",
            "",
        ]

        for ntype, group in grouped.items():
            if not group:
                continue
            label = ntype.replace("_", " ").title()
            lines.append(f"## {label} ({len(group)})")
            for note in group:
                nid = note.get("id", "")
                title = note.get("body", "").split("\n")[0].lstrip("# ").strip()
                if not title:
                    title = nid
                lines.append(f"- [[{NoteSchema.slugify(title)}]]")
            lines.append("")

        # Related topics
        related = self._related_topics(topic)
        if related:
            lines.append("## Related Topics")
            for rt in related:
                lines.append(f"- [[{NoteSchema.slugify(rt)}]]")
            lines.append("")

        return "\n".join(lines)

    def update_moc(self, moc_path: str) -> None:
        """Refresh an existing MOC with current links."""
        # Extract topic from existing MOC
        with open(moc_path, "r", encoding="utf-8") as fh:
            content = fh.read()
        topic = ""
        for line in content.split("\n"):
            if line.startswith("topic:"):
                topic = line.split(":", 1)[1].strip()
                break
        if topic:
            new_content = self.generate_moc(topic)
            with open(moc_path, "w", encoding="utf-8") as fh:
                fh.write(new_content)

    def list_mocs(self) -> List[str]:
        """List all existing MOC files in ``vault/topics/``."""
        if not os.path.isdir(self.moc_dir):
            return []
        return sorted(
            os.path.join(self.moc_dir, f)
            for f in os.listdir(self.moc_dir)
            if f.endswith(".md")
        )

    def suggest_moc(self, note: Dict[str, Any]) -> Optional[str]:
        """Suggest which MOC a note belongs to based on tags."""
        tags = [t.lower() for t in note.get("tags", [])]
        best_topic: Optional[str] = None
        best_count = 0

        for moc_path in self.list_mocs():
            topic = self._topic_from_moc_path(moc_path)
            if not topic:
                continue
            score = 1 if topic.lower() in tags else 0
            for tag in tags:
                if tag in topic.lower():
                    score += 1
            if score > best_count:
                best_count = score
                best_topic = topic

        return best_topic

    def auto_generate_all(self) -> int:
        """Generate MOCs for all topics with 3+ notes.

        Returns number of MOCs created/updated.
        """
        os.makedirs(self.moc_dir, exist_ok=True)
        topic_counts = self._count_topics()
        count = 0
        for topic, n in topic_counts.items():
            if n >= 3:
                content = self.generate_moc(topic)
                slug = NoteSchema.slugify(topic)
                path = os.path.join(self.moc_dir, f"{slug}.md")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(content)
                count += 1
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _notes_for_topic(self, topic: str) -> List[Dict[str, Any]]:
        """Find all notes with matching tags or topics."""
        if self._reader is None:
            return []
        topic_lower = topic.lower()
        results: List[Dict[str, Any]] = []
        for note in self._reader.read_all():
            tags = [t.lower() for t in note.get("tags", [])]
            if topic_lower in tags:
                results.append(note)
        return results

    def _categorize(self, notes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group notes by type."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for note in notes:
            ntype = note.get("type", "observation")
            grouped.setdefault(ntype, []).append(note)
        return grouped

    def _related_topics(self, topic: str) -> List[str]:
        """Find topics that co-occur with *topic* in notes."""
        if self._reader is None:
            return []
        topic_lower = topic.lower()
        co_topics: Dict[str, int] = {}
        for note in self._reader.read_all():
            tags = [t.lower() for t in note.get("tags", [])]
            if topic_lower in tags:
                for t in tags:
                    if t != topic_lower:
                        co_topics[t] = co_topics.get(t, 0) + 1
        return [t for t, _ in sorted(co_topics.items(), key=lambda x: -x[1])[:10]]

    def _count_topics(self) -> Dict[str, int]:
        """Count notes per topic across the vault."""
        if self._reader is None:
            return {}
        counts: Dict[str, int] = {}
        for note in self._reader.read_all():
            for tag in note.get("tags", []):
                counts[tag] = counts.get(tag, 0) + 1
        return counts

    def _topic_from_moc_path(self, moc_path: str) -> Optional[str]:
        """Extract topic from MOC frontmatter."""
        try:
            with open(moc_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("topic:"):
                        return line.split(":", 1)[1].strip()
                    if line.startswith("---") and "type: moc" not in "".join(
                        [l for l in fh]
                    ):
                        break
        except OSError:
            return None
        return None
