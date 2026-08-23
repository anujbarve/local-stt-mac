"""
Vault Reader
============

Inverse of ``vault/writer.py``.  Reads ``.md`` files from the vault and
returns structured dicts that other modules (search, graph, linker) consume.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from brain_contract import NoteType, NOTE_TYPE_DIRS
from vault.config import VAULT_ROOT
from vault.schema import _parse_frontmatter, _extract_wikilinks  # reuse


class VaultReader:
    """Read notes from the vault as structured dicts."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root

    # ------------------------------------------------------------------
    # Core reads
    # ------------------------------------------------------------------

    def read(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Read a single ``.md`` file → structured dict.

        Returns ``None`` if the file doesn't exist or is unparseable.
        """
        if not os.path.isfile(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError:
            return None

        if not raw.strip():
            return None

        fm, body = _parse_frontmatter(raw)

        if not fm:
            return None

        # Normalise note type
        note_type = fm.get("type", "observation")

        # Extract wikilinks from the body
        wikilinks = _extract_wikilinks(body)

        # Resolve entity lists (may be lists or bare strings from YAML)
        def _list_val(key: str) -> List[str]:
            v = fm.get(key, [])
            if isinstance(v, list):
                return [str(x) for x in v]
            if v:
                return [str(v)]
            return []

        return {
            "id": fm.get("id", ""),
            "type": note_type,
            "created": fm.get("created", ""),
            "updated": fm.get("updated", ""),
            "source": fm.get("source", ""),
            "priority": fm.get("priority", "medium"),
            "urgency": fm.get("urgency", "later"),
            "sentiment": fm.get("sentiment", "neutral"),
            "confidence": fm.get("confidence", 0.5),
            "people": _list_val("people"),
            "projects": _list_val("projects"),
            "tasks": _list_val("tasks"),
            "tools": _list_val("tools"),
            "dates": _list_val("dates"),
            "tags": _list_val("tags"),
            "category": fm.get("category", note_type),
            "related_notes": _list_val("related_notes"),
            "backlinks": _list_val("backlinks"),
            "body": body,
            "file_path": os.path.abspath(file_path),
            "wikilinks": wikilinks,
        }

    def read_all(self) -> List[Dict[str, Any]]:
        """Read every ``.md`` file in the vault → list of dicts."""
        results: List[Dict[str, Any]] = []
        for fpath in self._all_md_files():
            note = self.read(fpath)
            if note is not None:
                results.append(note)
        return results

    def read_by_type(self, note_type: NoteType) -> List[Dict[str, Any]]:
        """Read all notes of a specific type."""
        target = note_type.value if isinstance(note_type, NoteType) else str(note_type)
        return [n for n in self.read_all() if n.get("type") == target]

    def read_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Find and read a note by its frontmatter ``id``."""
        for note in self.read_all():
            if note.get("id") == note_id:
                return note
        return None

    def read_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Read all notes that have a specific tag."""
        tag_lower = tag.lower()
        return [n for n in self.read_all() if tag_lower in [t.lower() for t in n.get("tags", [])]]

    def read_directory(self, subdir: str) -> List[Dict[str, Any]]:
        """Read all notes in a specific vault subdirectory."""
        dir_path = os.path.join(self.vault_root, subdir)
        if not os.path.isdir(dir_path):
            return []
        results: List[Dict[str, Any]] = []
        for fname in os.listdir(dir_path):
            if fname.endswith(".md"):
                note = self.read(os.path.join(dir_path, fname))
                if note is not None:
                    results.append(note)
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Total number of readable notes."""
        return len(self.read_all())

    def list_files(self) -> List[str]:
        """List all ``.md`` file paths relative to ``vault_root``."""
        rels: List[str] = []
        for fpath in self._all_md_files():
            rels.append(os.path.relpath(fpath, self.vault_root))
        return sorted(rels)

    def extract_wikilinks(self, body: str) -> List[str]:
        """Extract ``[[wikilink]]`` targets from markdown body."""
        return _extract_wikilinks(body)

    def get_note_summary(self, note: Dict[str, Any]) -> str:
        """Return a short text summary (body truncated to 200 chars)."""
        body = note.get("body", "").strip()
        if len(body) <= 200:
            return body
        return body[:197] + "..."

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _all_md_files(self) -> List[str]:
        """Return absolute paths of all ``.md`` files under vault_root."""
        md_files: List[str] = []
        for dirpath, _dirnames, filenames in os.walk(self.vault_root):
            for fname in filenames:
                if fname.endswith(".md"):
                    md_files.append(os.path.join(dirpath, fname))
        return sorted(md_files)
