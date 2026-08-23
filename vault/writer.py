"""
Vault Writer
============

The primary bridge between the brain (Instance A) and the vault (Instance B).

Given an ``Understanding`` object, this module:
  1. Determines the target directory from the note type.
  2. Generates a filename from the content.
  3. Renders the markdown note (frontmatter + body).
  4. Writes the file.
  5. Updates backlinks for any linked entities.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from brain_contract import (
    Understanding,
    NoteType,
    EntityType,
    NOTE_TYPE_DIRS,
    ENTITY_PAGE_TYPES,
)
from vault.config import VAULT_ROOT
from vault.schema import NoteSchema
from vault.templates import NoteTemplate


# ---------------------------------------------------------------------------
# VaultWriter
# ---------------------------------------------------------------------------

class VaultWriter:
    """Write ``Understanding`` objects to the vault."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self._ensure_directories()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, understanding: Understanding) -> str:
        """Write a note to the vault.

        Returns the *file_path* (relative to project root) of the created note.
        """
        note_id = understanding.note_id or NoteSchema.generate_id()
        understanding.note_id = note_id

        subdir = self._determine_directory(understanding.type)
        filename = self._generate_filename(understanding, subdir)
        file_path = os.path.join(subdir, filename)

        # Render and write
        u_dict = understanding.to_dict()
        u_dict["note_id"] = note_id
        content = NoteTemplate.render(u_dict)
        os.makedirs(subdir, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(content)

        # Store path back on understanding
        understanding.file_path = file_path

        # Update backlinks for any entities that have pages
        self._update_backlinks(understanding, file_path)

        return file_path

    # ------------------------------------------------------------------

    def write_daily_entry(
        self, understanding: Understanding, date: Optional[str] = None
    ) -> str:
        """Append a transcription to the daily journal.

        Returns the file path of the daily note.
        """
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        daily_dir = os.path.join(self.vault_root, "daily")
        os.makedirs(daily_dir, exist_ok=True)
        file_path = os.path.join(daily_dir, f"{date}.md")

        # Build a simple entry block
        entry_lines = [
            f"\n## {understanding.timestamp}",
            "",
            f"**Type:** {understanding.type.value}",
            "",
            understanding.text,
            "",
        ]
        if understanding.summary:
            entry_lines += ["**Summary:**", understanding.summary, ""]

        entry_block = "\n".join(entry_lines)

        # Append or create
        if os.path.exists(file_path):
            with open(file_path, "a", encoding="utf-8") as fh:
                fh.write(entry_block)
        else:
            header = f"---\nid: daily_{date}\ntype: journal\ncreated: {date}\nupdated: {date}\nsource: audio\n---\n\n# Daily Journal — {date}\n"
            with open(file_path, "w", encoding="utf-8") as fh:
                fh.write(header + entry_block)

        return file_path

    # ------------------------------------------------------------------

    def update_note(self, file_path: str, updates: Dict[str, Any]) -> None:
        """Update specific fields in an existing note's frontmatter."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Note not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as fh:
            raw = fh.read()

        if not raw.startswith("---"):
            raise ValueError("No frontmatter found in note")

        end = raw.find("\n---", 3)
        if end == -1:
            raise ValueError("Unclosed frontmatter block")

        fm_block = raw[3:end].strip()
        body = raw[end + 4:].lstrip("\n")

        # Apply updates as simple key: value replacements
        for key, value in updates.items():
            pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
            new_line = f"{key}: {value}"
            if pattern.search(fm_block):
                fm_block = pattern.sub(new_line, fm_block)
            else:
                fm_block += f"\n{new_line}"

        # Add updated timestamp
        now = datetime.utcnow().isoformat() + "Z"
        upd_pattern = re.compile(r"^updated:.*$", re.MULTILINE)
        if upd_pattern.search(fm_block):
            fm_block = upd_pattern.sub(f"updated: {now}", fm_block)
        else:
            fm_block += f"\nupdated: {now}"

        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(f"---\n{fm_block}\n---\n\n{body}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_directory(self, note_type: NoteType) -> str:
        """Map note type to a vault subdirectory path."""
        subdir = NOTE_TYPE_DIRS.get(note_type, "daily")
        return os.path.join(self.vault_root, subdir)

    def _generate_filename(
        self, understanding: Understanding, subdir: str
    ) -> str:
        """Generate ``<slug>.md``, avoiding collisions by appending a counter."""
        # Derive a meaningful slug
        if understanding.action:
            base_slug = NoteSchema.slugify(understanding.action)
        elif understanding.topics:
            base_slug = NoteSchema.slugify(understanding.topics[0])
        else:
            base_slug = NoteSchema.slugify(understanding.text[:60])

        filename = f"{base_slug}.md"
        candidate = os.path.join(subdir, filename)

        # De-duplicate
        counter = 2
        while os.path.exists(candidate):
            candidate = os.path.join(subdir, f"{base_slug}-{counter}.md")
            counter += 1

        return os.path.basename(candidate)

    def _ensure_directories(self) -> None:
        """Create every vault subdirectory we might need."""
        dirs = set(NOTE_TYPE_DIRS.values())
        dirs.add("_archive")
        dirs.add("topics")
        for d in dirs:
            os.makedirs(os.path.join(self.vault_root, d), exist_ok=True)

    def _update_backlinks(
        self, understanding: Understanding, note_path: str
    ) -> None:
        """Add ``[[backlink]]`` references from entity pages to *note_path*."""
        entities = understanding.entities
        if not entities:
            return

        # Resolve the short wikilink target (basename without .md)
        target_link = os.path.splitext(os.path.basename(note_path))[0]

        for entity in entities:
            etype = entity.entity_type
            if etype not in ENTITY_PAGE_TYPES:
                continue

            page_dir = os.path.join(self.vault_root, ENTITY_PAGE_TYPES[etype])
            page_path = os.path.join(page_dir, f"{NoteSchema.slugify(entity.name)}.md")

            if os.path.exists(page_path):
                # Append backlink if not already present
                with open(page_path, "r", encoding="utf-8") as fh:
                    content = fh.read()

                backlink_wiki = f"[[{target_link}]]"
                if backlink_wiki not in content:
                    # Try to insert before the closing --- or at end
                    marker = "# Backlinks"
                    if marker in content:
                        content = content.replace(
                            marker,
                            f"{marker}\n- {backlink_wiki}",
                        )
                    else:
                        content += f"\n\n# Backlinks\n- {backlink_wiki}\n"
                    with open(page_path, "w", encoding="utf-8") as fh:
                        fh.write(content)
            else:
                # Create a minimal entity page with backlink
                page_dir = os.path.join(self.vault_root, ENTITY_PAGE_TYPES[etype])
                os.makedirs(page_dir, exist_ok=True)
                page_content = (
                    f"---\nid: entity_{NoteSchema.slugify(entity.name)}\n"
                    f"type: {etype.value}\nname: {entity.name}\n---\n\n"
                    f"# {entity.name}\n\n"
                    f"# Backlinks\n- [[{target_link}]]\n"
                )
                with open(page_path, "w", encoding="utf-8") as fh:
                    fh.write(page_content)
