"""
Vault Migration
===============

Detect and fix schema issues in vault notes.  Provides validation,
auto-repair, and versioned migration support.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from vault.config import VAULT_ROOT
from vault.schema import NoteSchema, _parse_frontmatter, _extract_wikilinks, REQUIRED_FIELDS


class SchemaMigrator:
    """Detect and fix schema issues in vault notes."""

    SCHEMA_VERSION = 2

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root

    def validate_all(self) -> Dict[str, Any]:
        """Scan all notes and report issues.

        Returns:
            {
                "total": 100,
                "valid": 95,
                "issues": [
                    {"file": "tasks/fix-vad.md", "issue": "missing_type_field"},
                    ...
                ]
            }
        """
        issues: List[Dict[str, str]] = []
        total = 0
        valid = 0

        for dirpath, _dn, fnames in os.walk(self.vault_root):
            if "_backups" in dirpath or "_corrections" in dirpath:
                continue
            for fname in fnames:
                if not fname.endswith(".md"):
                    continue
                total += 1
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, self.vault_root)
                note_issues = self._validate_note(fpath)
                if note_issues:
                    for issue in note_issues:
                        issues.append({"file": rel, "issue": issue})
                else:
                    valid += 1

        return {
            "total": total,
            "valid": valid,
            "issues": issues,
        }

    def auto_fix(self, dry_run: bool = True) -> Dict[str, Any]:
        """Automatically fix common issues.

        Returns: {"fixed": 5, "skipped": 0, "errors": 0}
        """
        fixed = 0
        skipped = 0
        errors = 0

        for dirpath, _dn, fnames in os.walk(self.vault_root):
            if "_backups" in dirpath or "_corrections" in dirpath:
                continue
            for fname in fnames:
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    result = self._fix_note(fpath, dry_run=dry_run)
                    if result == "fixed":
                        fixed += 1
                    elif result == "skipped":
                        skipped += 1
                except Exception:
                    errors += 1

        return {"fixed": fixed, "skipped": skipped, "errors": errors}

    def get_stats(self) -> Dict[str, Any]:
        """Quick vault statistics."""
        total = 0
        with_frontmatter = 0
        with_embedding = 0

        for dirpath, _dn, fnames in os.walk(self.vault_root):
            if "_backups" in dirpath or "_corrections" in dirpath:
                continue
            for fname in fnames:
                if not fname.endswith(".md"):
                    continue
                total += 1
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        raw = fh.read()
                    if raw.startswith("---"):
                        with_frontmatter += 1
                    if "embedding_id:" in raw:
                        with_embedding += 1
                except OSError:
                    pass

        return {
            "total_notes": total,
            "with_frontmatter": with_frontmatter,
            "with_embedding": with_embedding,
            "schema_version": self.SCHEMA_VERSION,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_note(self, fpath: str) -> List[str]:
        """Validate a single note. Returns list of issues."""
        issues: List[str] = []
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError:
            return ["read_error"]

        if not raw.strip():
            issues.append("empty_file")
            return issues

        if not raw.startswith("---"):
            issues.append("no_frontmatter")
            return issues

        fm, body = _parse_frontmatter(raw)

        if not fm:
            issues.append("empty_frontmatter")
            return issues

        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in fm or fm[field] is None:
                issues.append(f"missing_{field}_field")

        # Check type field is valid
        note_type = fm.get("type", "")
        if note_type:
            try:
                from brain_contract import NoteType
                NoteType(note_type)
            except (ValueError, KeyError):
                issues.append(f"invalid_type_{note_type}")

        # Check body is not empty
        if not body.strip():
            issues.append("empty_body")

        return issues

    def _fix_note(self, fpath: str, dry_run: bool = True) -> str:
        """Fix issues in a single note.

        Returns "fixed", "skipped", or "unchanged".
        """
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError:
            return "skipped"

        if not raw.startswith("---"):
            return "skipped"

        fm, body = _parse_frontmatter(raw)
        if not fm:
            return "skipped"

        changed = False

        # Fix missing 'updated' field
        if "updated" not in fm or not fm.get("updated"):
            now = datetime.now(timezone.utc).isoformat() + "Z"
            raw = _set_frontmatter_field(raw, "updated", now)
            changed = True

        # Fix missing 'source' field
        if "source" not in fm or not fm.get("source"):
            raw = _set_frontmatter_field(raw, "source", "audio")
            changed = True

        if changed and not dry_run:
            with open(fpath, "w", encoding="utf-8") as fh:
                fh.write(raw)
            return "fixed"

        return "fixed" if changed else "unchanged"


def _set_frontmatter_field(raw: str, key: str, value: str) -> str:
    """Set a field in YAML frontmatter."""
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.MULTILINE)
    new_line = f"{key}: {value}"
    if pattern.search(raw):
        return pattern.sub(new_line, raw)
    # Insert before closing ---
    end = raw.find("\n---", 3)
    if end != -1:
        return raw[:end] + f"\n{key}: {value}" + raw[end:]
    return raw
