"""
Vault Schema Validation
=======================

Defines the valid structure for each note type and validates
frontmatter before writing.  Also provides helpers for ID
generation and filename slugification.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from brain_contract import NoteType, EntityType

# ---------------------------------------------------------------------------
# Frontmatter field definitions per note type
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: List[str] = ["id", "type", "created", "updated", "source"]

TYPE_SPECIFIC_FIELDS: Dict[NoteType, List[str]] = {
    NoteType.TASK: ["status", "priority", "urgency", "due_date", "project"],
    NoteType.PROJECT_UPDATE: ["status", "priority", "progress", "people"],
    NoteType.PERSON_INFO: ["name", "relationship", "birthday", "organization"],
    NoteType.IDEA: ["stage", "origin", "potential_impact"],
    NoteType.LEARNING: ["source_type", "source_title", "confidence"],
    NoteType.DECISION: ["decision", "rationale", "alternatives", "reversible"],
    NoteType.MEETING: ["date", "participants", "type"],
    NoteType.REMINDER: ["reminder_date", "recurrence", "completed"],
    NoteType.PREFERENCE: ["category", "item", "sentiment"],
    NoteType.OBSERVATION: [],
    NoteType.LESSON: [],
    NoteType.QUESTION: [],
    NoteType.PLAN: ["stage", "milestones"],
    NoteType.CELEBRATION: ["achievement"],
    NoteType.FRUSTRATION: [],
    NoteType.GRATITUDE: ["recipient"],
    NoteType.JOURNAL: ["mood"],
    NoteType.CONTACT: ["name", "relationship", "organization"],
    NoteType.HEALTH: ["category", "metric"],
    NoteType.FINANCE: ["category", "amount"],
    NoteType.TRAVEL: ["destination", "dates"],
    NoteType.GOAL: ["target_date", "progress", "metric"],
    NoteType.HABIT: ["frequency", "streak"],
    NoteType.WISH: [],
    NoteType.MEDIA: ["title", "media_type", "rating"],
    NoteType.PLACE: ["location", "category"],
    NoteType.ACCOUNT: ["service", "username"],
    NoteType.TOOL_INFO: ["tool_name", "version"],
}

# Valid values for common fields
VALID_STATUSES = {"todo", "in_progress", "done", "blocked", "cancelled"}
VALID_PRIORITIES = {"critical", "high", "medium", "low", "someday"}
VALID_URGENCIES = {"immediate", "soon", "later", "someday"}
VALID_SENTIMENTS = {"positive", "negative", "neutral", "mixed", "uncertain"}

# Frontmatter keys that the writer always sets (not user-facing validation targets)
IMPLICIT_FIELDS = {"backlinks", "embedding_id", "related_notes"}


# ---------------------------------------------------------------------------
# NoteSchema
# ---------------------------------------------------------------------------

class NoteSchema:
    """Validate and generate note metadata."""

    @classmethod
    def validate(cls, frontmatter: Dict[str, Any], note_type: NoteType) -> List[str]:
        """Validate frontmatter dict for *note_type*.  Returns list of error strings (empty == valid)."""
        errors: List[str] = []

        # ---- required fields ----
        for field in REQUIRED_FIELDS:
            if field not in frontmatter or frontmatter[field] is None:
                errors.append(f"Missing required field: {field}")

        # ---- note-type-specific fields (must be present) ----
        for field in TYPE_SPECIFIC_FIELDS.get(note_type, []):
            if field not in frontmatter:
                errors.append(f"Missing {note_type.value}-specific field: {field}")

        # ---- semantic checks ----
        fm_type = frontmatter.get("type")
        if fm_type and fm_type != note_type.value:
            errors.append(
                f"Frontmatter type '{fm_type}' does not match expected '{note_type.value}'"
            )

        priority = frontmatter.get("priority")
        if priority is not None and priority not in VALID_PRIORITIES:
            errors.append(f"Invalid priority '{priority}'. Expected one of {VALID_PRIORITIES}")

        urgency = frontmatter.get("urgency")
        if urgency is not None and urgency not in VALID_URGENCIES:
            errors.append(f"Invalid urgency '{urgency}'. Expected one of {VALID_URGENCIES}")

        sentiment = frontmatter.get("sentiment")
        if sentiment is not None and sentiment not in VALID_SENTIMENTS:
            errors.append(f"Invalid sentiment '{sentiment}'. Expected one of {VALID_SENTIMENTS}")

        status = frontmatter.get("status")
        if status is not None and status not in VALID_STATUSES:
            errors.append(f"Invalid status '{status}'. Expected one of {VALID_STATUSES}")

        # ---- id format ----
        note_id = frontmatter.get("id", "")
        if note_id and not re.match(r"^note_\d{4}_\d{2}_\d{2}_\d{3}$", note_id):
            errors.append(f"Invalid id format '{note_id}'. Expected note_YYYY_MM_DD_NNN")

        return errors

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    @classmethod
    def generate_id(cls, counter: Optional[int] = None) -> str:
        """Generate a unique note ID: ``note_YYYY_MM_DD_NNN``.

        If *counter* is ``None`` a random 3-digit suffix is used.  Callers
        that track sequence numbers can pass an explicit counter.
        """
        today = datetime.utcnow()
        if counter is not None:
            seq = f"{counter:03d}"
        else:
            seq = f"{uuid.uuid4().int % 1000:03d}"
        return f"note_{today.year}_{today.month:02d}_{today.day:02d}_{seq}"

    # ------------------------------------------------------------------
    # Slugify
    # ------------------------------------------------------------------

    @classmethod
    def slugify(cls, text: str) -> str:
        """Convert *text* to a URL-safe filename slug.

        ``"Fix VAD Bug"`` → ``"fix-vad-bug"``
        """
        slug = text.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)       # drop punctuation
        slug = re.sub(r"[\s_]+", "-", slug)          # spaces/underscores → hyphens
        slug = re.sub(r"-{2,}", "-", slug)           # collapse runs
        slug = slug.strip("-")
        # Cap length at 80 chars
        return slug[:80] if slug else "untitled"


# ---------------------------------------------------------------------------
# VaultValidator  –  full-vault checks
# ---------------------------------------------------------------------------

class VaultValidator:
    """Scan the entire vault and report structural problems."""

    def __init__(self, vault_root: str):
        self.vault_root = vault_root

    # ------------------------------------------------------------------

    def validate_all(self) -> Dict[str, List[str]]:
        """Return ``{file_path: [error_strings]}`` for every ``.md`` file."""
        errors: Dict[str, List[str]] = {}
        for dirpath, _dirnames, filenames in os.walk(self.vault_root):
            for fname in filenames:
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, self.vault_root)
                try:
                    fm, _body = _parse_frontmatter_file(fpath)
                    note_type = NoteType(fm.get("type", "observation"))
                    errs = NoteSchema.validate(fm, note_type)
                    if errs:
                        errors[rel] = errs
                except Exception as exc:
                    errors[rel] = [f"Parse error: {exc}"]
        return errors

    # ------------------------------------------------------------------

    def find_orphans(self) -> List[str]:
        """Find notes with no backlinks and no outbound wikilinks."""
        # Build maps: outlinks per note, backlinks per note
        note_files: Dict[str, str] = {}   # basename_without_ext → full path
        outlinks: Dict[str, set] = {}     # rel_path → set of target basenames
        inbound: Dict[str, set] = {}      # basename → set of source rel_paths

        for dirpath, _dn, fnames in os.walk(self.vault_root):
            for fname in fnames:
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, self.vault_root)
                base = fname[:-3]
                note_files[base] = rel
                links = _extract_wikilinks_from_file(fpath)
                outlinks[rel] = {l for l in links if l in note_files or True}
                for target in links:
                    inbound.setdefault(target, set()).add(rel)

        orphans: List[str] = []
        for base, rel in note_files.items():
            has_out = bool(outlinks.get(rel))
            has_in = base in inbound
            if not has_out and not has_in:
                orphans.append(rel)
        return orphans

    # ------------------------------------------------------------------

    def find_broken_links(self) -> List[str]:
        """Return list of file paths whose wikilinks point to nonexistent notes."""
        known: Dict[str, bool] = {}
        broken: List[str] = []

        for dirpath, _dn, fnames in os.walk(self.vault_root):
            for fname in fnames:
                if fname.endswith(".md"):
                    known[fname[:-3]] = True

        for dirpath, _dn, fnames in os.walk(self.vault_root):
            for fname in fnames:
                if not fname.endswith(".md"):
                    continue
                fpath = os.path.join(dirpath, fname)
                rel = os.path.relpath(fpath, self.vault_root)
                links = _extract_wikilinks_from_file(fpath)
                for link in links:
                    if link not in known:
                        broken.append(rel)
                        break  # one report per file is enough
        return broken


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_frontmatter_file(fpath: str):
    """Read a .md file and return ``(frontmatter_dict, body_str)``."""
    with open(fpath, "r", encoding="utf-8") as fh:
        raw = fh.read()
    return _parse_frontmatter(raw)


def _parse_frontmatter(raw: str):
    """Parse YAML frontmatter from raw markdown text.

    Returns ``(frontmatter_dict, body_str)``.  A minimal YAML parser avoids
    a hard dependency on ``PyYAML``; for the subset we need (flat keys,
    lists of strings) this is sufficient.
    """
    if not raw.startswith("---"):
        return {}, raw

    # Find closing ---
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw

    fm_block = raw[3:end].strip()
    body = raw[end + 4:].lstrip("\n")
    return _parse_yaml_simple(fm_block), body


def _parse_yaml_simple(text: str) -> Dict[str, Any]:
    """Minimal YAML parser supporting flat key: value and key: list entries."""
    result: Dict[str, Any] = {}
    current_key: Optional[str] = None
    current_list: Optional[List[str]] = None

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        # Detect list items
        if stripped.startswith("- "):
            if current_list is not None:
                current_list.append(stripped[2:].strip().strip("'\""))
            continue

        # Detect key: value  or  key:
        if ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")

            # Save previous list
            if current_key and current_list is not None:
                result[current_key] = current_list

            current_key = key
            if value:
                result[key] = _coerce_value(value)
                current_list = None
            else:
                current_list = []
        else:
            current_list = None

    # Flush last key
    if current_key:
        if current_list is not None:
            result[current_key] = current_list
        elif current_key not in result:
            pass  # already set

    return result


def _coerce_value(v: str) -> Any:
    """Coerce a YAML scalar string to Python type."""
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    if v.lower() in ("null", "none", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _extract_wikilinks_from_file(fpath: str) -> List[str]:
    """Return list of [[wikilink]] targets found in *fpath*."""
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return []
    return _extract_wikilinks(raw)


def _extract_wikilinks(text: str) -> List[str]:
    """Extract ``[[target]]`` links from markdown text."""
    return re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+)?\]\]", text)
