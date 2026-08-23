"""Tests for vault/schema.py"""

import os
import shutil
import tempfile

import pytest

from brain_contract import NoteType
from vault.schema import NoteSchema, VaultValidator


# ---------------------------------------------------------------------------
# NoteSchema
# ---------------------------------------------------------------------------

class TestNoteSchema:
    """NoteSchema helpers."""

    def test_generate_id_format(self):
        nid = NoteSchema.generate_id()
        assert nid.startswith("note_")
        parts = nid.split("_")
        assert len(parts) == 5  # note, YYYY, MM, DD, NNN
        assert len(parts[1]) == 4
        assert len(parts[2]) == 2
        assert len(parts[3]) == 2
        assert len(parts[4]) == 3

    def test_generate_id_explicit_counter(self):
        nid = NoteSchema.generate_id(counter=7)
        assert nid.endswith("_007")

    def test_generate_id_deterministic_with_counter(self):
        id1 = NoteSchema.generate_id(counter=42)
        id2 = NoteSchema.generate_id(counter=42)
        assert id1 == id2

    def test_slugify_simple(self):
        assert NoteSchema.slugify("Fix VAD Bug") == "fix-vad-bug"

    def test_slugify_punctuation(self):
        assert NoteSchema.slugify("Hello, World! (v2)") == "hello-world-v2"

    def test_slugify_multiple_spaces(self):
        assert NoteSchema.slugify("  too   many   spaces  ") == "too-many-spaces"

    def test_slugify_empty(self):
        assert NoteSchema.slugify("") == "untitled"

    def test_slugify_max_length(self):
        long = "a" * 200
        assert len(NoteSchema.slugify(long)) <= 80

    def test_validate_valid_task(self):
        fm = {
            "id": "note_2026_08_23_001",
            "type": "task",
            "created": "2026-08-23T14:30:00Z",
            "updated": "2026-08-23T14:30:00Z",
            "source": "audio",
            "status": "todo",
            "priority": "high",
            "urgency": "soon",
            "due_date": None,
            "project": None,
        }
        errors = NoteSchema.validate(fm, NoteType.TASK)
        assert errors == []

    def test_validate_missing_required(self):
        fm = {"type": "task"}
        errors = NoteSchema.validate(fm, NoteType.TASK)
        assert any("Missing required field" in e for e in errors)

    def test_validate_type_mismatch(self):
        fm = {
            "id": "note_2026_08_23_001",
            "type": "idea",  # wrong!
            "created": "2026-08-23T14:30:00Z",
            "updated": "2026-08-23T14:30:00Z",
            "source": "audio",
            "status": "todo",
            "priority": "high",
            "urgency": "soon",
            "due_date": None,
            "project": None,
        }
        errors = NoteSchema.validate(fm, NoteType.TASK)
        assert any("does not match" in e for e in errors)

    def test_validate_invalid_priority(self):
        fm = {
            "id": "note_2026_08_23_001",
            "type": "task",
            "created": "2026-08-23T14:30:00Z",
            "updated": "2026-08-23T14:30:00Z",
            "source": "audio",
            "status": "todo",
            "priority": "mega-ultra",  # invalid
            "urgency": "soon",
        }
        errors = NoteSchema.validate(fm, NoteType.TASK)
        assert any("Invalid priority" in e for e in errors)

    def test_validate_id_format(self):
        fm = {
            "id": "bad-id",
            "type": "observation",
            "created": "2026-08-23T14:30:00Z",
            "updated": "2026-08-23T14:30:00Z",
            "source": "audio",
        }
        errors = NoteSchema.validate(fm, NoteType.OBSERVATION)
        assert any("Invalid id format" in e for e in errors)

    def test_validate_observation_minimal(self):
        fm = {
            "id": "note_2026_08_23_001",
            "type": "observation",
            "created": "2026-08-23T14:30:00Z",
            "updated": "2026-08-23T14:30:00Z",
            "source": "audio",
        }
        errors = NoteSchema.validate(fm, NoteType.OBSERVATION)
        assert errors == []


# ---------------------------------------------------------------------------
# VaultValidator
# ---------------------------------------------------------------------------

class TestVaultValidator:
    """VaultValidator on a temporary vault directory."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.vault_dir = os.path.join(self.tmpdir, "vault")
        os.makedirs(self.vault_dir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_note(self, rel_path: str, content: str):
        fpath = os.path.join(self.vault_dir, rel_path)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w") as fh:
            fh.write(content)

    def test_validate_all_valid(self):
        self._write_note("daily/n1.md", (
            "---\nid: note_2026_08_23_001\ntype: observation\n"
            "created: 2026-08-23\nupdated: 2026-08-23\nsource: audio\n---\n\n# Hi\n"
        ))
        validator = VaultValidator(self.vault_dir)
        errors = validator.validate_all()
        assert errors == {}

    def test_validate_all_bad_type(self):
        self._write_note("daily/n2.md", (
            "---\nid: note_2026_08_23_002\ntype: INVALID\n"
            "created: 2026-08-23\nupdated: 2026-08-23\nsource: audio\n---\n\n# Bad\n"
        ))
        validator = VaultValidator(self.vault_dir)
        errors = validator.validate_all()
        # Should report parse error or validation error
        assert len(errors) > 0

    def test_broken_links(self):
        self._write_note("tasks/note1.md", (
            "---\nid: n1\ntype: task\ncreated: x\nupdated: x\nsource: audio\n---\n\n"
            "See [[nonexistent-note]] for details.\n"
        ))
        validator = VaultValidator(self.vault_dir)
        broken = validator.find_broken_links()
        assert len(broken) == 1

    def test_no_broken_links_when_valid(self):
        self._write_note("tasks/note1.md", (
            "---\nid: n1\ntype: task\ncreated: x\nupdated: x\nsource: audio\n---\n\n"
            "See [[note2]] for details.\n"
        ))
        self._write_note("daily/note2.md", (
            "---\nid: n2\ntype: observation\ncreated: x\nupdated: x\nsource: audio\n---\n\n"
            "# Note 2\n"
        ))
        validator = VaultValidator(self.vault_dir)
        broken = validator.find_broken_links()
        assert broken == []
