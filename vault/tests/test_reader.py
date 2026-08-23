"""Tests for vault/reader.py"""

import os
import shutil
import tempfile

import pytest

from brain_contract import NoteType
from vault.reader import VaultReader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TASK = """\
---
id: note_2026_08_23_001
type: task
created: 2026-08-23T14:30:00Z
updated: 2026-08-23T14:30:00Z
source: audio
priority: high
urgency: soon
sentiment: neutral
confidence: 0.87
people: []
projects: []
tasks: []
tools: []
dates: []
tags:
  - bug
  - vad
category: task
related_notes: []
backlinks: []
embedding_id: vec_note_2026_08_23_001
---

# Fix VAD Threshold

The Silero VAD threshold is too sensitive.

See [[sarah-chen]] for context and [[voiceassist]] project details.
"""

SAMPLE_OBSERVATION = """\
---
id: note_2026_08_23_002
type: observation
created: 2026-08-23T15:00:00Z
updated: 2026-08-23T15:00:00Z
source: audio
priority: medium
tags:
  - nature
category: observation
---

# The Sky is Blue

Noticed the sky is very blue today.
"""

SAMPLE_IDEA = """\
---
id: note_2026_08_23_003
type: idea
created: 2026-08-23T16:00:00Z
updated: 2026-08-23T16:00:00Z
source: audio
tags:
  - technology
  - engineering
category: idea
---

# Circular Buffer for Audio

What if we use a circular buffer for the pre-roll?
"""


@pytest.fixture
def vault_with_notes(tmp_path):
    """Create a vault with sample notes."""
    vault_dir = str(tmp_path / "vault")
    os.makedirs(os.path.join(vault_dir, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(vault_dir, "daily"), exist_ok=True)
    os.makedirs(os.path.join(vault_dir, "ideas"), exist_ok=True)

    with open(os.path.join(vault_dir, "tasks", "fix-vad.md"), "w") as f:
        f.write(SAMPLE_TASK)
    with open(os.path.join(vault_dir, "daily", "sky.md"), "w") as f:
        f.write(SAMPLE_OBSERVATION)
    with open(os.path.join(vault_dir, "ideas", "circular-buffer.md"), "w") as f:
        f.write(SAMPLE_IDEA)

    return vault_dir


@pytest.fixture
def empty_vault(tmp_path):
    """Create an empty vault."""
    vault_dir = str(tmp_path / "vault")
    os.makedirs(vault_dir, exist_ok=True)
    return vault_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVaultReader:
    def test_read_single_note(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        note = reader.read(os.path.join(vault_with_notes, "tasks", "fix-vad.md"))
        assert note is not None
        assert note["id"] == "note_2026_08_23_001"
        assert note["type"] == "task"
        assert note["priority"] == "high"
        assert "VAD" in note["body"]

    def test_read_nonexistent_returns_none(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        note = reader.read("/nonexistent/path.md")
        assert note is None

    def test_read_all_finds_all_notes(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        notes = reader.read_all()
        assert len(notes) == 3

    def test_read_by_type_filters(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        tasks = reader.read_by_type(NoteType.TASK)
        assert len(tasks) == 1
        assert tasks[0]["type"] == "task"

    def test_read_by_type_observation(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        obs = reader.read_by_type(NoteType.OBSERVATION)
        assert len(obs) == 1

    def test_read_by_id_finds_exact(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        note = reader.read_by_id("note_2026_08_23_003")
        assert note is not None
        assert note["type"] == "idea"

    def test_read_by_id_not_found(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        note = reader.read_by_id("nonexistent_id")
        assert note is None

    def test_read_by_tag(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        notes = reader.read_by_tag("bug")
        assert len(notes) == 1
        assert "bug" in notes[0]["tags"]

    def test_read_by_tag_case_insensitive(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        notes = reader.read_by_tag("BUG")
        assert len(notes) == 1

    def test_read_directory(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        notes = reader.read_directory("tasks")
        assert len(notes) == 1

    def test_read_directory_nonexistent(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        notes = reader.read_directory("nonexistent")
        assert notes == []

    def test_wikilinks_extracted(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        note = reader.read(os.path.join(vault_with_notes, "tasks", "fix-vad.md"))
        assert "sarah-chen" in note["wikilinks"]
        assert "voiceassist" in note["wikilinks"]

    def test_body_extracted(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        note = reader.read(os.path.join(vault_with_notes, "tasks", "fix-vad.md"))
        assert "Fix VAD Threshold" in note["body"]
        assert "Silero VAD" in note["body"]

    def test_file_path_is_absolute(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        note = reader.read(os.path.join(vault_with_notes, "tasks", "fix-vad.md"))
        assert os.path.isabs(note["file_path"])

    def test_count(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        assert reader.count() == 3

    def test_list_files(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        files = reader.list_files()
        assert len(files) == 3
        assert all(f.endswith(".md") for f in files)

    def test_get_note_summary(self, vault_with_notes):
        reader = VaultReader(vault_with_notes)
        note = reader.read(os.path.join(vault_with_notes, "tasks", "fix-vad.md"))
        summary = reader.get_note_summary(note)
        assert len(summary) <= 200

    def test_graceful_skip_bad_files(self, vault_with_notes):
        # Write a bad file
        with open(os.path.join(vault_with_notes, "tasks", "bad.md"), "w") as f:
            f.write("This is not a note with frontmatter.")
        reader = VaultReader(vault_with_notes)
        notes = reader.read_all()
        # Bad file should be skipped
        assert len(notes) == 3

    def test_read_all_empty_vault(self, empty_vault):
        reader = VaultReader(empty_vault)
        notes = reader.read_all()
        assert notes == []

    def test_count_empty_vault(self, empty_vault):
        reader = VaultReader(empty_vault)
        assert reader.count() == 0
