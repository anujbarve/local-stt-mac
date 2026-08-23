"""Tests for vault/writer.py"""

import os
import shutil
import tempfile

import pytest

from brain_contract import Understanding, NoteType, Entity, EntityType, Urgency
from vault.writer import VaultWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_vault(tmp_path):
    """Create a temporary vault root."""
    root = str(tmp_path / "test_vault")
    os.makedirs(root, exist_ok=True)
    return root


def _make_task():
    return Understanding(
        text="I need to fix the VAD threshold by Friday",
        type=NoteType.TASK,
        priority=0.82,
        urgency=Urgency.SOON,
        confidence=0.87,
        entities=[
            Entity(name="VAD", entity_type=EntityType.TOOL),
            Entity(name="Friday", entity_type=EntityType.DATE),
        ],
        action="fix VAD threshold",
        topics=["technology", "engineering"],
        tags=["bug", "vad"],
    )


def _make_observation():
    return Understanding(
        text="The sky is blue today",
        type=NoteType.OBSERVATION,
        priority=0.2,
        topics=["nature"],
    )


def _make_meeting():
    return Understanding(
        text="Met with Alice and Bob about the project.",
        type=NoteType.MEETING,
        priority=0.6,
        entities=[
            Entity(name="Alice", entity_type=EntityType.PERSON),
            Entity(name="Bob", entity_type=EntityType.PERSON),
        ],
        action="project status update",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVaultWriter:
    def test_write_task(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        path = w.write(u)
        assert os.path.exists(path)
        assert path.startswith(os.path.join(tmp_vault, "tasks"))
        assert path.endswith(".md")

    def test_write_content_valid(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        path = w.write(u)
        with open(path) as fh:
            content = fh.read()
        assert content.startswith("---")
        assert "id: " in content
        assert "type: task" in content
        assert "fix VAD threshold" in content.lower() or "vad" in content.lower()

    def test_write_observation(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_observation()
        path = w.write(u)
        assert os.path.join(tmp_vault, "daily") in path

    def test_write_meeting(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_meeting()
        path = w.write(u)
        assert os.path.join(tmp_vault, "meetings") in path

    def test_write_sets_note_id(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        w.write(u)
        assert u.note_id is not None
        assert u.note_id.startswith("note_")

    def test_write_sets_file_path(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        path = w.write(u)
        assert u.file_path == path

    def test_write_preserves_existing_id(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        u.note_id = "note_2026_01_01_999"
        path = w.write(u)
        with open(path) as fh:
            content = fh.read()
        assert "id: note_2026_01_01_999" in content

    def test_filename_de_duplication(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        p1 = w.write(u)
        # Write the same thing again
        u2 = _make_task()
        p2 = w.write(u2)
        assert p1 != p2
        assert os.path.exists(p1)
        assert os.path.exists(p2)

    def test_directories_created(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        # All standard subdirectories should exist
        expected = ["daily", "tasks", "ideas", "meetings", "people",
                     "projects", "goals", "lessons", "_archive", "topics"]
        for d in expected:
            assert os.path.isdir(os.path.join(tmp_vault, d)), f"Missing: {d}"

    def test_write_daily_entry(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_observation()
        path = w.write_daily_entry(u, date="2026-08-23")
        assert os.path.exists(path)
        assert "2026-08-23" in path
        with open(path) as fh:
            content = fh.read()
        assert "Daily Journal" in content

    def test_write_daily_entry_append(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u1 = _make_observation()
        w.write_daily_entry(u1, date="2026-08-23")
        u2 = Understanding(text="Second note", type=NoteType.OBSERVATION)
        w.write_daily_entry(u2, date="2026-08-23")
        path = os.path.join(tmp_vault, "daily", "2026-08-23.md")
        with open(path) as fh:
            content = fh.read()
        assert content.count("## 2") == 2  # two entry headers

    def test_update_note(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        path = w.write(u)
        w.update_note(path, {"status": "in_progress"})
        with open(path) as fh:
            content = fh.read()
        assert "status: in_progress" in content
        assert "updated: " in content

    def test_update_note_missing_file(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        with pytest.raises(FileNotFoundError):
            w.update_note("/nonexistent/path.md", {"status": "done"})

    def test_backlink_entity_page_created(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        w.write(u)
        # Entity page for VAD tool should exist
        tool_path = os.path.join(tmp_vault, "tools", "vad.md")
        assert os.path.exists(tool_path)
        with open(tool_path) as fh:
            content = fh.read()
        assert "VAD" in content
        assert "[[" in content  # should contain a wikilink

    def test_backlink_appended_not_duplicated(self, tmp_vault):
        w = VaultWriter(tmp_vault)
        u = _make_task()
        w.write(u)
        # Write another note with the same entity
        u2 = Understanding(
            text="VAD is working well now",
            type=NoteType.OBSERVATION,
            entities=[Entity(name="VAD", entity_type=EntityType.TOOL)],
        )
        w.write(u2)
        tool_path = os.path.join(tmp_vault, "tools", "vad.md")
        with open(tool_path) as fh:
            content = fh.read()
        # Should have backlinks section but not duplicated links
        count = content.count("# Backlinks")
        assert count == 1
