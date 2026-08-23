"""Tests for vault/backup.py"""

import os
import shutil

import pytest

from vault.backup import VaultBackup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault_with_notes(tmp_path):
    """Create a vault with sample notes."""
    vault_dir = str(tmp_path / "vault")
    os.makedirs(os.path.join(vault_dir, "tasks"), exist_ok=True)
    os.makedirs(os.path.join(vault_dir, "daily"), exist_ok=True)

    with open(os.path.join(vault_dir, "tasks", "note1.md"), "w") as f:
        f.write("---\nid: n1\ntype: task\ncreated: x\nupdated: x\nsource: audio\n---\n\n# Task 1\n")
    with open(os.path.join(vault_dir, "daily", "2026-08-23.md"), "w") as f:
        f.write("---\nid: d1\ntype: journal\ncreated: x\nupdated: x\nsource: audio\n---\n\n# Journal\n")

    return vault_dir


@pytest.fixture
def backup(vault_with_notes):
    return VaultBackup(vault_with_notes)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVaultBackup:
    def test_full_backup_creates_tarball(self, backup):
        path = backup.full_backup()
        assert os.path.exists(path)
        assert path.endswith(".tar.gz")

    def test_full_backup_with_label(self, backup):
        path = backup.full_backup(label="test")
        assert "test" in path

    def test_list_backups(self, backup):
        backup.full_backup()
        backup.full_backup()
        backups = backup.list_backups()
        assert len(backups) == 2

    def test_verify_backup(self, backup):
        path = backup.full_backup()
        result = backup.verify(path)
        assert result["valid"] is True
        assert result["file_count"] >= 1

    def test_verify_nonexistent(self, backup):
        result = backup.verify("/nonexistent/backup.tar.gz")
        assert result["valid"] is False

    def test_restore_from_backup(self, backup, tmp_path):
        path = backup.full_backup()
        target = str(tmp_path / "restored")
        success = backup.restore(path, target=target)
        assert success is True
        assert os.path.isdir(target)

    def test_incremental_backup(self, backup):
        # Full backup first
        backup.full_backup()
        # Incremental
        incr_path = backup.incremental_backup()
        assert os.path.exists(incr_path)

    def test_incremental_no_changes(self, backup):
        backup.full_backup()
        incr_path = backup.incremental_backup()
        assert os.path.exists(incr_path)

    def test_delete_backup(self, backup):
        path = backup.full_backup()
        assert backup.delete_backup(path)
        assert not os.path.exists(path)
        assert len(backup.list_backups()) == 0

    def test_cleanup_keeps_last_n(self, backup):
        for _ in range(5):
            backup.full_backup()
        initial = len(backup.list_backups())
        deleted = backup.cleanup(keep_last=2)
        assert deleted >= 1
        assert len(backup.list_backups()) <= initial

    def test_cleanup_nothing_to_delete(self, backup):
        backup.full_backup()
        deleted = backup.cleanup(keep_last=5)
        assert deleted == 0
