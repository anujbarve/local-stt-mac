"""Tests for vault/migration.py"""

import os

import pytest

from vault.migration import SchemaMigrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault_dir(tmp_path):
    return str(tmp_path / "vault")


@pytest.fixture
def migrator(vault_dir):
    return SchemaMigrator(vault_root=vault_dir)


def _write_note(vault_dir, rel_path, content):
    fpath = os.path.join(vault_dir, rel_path)
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "w") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSchemaMigrator:
    def test_validate_all_empty_vault(self, migrator):
        result = migrator.validate_all()
        assert result["total"] == 0
        assert result["valid"] == 0

    def test_validate_all_valid_note(self, vault_dir, migrator):
        _write_note(vault_dir, "tasks/n1.md", (
            "---\nid: n1\ntype: task\ncreated: x\nupdated: x\nsource: audio\n---\n\n# Task\n"
        ))
        result = migrator.validate_all()
        assert result["total"] == 1
        assert result["valid"] == 1
        assert result["issues"] == []

    def test_validate_all_missing_type(self, vault_dir, migrator):
        _write_note(vault_dir, "tasks/n2.md", (
            "---\nid: n2\ncreated: x\nupdated: x\nsource: audio\n---\n\n# No type\n"
        ))
        result = migrator.validate_all()
        assert len(result["issues"]) >= 1
        assert any("missing_type" in i["issue"] for i in result["issues"])

    def test_validate_all_invalid_type(self, vault_dir, migrator):
        _write_note(vault_dir, "tasks/n3.md", (
            "---\nid: n3\ntype: INVALID_TYPE\ncreated: x\nupdated: x\nsource: audio\n---\n\n# Bad\n"
        ))
        result = migrator.validate_all()
        assert any("invalid_type" in i["issue"] for i in result["issues"])

    def test_validate_all_no_frontmatter(self, vault_dir, migrator):
        _write_note(vault_dir, "tasks/n4.md", "Just plain text, no frontmatter.\n")
        result = migrator.validate_all()
        assert any("no_frontmatter" in i["issue"] for i in result["issues"])

    def test_auto_fix_dry_run(self, vault_dir, migrator):
        _write_note(vault_dir, "tasks/n5.md", (
            "---\nid: n5\ntype: task\n---\n\n# Task\n"
        ))
        result = migrator.auto_fix(dry_run=True)
        # Dry run should not modify files
        with open(os.path.join(vault_dir, "tasks", "n5.md")) as f:
            content = f.read()
        assert "updated:" not in content

    def test_auto_fix_real(self, vault_dir, migrator):
        _write_note(vault_dir, "tasks/n6.md", (
            "---\nid: n6\ntype: task\n---\n\n# Task\n"
        ))
        result = migrator.auto_fix(dry_run=False)
        assert result["fixed"] >= 1
        # File should now have updated field
        with open(os.path.join(vault_dir, "tasks", "n6.md")) as f:
            content = f.read()
        assert "updated:" in content

    def test_get_stats(self, vault_dir, migrator):
        _write_note(vault_dir, "tasks/n1.md", (
            "---\nid: n1\ntype: task\ncreated: x\nupdated: x\nsource: audio\nembedding_id: vec_1\n---\n\n# Task\n"
        ))
        _write_note(vault_dir, "daily/n2.md", (
            "---\nid: n2\ntype: observation\ncreated: x\nupdated: x\nsource: audio\n---\n\n# Obs\n"
        ))
        stats = migrator.get_stats()
        assert stats["total_notes"] == 2
        assert stats["with_frontmatter"] == 2
        assert stats["with_embedding"] == 1

    def test_skips_backup_dir(self, vault_dir, migrator):
        _write_note(vault_dir, "_backups/old.md", "backup content")
        _write_note(vault_dir, "tasks/n1.md", (
            "---\nid: n1\ntype: task\ncreated: x\nupdated: x\nsource: audio\n---\n\n# Task\n"
        ))
        result = migrator.validate_all()
        assert result["total"] == 1
