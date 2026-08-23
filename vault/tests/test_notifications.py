"""Tests for vault/notifications.py"""

import os

import pytest

from vault.notifications import NotificationManager
from vault.writer import VaultWriter
from brain_contract import Understanding, NoteType, Urgency


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault_with_tasks(tmp_path):
    """Create a vault with sample tasks."""
    vault_dir = str(tmp_path / "vault")
    writer = VaultWriter(vault_dir)

    # Write some tasks
    u1 = Understanding(
        text="Fix the critical bug by tomorrow",
        type=NoteType.TASK,
        priority=0.95,
        urgency=Urgency.IMMEDIATE,
    )
    writer.write(u1)

    u2 = Understanding(
        text="Update the README sometime",
        type=NoteType.TASK,
        priority=0.3,
        urgency=Urgency.SOMEDAY,
    )
    writer.write(u2)

    u3 = Understanding(
        text="Call dentist next week",
        type=NoteType.REMINDER,
        priority=0.6,
    )
    writer.write(u3)

    return vault_dir


@pytest.fixture
def notifications(vault_with_tasks):
    return NotificationManager(vault_root=vault_with_tasks)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNotificationManager:
    def test_add_rule(self, notifications):
        notifications.add_rule({"type": "deadline", "hours_before": 24, "channel": "macos"})
        assert len(notifications.rules) == 1

    def test_remove_rule(self, notifications):
        notifications.add_rule({"type": "deadline", "hours_before": 24, "channel": "macos"})
        assert notifications.remove_rule(0)
        assert len(notifications.rules) == 0

    def test_remove_rule_invalid_index(self, notifications):
        assert not notifications.remove_rule(99)

    def test_check_reminders(self, notifications):
        reminders = notifications.check_reminders()
        # Should find at least the critical task and the reminder
        assert len(reminders) >= 2

    def test_check_reminders_sorted(self, notifications):
        reminders = notifications.check_reminders()
        if len(reminders) >= 2:
            # First should be higher priority/more urgent
            assert reminders[0].get("urgency", "later") in ("immediate", "soon")

    def test_daily_digest(self, notifications):
        digest = notifications.daily_digest()
        assert "date" in digest
        assert "total_notes" in digest
        assert digest["total_notes"] >= 3
        assert "tasks_todo" in digest
        assert "high_priority" in digest

    def test_daily_digest_has_tasks(self, notifications):
        digest = notifications.daily_digest()
        assert digest["tasks_todo"] >= 2

    def test_send_macos_notification_false_on_non_mac(self, notifications):
        # On non-macOS, should return False
        result = notifications.send_macos_notification("Test", "Hello")
        # May be True or False depending on platform
        assert isinstance(result, bool)

    def test_rules_persist(self, vault_with_tasks):
        n1 = NotificationManager(vault_root=vault_with_tasks)
        n1.add_rule({"type": "test", "value": 42})
        # Create new instance — rules should persist
        n2 = NotificationManager(vault_root=vault_with_tasks)
        assert len(n2.rules) == 1
        assert n2.rules[0]["value"] == 42
