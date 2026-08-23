"""Tests for vault/scheduler.py"""

import os

import pytest

from vault.scheduler import SmartScheduler
from vault.writer import VaultWriter
from brain_contract import Understanding, NoteType, Urgency


@pytest.fixture
def scheduler_with_tasks(tmp_path):
    vault_dir = str(tmp_path / "vault")
    writer = VaultWriter(vault_dir)

    writer.write(Understanding(
        text="Fix critical bug tomorrow",
        type=NoteType.TASK, priority=0.95, urgency=Urgency.IMMEDIATE,
    ))
    writer.write(Understanding(
        text="Update README sometime",
        type=NoteType.TASK, priority=0.3, urgency=Urgency.SOMEDAY,
    ))
    writer.write(Understanding(
        text="Call dentist next week",
        type=NoteType.REMINDER, priority=0.6, urgency=Urgency.LATER,
    ))

    return SmartScheduler(vault_root=vault_dir)


class TestSmartScheduler:
    def test_scan_deadlines(self, scheduler_with_tasks):
        deadlines = scheduler_with_tasks.scan_deadlines()
        assert len(deadlines) >= 2

    def test_scan_deadlines_sorted(self, scheduler_with_tasks):
        deadlines = scheduler_with_tasks.scan_deadlines()
        if len(deadlines) >= 2:
            # First should be most urgent
            assert deadlines[0].get("urgency") in ("immediate", "soon")

    def test_suggest_daily_schedule(self, scheduler_with_tasks):
        schedule = scheduler_with_tasks.suggest_daily_schedule()
        assert "date" in schedule
        assert "blocks" in schedule
        assert "overdue" in schedule
        assert "upcoming" in schedule

    def test_suggest_daily_schedule_blocks(self, scheduler_with_tasks):
        schedule = scheduler_with_tasks.suggest_daily_schedule(available_hours=4)
        assert schedule["total_hours_scheduled"] <= 4

    def test_detect_conflicts(self, scheduler_with_tasks):
        conflicts = scheduler_with_tasks.detect_conflicts()
        assert isinstance(conflicts, list)

    def test_weekly_overview(self, scheduler_with_tasks):
        overview = scheduler_with_tasks.weekly_overview()
        assert "total_tasks" in overview
        assert "completed" in overview
        assert "by_priority" in overview
        assert overview["total_tasks"] >= 2
