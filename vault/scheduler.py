"""
Vault Scheduler
===============

Smart scheduling: scan vault for deadlines, suggest daily schedules,
detect conflicts, and generate weekly overviews.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from vault.config import VAULT_ROOT
from vault.reader import VaultReader


class SmartScheduler:
    """Scan vault for deadlines and suggest schedules."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.reader = VaultReader(vault_root)

    def scan_deadlines(self) -> List[Dict[str, Any]]:
        """Scan all notes for deadlines and high-priority items."""
        deadlines: List[Dict[str, Any]] = []

        for note in self.reader.read_all():
            ntype = note.get("type", "")
            status = note.get("status", "todo")

            if status in ("done", "cancelled"):
                continue

            if ntype in ("task", "reminder"):
                priority = self._parse_priority(note.get("priority", "medium"))
                urgency = note.get("urgency", "later")
                title = note.get("body", "").split("\n")[0].lstrip("# ").strip()[:80]

                deadlines.append({
                    "note_id": note.get("id", ""),
                    "title": title,
                    "type": ntype,
                    "priority": priority,
                    "urgency": urgency,
                    "file_path": note.get("file_path", ""),
                })

        # Sort by urgency then priority
        urgency_order = {"immediate": 0, "soon": 1, "later": 2, "someday": 3}
        deadlines.sort(key=lambda d: (
            urgency_order.get(d.get("urgency", "later"), 2),
            -d.get("priority", 0.5),
        ))

        return deadlines

    def suggest_daily_schedule(self, available_hours: int = 8) -> Dict[str, Any]:
        """Generate a suggested daily schedule."""
        deadlines = self.scan_deadlines()
        now = datetime.now(timezone.utc)

        blocks: List[Dict[str, Any]] = []
        hours_used = 0.0

        for item in deadlines:
            if hours_used >= available_hours:
                break

            # Estimate duration based on priority
            priority = item.get("priority", 0.5)
            if priority >= 0.8:
                duration_hours = 2.0
            elif priority >= 0.5:
                duration_hours = 1.0
            else:
                duration_hours = 0.5

            if hours_used + duration_hours > available_hours:
                duration_hours = available_hours - hours_used

            start_hour = 9 + int(hours_used)
            start_min = int((hours_used % 1) * 60)
            time_str = f"{start_hour:02d}:{start_min:02d}"

            blocks.append({
                "time": time_str,
                "task": item["title"],
                "priority": item["priority"],
                "duration": f"{int(duration_hours * 60)}m",
                "note_id": item.get("note_id", ""),
            })

            hours_used += duration_hours

        overdue = [d for d in deadlines if d.get("urgency") == "immediate"]
        upcoming = [d for d in deadlines if d.get("urgency") in ("soon", "later")]

        return {
            "date": now.strftime("%Y-%m-%d"),
            "blocks": blocks,
            "overdue": overdue[:5],
            "upcoming": upcoming[:5],
            "total_hours_scheduled": round(hours_used, 1),
        }

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """Find overlapping deadlines or conflicting tasks."""
        deadlines = self.scan_deadlines()
        conflicts: List[Dict[str, Any]] = []

        # Group by urgency level
        immediate = [d for d in deadlines if d.get("urgency") == "immediate"]
        if len(immediate) > 3:
            conflicts.append({
                "notes": [d["note_id"] for d in immediate],
                "reason": f"{len(immediate)} tasks marked as immediate",
                "suggestion": "Consider prioritizing — not everything can be immediate",
            })

        # Check for same-type clustering
        by_type: Dict[str, List] = {}
        for d in deadlines:
            by_type.setdefault(d.get("type", ""), []).append(d)

        for ntype, items in by_type.items():
            if len(items) > 5:
                conflicts.append({
                    "notes": [d["note_id"] for d in items],
                    "reason": f"{len(items)} {ntype} items pending",
                    "suggestion": f"Review and batch-process {ntype} items",
                })

        return conflicts

    def weekly_overview(self) -> Dict[str, Any]:
        """Generate weekly task overview."""
        all_notes = self.reader.read_all()

        tasks_todo = 0
        tasks_done = 0
        tasks_in_progress = 0
        overdue = 0
        by_priority: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}

        for note in all_notes:
            if note.get("type") != "task":
                continue
            status = note.get("status", "todo")
            priority = self._parse_priority(note.get("priority", "medium"))
            urgency = note.get("urgency", "later")

            if status == "done":
                tasks_done += 1
            elif status == "in_progress":
                tasks_in_progress += 1
            elif status == "todo":
                tasks_todo += 1
                if urgency == "immediate":
                    overdue += 1

            if priority >= 0.8:
                by_priority["high"] += 1
            elif priority >= 0.4:
                by_priority["medium"] += 1
            else:
                by_priority["low"] += 1

        total = tasks_todo + tasks_done + tasks_in_progress
        completion_rate = (tasks_done / total * 100) if total > 0 else 0

        return {
            "total_tasks": total,
            "completed": tasks_done,
            "in_progress": tasks_in_progress,
            "todo": tasks_todo,
            "overdue": overdue,
            "by_priority": by_priority,
            "completion_rate": round(completion_rate, 1),
        }

    def _parse_priority(self, val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        mapping = {"critical": 0.95, "high": 0.8, "medium": 0.5, "low": 0.3, "someday": 0.1}
        return mapping.get(str(val).lower(), 0.5)
