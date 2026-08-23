"""
Vault Notifications
===================

Task reminders, deadline alerts, daily digest generation,
and webhook notification support.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from vault.config import VAULT_ROOT
from vault.reader import VaultReader


class NotificationManager:
    """Manage task reminders and webhook notifications."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.reader = VaultReader(vault_root)
        self.rules: List[Dict[str, Any]] = []
        self._load_rules()

    def add_rule(self, rule: Dict[str, Any]) -> None:
        """Add a notification rule.

        Rule format::

            {"type": "deadline", "hours_before": 24, "channel": "macos"}
            {"type": "priority", "min_priority": 0.9, "channel": "webhook"}
            {"type": "daily_digest", "time": "09:00", "channel": "macos"}
        """
        self.rules.append(rule)
        self._save_rules()

    def remove_rule(self, index: int) -> bool:
        """Remove a rule by index."""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)
            self._save_rules()
            return True
        return False

    def check_reminders(self) -> List[Dict[str, Any]]:
        """Scan vault for upcoming deadlines and high-priority items."""
        reminders: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)

        for note in self.reader.read_all():
            # Check for task deadlines
            if note.get("type") == "task":
                urgency = note.get("urgency", "later")
                priority = self._parse_priority(note.get("priority", "medium"))
                status = note.get("status", "todo")

                if status in ("done", "cancelled"):
                    continue

                if urgency in ("immediate", "soon") or priority >= 0.8:
                    reminders.append({
                        "note_id": note.get("id", ""),
                        "title": note.get("body", "").split("\n")[0].lstrip("# ").strip()[:80],
                        "type": "deadline",
                        "urgency": urgency,
                        "priority": priority,
                        "file_path": note.get("file_path", ""),
                    })

            # Check for reminders
            if note.get("type") == "reminder":
                reminders.append({
                    "note_id": note.get("id", ""),
                    "title": note.get("body", "").split("\n")[0].lstrip("# ").strip()[:80],
                    "type": "reminder",
                    "file_path": note.get("file_path", ""),
                })

        # Sort by priority/urgency
        urgency_order = {"immediate": 0, "soon": 1, "later": 2, "someday": 3}
        reminders.sort(key=lambda r: (
            urgency_order.get(r.get("urgency", "later"), 2),
            -r.get("priority", 0.5),
        ))

        return reminders

    def daily_digest(self) -> Dict[str, Any]:
        """Generate daily digest of tasks and reminders."""
        all_notes = self.reader.read_all()
        now = datetime.now(timezone.utc)

        tasks_todo = []
        tasks_in_progress = []
        reminders = []
        high_priority = []

        for note in all_notes:
            ntype = note.get("type", "")
            status = note.get("status", "todo")
            priority = self._parse_priority(note.get("priority", "medium"))
            title = note.get("body", "").split("\n")[0].lstrip("# ").strip()[:80]

            if ntype == "task":
                if status == "todo":
                    tasks_todo.append({"title": title, "priority": priority})
                elif status == "in_progress":
                    tasks_in_progress.append({"title": title, "priority": priority})

                if priority >= 0.8:
                    high_priority.append({"title": title, "priority": priority})

            if ntype == "reminder":
                reminders.append({"title": title})

        return {
            "date": now.strftime("%Y-%m-%d"),
            "total_notes": len(all_notes),
            "tasks_todo": len(tasks_todo),
            "tasks_in_progress": len(tasks_in_progress),
            "high_priority": high_priority[:5],
            "reminders": reminders[:5],
            "top_tasks_todo": sorted(tasks_todo, key=lambda t: -t["priority"])[:5],
        }

    def send_macos_notification(self, title: str, message: str) -> bool:
        """Send macOS notification via osascript."""
        if sys.platform != "darwin":
            return False
        try:
            script = f'display notification "{message}" with title "{title}"'
            subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def send_webhook(self, url: str, payload: Dict[str, Any]) -> bool:
        """Send webhook POST request."""
        try:
            import urllib.request
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_priority(self, val: Any) -> float:
        """Parse priority to float."""
        if isinstance(val, (int, float)):
            return float(val)
        mapping = {"critical": 0.95, "high": 0.8, "medium": 0.5, "low": 0.3, "someday": 0.1}
        return mapping.get(str(val).lower(), 0.5)

    def _rules_path(self) -> str:
        return os.path.join(self.vault_root, "_notifications_rules.json")

    def _load_rules(self) -> None:
        path = self._rules_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self.rules = json.load(fh)
            except (json.JSONDecodeError, OSError):
                self.rules = []

    def _save_rules(self) -> None:
        path = self._rules_path()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.rules, fh, indent=2)
