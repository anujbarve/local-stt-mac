"""
Vault Analytics
===============

Track and report usage patterns: note creation frequency,
search query patterns, model accuracy trends, and more.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from vault.config import VAULT_ROOT


class UsageAnalytics:
    """Track and report usage patterns."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.analytics_dir = os.path.join(vault_root, "_analytics")
        self.log_path = os.path.join(self.analytics_dir, "usage.jsonl")
        self._ensure_dir()

    def track(self, event: str, **kwargs: Any) -> None:
        """Log a usage event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        entry.update(kwargs)

        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def note_creation_stats(self, days: int = 30) -> Dict[str, Any]:
        """Note creation frequency by type and date."""
        events = self._read_events()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        by_type: Counter = Counter()
        by_date: Counter = Counter()

        for e in events:
            if e.get("event") != "note_created":
                continue
            ts = e.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt < cutoff:
                    continue
            except (ValueError, AttributeError):
                continue

            by_type[e.get("type", "unknown")] += 1
            by_date[ts[:10]] += 1

        return {
            "period_days": days,
            "total_created": sum(by_type.values()),
            "by_type": dict(by_type),
            "by_date": dict(sorted(by_date.items())),
        }

    def search_stats(self, days: int = 30) -> Dict[str, Any]:
        """Search query patterns and success rates."""
        events = self._read_events()
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        searches = []
        for e in events:
            if e.get("event") != "search":
                continue
            ts = e.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt < cutoff:
                    continue
            except (ValueError, AttributeError):
                continue
            searches.append(e)

        total = len(searches)
        successful = sum(1 for s in searches if s.get("result_count", 0) > 0)
        queries = [s.get("query", "") for s in searches]

        # Top queries
        query_counts: Counter = Counter(queries)
        top_queries = query_counts.most_common(10)

        return {
            "period_days": days,
            "total_searches": total,
            "successful": successful,
            "success_rate": round(successful / total * 100, 1) if total > 0 else 0,
            "top_queries": [{"query": q, "count": c} for q, c in top_queries],
        }

    def model_accuracy_trend(self) -> List[Dict[str, Any]]:
        """Track model accuracy over time from corrections."""
        # Read from corrections accuracy log
        accuracy_path = os.path.join(self.vault_root, "_corrections", "accuracy_log.json")
        if not os.path.exists(accuracy_path):
            return []

        try:
            with open(accuracy_path, "r", encoding="utf-8") as fh:
                values = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return []

        return [
            {"index": i, "accuracy": v}
            for i, v in enumerate(values)
        ]

    def generate_report(self, period: str = "weekly") -> Dict[str, Any]:
        """Generate usage report."""
        days = 7 if period == "weekly" else 30
        note_stats = self.note_creation_stats(days)
        search = self.search_stats(days)
        accuracy = self.model_accuracy_trend()

        return {
            "period": period,
            "notes": note_stats,
            "search": search,
            "accuracy_trend": accuracy[-10:] if accuracy else [],
        }

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get the most recent events."""
        events = self._read_events()
        return events[-limit:]

    def clear(self) -> int:
        """Clear all analytics data. Returns count removed."""
        events = self._read_events()
        count = len(events)
        if os.path.exists(self.log_path):
            os.unlink(self.log_path)
        return count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        os.makedirs(self.analytics_dir, exist_ok=True)

    def _read_events(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        events: List[Dict[str, Any]] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events
