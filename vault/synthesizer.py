"""
Vault Synthesizer
=================

Generate summaries, insights, daily digests, and weekly reports
from vault content.  Turns raw notes into actionable intelligence.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from vault.config import VAULT_ROOT
from vault.reader import VaultReader


class KnowledgeSynthesizer:
    """Generate summaries and insights from vault content."""

    def __init__(self, vault_root: str = VAULT_ROOT, reader: Any = None, graph: Any = None):
        self.vault_root = vault_root
        self.reader = reader or VaultReader(vault_root)
        self.graph = graph

    def daily_digest(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Generate daily digest."""
        if date is None:
            date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        all_notes = self.reader.read_all()

        by_type: Dict[str, int] = Counter()
        highlights: List[str] = []
        deadlines: List[Dict] = []

        for note in all_notes:
            ntype = note.get("type", "observation")
            by_type[ntype] += 1

            # Collect highlights from high-priority notes
            priority = self._parse_priority(note.get("priority", "medium"))
            if priority >= 0.8:
                title = note.get("body", "").split("\n")[0].lstrip("# ").strip()[:80]
                highlights.append(title)

            # Collect upcoming deadlines
            if ntype in ("task", "reminder"):
                urgency = note.get("urgency", "later")
                if urgency in ("immediate", "soon"):
                    deadlines.append({
                        "title": note.get("body", "").split("\n")[0].lstrip("# ").strip()[:60],
                        "urgency": urgency,
                    })

        return {
            "date": date,
            "notes_created": len(all_notes),
            "by_type": dict(by_type),
            "highlights": highlights[:5],
            "upcoming_deadlines": deadlines[:5],
        }

    def weekly_report(self, week_start: Optional[str] = None) -> Dict[str, Any]:
        """Generate weekly report."""
        all_notes = self.reader.read_all()

        completed = 0
        new_ideas = 0
        decisions = 0
        topics: Counter = Counter()

        for note in all_notes:
            ntype = note.get("type", "")
            status = note.get("status", "todo")

            if ntype == "task" and status == "done":
                completed += 1
            elif ntype == "idea":
                new_ideas += 1
            elif ntype == "decision":
                decisions += 1

            for tag in note.get("tags", []):
                topics[tag] += 1

        total = len(all_notes)
        task_count = sum(1 for n in all_notes if n.get("type") == "task")
        productivity = (completed / task_count * 100) if task_count > 0 else 0

        now = datetime.now(timezone.utc)
        week_num = now.isocalendar()[1]

        return {
            "week": f"{now.year}-W{week_num:02d}",
            "total_notes": total,
            "completed_tasks": completed,
            "new_ideas": new_ideas,
            "decisions_made": decisions,
            "top_topics": [t for t, _ in topics.most_common(5)],
            "productivity_score": round(productivity, 1),
        }

    def topic_summary(self, topic: str) -> Dict[str, Any]:
        """Generate summary of all notes about a topic."""
        topic_lower = topic.lower()
        matching: List[Dict] = []
        entities: Counter = Counter()

        for note in self.reader.read_all():
            tags = [t.lower() for t in note.get("tags", [])]
            body = note.get("body", "").lower()
            if topic_lower in tags or topic_lower in body:
                matching.append(note)
                for e in note.get("entities", []):
                    if isinstance(e, dict):
                        entities[e.get("name", "")] += 1

        recent = sorted(matching, key=lambda n: n.get("created", ""), reverse=True)[:5]

        return {
            "topic": topic,
            "note_count": len(matching),
            "recent_notes": [
                {"id": n.get("id", ""), "type": n.get("type", ""), "preview": n.get("body", "")[:80]}
                for n in recent
            ],
            "key_entities": [e for e, _ in entities.most_common(5)],
        }

    def insight_report(self) -> List[Dict[str, Any]]:
        """Generate cross-cutting insights."""
        all_notes = self.reader.read_all()
        insights: List[Dict[str, Any]] = []

        # Insight 1: Most active topics
        topics: Counter = Counter()
        for note in all_notes:
            for tag in note.get("tags", []):
                topics[tag] += 1
        if topics:
            top_topic, top_count = topics.most_common(1)[0]
            insights.append({
                "type": "topic_activity",
                "message": f"Most active topic: '{top_topic}' ({top_count} notes)",
                "confidence": 0.9,
            })

        # Insight 2: Task completion rate
        tasks = [n for n in all_notes if n.get("type") == "task"]
        done = [n for n in tasks if n.get("status") == "done"]
        if tasks:
            rate = len(done) / len(tasks) * 100
            insights.append({
                "type": "productivity",
                "message": f"Task completion rate: {rate:.0f}% ({len(done)}/{len(tasks)})",
                "confidence": 0.95,
            })

        # Insight 3: Orphan notes (no connections)
        if self.graph:
            orphans = self.graph.orphans()
            if orphans:
                insights.append({
                    "type": "orphan_notes",
                    "message": f"{len(orphans)} notes have no connections — consider linking them",
                    "confidence": 0.8,
                    "related_notes": orphans[:5],
                })

        return insights

    def _parse_priority(self, val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        mapping = {"critical": 0.95, "high": 0.8, "medium": 0.5, "low": 0.3, "someday": 0.1}
        return mapping.get(str(val).lower(), 0.5)
