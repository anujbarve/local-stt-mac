"""
brain.insights — Insight generation from note patterns.

Detects usage patterns, cross-references, stale projects,
and generates actionable insights.

Usage:
    from brain.insights import InsightGenerator

    gen = InsightGenerator()
    insights = gen.daily_insights(notes)
    report = gen.weekly_report(notes)
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class Insight:
    """A generated insight."""
    type: str       # "pattern", "alert", "suggestion", "trend"
    message: str
    severity: str   # "info", "warning", "action"
    related_notes: List[str]


class InsightGenerator:
    """Generate insights from note patterns and history."""

    def __init__(self):
        pass

    def daily_insights(self, notes: List[Dict]) -> List[Insight]:
        """Generate daily insights from recent notes.

        Returns list of Insight objects.
        """
        insights = []
        now = datetime.now(timezone.utc)
        today = now.date()

        # Notes from today
        today_notes = [n for n in notes if self._is_same_day(n, today)]
        if not today_notes:
            insights.append(Insight(
                type="alert",
                message="No notes captured today. Consider a quick journal entry.",
                severity="info",
                related_notes=[],
            ))
            return insights

        # Type breakdown
        type_counts = Counter(n.get("type", "unknown") for n in today_notes)
        dominant = type_counts.most_common(1)[0]
        if dominant[1] >= 3:
            insights.append(Insight(
                type="pattern",
                message=f"You've captured {dominant[1]} {dominant[0]} notes today — focused session!",
                severity="info",
                related_notes=[str(n.get("id", "")) for n in today_notes if n.get("type") == dominant[0]],
            ))

        # High-priority items
        high_priority = [n for n in today_notes if n.get("priority", 0) > 0.8]
        if high_priority:
            insights.append(Insight(
                type="alert",
                message=f"{len(high_priority)} high-priority items captured today.",
                severity="warning",
                related_notes=[str(n.get("id", "")) for n in high_priority],
            ))

        # Overdue tasks
        overdue = self._find_overdue(notes, now)
        if overdue:
            insights.append(Insight(
                type="alert",
                message=f"{len(overdue)} tasks are overdue!",
                severity="action",
                related_notes=[str(n.get("id", "")) for n in overdue],
            ))

        return insights

    def weekly_report(self, notes: List[Dict]) -> Dict[str, Any]:
        """Generate weekly summary.

        Returns comprehensive weekly stats.
        """
        now = datetime.now(timezone.utc)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        week_notes = [
            n for n in notes
            if self._is_after(n, week_start)
        ]

        # Type breakdown
        by_type: Counter = Counter(n.get("type", "unknown") for n in week_notes)

        # Topic frequency
        topic_counter: Counter = Counter()
        for n in week_notes:
            for t in n.get("topics", []):
                topic_counter[t] += 1

        # Entity frequency
        entity_counter: Counter = Counter()
        for n in week_notes:
            for e in n.get("entities", []):
                name = e.get("name", "") if isinstance(e, dict) else getattr(e, "name", "")
                if name:
                    entity_counter[name] += 1

        # Priority stats
        priorities = [n.get("priority", 0.5) for n in week_notes]
        avg_priority = sum(priorities) / len(priorities) if priorities else 0

        # Productivity score (rough heuristic)
        task_count = by_type.get("task", 0)
        completed = sum(1 for n in week_notes if n.get("type") == "task" and n.get("completed"))
        productivity = completed / task_count if task_count > 0 else 0.5

        return {
            "week": week_start.strftime("%Y-W%W"),
            "total_notes": len(week_notes),
            "by_type": dict(by_type),
            "completed_tasks": completed,
            "new_ideas": by_type.get("idea", 0),
            "decisions_made": by_type.get("decision", 0),
            "meetings": by_type.get("meeting", 0),
            "top_topics": [t for t, _ in topic_counter.most_common(5)],
            "top_entities": [e for e, _ in entity_counter.most_common(5)],
            "avg_priority": round(avg_priority, 2),
            "productivity_score": round(productivity, 2),
        }

    def detect_patterns(self, notes: List[Dict]) -> List[Dict]:
        """Detect recurring patterns in notes.

        Returns list of patterns with examples.
        """
        patterns = []

        # Time-of-day patterns
        hour_counts: Counter = Counter()
        for n in notes:
            ts = n.get("timestamp", "")
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                hour_counts[dt.hour] += 1
            except (ValueError, TypeError):
                continue

        if hour_counts:
            peak_hour = hour_counts.most_common(1)[0]
            patterns.append({
                "pattern": f"Most notes captured around {peak_hour[0]}:00",
                "frequency": f"{peak_hour[1]} notes",
                "examples": [],
                "suggestion": f"Consider using this time for focused capture.",
            })

        # Topic repetition
        topic_notes: Dict[str, List[str]] = defaultdict(list)
        for n in notes:
            for t in n.get("topics", []):
                topic_notes[t].append(n.get("text", "")[:60])

        for topic, texts in topic_notes.items():
            if len(texts) >= 3:
                patterns.append({
                    "pattern": f"Frequent topic: {topic}",
                    "frequency": f"{len(texts)} notes",
                    "examples": texts[:3],
                    "suggestion": f"Consider creating a MOC for '{topic}'.",
                })

        # Entity co-occurrence
        entity_pairs: Counter = Counter()
        for n in notes:
            entities = [e.get("name", "") if isinstance(e, dict) else getattr(e, "name", "") for e in n.get("entities", [])]
            entities = [e for e in entities if e]
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    pair = tuple(sorted([entities[i], entities[j]]))
                    entity_pairs[pair] += 1

        for (e1, e2), count in entity_pairs.most_common(3):
            if count >= 2:
                patterns.append({
                    "pattern": f"Frequent co-occurrence: {e1} + {e2}",
                    "frequency": f"{count} times",
                    "examples": [],
                    "suggestion": f"These entities are closely related in your notes.",
                })

        return patterns

    def cross_references(self, note_id: str, notes: List[Dict]) -> List[Dict]:
        """Find related notes and cross-references.

        Args:
            note_id: ID of the note to find references for.
            notes: All notes.

        Returns:
            List of related notes with relevance scores.
        """
        # Find the target note
        target = None
        for n in notes:
            if str(n.get("id", n.get("note_id", ""))) == str(note_id):
                target = n
                break

        if target is None:
            return []

        target_topics = set(target.get("topics", []))
        target_entities = set()
        for e in target.get("entities", []):
            name = e.get("name", "") if isinstance(e, dict) else getattr(e, "name", "")
            if name:
                target_entities.add(name.lower())

        target_type = target.get("type", "")

        related = []
        for n in notes:
            if str(n.get("id", n.get("note_id", ""))) == str(note_id):
                continue

            score = 0.0
            connection_types = []

            # Same topics
            n_topics = set(n.get("topics", []))
            common_topics = target_topics & n_topics
            if common_topics:
                score += 0.3 * len(common_topics)
                connection_types.append("same_topic")

            # Same entities
            n_entities = set()
            for e in n.get("entities", []):
                name = e.get("name", "") if isinstance(e, dict) else getattr(e, "name", "")
                if name:
                    n_entities.add(name.lower())
            common_entities = target_entities & n_entities
            if common_entities:
                score += 0.4 * len(common_entities)
                connection_types.append("same_entity")

            # Same type
            if n.get("type") == target_type:
                score += 0.1
                connection_types.append("same_type")

            if score > 0:
                related.append({
                    "note_id": str(n.get("id", n.get("note_id", ""))),
                    "text": n.get("text", "")[:100],
                    "relevance": round(min(score, 1.0), 2),
                    "connection_type": connection_types[0] if connection_types else "unknown",
                })

        related.sort(key=lambda x: x["relevance"], reverse=True)
        return related[:10]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_same_day(self, note: Dict, date) -> bool:
        """Check if a note was created on a given date."""
        ts = note.get("timestamp", note.get("created", ""))
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return dt.date() == date
        except (ValueError, TypeError):
            return False

    def _is_after(self, note: Dict, dt: datetime) -> bool:
        """Check if a note was created after a datetime."""
        ts = note.get("timestamp", note.get("created", ""))
        try:
            note_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return note_dt >= dt
        except (ValueError, TypeError):
            return False

    def _find_overdue(self, notes: List[Dict], now: datetime) -> List[Dict]:
        """Find notes with deadlines in the past."""
        overdue = []
        for n in notes:
            deadline = n.get("deadline")
            if not deadline:
                continue
            try:
                if isinstance(deadline, str):
                    dl = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
                else:
                    dl = deadline
                if dl < now and n.get("type") == "task":
                    overdue.append(n)
            except (ValueError, TypeError):
                continue
        return overdue
