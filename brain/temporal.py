"""
brain.temporal — Temporal reasoning engine.

Parses natural language time references ("next Friday", "in 2 hours"),
extracts deadlines, detects recurring patterns, and suggests scheduling.

Usage:
    from brain.temporal import TemporalReasoner

    reasoner = TemporalReasoner()
    result = reasoner.parse("I need to fix this by Friday")
    print(result.deadline)  # datetime of next Friday
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


@dataclass
class TemporalResult:
    """Result of parsing temporal information from text."""
    has_time: bool = False
    deadline: Optional[datetime] = None
    relative: str = "unknown"  # "past", "today", "future", "unknown"
    recurring: Optional[str] = None  # "daily", "weekly", "monthly"
    confidence: float = 0.0
    raw_expression: Optional[str] = None
    duration_hours: Optional[float] = None


@dataclass
class ScheduledItem:
    """A scheduled task with timing information."""
    note_id: str
    title: str
    deadline: datetime
    priority: float
    urgency: str
    suggested_time: str
    duration_minutes: int = 60


# ---------------------------------------------------------------------------
# Day/time name mappings
# ---------------------------------------------------------------------------

DAY_NAMES = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

RELATIVE_DAY = {
    "today": 0,
    "tomorrow": 1,
    "day after tomorrow": 2,
    "yesterday": -1,
}

TIME_OF_DAY = {
    "morning": 9,
    "afternoon": 14,
    "evening": 18,
    "night": 20,
    "end of day": 17,
    "eod": 17,
    "start of day": 9,
}

MONTH_NAMES = {
    "january": 1, "jan": 1, "february": 2, "feb": 2,
    "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6,
    "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

RECURRING_PATTERNS = {
    "daily": "daily",
    "every day": "daily",
    "each day": "daily",
    "weekly": "weekly",
    "every week": "weekly",
    "each week": "weekly",
    "every monday": "weekly",
    "every tuesday": "weekly",
    "every wednesday": "weekly",
    "every thursday": "weekly",
    "every friday": "weekly",
    "monthly": "monthly",
    "every month": "monthly",
    "each month": "monthly",
}


class TemporalReasoner:
    """Parse and reason about time expressions in text."""

    def __init__(self, now: Optional[datetime] = None):
        self.now = now or datetime.now(timezone.utc)

    def parse(self, text: str) -> TemporalResult:
        """Extract temporal information from text.

        Args:
            text: Input text to parse.

        Returns:
            TemporalResult with deadline, relative timing, recurrence, etc.
        """
        text_lower = text.lower()

        # Check for recurring patterns first
        for pattern, freq in RECURRING_PATTERNS.items():
            if pattern in text_lower:
                return TemporalResult(
                    has_time=True,
                    recurring=freq,
                    confidence=0.9,
                    raw_expression=pattern,
                )

        # Try to extract a deadline
        deadline = None
        raw = None
        confidence = 0.0

        # "by <day>"
        m = re.search(r'\bby\s+(next\s+)?(\w+day)\b', text_lower)
        if m:
            day_name = m.group(2)
            next_prefix = m.group(1) is not None
            deadline = self._next_weekday(day_name, next_prefix)
            raw = m.group(0)
            confidence = 0.85

        # "before <day>"
        if not deadline:
            m = re.search(r'\bbefore\s+(next\s+)?(\w+day)\b', text_lower)
            if m:
                day_name = m.group(2)
                next_prefix = m.group(1) is not None
                deadline = self._next_weekday(day_name, next_prefix)
                raw = m.group(0)
                confidence = 0.85

        # "tomorrow" / "today" / "day after tomorrow"
        if not deadline:
            for expr, offset in RELATIVE_DAY.items():
                if re.search(r'\b' + expr + r'\b', text_lower):
                    deadline = self.now + timedelta(days=offset)
                    raw = expr
                    confidence = 0.8
                    break

        # "in N hours/minutes/days/weeks"
        if not deadline:
            m = re.search(r'\bin\s+(\d+)\s+(hour|hr|minute|min|day|week)s?\b', text_lower)
            if m:
                num = int(m.group(1))
                unit = m.group(2)
                if unit in ("hour", "hr"):
                    deadline = self.now + timedelta(hours=num)
                elif unit in ("minute", "min"):
                    deadline = self.now + timedelta(minutes=num)
                elif unit == "day":
                    deadline = self.now + timedelta(days=num)
                elif unit == "week":
                    deadline = self.now + timedelta(weeks=num)
                raw = m.group(0)
                confidence = 0.9

        # "end of week" / "end of month"
        if not deadline:
            if "end of week" in text_lower or "eow" in text_lower:
                days_until_friday = (4 - self.now.weekday()) % 7
                if days_until_friday == 0:
                    days_until_friday = 7
                deadline = self.now + timedelta(days=days_until_friday)
                deadline = deadline.replace(hour=17, minute=0, second=0)
                raw = "end of week"
                confidence = 0.75
            elif "end of month" in text_lower or "eom" in text_lower:
                if self.now.month == 12:
                    next_month = self.now.replace(year=self.now.year + 1, month=1, day=1)
                else:
                    next_month = self.now.replace(month=self.now.month + 1, day=1)
                deadline = next_month - timedelta(days=1)
                deadline = deadline.replace(hour=17, minute=0, second=0)
                raw = "end of month"
                confidence = 0.75

        # "<month> <day>" e.g. "march 15th"
        if not deadline:
            m = re.search(
                r'\b(' + '|'.join(MONTH_NAMES.keys()) + r')\s+(\d{1,2})(st|nd|rd|th)?\b',
                text_lower,
            )
            if m:
                month_name = m.group(1)
                day_num = int(m.group(2))
                month = MONTH_NAMES[month_name]
                year = self.now.year
                try:
                    deadline = datetime(year, month, day_num, tzinfo=timezone.utc)
                    if deadline < self.now:
                        deadline = datetime(year + 1, month, day_num, tzinfo=timezone.utc)
                    raw = m.group(0)
                    confidence = 0.8
                except ValueError:
                    pass

        # "ASAP" / "urgently"
        if not deadline:
            if re.search(r'\b(asap|as soon as possible|urgently)\b', text_lower):
                deadline = self.now + timedelta(hours=2)
                raw = "ASAP"
                confidence = 0.6

        # Determine relative direction
        relative = "unknown"
        if deadline:
            if deadline.date() == self.now.date():
                relative = "today"
            elif deadline > self.now:
                relative = "future"
            elif deadline < self.now:
                relative = "past"

        # Extract time of day hints
        if deadline:
            for tod_name, hour in TIME_OF_DAY.items():
                if tod_name in text_lower:
                    deadline = deadline.replace(hour=hour, minute=0, second=0)
                    break

        return TemporalResult(
            has_time=deadline is not None,
            deadline=deadline,
            relative=relative,
            recurring=None,
            confidence=confidence,
            raw_expression=raw,
        )

    def suggest_schedule(
        self,
        notes: List[Dict],
        available_hours: int = 8,
    ) -> List[ScheduledItem]:
        """Suggest a daily schedule based on deadlines and priorities.

        Args:
            notes: List of note dicts with id, text, priority, urgency, deadline.
            available_hours: Working hours per day.

        Returns:
            Sorted list of ScheduledItem.
        """
        items = []
        now = self.now

        for note in notes:
            deadline_str = note.get("deadline")
            if not deadline_str:
                continue

            try:
                if isinstance(deadline_str, str):
                    deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
                else:
                    deadline = deadline_str
            except (ValueError, TypeError):
                continue

            # Only schedule items due today or soon
            if deadline.date() > (now + timedelta(days=7)).date():
                continue

            priority = note.get("priority", 0.5)
            urgency = note.get("urgency", "later")

            # Estimate duration based on complexity
            duration = 60  # default 1 hour
            if priority > 0.8:
                duration = 90
            elif priority < 0.3:
                duration = 30

            items.append(ScheduledItem(
                note_id=note.get("id", note.get("note_id", "")),
                title=note.get("text", note.get("title", ""))[:80],
                deadline=deadline,
                priority=priority,
                urgency=urgency,
                suggested_time="",
                duration_minutes=duration,
            ))

        # Sort by priority * urgency weight
        urgency_weight = {"immediate": 3, "soon": 2, "later": 1, "someday": 0.5}
        items.sort(
            key=lambda x: x.priority * urgency_weight.get(x.urgency, 1),
            reverse=True,
        )

        # Assign time slots
        hour = 9  # start at 9 AM
        for item in items:
            slot_hour = min(hour, 9 + available_hours - 1)
            item.suggested_time = f"{slot_hour}:00"
            hour += max(1, item.duration_minutes // 60)

        return items

    def detect_recurring(self, notes: List[Dict]) -> List[Dict]:
        """Detect recurring patterns in notes.

        Returns list of recurring patterns:
            - pattern: str
            - examples: List[str]
            - frequency: str
        """
        patterns: Dict[str, List[str]] = {}

        for note in notes:
            text = note.get("text", "").lower()
            for pattern, freq in RECURRING_PATTERNS.items():
                if pattern in text:
                    if freq not in patterns:
                        patterns[freq] = []
                    patterns[freq].append(note.get("text", "")[:60])

        return [
            {"pattern": freq, "examples": examples[:3], "frequency": freq}
            for freq, examples in patterns.items()
        ]

    def time_until(self, deadline: datetime) -> str:
        """Human-readable time until deadline.

        Returns: "in 2 hours", "tomorrow", "next Friday", etc.
        """
        now = self.now
        diff = deadline - now

        if diff.total_seconds() < 0:
            return "overdue"

        days = diff.days
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60

        if days > 7:
            return f"in {days // 7} week{'s' if days // 7 > 1 else ''}"
        elif days > 1:
            return f"in {days} days"
        elif days == 1:
            if hours > 0:
                return f"tomorrow in {hours}h"
            return "tomorrow"
        elif hours > 0:
            return f"in {hours}h {minutes}m" if minutes > 0 else f"in {hours} hours"
        else:
            return f"in {minutes} minutes"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _next_weekday(self, day_name: str, next_week: bool = False) -> datetime:
        """Get the next occurrence of a weekday."""
        target = DAY_NAMES.get(day_name.lower())
        if target is None:
            return self.now + timedelta(days=1)

        current = self.now.weekday()
        days_ahead = (target - current) % 7
        if days_ahead == 0:
            days_ahead = 7
        if next_week:
            days_ahead += 7

        return (self.now + timedelta(days=days_ahead)).replace(
            hour=17, minute=0, second=0, microsecond=0,
        )
