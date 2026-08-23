"""
Vault Query Parser
==================

Parse natural language queries into structured search plans using regex
and keyword matching — no external NLP dependencies.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pattern dictionaries
# ---------------------------------------------------------------------------

INTENT_PATTERNS: Dict[str, str] = {
    "list": r"\b(what|show|list|find|get|which|give me)\b",
    "search": r"\b(when|where|who|tell me about)\b",
    "explain": r"\b(why|explain|how does|how do)\b",
    "summarize": r"\b(summarize|summary|how many|count|total|overview)\b",
}

ENTITY_TYPE_PATTERNS: Dict[str, str] = {
    "task": r"\b(tasks?|to-?do|action items?|errands?)\b",
    "person": r"\b(people|person|who|contacts?|friend)\b",
    "project": r"\b(projects?|initiatives?|repos?)\b",
    "meeting": r"\b(meetings?|calls?|discussions?|standups?)\b",
    "idea": r"\b(ideas?|thoughts?|brainstorms?|notions?)\b",
    "learning": r"\b(learnings?|insights?|discoveries?|lessons?)\b",
    "decision": r"\b(decisions?|choices?|trade-?offs?)\b",
    "reminder": r"\b(reminders?|alerts?|alarms?)\b",
    "journal": r"\b(journal|diary|entries?|log)\b",
}

PRIORITY_PATTERNS: Dict[str, str] = {
    "critical": r"\b(critical|urgent|emergency|asap|fire)\b",
    "high": r"\b(high|important|priority|p0|p1)\b",
    "medium": r"\b(medium|normal|standard)\b",
    "low": r"\b(low|minor|backlog|someday|nice-?to-?have)\b",
}

STATUS_PATTERNS: Dict[str, str] = {
    "todo": r"\b(todo|pending|open|new)\b",
    "in_progress": r"\b(in progress|working on|started|active)\b",
    "done": r"\b(done|completed|finished|closed|shipped)\b",
    "blocked": r"\b(blocked|stuck|waiting|held up)\b",
    "cancelled": r"\b(cancelled|abandoned|dropped)\b",
}

TEMPORAL_PATTERNS: Dict[str, str] = {
    "today": r"\b(today|tonight|this evening)\b",
    "yesterday": r"\b(yesterday)\b",
    "this_week": r"\b(this week|current week|this sprint)\b",
    "last_week": r"\b(last week|previous week)\b",
    "this_month": r"\b(this month|current month)\b",
    "last_month": r"\b(last month|previous month)\b",
    "this_year": r"\b(this year|current year)\b",
}


class QueryParser:
    """Parse natural language queries into structured plans."""

    def parse(self, query: str) -> Dict[str, Any]:
        """Parse a query into a QueryPlan dict."""
        intent = self._detect_intent(query)
        entity_type = self._extract_entity_type(query)
        filters = self._extract_filters(query)
        temporal = self._extract_temporal(query)
        expand = self._is_vague(query)

        # Merge entity_type into filters if found
        if entity_type:
            filters["type"] = entity_type

        return {
            "intent": intent,
            "entity_type": entity_type,
            "filters": filters,
            "temporal": temporal,
            "text": query,
            "expand": expand,
        }

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    def _detect_intent(self, query: str) -> str:
        """Classify query intent."""
        lower = query.lower()

        scores: Dict[str, int] = {}
        for intent, pattern in INTENT_PATTERNS.items():
            matches = re.findall(pattern, lower)
            scores[intent] = len(matches)

        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        if scores[best] == 0:
            return "search"  # default
        return best

    # ------------------------------------------------------------------
    # Entity type extraction
    # ------------------------------------------------------------------

    def _extract_entity_type(self, query: str) -> Optional[str]:
        """Extract target entity type from query nouns."""
        lower = query.lower()
        for etype, pattern in ENTITY_TYPE_PATTERNS.items():
            if re.search(pattern, lower):
                return etype
        return None

    # ------------------------------------------------------------------
    # Filter extraction
    # ------------------------------------------------------------------

    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filter conditions."""
        lower = query.lower()
        filters: Dict[str, Any] = {}

        # Priority
        for priority, pattern in PRIORITY_PATTERNS.items():
            if re.search(pattern, lower):
                filters["priority"] = priority
                break

        # Status
        for status, pattern in STATUS_PATTERNS.items():
            if re.search(pattern, lower):
                filters["status"] = status
                break

        # Tags / topics (look for quoted words or after "about" / "involving")
        tag_match = re.search(r'"([^"]+)"', query)
        if tag_match:
            filters["tags"] = [tag_match.group(1).lower()]
        else:
            about_match = re.search(r"(?:about|involving|tagged|related to)\s+(\w+)", lower)
            if about_match:
                filters["tags"] = [about_match.group(1)]

        # People (after "assigned to" / "by" / "with" / "for")
        people_match = re.search(r"(?:assigned to|by|with|for|about)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", query)
        if people_match:
            filters["people"] = [people_match.group(1)]

        return filters

    # ------------------------------------------------------------------
    # Temporal extraction
    # ------------------------------------------------------------------

    def _extract_temporal(self, query: str) -> Dict[str, Any]:
        """Extract time references."""
        lower = query.lower()

        for rel, pattern in TEMPORAL_PATTERNS.items():
            if re.search(pattern, lower):
                return {"relative": rel, "absolute": None}

        # Absolute date (YYYY-MM-DD)
        date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", query)
        if date_match:
            return {"relative": None, "absolute": date_match.group(1)}

        return {"relative": None, "absolute": None}

    # ------------------------------------------------------------------
    # Vagueness detection
    # ------------------------------------------------------------------

    def _is_vague(self, query: str) -> bool:
        """True if the query is too vague for precise search."""
        words = query.split()
        if len(words) <= 2:
            return True
        lower = query.lower()
        vague_markers = r"\b(things?|stuff|everything|anything|something|all)\b"
        return bool(re.search(vague_markers, lower))
