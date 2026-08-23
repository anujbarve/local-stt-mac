"""
Vault Note Templates
====================

Render markdown notes (YAML frontmatter + body) from an ``Understanding``
dict.  Each note type has a dedicated sub-class that formats the body
differently while sharing the same frontmatter structure.
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from typing import Any, Dict, List, Optional

from brain_contract import (
    NoteType,
    EntityType,
    Urgency,
    NOTE_TYPE_DIRS,
    ENTITY_PAGE_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _priority_label(priority: float) -> str:
    """Map 0-1 float to a human label."""
    if priority >= 0.9:
        return "critical"
    if priority >= 0.7:
        return "high"
    if priority >= 0.4:
        return "medium"
    if priority >= 0.2:
        return "low"
    return "someday"


def _entity_list(entities: List[Dict], etype: str) -> List[str]:
    """Return names of entities of *etype*."""
    return [e["name"] for e in entities if e.get("entity_type") == etype]


def _format_entity_attributes(entities: List[Dict]) -> str:
    """Render a markdown list of entity attributes."""
    lines: List[str] = []
    for e in entities:
        attrs = e.get("attributes", {})
        if attrs:
            for k, v in attrs.items():
                lines.append(f"- **{k}:** {v}")
        else:
            lines.append(f"- {e['name']} ({e.get('entity_type', 'unknown')})")
    return "\n".join(lines) if lines else "_No details yet._"


def _tag_list(tags: List[str], topics: List[str]) -> str:
    """Format tags/topics as YAML list items."""
    items = [f"  - {t}" for t in (tags or [])]
    for t in (topics or []):
        if t not in (tags or []):
            items.append(f"  - {t}")
    return "\n".join(items) if items else "  []"


# ---------------------------------------------------------------------------
# Base template
# ---------------------------------------------------------------------------

class NoteTemplate:
    """Generate note content from an Understanding dict."""

    @staticmethod
    def render(understanding: Dict[str, Any]) -> str:
        """Render a complete .md file with frontmatter and body."""
        fm = NoteTemplate.render_frontmatter(understanding)
        body = NoteTemplate.render_body(understanding)
        return f"{fm}\n\n{body}\n"

    @staticmethod
    def render_frontmatter(u: Dict[str, Any]) -> str:
        """Render the YAML frontmatter block."""
        entities = u.get("entities", [])
        people = _entity_list(entities, "person")
        projects = _entity_list(entities, "project")
        tasks = _entity_list(entities, "task")
        tools = _entity_list(entities, "tool")
        dates = _entity_list(entities, "date")

        created = u.get("timestamp", datetime.utcnow().isoformat() + "Z")
        updated = created

        fm_lines = [
            "---",
            f"id: {u.get('note_id', '')}",
            f"type: {u.get('type', 'observation')}",
            f"created: {created}",
            f"updated: {updated}",
            f"source: audio",
            "",
            "# Classification",
            f"priority: {_priority_label(u.get('priority', 0.5))}",
            f"urgency: {u.get('urgency', 'later')}",
            f"sentiment: {u.get('sentiment', 'neutral')}",
            f"confidence: {u.get('confidence', 0.5)}",
            "",
            "# Entities",
            f"people: {people or '[]'}",
            f"projects: {projects or '[]'}",
            f"tasks: {tasks or '[]'}",
            f"tools: {tools or '[]'}",
            f"dates: {dates or '[]'}",
            "",
            "# Organization",
        ]

        # tags/topics
        all_tags = list(dict.fromkeys(u.get("tags", []) + u.get("topics", [])))
        fm_lines.append("tags:")
        for t in all_tags:
            fm_lines.append(f"  - {t}")
        if not all_tags:
            fm_lines.append("  []")

        fm_lines.append(f"category: {u.get('type', 'observation')}")
        fm_lines.append("")
        fm_lines.append("# Links")
        fm_lines.append(f"related_notes: {u.get('related_notes', []) or '[]'}")
        fm_lines.append("backlinks: []")

        # embedding id (placeholder – writer fills real one)
        fm_lines.append("")
        fm_lines.append("# Embedding")
        fm_lines.append(f"embedding_id: vec_{u.get('note_id', 'unknown')}")
        fm_lines.append("---")

        return "\n".join(fm_lines)

    @staticmethod
    def render_body(u: Dict[str, Any]) -> str:
        """Render the markdown body — dispatch to per-type renderer."""
        note_type_str = u.get("type", "observation")
        try:
            note_type = NoteType(note_type_str)
        except ValueError:
            note_type = NoteType.OBSERVATION

        renderer = _BODY_RENDERERS.get(note_type, _render_generic)
        return renderer(u)


# ---------------------------------------------------------------------------
# Per-type body renderers
# ---------------------------------------------------------------------------

def _render_task(u: Dict[str, Any]) -> str:
    action = u.get("action") or u["text"][:80]
    summary = u.get("summary") or u["text"]
    return textwrap.dedent(f"""\
        # {action}

        {summary}

        ## Details
        - **Status:** todo
        - **Priority:** {_priority_label(u.get('priority', 0.5))}
        - **Urgency:** {u.get('urgency', 'later')}
        """)


def _render_person(u: Dict[str, Any]) -> str:
    entities = u.get("entities", [])
    person = next((e for e in entities if e.get("entity_type") == "person"), None)
    name = person["name"] if person else "Unknown"
    return textwrap.dedent(f"""\
        # {name}

        ## Information
        {_format_entity_attributes(entities)}

        ## Activity
        - First mentioned: {u.get('timestamp', 'unknown')}
        """)


def _render_meeting(u: Dict[str, Any]) -> str:
    participants = [e["name"] for e in u.get("entities", []) if e.get("entity_type") == "person"]
    return textwrap.dedent(f"""\
        # Meeting Notes

        **Date:** {u.get('timestamp', 'unknown')}

        ## Participants
        {chr(10).join(f'- {p}' for p in participants) if participants else '_No participants recorded._'}

        ## Notes
        {u.get('summary') or u['text']}

        ## Action Items
        {u.get('action') or '_None identified._'}
        """)


def _render_idea(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Idea

        {u.get('summary') or u['text']}

        ## Details
        - **Stage:** captured
        - **Potential Impact:** _to be evaluated_
        """)


def _render_learning(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Learning

        {u.get('summary') or u['text']}

        ## Source
        - **Type:** note
        - **Confidence:** {u.get('confidence', 0.5)}
        """)


def _render_decision(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Decision

        {u.get('summary') or u['text']}

        ## Details
        - **Rationale:** _pending_
        - **Reversible:** _unknown_
        """)


def _render_reminder(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Reminder

        {u.get('summary') or u['text']}

        ## Details
        - **Completed:** false
        - **Recurrence:** none
        """)


def _render_observation(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Observation

        {u.get('summary') or u['text']}

        ## Context
        {u.get('action') or ''}
        """)


def _render_preference(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Preference

        {u.get('summary') or u['text']}
        """)


def _render_plan(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Plan

        {u.get('summary') or u['text']}

        ## Milestones
        - _pending_
        """)


def _render_celebration(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # 🎉 Celebration

        {u.get('summary') or u['text']}
        """)


def _render_journal(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Journal Entry

        **Date:** {u.get('timestamp', 'unknown')}

        {u.get('summary') or u['text']}
        """)


def _render_health(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Health

        {u.get('summary') or u['text']}

        ## Metric
        - **Category:** general
        """)


def _render_finance(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Finance

        {u.get('summary') or u['text']}
        """)


def _render_travel(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Travel

        {u.get('summary') or u['text']}
        """)


def _render_goal(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Goal

        {u.get('summary') or u['text']}

        ## Progress
        - **Target date:** _pending_
        - **Metric:** _pending_
        """)


def _render_habit(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Habit

        {u.get('summary') or u['text']}

        ## Tracking
        - **Frequency:** _pending_
        - **Streak:** 0
        """)


def _render_media(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Media

        {u.get('summary') or u['text']}
        """)


def _render_wish(u: Dict[str, Any]) -> str:
    return textwrap.dedent(f"""\
        # Wish

        {u.get('summary') or u['text']}
        """)


def _render_generic(u: Dict[str, Any]) -> str:
    """Fallback renderer for any note type without a dedicated one."""
    return textwrap.dedent(f"""\
        # {u.get('type', 'Note').replace('_', ' ').title()}

        {u.get('summary') or u['text']}
        """)


# Map NoteType → renderer function
_BODY_RENDERERS = {
    NoteType.TASK: _render_task,
    NoteType.PERSON_INFO: _render_person,
    NoteType.CONTACT: _render_person,
    NoteType.MEETING: _render_meeting,
    NoteType.IDEA: _render_idea,
    NoteType.QUESTION: _render_idea,
    NoteType.LEARNING: _render_learning,
    NoteType.LESSON: _render_learning,
    NoteType.DECISION: _render_decision,
    NoteType.REMINDER: _render_reminder,
    NoteType.OBSERVATION: _render_observation,
    NoteType.PREFERENCE: _render_preference,
    NoteType.PLAN: _render_plan,
    NoteType.GOAL: _render_plan,
    NoteType.CELEBRATION: _render_celebration,
    NoteType.GRATITUDE: _render_celebration,
    NoteType.JOURNAL: _render_journal,
    NoteType.HEALTH: _render_health,
    NoteType.FINANCE: _render_finance,
    NoteType.TRAVEL: _render_travel,
    NoteType.HABIT: _render_habit,
    NoteType.WISH: _render_wish,
    NoteType.MEDIA: _render_media,
    NoteType.PLACE: _render_generic,
    NoteType.ACCOUNT: _render_generic,
    NoteType.TOOL_INFO: _render_generic,
    NoteType.PROJECT_UPDATE: _render_generic,
    NoteType.FRUSTRATION: _render_observation,
}
