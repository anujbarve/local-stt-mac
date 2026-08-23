"""
brain.recommend — Contextual recommendations.

Suggests related notes, actions, detects stale projects,
and analyzes priority trends.

Usage:
    from brain.recommend import RecommendationEngine

    engine = RecommendationEngine()
    related = engine.recommend_related(note, all_notes)
    stale = engine.find_stale_projects(notes)
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional


class RecommendationEngine:
    """Generate contextual recommendations based on note content."""

    def __init__(self):
        pass

    def recommend_related(
        self,
        note: Dict,
        all_notes: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Recommend related notes.

        Args:
            note: The current note.
            all_notes: All other notes.
            top_k: Maximum recommendations.

        Returns:
            List of recommendations with relevance and reason.
        """
        note_id = str(note.get("id", note.get("note_id", "")))
        note_topics = set(note.get("topics", []))
        note_entities = set()
        for e in note.get("entities", []):
            name = e.get("name", "") if isinstance(e, dict) else getattr(e, "name", "")
            if name:
                note_entities.add(name.lower())
        note_type = note.get("type", "")
        note_text = note.get("text", "").lower()

        scored = []
        for n in all_notes:
            n_id = str(n.get("id", n.get("note_id", "")))
            if n_id == note_id:
                continue

            score = 0.0
            reason = "similar_content"

            # Topic overlap
            n_topics = set(n.get("topics", []))
            common_topics = note_topics & n_topics
            if common_topics:
                score += 0.3 * len(common_topics)
                reason = "similar_topic"

            # Entity overlap
            n_entities = set()
            for e in n.get("entities", []):
                name = e.get("name", "") if isinstance(e, dict) else getattr(e, "name", "")
                if name:
                    n_entities.add(name.lower())
            common_entities = note_entities & n_entities
            if common_entities:
                score += 0.4 * len(common_entities)
                reason = "same_entity"

            # Same type
            if n.get("type") == note_type:
                score += 0.1

            # Text keyword overlap
            n_words = set(n.get("text", "").lower().split())
            note_words = set(note_text.split())
            common_words = (n_words & note_words) - {"the", "a", "an", "is", "to", "and", "of", "in", "for", "i", "it"}
            if len(common_words) >= 3:
                score += 0.1 * min(len(common_words), 5)

            if score > 0:
                scored.append({
                    "note_id": n_id,
                    "text": n.get("text", "")[:100],
                    "relevance": round(min(score, 1.0), 2),
                    "reason": reason,
                })

        scored.sort(key=lambda x: x["relevance"], reverse=True)
        return scored[:top_k]

    def suggest_actions(
        self,
        note: Dict,
        context: Optional[Dict] = None,
    ) -> List[Dict]:
        """Suggest actions based on note content.

        Returns list of action suggestions.
        """
        suggestions = []
        note_type = note.get("type", "")
        text = note.get("text", "").lower()
        priority = note.get("priority", 0.5)

        # Task-related suggestions
        if note_type == "task":
            if priority > 0.8:
                suggestions.append({
                    "action": "prioritize",
                    "target": note.get("id", ""),
                    "confidence": 0.9,
                    "reason": "High priority task",
                })
            suggestions.append({
                "action": "set_deadline",
                "target": note.get("id", ""),
                "confidence": 0.7,
                "reason": "Tasks benefit from deadlines",
            })

        # Meeting follow-ups
        if note_type == "meeting":
            suggestions.append({
                "action": "create_action_items",
                "target": note.get("id", ""),
                "confidence": 0.8,
                "reason": "Extract action items from meeting",
            })

        # Ideas need development
        if note_type == "idea":
            suggestions.append({
                "action": "develop_idea",
                "target": note.get("id", ""),
                "confidence": 0.6,
                "reason": "Ideas benefit from elaboration",
            })

        # Questions need answers
        if note_type == "question":
            suggestions.append({
                "action": "research_answer",
                "target": note.get("id", ""),
                "confidence": 0.7,
                "reason": "Questions need research",
            })

        # Keyword-based suggestions
        if any(w in text for w in ["bug", "broken", "error", "fix"]):
            suggestions.append({
                "action": "create_bug_ticket",
                "target": note.get("id", ""),
                "confidence": 0.7,
                "reason": "Bug-related content detected",
            })

        if any(w in text for w in ["deploy", "release", "ship"]):
            suggestions.append({
                "action": "check_deployment_readiness",
                "target": note.get("id", ""),
                "confidence": 0.6,
                "reason": "Deployment-related content",
            })

        return suggestions

    def find_stale_projects(
        self,
        notes: List[Dict],
        stale_days: int = 14,
    ) -> List[Dict]:
        """Find projects that haven't been updated recently.

        Args:
            notes: All notes.
            stale_days: Days without updates to consider stale.

        Returns:
            List of stale projects.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=stale_days)

        # Collect project mentions with timestamps
        project_notes: Dict[str, List[Dict]] = defaultdict(list)
        for n in notes:
            entities = n.get("entities", [])
            ts_str = n.get("timestamp", "")

            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                ts = now

            for e in entities:
                etype = e.get("entity_type", "") if isinstance(e, dict) else getattr(e, "entity_type", "")
                name = e.get("name", "") if isinstance(e, dict) else getattr(e, "name", "")
                if etype in ("project", "tool") and name:
                    project_notes[name].append({"timestamp": ts, "note": n})

        stale = []
        for project, pnotes in project_notes.items():
            latest = max(pnotes, key=lambda x: x["timestamp"])
            if latest["timestamp"] < cutoff:
                days_stale = (now - latest["timestamp"]).days
                stale.append({
                    "project": project,
                    "last_updated": latest["timestamp"].isoformat(),
                    "days_stale": days_stale,
                    "related_notes": [str(pn["note"].get("id", "")) for pn in pnotes[:5]],
                })

        stale.sort(key=lambda x: x["days_stale"], reverse=True)
        return stale

    def priority_trends(
        self,
        notes: List[Dict],
        window_days: int = 30,
    ) -> Dict[str, Any]:
        """Analyze priority changes over time.

        Returns trends by topic.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=window_days)
        mid = now - timedelta(days=window_days // 2)

        # Collect priorities by topic in two time windows
        recent: Dict[str, List[float]] = defaultdict(list)
        older: Dict[str, List[float]] = defaultdict(list)

        for n in notes:
            ts_str = n.get("timestamp", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            priority = n.get("priority", 0.5)
            topics = n.get("topics", ["untagged"])

            for topic in topics:
                if ts >= mid:
                    recent[topic].append(priority)
                elif ts >= cutoff:
                    older[topic].append(priority)

        increasing = []
        decreasing = []
        stable = []

        all_topics = set(list(recent.keys()) + list(older.keys()))
        for topic in all_topics:
            r_avg = sum(recent.get(topic, [])) / len(recent[topic]) if recent.get(topic) else 0
            o_avg = sum(older.get(topic, [])) / len(older[topic]) if older.get(topic) else 0

            if not recent.get(topic) or not older.get(topic):
                continue

            diff = r_avg - o_avg
            if diff > 0.1:
                increasing.append({"topic": topic, "from": round(o_avg, 2), "to": round(r_avg, 2)})
            elif diff < -0.1:
                decreasing.append({"topic": topic, "from": round(o_avg, 2), "to": round(r_avg, 2)})
            else:
                stable.append({"topic": topic, "priority": round(r_avg, 2)})

        increasing.sort(key=lambda x: x["to"] - x["from"], reverse=True)
        decreasing.sort(key=lambda x: x["from"] - x["to"], reverse=True)

        return {
            "increasing": increasing[:5],
            "decreasing": decreasing[:5],
            "stable": stable[:5],
        }
