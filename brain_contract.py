"""
Brain-Vault Interface Contract
==============================

This file defines the shared types between the brain (intelligence) layer
and the vault (storage/retrieval) layer. BOTH instances must use these
exact types. Do NOT modify this file after Week 1 without coordinating.

Rules:
  - Instance A (brain/) PRODUCES Understanding objects
  - Instance B (vault/) CONSUMES Understanding objects
  - Both instances READ these types
  - Neither instance MODIFIES these types without agreement

Last coordinated: 2026-08-23
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# ENUMS — Canonical values used by both sides
# =============================================================================

class NoteType(str, Enum):
    """All recognized note types. Both brain and vault use these."""
    OBSERVATION = "observation"
    TASK = "task"
    IDEA = "idea"
    LEARNING = "learning"
    DECISION = "decision"
    MEETING = "meeting"
    REMINDER = "reminder"
    PERSON_INFO = "person_info"
    PROJECT_UPDATE = "project_update"
    PREFERENCE = "preference"
    REFLECTION = "reflection"
    QUESTION = "question"
    PLAN = "plan"
    CELEBRATION = "celebration"
    FRUSTRATION = "frustration"
    GRATITUDE = "gratitude"
    JOURNAL = "journal"
    CONTACT = "contact"
    HEALTH = "health"
    FINANCE = "finance"
    TRAVEL = "travel"
    GOAL = "goal"
    LESSON = "lesson"
    HABIT = "habit"
    WISH = "wish"
    MEDIA = "media"
    PLACE = "place"
    ACCOUNT = "account"
    TOOL_INFO = "tool_info"


class EntityType(str, Enum):
    """All recognized entity types."""
    PERSON = "person"
    PROJECT = "project"
    TASK = "task"
    TOOL = "tool"
    DATE = "date"
    TIME = "time"
    ORG = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    ACTION = "action"
    DEADLINE = "deadline"
    PRODUCT = "product"
    MEDICATION = "medication"
    ACCOUNT = "account"
    SKILL = "skill"
    TOPIC = "topic"
    EVENT = "event"
    FILE = "file"
    URL = "url"
    MONEY = "money"
    DURATION = "duration"
    RECIPE = "recipe"


class Priority(str, Enum):
    """Priority levels. The NN outputs a float, this maps to discrete."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    SOMEDAY = "someday"

    @classmethod
    def from_score(cls, score: float) -> "Priority":
        """Convert 0.0-1.0 neural network output to priority level."""
        if score >= 0.9:
            return cls.CRITICAL
        elif score >= 0.7:
            return cls.HIGH
        elif score >= 0.4:
            return cls.MEDIUM
        elif score >= 0.2:
            return cls.LOW
        else:
            return cls.SOMEDAY

    def to_score(self) -> float:
        """Convert priority to a 0.0-1.0 score."""
        return {
            Priority.CRITICAL: 0.95,
            Priority.HIGH: 0.8,
            Priority.MEDIUM: 0.5,
            Priority.LOW: 0.3,
            Priority.SOMEDAY: 0.1,
        }[self]


class Urgency(str, Enum):
    """Urgency levels."""
    IMMEDIATE = "immediate"
    SOON = "soon"
    LATER = "later"
    SOMEDAY = "someday"


class Sentiment(str, Enum):
    """Sentiment values."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    UNCERTAIN = "uncertain"


class ProjectStatus(str, Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


# =============================================================================
# CORE DATA TYPES — Shared between brain and vault
# =============================================================================

@dataclass
class Entity:
    """A single extracted entity from text."""
    name: str
    entity_type: EntityType
    confidence: float = 1.0
    start_char: Optional[int] = None   # character offset in original text
    end_char: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["entity_type"] = self.entity_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        if isinstance(data.get("entity_type"), str):
            data["entity_type"] = EntityType(data["entity_type"])
        return cls(**data)


@dataclass
class TemporalInfo:
    """Temporal information extracted from text."""
    has_date: bool = False
    has_time: bool = False
    relative: Optional[str] = None     # "past", "today", "future"
    absolute_date: Optional[str] = None  # "2026-08-25"
    absolute_time: Optional[str] = None  # "17:00"
    month: Optional[int] = None
    confidence: float = 0.0
    raw_expression: Optional[str] = None  # "next Friday", "tomorrow"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Understanding:
    """
    The complete understanding of a piece of text.
    
    PRODUCED BY: brain/understand.py
    CONSUMED BY: vault/writer.py
    
    This is the primary interface object. The brain fills it in,
    the vault writes it to disk.
    """
    # Source
    text: str                                        # original transcription
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    # Classification
    type: NoteType = NoteType.OBSERVATION
    priority: float = 0.5                            # 0.0 to 1.0
    urgency: Urgency = Urgency.LATER
    sentiment: Sentiment = Sentiment.NEUTRAL
    confidence: float = 0.5                          # model's overall confidence

    # Entities
    entities: List[Entity] = field(default_factory=list)

    # Content
    action: Optional[str] = None                     # extracted action phrase
    summary: Optional[str] = None                    # extractive summary
    topics: List[str] = field(default_factory=list)  # multi-label topics
    tags: List[str] = field(default_factory=list)    # auto-generated tags

    # Embedding
    embedding: Optional[List[float]] = None          # 384-dim vector

    # Temporal
    temporal: Optional[TemporalInfo] = None

    # Links
    related_notes: List[str] = field(default_factory=list)  # note IDs

    # Vault metadata (filled by vault/writer.py after writing)
    note_id: Optional[str] = None
    file_path: Optional[str] = None

    # Extensible
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage / cross-module passing."""
        d = {
            "text": self.text,
            "timestamp": self.timestamp,
            "type": self.type.value,
            "priority": self.priority,
            "urgency": self.urgency.value,
            "sentiment": self.sentiment.value,
            "confidence": self.confidence,
            "entities": [e.to_dict() for e in self.entities],
            "action": self.action,
            "summary": self.summary,
            "topics": self.topics,
            "tags": self.tags,
            "embedding": self.embedding,
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "related_notes": self.related_notes,
            "note_id": self.note_id,
            "file_path": self.file_path,
            "metadata": self.metadata,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Understanding":
        """Deserialize from dict."""
        entities = [Entity.from_dict(e) for e in data.get("entities", [])]
        temporal = TemporalInfo.from_dict(data["temporal"]) if data.get("temporal") else None

        return cls(
            text=data["text"],
            timestamp=data.get("timestamp", ""),
            type=NoteType(data.get("type", "observation")),
            priority=data.get("priority", 0.5),
            urgency=Urgency(data.get("urgency", "later")),
            sentiment=Sentiment(data.get("sentiment", "neutral")),
            confidence=data.get("confidence", 0.5),
            entities=entities,
            action=data.get("action"),
            summary=data.get("summary"),
            topics=data.get("topics", []),
            tags=data.get("tags", []),
            embedding=data.get("embedding"),
            temporal=temporal,
            related_notes=data.get("related_notes", []),
            note_id=data.get("note_id"),
            file_path=data.get("file_path"),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Understanding":
        return cls.from_dict(json.loads(json_str))


@dataclass
class QueryResult:
    """
    Result from a vault search.
    
    PRODUCED BY: vault/search.py
    CONSUMED BY: vault/answer.py, vault/query.py
    """
    notes: List[Dict[str, Any]]        # matching note dicts
    scores: List[float]                # relevance scores (higher = more relevant)
    strategy: str                      # "structural", "semantic", "graph", "hybrid"
    total: int                         # total number of matches
    query_parsed: Optional[Dict[str, Any]] = None  # parsed query structure

    def top_n(self, n: int) -> "QueryResult":
        """Return top N results."""
        indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)[:n]
        return QueryResult(
            notes=[self.notes[i] for i in indices],
            scores=[self.scores[i] for i in indices],
            strategy=self.strategy,
            total=self.total,
            query_parsed=self.query_parsed,
        )


@dataclass
class Correction:
    """
    A user correction for model improvement.
    
    PRODUCED BY: brain/correct.py
    CONSUMED BY: brain/train.py
    """
    text: str                          # original text
    predicted: Understanding           # what the model predicted
    corrected: Understanding           # what the user says it should be
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    applied: bool = False              # has this been used for retraining?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "predicted": self.predicted.to_dict(),
            "corrected": self.corrected.to_dict(),
            "timestamp": self.timestamp,
            "applied": self.applied,
        }


# =============================================================================
# CONSTANTS — Shared configuration
# =============================================================================

# Note type to vault directory mapping
NOTE_TYPE_DIRS = {
    NoteType.OBSERVATION: "daily",
    NoteType.TASK: "tasks",
    NoteType.IDEA: "ideas",
    NoteType.LEARNING: "learnings",
    NoteType.DECISION: "decisions",
    NoteType.MEETING: "meetings",
    NoteType.REMINDER: "reminders",
    NoteType.PERSON_INFO: "people",
    NoteType.PROJECT_UPDATE: "projects",
    NoteType.PREFERENCE: "preferences",
    NoteType.REFLECTION: "lessons",
    NoteType.QUESTION: "ideas",
    NoteType.PLAN: "goals",
    NoteType.CELEBRATION: "daily",
    NoteType.FRUSTRATION: "daily",
    NoteType.GRATITUDE: "daily",
    NoteType.JOURNAL: "daily",
    NoteType.CONTACT: "people",
    NoteType.HEALTH: "health",
    NoteType.FINANCE: "finance",
    NoteType.TRAVEL: "travel",
    NoteType.GOAL: "goals",
    NoteType.LESSON: "lessons",
    NoteType.HABIT: "habits",
    NoteType.WISH: "wishes",
    NoteType.MEDIA: "media",
    NoteType.PLACE: "places",
    NoteType.ACCOUNT: "accounts",
    NoteType.TOOL_INFO: "tools",
}

# Entity types that become vault entity pages
ENTITY_PAGE_TYPES = {
    EntityType.PERSON: "people",
    EntityType.PROJECT: "projects",
    EntityType.TOOL: "tools",
    EntityType.ORG: "people",
    EntityType.TOPIC: "topics",
}

# Topic taxonomy (expandable)
TOPICS = [
    "technology", "health", "finance", "career", "relationships",
    "education", "entertainment", "food", "travel", "home",
    "fitness", "creativity", "spirituality", "science", "philosophy",
    "engineering", "design", "music", "gaming", "sports",
    "parenting", "cooking", "gardening", "reading", "writing",
    "politics", "environment", "productivity", "psychology",
]

# Relationship types (for head 10 and graph)
RELATION_TYPES = [
    "works_on", "mentioned_in", "created", "uses", "depends_on",
    "related_to", "decided_by", "learned_from", "attended",
    "assigned_to", "lives_in", "prefers", "authored",
    "discussed_in", "follows_up_on", "blocks", "enables",
]
