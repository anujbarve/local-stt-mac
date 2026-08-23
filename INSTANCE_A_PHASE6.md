# Instance A: Phase 6 — Brain Intelligence Layer

## Your Mission (Phase 6)

Build the intelligence layer that makes the second brain truly smart — temporal reasoning, topic clustering, insight generation, and contextual recommendations. The brain moves from understanding to *understanding with context*.

**You are the brain. You reason about time. You find patterns. You generate insights.**

## Rules

1. **DO NOT** touch `vault/`, `transcribe.py`, `scripts/unified_cli.py`, or any voiceassist module
2. **DO NOT** modify `brain_contract.py` (FROZEN)
3. **DO** import from `brain_contract.py` for shared types
4. **DO** write tests for every new module
5. **DO** handle errors gracefully
6. **DO** keep all imports within `brain/`

## File Ownership (STRICT)

```
brain/temporal.py       # NEW: temporal reasoning engine
brain/clustering.py     # NEW: topic clustering
brain/insights.py       # NEW: insight generation
brain/recommend.py      # NEW: contextual recommendations
brain/tests/            # NEW tests
```

### OFF LIMITS:
```
vault/                    # Instance B owns
transcribe.py             # Modified during merge only
brain_contract.py         # FROZEN
scripts/unified_cli.py    # Instance B owns
scripts/brain_cli.py      # Already built
```

---

## Build Order

### Step 1: `brain/temporal.py` — Temporal Reasoning Engine

Parse natural language time references and extract deadlines.

```python
class TemporalReasoner:
    """Parse and reason about time expressions in text."""

    def parse(self, text: str) -> TemporalResult:
        """Extract temporal information from text.

        Returns:
            TemporalResult with:
                - has_time: bool
                - deadline: Optional[datetime]
                - relative: str ("past", "today", "future")
                - recurring: Optional[str] ("daily", "weekly", "monthly")
                - confidence: float
                - raw_expression: str
        """
        ...

    def suggest_schedule(
        self,
        notes: List[Dict],
        now: Optional[datetime] = None,
    ) -> List[ScheduledItem]:
        """Suggest a schedule based on deadlines and priorities.

        Returns sorted list of ScheduledItem:
            - note_id: str
            - title: str
            - deadline: datetime
            - priority: float
            - suggested_time: str ("9:00 AM", "2:00 PM")
        """
        ...

    def detect_recurring(self, notes: List[Dict]) -> List[Dict]:
        """Detect recurring patterns in notes.

        Returns list of recurring patterns:
            - pattern: str ("every Monday")
            - examples: List[str]
            - frequency: str
        """
        ...

    def time_until(self, deadline: str, now: Optional[datetime] = None) -> str:
        """Human-readable time until deadline.

        Returns: "in 2 hours", "tomorrow", "next Friday", etc.
        """
        ...
```

**Supported patterns:**
```
"by Friday" → deadline: next Friday
"in 2 hours" → deadline: now + 2h
"end of month" → deadline: last day of month
"tomorrow morning" → deadline: tomorrow 9:00
"next Tuesday at 3pm" → deadline: next Tuesday 15:00
"every Monday" → recurring: weekly on Monday
"daily standup" → recurring: daily
```

---

### Step 2: `brain/clustering.py` — Topic Clustering

Auto-cluster notes by semantic similarity.

```python
class TopicClusterer:
    """Cluster notes by semantic similarity and topic."""

    def __init__(self, embedder=None):
        self.embedder = embedder

    def cluster(
        self,
        notes: List[Dict],
        n_clusters: Optional[int] = None,
    ) -> List[Cluster]:
        """Cluster notes into groups.

        Returns list of Cluster:
            - id: int
            - label: str (auto-generated from top notes)
            - notes: List[str] (note IDs)
            - centroid: List[float] (embedding centroid)
            - size: int
        """
        ...

    def find_emerging_topics(
        self,
        notes: List[Dict],
        time_window_days: int = 7,
    ) -> List[Dict]:
        """Detect topics that are gaining traction.

        Returns list of emerging topics:
            - topic: str
            - count_recent: int
            - count_previous: int
            - growth_rate: float
        """
        ...

    def suggest_topic_hierarchy(
        self,
        notes: List[Dict],
    ) -> Dict[str, List[str]]:
        """Suggest parent-child topic relationships.

        Returns:
            {
                "technology": ["python", "rust", "web"],
                "health": ["fitness", "nutrition", "sleep"],
            }
        """
        ...

    def find_outliers(
        self,
        notes: List[Dict],
    ) -> List[Dict]:
        """Find notes that don't fit any cluster well.

        Returns list of outlier notes with low cluster affinity.
        """
        ...
```

---

### Step 3: `brain/insights.py` — Insight Generation

Detect patterns and generate actionable insights.

```python
class InsightGenerator:
    """Generate insights from note patterns and history."""

    def __init__(self, reader=None, graph=None):
        self.reader = reader
        self.graph = graph

    def daily_insights(self, notes: List[Dict]) -> List[Insight]:
        """Generate daily insights.

        Returns list of Insight:
            - type: str ("pattern", "alert", "suggestion")
            - message: str
            - severity: str ("info", "warning", "action")
            - related_notes: List[str]
        """
        ...

    def weekly_report(self, notes: List[Dict]) -> Dict[str, Any]:
        """Generate weekly summary.

        Returns:
            {
                "total_notes": 42,
                "by_type": {"task": 15, "meeting": 8, ...},
                "completed_tasks": 12,
                "new_ideas": 5,
                "top_topics": ["voiceassist", "planning"],
                "stale_projects": ["old-project"],
                "productivity_score": 0.78,
            }
        """
        ...

    def detect_patterns(self, notes: List[Dict]) -> List[Dict]:
        """Detect recurring patterns in notes.

        Returns list of patterns:
            - pattern: str
            - frequency: str
            - examples: List[str]
            - suggestion: str
        """
        ...

    def cross_references(self, note_id: str, notes: List[Dict]) -> List[Dict]:
        """Find related notes and cross-references.

        Returns list of related notes:
            - note_id: str
            - relevance: float
            - connection_type: str ("same_topic", "same_entity", "temporal")
        """
        ...
```

---

### Step 4: `brain/recommend.py` — Contextual Recommendations

Suggest related notes and actions.

```python
class RecommendationEngine:
    """Generate contextual recommendations based on note content."""

    def __init__(self, search=None, graph=None):
        self.search = search
        self.graph = graph

    def recommend_related(
        self,
        note: Dict,
        all_notes: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """Recommend related notes.

        Returns list of recommendations:
            - note_id: str
            - title: str
            - relevance: float
            - reason: str ("similar_topic", "same_entity", "follows_up")
        """
        ...

    def suggest_actions(
        self,
        note: Dict,
        context: Optional[Dict] = None,
    ) -> List[Dict]:
        """Suggest actions based on note content.

        Returns list of suggestions:
            - action: str
            - target: str (note_id or entity)
            - confidence: float
        """
        ...

    def find_stale_projects(
        self,
        notes: List[Dict],
        stale_days: int = 14,
    ) -> List[Dict]:
        """Find projects that haven't been updated recently.

        Returns list of stale projects:
            - project: str
            - last_updated: str
            - days_stale: int
            - related_notes: List[str]
        """
        ...

    def priority_trends(
        self,
        notes: List[Dict],
        window_days: int = 30,
    ) -> Dict[str, Any]:
        """Analyze priority changes over time.

        Returns:
            {
                "increasing": [{"topic": "X", "from": 0.3, "to": 0.8}],
                "decreasing": [{"topic": "Y", "from": 0.7, "to": 0.2}],
                "stable": [{"topic": "Z", "priority": 0.5}],
            }
        """
        ...
```

---

### Step 5: Tests

#### `brain/tests/test_temporal.py`
```python
class TestTemporalReasoner:
    def test_parse_relative_deadline(self): ...
    def test_parse_recurring(self): ...
    def test_suggest_schedule(self): ...
    def test_time_until(self): ...
    def test_detect_recurring(self): ...
```

#### `brain/tests/test_clustering.py`
```python
class TestTopicClusterer:
    def test_cluster_returns_clusters(self): ...
    def test_cluster_determines_count(self): ...
    def test_find_emerging_topics(self): ...
    def test_suggest_topic_hierarchy(self): ...
    def test_find_outliers(self): ...
```

#### `brain/tests/test_insights.py`
```python
class TestInsightGenerator:
    def test_daily_insights(self): ...
    def test_weekly_report(self): ...
    def test_detect_patterns(self): ...
    def test_cross_references(self): ...
```

#### `brain/tests/test_recommend.py`
```python
class TestRecommendationEngine:
    def test_recommend_related(self): ...
    def test_suggest_actions(self): ...
    def test_find_stale_projects(self): ...
    def test_priority_trends(self): ...
```

---

## Commit Messages

```
feat(brain): add temporal reasoning engine
feat(brain): add topic clustering
feat(brain): add insight generation
feat(brain): add contextual recommendations
test(brain): add temporal, clustering, insight, recommendation tests
```
