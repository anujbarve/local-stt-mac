# Instance B: Phase 6 — Vault Intelligence & Extensibility

## Your Mission (Phase 6)

Make the vault intelligent — knowledge graph analysis, smart scheduling, knowledge synthesis, a plugin system, and a web dashboard. The vault transforms from passive storage to an active knowledge partner.

**You are the vault. You analyze. You synthesize. You extend. You serve.**

## Rules

1. **DO NOT** touch `brain/` modules — only import from `brain_contract.py`
2. **DO NOT** modify `brain_contract.py` (FROZEN)
3. **DO NOT** modify `transcribe.py` — modified during merge only
4. **DO NOT** modify `scripts/brain_cli.py` — Instance A owns
5. **DO** import from `brain_contract.py` for shared types
6. **DO** import from `vault/` modules (already built)
7. **DO** write tests for every new module
8. **DO** handle errors gracefully
9. **DO** keep all imports within `vault/` — no cross-boundary imports

## File Ownership (STRICT)

```
vault/graph.py          # ENHANCE: path finding, communities, centrality
vault/scheduler.py      # NEW: smart scheduling
vault/synthesizer.py    # NEW: knowledge synthesis
vault/plugins.py        # NEW: plugin system
vault/plugin_registry.py # NEW: plugin registry
vault/web/              # NEW: web dashboard
vault/analytics.py      # NEW: usage analytics
vault/graph_query.py    # NEW: graph query language
vault/tests/            # NEW tests
tests/test_intelligence.py # NEW: E2E intelligence tests
scripts/unified_cli.py  # ENHANCE: new commands
```

### OFF LIMITS:
```
brain/                    # Instance A owns
transcribe.py             # Modified during merge only
brain_contract.py         # FROZEN
scripts/brain_cli.py      # Instance A owns
```

---

## Build Order

### Step 1: Enhance `vault/graph.py` — Graph Intelligence

Add path finding, community detection, and centrality analysis:

```python
class VaultGraph:
    # ... existing methods (build, num_nodes, num_edges, orphans) ...

    def shortest_path(self, from_id: str, to_id: str) -> Optional[List[str]]:
        """Find shortest path between two notes.

        Returns list of note_ids forming the path, or None.
        """
        ...

    def communities(self) -> List[List[str]]:
        """Detect communities using label propagation.

        Returns list of communities, each a list of note_ids.
        """
        ...

    def centrality(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Compute betweenness centrality for notes.

        Returns list of {note_id, centrality_score} sorted by score.
        """
        ...

    def similarity_cluster(self, threshold: float = 0.7) -> List[List[str]]:
        """Group notes by embedding similarity above threshold.

        Returns list of clusters, each a list of note_ids.
        """
        ...

    def graph_stats(self) -> Dict[str, Any]:
        """Compute graph statistics.

        Returns:
            {
                "num_nodes": 100,
                "num_edges": 250,
                "density": 0.05,
                "avg_degree": 5.0,
                "num_communities": 8,
                "largest_community": 15,
                "connected_components": 3,
            }
        """
        ...
```

---

### Step 2: `vault/graph_query.py` — Graph Query Language

Simple query interface for graph traversal:

```python
class GraphQueryEngine:
    """Execute graph queries on the vault knowledge graph."""

    def __init__(self, graph: VaultGraph):
        self.graph = graph

    def find_path(self, from_id: str, to_id: str) -> Dict[str, Any]:
        """Find path between two notes.

        Returns {"path": [...], "distance": 3}
        """
        ...

    def find_cluster(self, topic: str) -> Dict[str, Any]:
        """Find all notes related to a topic.

        Returns {"notes": [...], "count": 5}
        """
        ...

    def find_central(self, top_k: int = 10) -> Dict[str, Any]:
        """Find most central/important notes.

        Returns {"notes": [{"id": ..., "score": ...}], ...}
        """
        ...

    def find_communities(self) -> Dict[str, Any]:
        """List all communities.

        Returns {"communities": [{"id": 0, "size": 15, "label": "tech"}, ...]}
        """
        ...

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute a natural language graph query.

        Supports:
            "path from A to B"
            "cluster technology"
            "central notes"
            "communities"
        """
        ...
```

---

### Step 3: `vault/scheduler.py` — Smart Scheduling

```python
class SmartScheduler:
    """Scan vault for deadlines and suggest schedules."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root

    def scan_deadlines(self) -> List[Dict]:
        """Scan all notes for deadlines.

        Returns list of:
            - note_id: str
            - title: str
            - deadline: datetime
            - priority: float
            - urgency: str
        """
        ...

    def suggest_daily_schedule(
        self,
        available_hours: int = 8,
    ) -> Dict[str, Any]:
        """Generate a suggested daily schedule.

        Returns:
            {
                "date": "2026-08-24",
                "blocks": [
                    {"time": "9:00 AM", "task": "Fix VAD threshold", "priority": 0.9, "duration": "2h"},
                    {"time": "11:00 AM", "task": "Team standup", "priority": 0.6, "duration": "30m"},
                    ...
                ],
                "overdue": [...],
                "upcoming": [...],
            }
        """
        ...

    def detect_conflicts(self) -> List[Dict]:
        """Find overlapping deadlines or conflicting tasks.

        Returns list of conflicts:
            - notes: List[str] (conflicting note IDs)
            - reason: str
            - suggestion: str
        """
        ...

    def weekly_overview(self) -> Dict[str, Any]:
        """Generate weekly task overview.

        Returns:
            {
                "total_tasks": 15,
                "completed": 8,
                "overdue": 2,
                "upcoming": 5,
                "by_priority": {"high": 3, "medium": 7, "low": 5},
            }
        """
        ...
```

---

### Step 4: `vault/synthesizer.py` — Knowledge Synthesis

```python
class KnowledgeSynthesizer:
    """Generate summaries and insights from vault content."""

    def __init__(self, reader=None, graph=None):
        self.reader = reader
        self.graph = graph

    def daily_digest(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Generate daily digest.

        Returns:
            {
                "date": "2026-08-24",
                "notes_created": 5,
                "by_type": {"task": 2, "meeting": 1, "idea": 2},
                "highlights": ["Fixed VAD threshold", "New project idea"],
                "upcoming_deadlines": [...],
            }
        """
        ...

    def weekly_report(self, week_start: Optional[str] = None) -> Dict[str, Any]:
        """Generate weekly report.

        Returns:
            {
                "week": "2026-W34",
                "total_notes": 42,
                "completed_tasks": 12,
                "new_ideas": 5,
                "decisions_made": 3,
                "top_topics": ["voiceassist", "planning"],
                "productivity_score": 0.78,
            }
        """
        ...

    def topic_summary(self, topic: str) -> Dict[str, Any]:
        """Generate summary of all notes about a topic.

        Returns:
            {
                "topic": "voiceassist",
                "note_count": 15,
                "recent_notes": [...],
                "key_entities": ["VAD", "Sarah"],
                "related_topics": ["technology", "engineering"],
            }
        """
        ...

    def insight_report(self) -> List[Dict]:
        """Generate cross-cutting insights.

        Returns list of insights:
            - type: str
            - message: str
            - confidence: float
            - related_notes: List[str]
        """
        ...
```

---

### Step 5: `vault/plugins.py` — Plugin System

```python
class BasePlugin:
    """Base class for vault plugins."""

    name: str = "base"
    version: str = "0.1.0"

    def on_note_created(self, note: Dict) -> None:
        """Called when a new note is created."""
        pass

    def on_search(self, query: str, results: List[Dict]) -> None:
        """Called when a search is performed."""
        pass

    def on_schedule(self, schedule: Dict) -> None:
        """Called when a schedule is generated."""
        pass

    def on_backup(self, backup_path: str) -> None:
        """Called when a backup is created."""
        pass


class PluginManager:
    """Manage plugin lifecycle and event dispatch."""

    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.enabled: Dict[str, bool] = {}

    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin."""
        ...

    def enable(self, name: str) -> None:
        """Enable a plugin."""
        ...

    def disable(self, name: str) -> None:
        """Disable a plugin."""
        ...

    def fire(self, event: str, *args, **kwargs) -> None:
        """Fire an event to all enabled plugins."""
        ...

    def list_plugins(self) -> List[Dict]:
        """List all registered plugins."""
        ...

    def load_from_config(self, config_path: str) -> None:
        """Load plugins from YAML/JSON config."""
        ...
```

---

### Step 6: `vault/web/` — Web Dashboard

```python
# vault/web/__init__.py
def create_web_app(vault_root: str = VAULT_ROOT):
    """Create Flask/FastAPI web app for the vault dashboard."""
    ...

# vault/web/templates/ (or static files)
# - index.html (dashboard)
# - graph.html (knowledge graph visualization)
# - search.html (search interface)
# - notes.html (note browser)
```

**Endpoints:**
- `GET /` — Dashboard with stats
- `GET /notes` — Browse notes with filters
- `GET /notes/<id>` — View single note
- `GET /search?q=...` — Search notes
- `GET /graph` — Knowledge graph visualization
- `GET /graph/data` — Graph JSON for D3.js
- `GET /schedule` — Daily/weekly schedule
- `GET /analytics` — Usage analytics

---

### Step 7: `vault/analytics.py` — Usage Analytics

```python
class UsageAnalytics:
    """Track and report usage patterns."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.log_path = os.path.join(vault_root, "_analytics", "usage.jsonl")

    def track(self, event: str, **kwargs) -> None:
        """Log a usage event."""
        ...

    def note_creation_stats(self, days: int = 30) -> Dict[str, Any]:
        """Note creation frequency by type and date."""
        ...

    def search_stats(self, days: int = 30) -> Dict[str, Any]:
        """Search query patterns and success rates."""
        ...

    def model_accuracy_trend(self) -> List[Dict]:
        """Track model accuracy over time from corrections."""
        ...

    def generate_report(self, period: str = "weekly") -> Dict[str, Any]:
        """Generate usage report."""
        ...
```

---

### Step 8: Tests

#### `vault/tests/test_graph_intelligence.py`
```python
class TestGraphIntelligence:
    def test_shortest_path(self): ...
    def test_communities(self): ...
    def test_centrality(self): ...
    def test_similarity_cluster(self): ...
    def test_graph_stats(self): ...
```

#### `vault/tests/test_graph_query.py`
```python
class TestGraphQueryEngine:
    def test_find_path(self): ...
    def test_find_cluster(self): ...
    def test_find_central(self): ...
    def test_execute_query(self): ...
```

#### `vault/tests/test_scheduler.py`
```python
class TestSmartScheduler:
    def test_scan_deadlines(self): ...
    def test_suggest_daily_schedule(self): ...
    def test_detect_conflicts(self): ...
    def test_weekly_overview(self): ...
```

#### `vault/tests/test_synthesizer.py`
```python
class TestKnowledgeSynthesizer:
    def test_daily_digest(self): ...
    def test_weekly_report(self): ...
    def test_topic_summary(self): ...
    def test_insight_report(self): ...
```

#### `vault/tests/test_plugins.py`
```python
class TestPluginSystem:
    def test_register_plugin(self): ...
    def test_enable_disable(self): ...
    def test_fire_event(self): ...
    def test_load_from_config(self): ...
```

#### `vault/tests/test_analytics.py`
```python
class TestUsageAnalytics:
    def test_track_event(self): ...
    def test_note_creation_stats(self): ...
    def test_search_stats(self): ...
    def test_generate_report(self): ...
```

---

## Commit Messages

```
feat(vault): enhance graph with path finding and communities
feat(vault): add graph query language
feat(vault): add smart scheduling
feat(vault): add knowledge synthesis
feat(vault): add plugin system
feat(vault): add web dashboard
feat(vault): add usage analytics
test(vault): add graph, scheduler, synthesizer, plugin, analytics tests
docs: add plugin development guide
```
