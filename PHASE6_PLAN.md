# Phase 6 — Intelligence & Extensibility

## Current State (End of Phase 5)

The second brain now has:
- **Brain**: 10-task transformer, ONNX export, confidence calibration, incremental learning, A/B testing, knowledge distillation
- **Vault**: Markdown storage, FAISS search, wikilink graph, auto-linking, MOC, LLM fallback, correction vault, backup/restore, REST API, structured logging
- **Integration**: Voice transcription → brain → vault, unified CLI, interactive corrections
- **Tests**: ~400+ passing across brain, vault, and integration

**What's next**: The system works end-to-end and is production-ready. Phase 6 adds the intelligence layer that makes the second brain truly *smart* — it learns patterns, connects knowledge, and provides proactive insights.

---

## Phase 6 Goals

1. **Knowledge Graph Intelligence**: Path finding, community detection, centrality analysis, similarity clustering
2. **Smart Scheduling**: Deadline detection, priority-based scheduling, time-block suggestions
3. **Knowledge Synthesis**: Auto-generate daily/weekly digests, topic summaries, insight reports
4. **Plugin System**: Extensible architecture for custom integrations (RSS, calendar, email, Slack)
5. **Web Dashboard**: Local web UI for browsing vault, viewing graph, searching notes
6. **Usage Analytics**: Track note access patterns, search effectiveness, model accuracy trends

---

## Instance A (Brain) — Phase 6 Tasks

### Task 1: Temporal Reasoning Engine
**Files**: `brain/temporal.py` (NEW)

- Parse natural language time references ("next Friday", "in 2 hours", "end of month")
- Extract deadlines from text with high accuracy
- Suggest optimal scheduling based on urgency + priority
- Detect recurring patterns ("every Monday", "daily standup")

### Task 2: Topic Clustering
**Files**: `brain/clustering.py` (NEW)

- Auto-cluster notes by semantic similarity
- Detect emerging topics from note patterns
- Generate topic hierarchies (parent → child topics)
- Suggest new topic labels based on note content

### Task 3: Insight Generation
**Files**: `brain/insights.py` (NEW)

- "You've been working on X a lot this week" — usage patterns
- "This note connects to 3 other notes about Y" — cross-references
- "Your priority on Z increased from 0.3 to 0.8" — trend detection
- "You haven't touched project A in 2 weeks" — stale project alerts

### Task 4: Contextual Recommendations
**Files**: `brain/recommend.py` (NEW)

- "Based on this note, you might want to review..." — related notes
- "This meeting note mentions a task from last week" — action item linking
- "Your daily journal shows stress about X" — pattern recognition

---

## Instance B (Vault) — Phase 6 Tasks

### Task 5: Knowledge Graph Intelligence
**Files**: `vault/graph.py` (ENHANCE)

- **Path finding**: Shortest path between any two notes
- **Community detection**: Group related notes into clusters
- **Centrality analysis**: Find the most connected/important notes
- **Similarity clustering**: Group notes by embedding similarity
- **Graph statistics**: Density, diameter, clustering coefficient

### Task 6: Smart Scheduling
**Files**: `vault/scheduler.py` (NEW)

- Scan vault for deadlines and high-priority tasks
- Generate daily/weekly schedule suggestions
- Detect conflicts and overlapping deadlines
- Integrate with macOS Calendar (via EventKit)

### Task 7: Knowledge Synthesis
**Files**: `vault/synthesizer.py` (NEW)

- **Daily digest**: "Today you captured 5 notes, 2 tasks, 1 meeting"
- **Weekly report**: "This week: 3 tasks completed, 2 new ideas, 1 decision"
- **Topic summary**: "All notes about 'voiceassist' project: overview"
- **Insight report**: "Cross-cutting themes in your recent notes"

### Task 8: Plugin System
**Files**: `vault/plugins.py` (NEW), `vault/plugin_registry.py` (NEW)

- Plugin interface: `BasePlugin` with `on_note_created`, `on_search`, `on_schedule`
- Built-in plugins: RSS monitor, calendar sync, email digest
- Plugin configuration via YAML/JSON
- Plugin lifecycle: install, enable, disable, uninstall

### Task 9: Web Dashboard
**Files**: `vault/web/` (NEW directory)

- Local web UI (Flask or FastAPI)
- Note browser with search and filters
- Knowledge graph visualization (D3.js force graph)
- Dashboard with daily/weekly stats
- Real-time updates via WebSocket

### Task 10: Usage Analytics
**Files**: `vault/analytics.py` (NEW)

- Track note creation frequency by type
- Track search queries and results
- Track model accuracy over time (from corrections)
- Generate usage reports (daily, weekly, monthly)
- Export analytics as JSON/CSV

---

## Shared Tasks

### Task 11: Graph Query Language
**Files**: `vault/graph_query.py` (NEW)

- Simple query language for graph traversal
- `FIND path FROM note_a TO note_b`
- `CLUSTER notes WHERE topic = "technology"`
- `CENTRALITY TOP 10`
- `COMMUNITIES LIST`

### Task 12: End-to-End Intelligence Test
**Files**: `tests/test_intelligence.py` (NEW)

- Test: Create notes → graph builds → clusters detected → insights generated
- Test: Deadline detection → schedule suggestion → notification sent
- Test: Plugin installed → events fired → plugin responds

---

## Build Order

### Week 1: Graph Intelligence & Temporal
1. Task 1 (Temporal reasoning) — time parsing foundation
2. Task 5 (Graph intelligence) — path finding, communities, centrality
3. Task 11 (Graph query language) — query interface for graph

### Week 2: Synthesis & Scheduling
4. Task 7 (Knowledge synthesis) — digests and reports
5. Task 6 (Smart scheduling) — deadline detection and scheduling
6. Task 2 (Topic clustering) — semantic grouping

### Week 3: Intelligence Layer
7. Task 3 (Insight generation) — pattern detection and alerts
8. Task 4 (Contextual recommendations) — related note suggestions
9. Task 10 (Usage analytics) — tracking and reporting

### Week 4: Extensibility & UI
10. Task 8 (Plugin system) — extensibility architecture
11. Task 9 (Web dashboard) — local web interface
12. Task 12 (E2E intelligence test) — verify everything works together

---

## File Ownership (Conflict-Free)

### Instance A owns:
```
brain/temporal.py       # NEW: temporal reasoning engine
brain/clustering.py     # NEW: topic clustering
brain/insights.py       # NEW: insight generation
brain/recommend.py      # NEW: contextual recommendations
brain/tests/            # NEW tests
```

### Instance B owns:
```
vault/graph.py          # ENHANCE: graph intelligence
vault/scheduler.py      # NEW: smart scheduling
vault/synthesizer.py    # NEW: knowledge synthesis
vault/plugins.py        # NEW: plugin system
vault/plugin_registry.py # NEW: plugin registry
vault/web/              # NEW: web dashboard
vault/analytics.py      # NEW: usage analytics
vault/graph_query.py    # NEW: graph query language
vault/tests/            # NEW tests
tests/test_intelligence.py # NEW: E2E intelligence tests
```

### Shared (merge time only):
```
brain_contract.py       # ENHANCE: temporal types if needed
transcribe.py           # Modified during merge only
```

### OFF LIMITS:
```
brain/                  # Instance A owns
vault/                  # Instance B owns (except shared)
brain_contract.py       # FROZEN unless coordinated
```

---

## Success Criteria

- [ ] All ~400 existing tests still pass
- [ ] New tests for every new module (target: 500+ total)
- [ ] Graph path finding works between any two notes
- [ ] Community detection clusters related notes
- [ ] Daily digest generates accurate summary
- [ ] Plugin system loads and fires events
- [ ] Web dashboard serves local UI with graph visualization
- [ ] Temporal parsing handles "next Friday", "in 2 hours", "end of month"
- [ ] Insight generation detects patterns across notes

---

## Commit Messages

```
feat(brain): add temporal reasoning engine
feat(brain): add topic clustering
feat(brain): add insight generation
feat(brain): add contextual recommendations
feat(vault): enhance graph with path finding and communities
feat(vault): add smart scheduling
feat(vault): add knowledge synthesis
feat(vault): add plugin system
feat(vault): add web dashboard
feat(vault): add usage analytics
feat(vault): add graph query language
test: add E2E intelligence tests
docs: update architecture for Phase 6
```
