# Instance B: Vault Retrieval & Reasoning Layer

## Your Mission (Phase 2)

Build the retrieval, search, and reasoning modules that make the vault queryable. You own the storage layer — reading, indexing, searching, and answering questions about stored notes.

**You are the vault. You store. You retrieve. You search. You answer.**

## Rules

1. **DO NOT** touch `brain/`, `transcribe.py`, or any brain module
2. **DO NOT** modify `brain_contract.py`
3. **DO** import types from `brain_contract.py` — never redefine them
4. **DO** import from `vault/config.py`, `vault/schema.py`, `vault/templates.py`, `vault/writer.py` (already built)
5. **DO** write tests for every new module
6. **DO** reuse `_parse_frontmatter` and `_extract_wikilinks` from `vault/schema.py`

## Your Existing Modules (read-only during this phase)

```
vault/__init__.py    # module docstring + exports
vault/config.py      # VAULT_ROOT, INDEX_DIR, EMBEDDING_DIM, paths
vault/schema.py      # NoteSchema, VaultValidator, _parse_frontmatter, _extract_wikilinks
vault/templates.py   # NoteTemplate, per-type renderers
vault/writer.py      # VaultWriter — Understanding → vault file
```

## File Ownership

### Instance B owns (ONLY touches these):
```
vault/reader.py       # NEW: vault file → structured dict
vault/index.py        # NEW: FAISS vector index
vault/search.py       # NEW: hybrid retrieval
vault/graph.py        # NEW: wikilink / backlink graph
vault/moc.py          # NEW: Map of Content
vault/query.py        # NEW: NL query parser
vault/llm.py          # NEW: Gemma 3 1B wrapper
vault/answer.py       # NEW: query answering
vault/pipeline.py     # NEW: end-to-end Understanding → vault write + index + link
vault/linker.py       # NEW: auto-linking via entity resolution
vault/tests/          # NEW tests for all modules
```

### OFF LIMITS (do not touch):
```
brain/                # Instance A owns this
transcribe.py         # Only during merge (Week 9)
brain_contract.py     # Frozen
vault/config.py       # Already built, read-only
vault/schema.py       # Already built, read-only
vault/templates.py    # Already built, read-only
vault/writer.py       # Already built, read-only
```

---

## Build Order

### Step 1: `vault/reader.py` — Read Notes from Disk

**Purpose:** Inverse of `vault/writer.py`. Reads `.md` files and returns structured dicts.

**Key reuse:** Import `_parse_frontmatter` and `_extract_wikilinks` from `vault/schema.py`:
```python
from vault.schema import _parse_frontmatter, _extract_wikilinks
```

**Interface:**

```python
class VaultReader:
    """Read notes from the vault as structured dicts."""

    def __init__(self, vault_root: str = VAULT_ROOT): ...

    def read(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Read a single note file → structured dict.

        Returns None if file doesn't exist or is unparseable.

        Output dict keys:
            id, type, created, updated, source,
            priority, urgency, sentiment, confidence,
            people, projects, tasks, tools, dates,
            tags, category, related_notes, backlinks,
            body (raw markdown body text),
            file_path (absolute path),
            wikilinks (list of [[target]] strings)
        """

    def read_all(self) -> List[Dict[str, Any]]:
        """Read every .md file in the vault → list of dicts.

        Walks all subdirectories. Skips files without valid frontmatter.
        """

    def read_by_type(self, note_type: NoteType) -> List[Dict[str, Any]]:
        """Read all notes of a specific type."""

    def read_by_id(self, note_id: str) -> Optional[Dict[str, Any]]:
        """Find and read a note by its frontmatter ID."""

    def read_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Read all notes that have a specific tag."""

    def read_directory(self, subdir: str) -> List[Dict[str, Any]]:
        """Read all notes in a specific vault subdirectory."""

    def count(self) -> int:
        """Total number of readable notes."""

    def list_files(self) -> List[str]:
        """List all .md file paths (relative to vault_root)."""

    def extract_wikilinks(self, body: str) -> List[str]:
        """Extract [[wikilink]] targets from markdown body."""

    def get_note_summary(self, note: Dict[str, Any]) -> str:
        """Return short text summary (body truncated to 200 chars)."""
```

**Return format** — every dict must have:
```python
{
    "id": "note_2026_08_23_001",
    "type": "task",
    "created": "2026-08-23T14:30:00Z",
    "updated": "2026-08-23T14:30:00Z",
    "source": "audio",
    "priority": "high",
    "urgency": "soon",
    "sentiment": "neutral",
    "confidence": 0.87,
    "people": ["Sarah"],
    "projects": ["voiceassist"],
    "tasks": [],
    "tools": ["VAD"],
    "dates": ["Friday"],
    "tags": ["bug", "vad"],
    "category": "task",
    "related_notes": [],
    "backlinks": [],
    "body": "# Fix VAD Threshold\n\nThe Silero VAD...",
    "file_path": "/absolute/path/to/vault/tasks/fix-vad-threshold.md",
    "wikilinks": ["sarah-chen", "voiceassist"],
}
```

**Error handling:** skip files that:
- Don't start with `---` (no frontmatter)
- Have unparseable YAML
- Have missing `type` field (default to "observation")

---

### Step 2: `vault/index.py` — FAISS Vector Index

**Purpose:** Build and query a FAISS index of note embeddings.

**Interface:**

```python
class VaultIndex:
    """FAISS-based vector index for semantic search."""

    def __init__(
        self,
        index_dir: str = INDEX_DIR,
        embedding_dim: int = EMBEDDING_DIM,
    ): ...

    def build(self, reader: VaultReader, embedder=None) -> int:
        """Scan vault, embed all notes, build index.

        Returns number of notes indexed.

        Algorithm:
            1. reader.read_all() → all note dicts
            2. For each note, embed the body text → 384-dim vector
            3. Store in FAISS index
            4. Save index + ID mapping to disk
        """

    def rebuild(self, reader: VaultReader, embedder=None) -> int:
        """Clear and rebuild entire index."""

    def add(self, note_id: str, file_path: str, embedding: List[float]) -> None:
        """Add a single note to the index."""

    def remove(self, note_id: str) -> bool:
        """Remove a note (mark as removed, filter during search)."""

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find most similar notes.

        Returns:
            [{"note_id": "...", "file_path": "...", "score": 0.87, "rank": 1}, ...]
        """

    def search_by_text(self, text: str, embedder, top_k: int = 10) -> List[Dict]:
        """Convenience: embed text then search."""

    def save(self) -> None:
        """Save index + ID mapping to disk.

        Files:
            {index_dir}/notes.index    — FAISS binary
            {index_dir}/notes_map.json — ID mapping
        """

    def load(self) -> bool:
        """Load index from disk. Returns True if successful."""

    def clear(self) -> None:
        """Delete all data from index."""

    @property
    def size(self) -> int: ...

    def stats(self) -> Dict[str, Any]: ...
```

**ID Mapping:** FAISS uses integer IDs. We maintain a positional list:
```python
self._id_map: List[Dict]  # [{note_id, file_path, removed}, ...]
```

**Stub mode:** When FAISS isn't installed, use brute-force cosine similarity:
```python
try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
```

**Save format:**
```
vault_index/
├── notes.index       # FAISS binary
└── notes_map.json    # [{"note_id": "...", "file_path": "...", "removed": false}, ...]
```

---

### Step 3: `vault/search.py` — Hybrid Retrieval

**Purpose:** Multi-strategy search combining structural, semantic, and graph approaches.

**Interface:**

```python
class VaultSearch:
    """Hybrid search across the vault."""

    def __init__(
        self,
        reader: VaultReader,
        index: VaultIndex,
        graph: "VaultGraph",  # forward reference
    ): ...

    def search(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = SEARCH_TOP_K,
        embedder=None,
    ) -> QueryResult:
        """Main search entry point.

        Strategies:
            "structural" — filter by frontmatter fields
            "semantic"   — FAISS cosine similarity
            "graph"      — follow wikilinks from seed notes
            "hybrid"     — combine all three with weighted scoring
        """

    def structural_search(
        self,
        filters: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Filter by frontmatter fields.

        filters examples:
            {"type": "task"}
            {"priority": "high"}
            {"tags": "python"}
            {"people": "Sarah"}
            {"projects": "voiceassist"}
        """

    def semantic_search(
        self,
        query_text: str,
        embedder,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Embed query, search FAISS index."""

    def graph_search(
        self,
        seed_note: str,
        depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Find notes connected to seed_note via wikilinks."""

    def _merge_results(
        self,
        result_sets: List[List[Dict]],
        weights: List[float],
        top_k: int,
    ) -> List[Dict]:
        """Merge multiple result sets with weighted scores."""

    def _note_to_result(self, note: Dict, score: float, rank: int) -> Dict:
        """Convert note dict to search result format."""
```

**Hybrid scoring:**
```python
# Default weights
STRUCTURAL_WEIGHT = 0.3
SEMANTIC_WEIGHT = 0.5
GRAPH_WEIGHT = 0.2

# For each note, combined_score = sum(weight_i * score_i)
# Sort by combined_score descending
```

---

### Step 4: `vault/graph.py` — Wikilink / Backlink Graph

**Purpose:** Maintain the knowledge graph of wikilinks between notes.

**Interface:**

```python
class VaultGraph:
    """Knowledge graph of note connections."""

    def __init__(self, vault_root: str = VAULT_ROOT): ...

    def build(self, reader: VaultReader) -> int:
        """Scan vault for [[wikilinks]] and build graph.

        Returns number of edges found.
        """

    def add_link(self, source: str, target: str) -> None:
        """Add a directed edge: source → target."""

    def remove_link(self, source: str, target: str) -> None: ...

    def backlinks(self, note: str) -> List[str]:
        """Notes that link TO this note."""

    def forward_links(self, note: str) -> List[str]:
        """Notes that this note links TO."""

    def neighbors(self, note: str, depth: int = 1) -> List[str]:
        """All notes within `depth` hops."""

    def orphans(self) -> List[str]:
        """Notes with no links in or out."""

    def shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """Find shortest path between two notes."""

    def communities(self) -> List[List[str]]:
        """Detect clusters of densely connected notes.

        Simple implementation: connected components via BFS.
        """

    def save(self, path: str = None) -> None:
        """Save graph to JSON."""

    def load(self, path: str = None) -> None:
        """Load graph from JSON."""

    @property
    def num_nodes(self) -> int: ...

    @property
    def num_edges(self) -> int: ...
```

**Internal structure:**
```python
# Adjacency list
self._outgoing: Dict[str, Set[str]]  # source → {targets}
self._incoming: Dict[str, Set[str]]  # target → {sources}
```

**Wikilink extraction:** Reuse `_extract_wikilinks` from `vault/schema.py`:
```python
from vault.schema import _extract_wikilinks
```

---

### Step 5: `vault/moc.py` — Map of Content

**Purpose:** Auto-generate and maintain topic index pages.

**Interface:**

```python
class VaultMOC:
    """Map of Content generator."""

    def __init__(
        self,
        vault_root: str = VAULT_ROOT,
        reader: VaultReader = None,
        graph: VaultGraph = None,
    ): ...

    def generate_moc(self, topic: str) -> str:
        """Generate a MOC markdown page for a topic.

        Scans all notes with matching tags/topics, groups by type,
        returns formatted markdown.
        """

    def update_moc(self, moc_path: str) -> None:
        """Refresh an existing MOC with current links."""

    def list_mocs(self) -> List[str]:
        """List all existing MOC files in vault/topics/."""

    def suggest_moc(self, note: Dict) -> Optional[str]:
        """Suggest which MOC a note should belong to.

        Checks note's tags against existing MOC topics.
        """

    def auto_generate_all(self) -> int:
        """Generate MOCs for all topics that have 3+ notes.

        Returns number of MOCs created/updated.
        """
```

**MOC format:**
```markdown
---
id: moc_technology
type: moc
created: 2026-08-23
updated: 2026-08-23
topic: technology
note_count: 42
---

# Technology

## Tasks (12)
- [[fix-vad-threshold]]
- [[update-readme]]
- ...

## Ideas (8)
- [[circular-buffer-pre-roll]]
- ...

## Learnings (15)
- [[zettelkasten-method]]
- ...

## Related Topics
- [[python]]
- [[engineering]]
```

---

### Step 6: `vault/linker.py` — Auto-Linking

**Purpose:** Automatically link new notes to existing ones via entity resolution.

**Interface:**

```python
class VaultLinker:
    """Auto-link notes via entity resolution."""

    def __init__(
        self,
        reader: VaultReader,
        graph: VaultGraph,
        confidence_threshold: float = LINK_CONFIDENCE_THRESHOLD,
    ): ...

    def link(
        self,
        understanding: Understanding,
        existing_notes: List[Dict] = None,
    ) -> List[str]:
        """Find and add links for a new note.

        Returns list of note_ids that were linked.
        """

    def find_related(
        self,
        note: Dict,
        candidates: List[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Find notes related to `note` by:
            1. Shared entities (person, project, tool)
            2. Shared tags
            3. Embedding similarity
            4. Graph proximity

        Returns list of {"note_id": ..., "score": ..., "reason": ...}
        """

    def merge_entity_pages(
        self,
        entity_name: str,
        pages: List[str],
    ) -> str:
        """Merge multiple entity pages into one.

        Returns path of merged page.
        """

    def suggest_links(
        self,
        note: Dict,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Suggest links without actually adding them.

        Returns list of {"note_id": ..., "score": ..., "reason": ...}
        """
```

---

### Step 7: `vault/query.py` — NL Query Parser

**Purpose:** Parse natural language queries into structured search plans.

**Interface:**

```python
class QueryParser:
    """Parse natural language queries into structured plans."""

    def parse(self, query: str) -> Dict[str, Any]:
        """Parse query into a QueryPlan dict.

        Returns:
            {
                "intent": "search" | "list" | "summarize" | "explain",
                "entity_type": "task" | "person" | "project" | ... | None,
                "filters": {
                    "type": "task",
                    "priority": "high",
                    "tags": ["python"],
                    "people": ["Sarah"],
                },
                "temporal": {
                    "relative": "today" | "this_week" | "this_month" | None,
                    "absolute": "2026-08-23" | None,
                },
                "text": "original query text",
                "expand": False,  # True if query is vague
            }
        """

    def _detect_intent(self, query: str) -> str:
        """Detect intent from query verbs.

        "what" / "show" / "list" → "list"
        "when" / "what time" → "search"
        "why" / "explain" → "explain"
        "summarize" / "how many" → "summarize"
        """

    def _extract_entity_type(self, query: str) -> Optional[str]:
        """Extract target entity type from query nouns.

        "projects" → "project"
        "tasks" → "task"
        "people" → "person"
        "meetings" → "meeting"
        """

    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filter conditions.

        "involving Python" → {"tags": ["python"]}
        "assigned to Sarah" → {"people": ["Sarah"]}
        "high priority" → {"priority": "high"}
        "blocked" → {"status": "blocked"}
        """

    def _extract_temporal(self, query: str) -> Dict[str, Any]:
        """Extract time references.

        "today" → {"relative": "today"}
        "this week" → {"relative": "this_week"}
        "last Tuesday" → {"absolute": "2026-08-19"}
        """
```

**No external NLP dependency** — use regex patterns and keyword matching:
```python
INTENT_PATTERNS = {
    "list": r"\b(what|show|list|find|get|which)\b",
    "search": r"\b(when|where|who)\b",
    "explain": r"\b(why|explain|how)\b",
    "summarize": r"\b(summarize|how many|count|total)\b",
}

ENTITY_TYPE_PATTERNS = {
    "task": r"\b(tasks?|to-?do|action items?)\b",
    "person": r"\b(people|person|who|contacts?)\b",
    "project": r"\b(projects?|initiatives?)\b",
    "meeting": r"\b(meetings?|calls?|discussions?)\b",
    "idea": r"\b(ideas?|thoughts?|brainstorms?)\b",
    "learning": r"\b(learnings?|insights?|discoveries?)\b",
    "decision": r"\b(decisions?|choices?)\b",
}
```

---

### Step 8: `vault/llm.py` — Gemma 3 1B Wrapper

**Purpose:** Deep reasoning when the NN isn't enough.

**Interface:**

```python
class LocalLLM:
    """Local LLM for deep reasoning."""

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ):
        """Load model. Lazy — only loads on first use."""
        self._model = None

    def _ensure_loaded(self) -> None:
        """Load model on first use."""

    def analyze(self, text: str, context: Optional[List[Dict]] = None) -> Understanding:
        """Full analysis of transcription.

        Prompt template:
            Analyze this text and extract structured information.
            Text: "{text}"
            {Context from recent notes: {context}}

            Extract:
            1. Primary type (task/idea/observation/...)
            2. All entities with types
            3. Priority with reasoning
            4. Urgency
            5. Action items
            6. Summary
            7. Related topics
            8. Sentiment

            Respond in JSON.
        """

    def answer(self, query: str, vault_context: List[Dict]) -> str:
        """Answer a question using vault context.

        Prompt template:
            Based on the following notes from my knowledge base:
            {formatted notes}

            Answer this question: {query}

            Be specific. Cite which note the information comes from.
        """

    def find_connections(
        self,
        new_note: Dict,
        related_notes: List[Dict],
    ) -> List[Dict]:
        """Discover non-obvious connections between notes."""

    @property
    def is_available(self) -> bool:
        """Check if LLM is loaded and available."""
```

**Graceful fallback:** If mlx/gemma not installed:
```python
try:
    import mlx_lm
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
```

When `HAS_LLM = False`, methods return empty/default values and `is_available = False`.

---

### Step 9: `vault/answer.py` — Query Answering

**Purpose:** End-to-end question answering with vault context.

**Interface:**

```python
class VaultAnswerer:
    """Answer questions using vault search + LLM."""

    def __init__(
        self,
        search: VaultSearch,
        reader: VaultReader,
        llm: LocalLLM = None,
    ): ...

    def answer(self, query: str) -> Dict[str, Any]:
        """Full question answering pipeline.

        Steps:
            1. Parse query → QueryPlan
            2. Search vault → results
            3. Format context from top results
            4. If LLM available, generate answer
            5. Return structured answer

        Returns:
            {
                "query": "original query",
                "answer": "text answer",
                "sources": [{"note_id": "...", "title": "...", "relevance": 0.87}],
                "strategy": "hybrid",
                "confidence": 0.82,
            }
        """

    def summarize(self, topic: str) -> str:
        """Aggregate notes about a topic into a summary."""

    def explain(self, note_id: str) -> str:
        """Explain a decision or lesson with full context."""
```

---

### Step 10: `vault/pipeline.py` — End-to-End Processor

**Purpose:** The main vault entry point. Orchestrates write + index + link.

**Interface:**

```python
class VaultPipeline:
    """End-to-end: Understanding → vault write + index + link."""

    def __init__(
        self,
        vault_root: str = VAULT_ROOT,
        index_dir: str = INDEX_DIR,
    ):
        self.reader = VaultReader(vault_root)
        self.writer = VaultWriter(vault_root)
        self.index = VaultIndex(index_dir)
        self.graph = VaultGraph(vault_root)
        self.linker = VaultLinker(self.reader, self.graph)
        self.search = VaultSearch(self.reader, self.index, self.graph)
        self.answerer = VaultAnswerer(self.search, self.reader)

    def process(self, understanding: Understanding) -> str:
        """Full pipeline: Understanding → vault.

        Steps:
            1. Write note to vault (vault/writer.py)
            2. Add to search index (vault/index.py)
            3. Update graph (vault/graph.py)
            4. Auto-link to related notes (vault/linker.py)

        Returns file_path of created note.
        """

    def process_batch(self, understandings: List[Understanding]) -> List[str]:
        """Process multiple understandings efficiently."""

    def query(self, text: str) -> Dict[str, Any]:
        """Search and answer a question about the vault."""

    def rebuild_index(self) -> int:
        """Rebuild the entire search index from vault files."""

    def rebuild_graph(self) -> int:
        """Rebuild the entire graph from vault files."""

    def stats(self) -> Dict[str, Any]:
        """Vault statistics: note count, index size, graph edges."""
```

---

### Step 11: Tests

#### `vault/tests/test_reader.py`
```python
class TestVaultReader:
    def test_read_single_note(self): ...
    def test_read_nonexistent_returns_none(self): ...
    def test_read_all_finds_all_notes(self): ...
    def test_read_by_type_filters(self): ...
    def test_read_by_id_finds_exact(self): ...
    def test_read_by_tag(self): ...
    def test_read_directory(self): ...
    def test_wikilinks_extracted(self): ...
    def test_body_extracted(self): ...
    def test_count(self): ...
    def test_graceful_skip_bad_files(self): ...
    def test_read_all_empty_vault(self): ...
```

#### `vault/tests/test_index.py`
```python
class TestVaultIndex:
    def test_build_from_empty_vault(self): ...
    def test_build_from_notes(self): ...
    def test_add_single(self): ...
    def test_search_returns_results(self): ...
    def test_search_scores_descending(self): ...
    def test_search_by_text(self): ...
    def test_remove(self): ...
    def test_save_and_load(self): ...
    def test_clear(self): ...
    def test_stats(self): ...
    def test_graceful_no_faiss(self): ...
    def test_empty_search(self): ...
    def test_load_no_index_returns_false(self): ...
```

#### `vault/tests/test_search.py`
```python
class TestVaultSearch:
    def test_structural_search_by_type(self): ...
    def test_structural_search_by_tag(self): ...
    def test_semantic_search(self): ...
    def test_graph_search(self): ...
    def test_hybrid_search(self): ...
    def test_empty_vault(self): ...
```

#### `vault/tests/test_graph.py`
```python
class TestVaultGraph:
    def test_build_from_vault(self): ...
    def test_add_and_remove_link(self): ...
    def test_backlinks(self): ...
    def test_forward_links(self): ...
    def test_neighbors(self): ...
    def test_orphans(self): ...
    def test_shortest_path(self): ...
    def test_communities(self): ...
    def test_save_and_load(self): ...
```

---

## Dependencies

Already in project:
```
torch>=2.0.0
numpy>=1.24.0
```

Need to add (optional, graceful fallback if missing):
```
faiss-cpu>=1.8.0              # Vector search (or brute-force fallback)
sentence-transformers>=3.0.0  # Embeddings (or stub embedder)
```

For LLM (optional, Apple Silicon only):
```
mlx>=0.15.0
```

---

## Commit Messages

Follow the convention:
```
feat(vault): add VaultReader for reading notes from disk
feat(vault): add VaultIndex with FAISS vector search
feat(vault): add VaultSearch with hybrid retrieval strategies
feat(vault): add VaultGraph for wikilink knowledge graph
feat(vault): add VaultMOC for Map of Content generation
feat(vault): add QueryParser for natural language queries
feat(vault): add LocalLLM wrapper for Gemma 3 1B
feat(vault): add VaultAnswerer for query answering
feat(vault): add VaultPipeline for end-to-end processing
feat(vault): add VaultLinker for auto-linking notes
test(vault): add reader tests
test(vault): add index tests
test(vault): add search tests
test(vault): add graph tests
```

---

## Integration Point

At merge time (Week 15), Instance A's scripts connect to your modules:

```python
# Instance A's scripts/train.py will call:
from brain.train import load_vault_data
# This reads vault files using vault/reader.py internally

# Instance A's transcribe.py will call:
from vault.pipeline import VaultPipeline
pipeline = VaultPipeline()
pipeline.process(understanding)
```

You do NOT import from `brain/` during parallel build. Only `brain_contract.py`.
