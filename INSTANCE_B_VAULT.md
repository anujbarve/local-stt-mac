# Instance B: The Vault (`vault/`)

## Your Mission

Build the storage, retrieval, and reasoning layer. You consume `Understanding` objects from the brain (Instance A) and write them to a markdown vault. You also handle queries, search, and the LLM.

**You are the memory. You store. You retrieve. You connect. You reason.**

## Your Directory

```
vault/
├── __init__.py
├── config.py          # Vault paths, settings
├── schema.py          # Entity/field definitions, validation
├── templates.py       # Note templates for all 21 domain types
├── writer.py          # Understanding dict → markdown file
├── reader.py          # Markdown file → structured dict
├── index.py           # FAISS index builder and manager
├── search.py          # Hybrid retrieval (structural + semantic + graph)
├── graph.py           # Wikilink/backlink graph traversal
├── linker.py          # Auto-linking (entity resolution across notes)
├── moc.py             # Map of Content auto-generation
├── query.py           # Natural language → structured query parser
├── llm.py             # Gemma 3 1B integration via MLX
├── answer.py          # Query answering with vault context
├── pipeline.py        # End-to-end: Understanding → vault write
├── tests/
    ├── test_writer.py
    ├── test_reader.py
    ├── test_search.py
    ├── test_graph.py
    ├── test_query.py
    ├── test_pipeline.py
    └── test_llm.py
```

## Rules

1. **DO NOT** touch `brain/`, `transcribe.py`, or any existing voiceassist module
2. **DO NOT** modify `brain_contract.py` after initial setup
3. **DO** import types from `brain_contract.py` — never redefine them
4. **DO** keep your modules self-contained — no side effects on import
5. **DO** write tests for every module
6. **ONLY YOU** may modify `transcribe.py` (during Phase 4 wiring)

## Shared Types

Import from `brain_contract.py` at the project root:

```python
from brain_contract import (
    Understanding, Entity, TemporalInfo,
    NoteType, EntityType, Priority, Urgency, Sentiment,
    QueryResult, Correction,
    NOTE_TYPE_DIRS, ENTITY_PAGE_TYPES, TOPICS, RELATION_TYPES,
)
```

## Module Specifications

---

### 1. `vault/config.py` — Configuration

```python
"""Vault configuration."""

import os

# Paths
VAULT_ROOT = os.getenv("VAULT_PATH", os.path.join(os.getcwd(), "vault"))
INDEX_DIR = os.path.join(os.getcwd(), "vault_index")
CHECKPOINT_DIR = os.path.join(os.getcwd(), "checkpoints", "vault")

# FAISS settings
EMBEDDING_DIM = 384          # must match brain/embedder.py
FAISS_INDEX_TYPE = "flat"    # "flat" or "ivf" for larger vaults

# LLM settings
LLM_MODEL = "google/gemma-3-1b-it"
LLM_MLX_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "gemma-3-1b")
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.3

# Search settings
SEARCH_TOP_K = 10            # max results to return
SEARCH_MIN_SCORE = 0.3       # minimum similarity score
SEARCH_EXPAND_THRESHOLD = 5  # if fewer results, expand search

# Auto-linking settings
LINK_CONFIDENCE_THRESHOLD = 0.75  # minimum entity match confidence
AUTO_LINK_MAX_DISTANCE = 3        # max graph hops for link suggestions

# Daily journal
DAILY_DIR = os.path.join(VAULT_ROOT, "daily")
```

---

### 2. `vault/schema.py` — Schema Validation

**Purpose:** Define and validate the structure of notes, entities, and the vault.

```python
class NoteSchema:
    """Defines the valid frontmatter fields for each note type."""
    
    REQUIRED_FIELDS = ["id", "type", "created", "updated", "source"]
    
    TYPE_SPECIFIC_FIELDS = {
        NoteType.TASK: ["status", "priority", "urgency", "due_date", "project"],
        NoteType.PROJECT_UPDATE: ["status", "priority", "progress", "people"],
        NoteType.PERSON_INFO: ["name", "relationship", "birthday", "organization"],
        NoteType.IDEA: ["stage", "origin", "potential_impact"],
        NoteType.LEARNING: ["source_type", "source_title", "confidence"],
        NoteType.DECISION: ["decision", "rationale", "alternatives", "reversible"],
        NoteType.MEETING: ["date", "participants", "type", "project"],
        NoteType.REMINDER: ["reminder_date", "recurrence", "completed"],
        NoteType.PREFERENCE: ["category", "item", "sentiment"],
        # ... etc for all 21 types
    }
    
    @classmethod
    def validate(cls, frontmatter: Dict, note_type: NoteType) -> List[str]:
        """Validate frontmatter. Returns list of errors (empty = valid)."""
        ...
    
    @classmethod
    def generate_id(cls) -> str:
        """Generate a unique note ID: note_YYYY_MM_DD_NNN"""
        ...
    
    @classmethod
    def slugify(cls, text: str) -> str:
        """Convert text to URL-safe filename: 'Fix VAD Bug' → 'fix-vad-bug'"""
        ...


class VaultValidator:
    """Validate the entire vault structure."""
    
    def validate_all(self, vault_path: str) -> Dict[str, List[str]]:
        """Scan all notes and return validation errors."""
        ...
    
    def find_orphans(self, vault_path: str) -> List[str]:
        """Find notes with no backlinks."""
        ...
    
    def find_broken_links(self, vault_path: str) -> List[str]:
        """Find wikilinks that point to nonexistent notes."""
        ...
```

---

### 3. `vault/templates.py` — Note Templates

**Purpose:** Generate YAML frontmatter + markdown body for each note type.

```python
class NoteTemplate:
    """Generate note content from an Understanding object."""
    
    @staticmethod
    def render(understanding: Dict) -> str:
        """
        Render a complete markdown note from an Understanding dict.
        
        Returns:
            Complete .md file content with frontmatter and body.
        """
        ...
    
    @staticmethod
    def render_frontmatter(understanding: Dict) -> str:
        """Render just the YAML frontmatter block."""
        ...
    
    @staticmethod
    def render_body(understanding: Dict) -> str:
        """Render the markdown body based on note type."""
        ...


# Per-type renderers
class TaskTemplate(NoteTemplate):
    @staticmethod
    def render_body(u: Dict) -> str:
        return f"""# {u.get('action', u['text'][:80])}

{u.get('summary', u['text'])}

## Details
- **Status:** todo
- **Priority:** {u.get('priority_label', 'medium')}
- **Urgency:** {u.get('urgency', 'later')}
"""

class PersonTemplate(NoteTemplate):
    @staticmethod
    def render_body(u: Dict) -> str:
        entities = u.get("entities", [])
        person = next((e for e in entities if e.get("type") == "person"), None)
        name = person["name"] if person else "Unknown"
        return f"""# {name}

## Information
{_render_entity_attributes(entities)}

## Activity
- First mentioned: {u.get('timestamp', 'unknown')}
"""
# ... similar renderers for all 21 types
```

---

### 4. `vault/writer.py` — Note Writer

**Purpose:** Take an `Understanding` object and write it to the vault as a markdown file.

**This is the primary bridge between brain and vault.**

```python
class VaultWriter:
    """Write Understanding objects to the vault."""
    
    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self._ensure_directories()
    
    def write(self, understanding: Understanding) -> str:
        """
        Write a note to the vault.
        
        Args:
            understanding: The Understanding object from the brain.
        
        Returns:
            File path of the created note.
        
        Steps:
            1. Determine target directory from note type
            2. Generate filename from content
            3. Render markdown with frontmatter
            4. Write file
            5. Update backlinks for any linked entities
            6. Return file path
        """
        ...
    
    def write_daily_entry(self, understanding: Understanding, 
                          date: Optional[str] = None) -> str:
        """Append a transcription to the daily journal."""
        ...
    
    def update_note(self, file_path: str, updates: Dict) -> None:
        """Update specific fields in an existing note."""
        ...
    
    def _determine_directory(self, note_type: NoteType) -> str:
        """Map note type to vault subdirectory."""
        subdir = NOTE_TYPE_DIRS.get(note_type, "daily")
        return os.path.join(self.vault_root, subdir)
    
    def _generate_filename(self, understanding: Understanding) -> str:
        """Generate a descriptive filename."""
        # e.g., "fix-vad-threshold.md" or "sarah-chen-birthday.md"
        ...
    
    def _ensure_directories(self) -> None:
        """Create vault directory structure if it doesn't exist."""
        dirs = set(NOTE_TYPE_DIRS.values())
        dirs.add("_archive")
        dirs.add("topics")
        for d in dirs:
            os.makedirs(os.path.join(self.vault_root, d), exist_ok=True)
    
    def _update_backlinks(self, understanding: Understanding, 
                          note_path: str) -> None:
        """Add backlinks from entity pages to this note."""
        ...


# Example output for a task note:
WRITER_EXAMPLE = """---
id: note_2026_08_23_001
type: task
created: 2026-08-23T14:30:00Z
updated: 2026-08-23T14:30:00Z
source: audio
transcription_id: null

# Classification
priority: high
urgency: soon
sentiment: neutral
confidence: 0.87

# Entities
people: []
projects:
  - voiceassist
tasks:
  - fix-vad-threshold
tools:
  - silero-vad
dates:
  - 2026-08-29

# Organization
tags:
  - bug
  - vad
  - audio
topics:
  - technology
  - engineering
category: technical

# Links
related_notes:
  - note_2026_08_20_003
backlinks: []

# Embedding
embedding_id: vec_abc123
---

# Fix VAD Threshold

The Silero VAD threshold is too sensitive, causing false positives
during wake word detection.

## Action
Update `vad.py` line 42: increase threshold from 0.5 to 0.65

## Context
Related to the latency issue discussed in sprint planning on 2026-08-20.
"""
```

---

### 5. `vault/reader.py` — Note Reader

**Purpose:** Parse markdown files back into structured dicts. The reverse of writer.

```python
class VaultReader:
    """Read and parse notes from the vault."""
    
    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
    
    def read(self, file_path: str) -> Dict[str, Any]:
        """
        Read a markdown file and return parsed structure.
        
        Returns:
            {
                "frontmatter": { ... },  # YAML dict
                "body": "...",           # markdown body text
                "raw": "...",            # full file content
                "file_path": "vault/tasks/fix-vad-threshold.md"
            }
        """
        ...
    
    def read_all(self, subdirectory: Optional[str] = None) -> List[Dict]:
        """Read all notes in vault or a subdirectory."""
        ...
    
    def read_by_type(self, note_type: NoteType) -> List[Dict]:
        """Read all notes of a specific type."""
        ...
    
    def read_by_id(self, note_id: str) -> Optional[Dict]:
        """Find and read a note by its ID."""
        ...
    
    def search_frontmatter(self, field: str, value: Any) -> List[Dict]:
        """Search notes by frontmatter field value."""
        ...
    
    def get_frontmatter(self, file_path: str) -> Dict[str, Any]:
        """Extract just the YAML frontmatter."""
        ...
    
    def get_wikilinks(self, file_path: str) -> List[str]:
        """Extract all [[wikilinks]] from a note."""
        ...
    
    def get_backlinks(self, file_path: str) -> List[Dict]:
        """Find all notes that link TO this note."""
        ...
```

---

### 6. `vault/index.py` — FAISS Vector Index

**Purpose:** Build and manage the FAISS index for semantic search.

```python
class VaultIndex:
    """FAISS-based vector index for the vault."""
    
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.index: Optional[faiss.Index] = None
        self.id_map: List[str] = []  # FAISS position → note file path
        self._load_or_create()
    
    def add_note(self, file_path: str, embedding: List[float]) -> None:
        """Add a note's embedding to the index."""
        ...
    
    def remove_note(self, file_path: str) -> None:
        """Remove a note from the index."""
        ...
    
    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar notes. Returns [(file_path, score), ...]"""
        ...
    
    def rebuild(self, vault_root: str = VAULT_ROOT) -> None:
        """Rebuild the entire index from vault files."""
        ...
    
    def save(self) -> None:
        """Save index to disk."""
        ...
    
    def _load_or_create(self) -> None:
        """Load existing index or create new one."""
        ...
    
    def stats(self) -> Dict[str, int]:
        """Index statistics."""
        ...
```

---

### 7. `vault/search.py` — Hybrid Retrieval

**Purpose:** The main search engine. Combines structural, semantic, and graph search.

```python
class VaultSearch:
    """Multi-strategy search across the vault."""
    
    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.reader = VaultReader(vault_root)
        self.index = VaultIndex()
        self.graph = VaultGraph(vault_root)
        self.embedder = None  # lazy load from brain
    
    def search(self, query: str, top_k: int = 10) -> QueryResult:
        """
        Main search entry point.
        
        Strategies:
            1. Structural: parse query for entity type + field filters
            2. Semantic: embed query, FAISS nearest neighbors
            3. Graph: follow wikilinks from seed notes
        
        Returns QueryResult with merged, ranked results.
        """
        ...
    
    def structural_search(self, query: str) -> List[Dict]:
        """Search by parsing frontmatter fields."""
        # "Which projects are active?" → search projects/ WHERE status=active
        ...
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Embedding similarity search via FAISS."""
        ...
    
    def graph_search(self, seed_notes: List[str], hops: int = 2) -> List[Dict]:
        """Expand results by following wikilinks."""
        ...
    
    def entity_search(self, entity_name: str) -> List[Dict]:
        """Find all notes mentioning a specific entity."""
        ...
    
    def temporal_search(self, time_expression: str) -> List[Dict]:
        """Search by time: 'today', 'this week', 'last month'."""
        ...
    
    def _merge_results(self, *result_lists: List[Dict]) -> List[Dict]:
        """Merge and deduplicate results from multiple strategies."""
        ...
    
    def _rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Rank results by relevance."""
        ...
```

---

### 8. `vault/graph.py` — Knowledge Graph

**Purpose:** Maintain the wikilink/backlink graph for traversal.

```python
class VaultGraph:
    """Knowledge graph based on wikilinks."""
    
    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.adjacency: Dict[str, Set[str]] = {}  # note_path → set of linked notes
        self._build_graph()
    
    def add_link(self, from_note: str, to_note: str) -> None:
        """Add a directed link between notes."""
        ...
    
    def remove_link(self, from_note: str, to_note: str) -> None:
        """Remove a link."""
        ...
    
    def get_neighbors(self, note_path: str, hops: int = 1) -> List[str]:
        """Get notes within N hops."""
        ...
    
    def get_backlinks(self, note_path: str) -> List[str]:
        """Get all notes that link to this note."""
        ...
    
    def find_path(self, from_note: str, to_note: str) -> Optional[List[str]]:
        """Find shortest path between two notes (BFS)."""
        ...
    
    def find_clusters(self) -> List[List[str]]:
        """Find connected components (topic clusters)."""
        ...
    
    def get_orphans(self) -> List[str]:
        """Find notes with no links in or out."""
        ...
    
    def _build_graph(self) -> None:
        """Scan all notes and build adjacency list from wikilinks."""
        ...
    
    def stats(self) -> Dict[str, int]:
        """Graph statistics: nodes, edges, clusters, orphans."""
        ...
```

---

### 9. `vault/linker.py` — Auto-Linking

**Purpose:** Automatically link new notes to existing notes via entity resolution.

```python
class AutoLinker:
    """Automatically create links between notes."""
    
    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.reader = VaultReader(vault_root)
        self.entity_index: Dict[str, List[str]] = {}  # entity_name → [note_paths]
        self._build_entity_index()
    
    def find_related_notes(self, understanding: Understanding) -> List[str]:
        """
        Given a new Understanding, find existing notes it should link to.
        
        Strategies:
            1. Entity overlap: notes that mention the same people/tools/projects
            2. Topic overlap: notes with matching topic tags
            3. Embedding similarity: notes with similar content
            4. Temporal proximity: notes from the same time period
        """
        ...
    
    def create_links(self, new_note_path: str, 
                     related_notes: List[str]) -> None:
        """Add wikilinks between the new note and related notes."""
        ...
    
    def suggest_links(self, file_path: str) -> List[Dict[str, Any]]:
        """Suggest links for an existing note (for manual review)."""
        ...
    
    def _build_entity_index(self) -> None:
        """Scan vault and index all entities mentioned in notes."""
        ...
```

---

### 10. `vault/moc.py` — Map of Content

**Purpose:** Auto-generate and maintain hub/index notes.

```python
class MOCGenerator:
    """Generate and maintain Maps of Content."""
    
    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.moc_dir = os.path.join(vault_root, "topics")
    
    def generate_topic_moc(self, topic: str) -> str:
        """
        Generate or update a MOC for a topic.
        
        Scans all notes with matching topic tags,
        creates a hub note with categorized links.
        """
        ...
    
    def generate_all_mocs(self) -> List[str]:
        """Regenerate all topic MOCs from current vault state."""
        ...
    
    def update_index(self) -> None:
        """Update the master _index.md with all MOC links."""
        ...
    
    def _categorize_notes(self, notes: List[Dict]) -> Dict[str, List[Dict]]:
        """Group notes by type within a topic."""
        ...
```

---

### 11. `vault/query.py` — Query Parser

**Purpose:** Parse natural language queries into structured search operations.

```python
class QueryParser:
    """Parse natural language queries into structured operations."""
    
    def parse(self, query: str) -> Dict[str, Any]:
        """
        Parse a query into structured form.
        
        Examples:
            "What projects are active?"
            → {"intent": "filter", "type": "project", "filters": {"status": "active"}}
            
            "When is Sarah's birthday?"
            → {"intent": "lookup", "entity": "Sarah", "field": "birthday"}
            
            "What did I learn about neural networks?"
            → {"intent": "search", "type": "learning", "topic": "neural networks"}
            
            "Summarize my week"
            → {"intent": "aggregate", "time": "this_week", "action": "summarize"}
            
            "How does this connect to my other ideas?"
            → {"intent": "graph", "seed": "current_note", "action": "find_connections"}
        """
        ...
    
    def _detect_intent(self, query: str) -> str:
        """Classify query intent: filter, lookup, search, aggregate, graph, action."""
        ...
    
    def _detect_entity_type(self, query: str) -> Optional[str]:
        """Detect what type of entity the query is about."""
        ...
    
    def _detect_temporal(self, query: str) -> Optional[str]:
        """Detect time expressions: today, this week, last month, etc."""
        ...
    
    def _detect_filters(self, query: str) -> Dict[str, Any]:
        """Extract filter conditions from query."""
        ...
```

---

### 12. `vault/llm.py` — Gemma 3 1B Integration

**Purpose:** Local LLM for deep understanding, query answering, and knowledge distillation.

```python
class LocalLLM:
    """Gemma 3 1B via MLX for local inference."""
    
    def __init__(self, model_path: Optional[str] = None):
        import mlx_lm
        self.model_path = model_path or LLM_MLX_MODEL_PATH
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def _ensure_loaded(self) -> None:
        """Lazy load model on first use."""
        if not self._loaded:
            import mlx_lm
            self.model, self.tokenizer = mlx_lm.load(self.model_path)
            self._loaded = True
    
    def generate(self, prompt: str, max_tokens: int = LLM_MAX_TOKENS,
                 temperature: float = LLM_TEMPERATURE) -> str:
        """Generate text from a prompt."""
        self._ensure_loaded()
        ...
    
    def analyze(self, text: str, context: Optional[List[Dict]] = None) -> Understanding:
        """
        Deep analysis of text using the LLM.
        
        This is the 'teacher' for knowledge distillation.
        Produces a full Understanding with higher accuracy than the NN.
        """
        prompt = self._build_analysis_prompt(text, context)
        response = self.generate(prompt)
        return self._parse_analysis_response(response)
    
    def answer(self, query: str, vault_context: List[Dict]) -> str:
        """
        Answer a question using vault context.
        
        Used for complex queries that need reasoning over multiple notes.
        """
        prompt = self._build_answer_prompt(query, vault_context)
        return self.generate(prompt)
    
    def summarize_notes(self, notes: List[Dict]) -> str:
        """Summarize multiple notes into a coherent overview."""
        ...
    
    def find_connections(self, note: Dict, candidates: List[Dict]) -> List[Dict]:
        """Discover non-obvious connections between notes."""
        ...
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Deep entity extraction (better than NN for complex cases)."""
        ...
    
    def _build_analysis_prompt(self, text: str, context: Optional[List[Dict]]) -> str: ...
    def _build_answer_prompt(self, query: str, vault_context: List[Dict]) -> str: ...
    def _parse_analysis_response(self, response: str) -> Understanding: ...
```

---

### 13. `vault/answer.py` — Query Answering

**Purpose:** Take a parsed query, retrieve context, and generate an answer.

```python
class QueryAnswerer:
    """Answer natural language queries about the vault."""
    
    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.search = VaultSearch(vault_root)
        self.llm = LocalLLM()
        self.parser = QueryParser()
    
    def answer(self, query: str) -> str:
        """
        Main entry point for answering questions.
        
        1. Parse the query
        2. Retrieve relevant notes
        3. For simple queries: return structured result
        4. For complex queries: use LLM to synthesize answer
        """
        parsed = self.parser.parse(query)
        
        # Simple structural queries
        if parsed["intent"] == "filter":
            results = self.search.structural_search(query)
            return self._format_list_answer(query, results)
        
        if parsed["intent"] == "lookup":
            results = self.search.entity_search(parsed.get("entity", ""))
            return self._format_lookup_answer(parsed, results)
        
        # Complex queries → LLM
        results = self.search.search(query, top_k=10)
        return self.llm.answer(query, results.notes)
    
    def _format_list_answer(self, query: str, results: List[Dict]) -> str: ...
    def _format_lookup_answer(self, parsed: Dict, results: List[Dict]) -> str: ...
```

---

### 14. `vault/pipeline.py` — End-to-End Pipeline

**Purpose:** Wire everything together. This is the entry point that `transcribe.py` will call.

```python
class VaultPipeline:
    """
    End-to-end pipeline: Understanding → vault.
    
    This is what transcribe.py calls after the brain processes a transcription.
    """
    
    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.writer = VaultWriter(vault_root)
        self.linker = AutoLinker(vault_root)
        self.index = VaultIndex()
        self.moc = MOCGenerator(vault_root)
    
    def process(self, understanding: Understanding) -> str:
        """
        Process a Understanding object and write to vault.
        
        Steps:
            1. Write the note to the vault
            2. Find and create links to related notes
            3. Update the FAISS index with the new embedding
            4. Update MOCs if new topics appeared
            5. Return the file path
        """
        # 1. Write note
        note_path = self.writer.write(understanding)
        
        # 2. Auto-link
        related = self.linker.find_related_notes(understanding)
        if related:
            self.linker.create_links(note_path, related)
        
        # 3. Update index
        if understanding.embedding:
            self.index.add_note(note_path, understanding.embedding)
        
        # 4. Update MOCs (every 10 notes to avoid overhead)
        # (handled lazily)
        
        return note_path
    
    def rebuild_index(self) -> None:
        """Rebuild the entire FAISS index from vault."""
        ...
    
    def rebuild_mocs(self) -> None:
        """Regenerate all Maps of Content."""
        ...
    
    def vault_stats(self) -> Dict[str, Any]:
        """Get vault statistics."""
        ...
```

---

## Testing

Every module must have tests. Run with:

```bash
python -m pytest vault/tests/ -v
```

### Test requirements:

| Module | What to test |
|---|---|
| `schema.py` | Validation, ID generation, slugify |
| `templates.py` | Render for each note type, frontmatter validity |
| `writer.py` | Write note, correct directory, backlink creation |
| `reader.py` | Parse frontmatter, extract wikilinks, roundtrip |
| `index.py` | Add/search/remove, save/load, rebuild |
| `search.py` | Structural/semantic/graph search, merge results |
| `graph.py` | Add/remove links, path finding, clusters, orphans |
| `linker.py` | Entity resolution, related note discovery |
| `moc.py` | Topic MOC generation, index update |
| `query.py` | Parse various query types |
| `llm.py` | Generate, analyze (mock), answer (mock) |
| `answer.py` | Answer simple and complex queries |
| `pipeline.py` | End-to-end: Understanding → vault file |

### Quick smoke test:

```bash
python -c "
from vault.pipeline import VaultPipeline
from brain_contract import Understanding, NoteType, Entity, EntityType

# Simulate a brain output
u = Understanding(
    text='I need to fix the VAD threshold by Friday',
    type=NoteType.TASK,
    priority=0.82,
    entities=[
        Entity(name='VAD', entity_type=EntityType.TOOL),
        Entity(name='Friday', entity_type=EntityType.DATE),
    ],
    action='fix VAD threshold',
    topics=['technology'],
    embedding=[0.1] * 384,
)

pipeline = VaultPipeline()
path = pipeline.process(u)
print(f'Written to: {path}')
"
```

---

## What You Consume

Instance A gives you an `Understanding` object like this:

```python
Understanding(
    text="I need to fix the VAD threshold by Friday",
    type=NoteType.TASK,
    priority=0.82,
    urgency=Urgency.SOON,
    entities=[
        Entity(name="VAD", entity_type=EntityType.TOOL),
        Entity(name="Friday", entity_type=EntityType.DATE),
    ],
    action="fix VAD threshold",
    topics=["technology", "engineering"],
    embedding=[0.23, -0.11, ...],  # 384 floats
    ...
)
```

You turn that into a markdown file at `vault/tasks/fix-vad-threshold.md`, link it to existing notes, index it for search, and update the knowledge graph. That's your job.
