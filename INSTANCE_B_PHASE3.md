# Instance B: Phase 3 — Vault Integration & Documentation

## Your Mission (Phase 3)

Wire the vault into the main pipeline, add documentation, handle errors gracefully, and polish the system. This is the final phase — make the system production-ready.

**You are the vault. You store. You document. You harden.**

## Rules

1. **DO NOT** modify `brain/` modules — only import from `brain_contract.py`
2. **DO NOT** modify `brain_contract.py`
3. **DO** import from `brain_contract.py` for shared types
4. **DO** write tests for every new module
5. **DO** handle errors gracefully — never crash the pipeline
6. **DO** add docstrings to all public methods

## File Ownership

### Instance B owns (ONLY touches these):
```
vault/pipeline.py       # ENHANCE: add error handling, dry-run mode
vault/writer.py         # ENHANCE: add backup, atomic writes
vault/config.py         # ENHANCE: add new config constants
vault/__init__.py       # ENHANCE: add version, public API exports
vault/tests/            # NEW tests
tests/test_vault_integration.py # NEW: end-to-end vault tests
docs/README.md          # NEW: project documentation
docs/USAGE.md           # NEW: usage guide
docs/ARCHITECTURE.md    # NEW: architecture documentation
```

### Shared (both instances touch during integration):
```
transcribe.py           # BOTH modify: add vault pipeline wiring
requirements.txt        # add new deps
```

### OFF LIMITS:
```
brain/                  # Instance A owns
scripts/                # Instance A owns
```

---

## Build Order

### Step 1: Enhance `vault/config.py`

Add integration and production constants:

```python
# Integration
BRAIN_CHECKPOINT_DIR = os.getenv("BRAIN_CHECKPOINT", os.path.join(os.getcwd(), "checkpoints", "brain"))

# Production
DRY_RUN = os.getenv("VAULT_DRY_RUN", "false").lower() == "true"
BACKUP_ENABLED = True
BACKUP_DIR = os.path.join(VAULT_ROOT, "_backups")
ATOMIC_WRITES = True

# Version
VAULT_VERSION = "0.2.0"
```

---

### Step 2: Enhance `vault/pipeline.py`

Add error handling, dry-run mode, and stats:

```python
class VaultPipeline:
    # ... existing methods ...

    def process_safe(
        self,
        understanding: Understanding,
    ) -> Optional[str]:
        """Process with full error handling. Never raises.

        Returns file_path on success, None on failure.
        Logs errors but continues processing.
        """

    def dry_run(self, understanding: Understanding) -> Dict[str, Any]:
        """Simulate processing without writing anything.

        Returns dict describing what WOULD happen:
        {
            "file_path": "vault/tasks/fix-vad-threshold.md",
            "note_type": "task",
            "entities_linked": ["sarah-chen", "voiceassist"],
            "index_added": True,
            "graph_edges_added": 3,
        }
        """

    def process_batch_safe(
        self,
        understandings: List[Understanding],
    ) -> List[Optional[str]]:
        """Process multiple understandings with error handling.

        Returns list of file_paths (None for failures).
        """

    def health_check(self) -> Dict[str, Any]:
        """Check vault health.

        Returns:
        {
            "vault_exists": True,
            "note_count": 42,
            "index_size": 42,
            "graph_edges": 128,
            "orphan_count": 3,
            "broken_links": 0,
            "disk_usage_mb": 1.2,
        }
        """

    def backup(self, note_path: str) -> Optional[str]:
        """Create a backup before overwriting a note.

        Returns backup path or None if backup disabled.
        """
```

**Error handling pattern:**
```python
def process_safe(self, understanding):
    try:
        return self.process(understanding)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        # Try to save just the text as a fallback
        try:
            return self._fallback_write(understanding)
        except Exception:
            logger.error(f"Fallback write also failed: {e}")
            return None
```

---

### Step 3: Enhance `vault/writer.py`

Add atomic writes and backup:

```python
class VaultWriter:
    # ... existing methods ...

    def write_atomic(self, understanding: Understanding) -> str:
        """Write note atomically (write to temp, then rename).

        Prevents corruption if process crashes mid-write.
        """

    def backup_and_write(self, understanding: Understanding) -> str:
        """Backup existing note before overwriting.
        
        Only backs up if file already exists.
        """

    def write_if_new(self, understanding: Understanding) -> Optional[str]:
        """Write only if a note with this content doesn't already exist.

        Prevents duplicate notes from repeated transcriptions.
        Returns file_path if written, None if already exists.
        """
```

---

### Step 4: Enhance `vault/__init__.py`

Add version and public API exports:

```python
"""
vault — Storage, retrieval, and reasoning layer for the second brain.

Version: 0.2.0
"""

from vault.config import VAULT_ROOT, INDEX_DIR, EMBEDDING_DIM, VAULT_VERSION
from vault.pipeline import VaultPipeline
from vault.reader import VaultReader
from vault.search import VaultSearch

__version__ = VAULT_VERSION
__all__ = [
    "VAULT_ROOT", "INDEX_DIR", "EMBEDDING_DIM", "VAULT_VERSION",
    "VaultPipeline", "VaultReader", "VaultSearch",
]
```

---

### Step 5: Add documentation

#### `docs/README.md`

```markdown
# Second Brain 🧠

A fully local, voice-driven personal knowledge system.

## Quick Start

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Train the brain model
python scripts/train.py --quick

# Transcribe and understand
python transcribe.py --brain --vault

# Or use the brain CLI directly
python scripts/brain_cli.py understand "I need to fix the VAD threshold"
\`\`\`

## Architecture

\`\`\`
Audio Input → Transcription → Brain Understanding → Vault Storage
                                    ↓
                              LLM Fallback (if uncertain)
                                    ↓
                              User Correction (optional)
\`\`\`

## Modules

### Brain (`brain/`)
- **config.py** — Hyperparameters and constants
- **tokenizer.py** — Custom word-level tokenizer
- **backbone.py** — 4-layer transformer encoder
- **heads.py** — 10 task-specific heads
- **model.py** — Multi-task model
- **embedder.py** — Sentence embeddings (384-dim)
- **data.py** — Synthetic data generator
- **train.py** — Training pipeline with metrics
- **understand.py** — Main inference API
- **entity_linker.py** — Entity resolution
- **correct.py** — Correction logging

### Vault (`vault/`)
- **config.py** — Paths and settings
- **schema.py** — Note validation
- **templates.py** — Markdown renderers
- **writer.py** — Understanding → vault file
- **reader.py** — Vault file → structured dict
- **index.py** — FAISS vector index
- **search.py** — Hybrid retrieval
- **graph.py** — Wikilink knowledge graph
- **moc.py** — Map of Content
- **query.py** — NL query parser
- **llm.py** — Gemma 3 1B wrapper
- **answer.py** — Query answering
- **pipeline.py** — End-to-end processor
- **linker.py** — Auto-linking

### Scripts
- **scripts/train.py** — CLI training pipeline
- **scripts/evaluate.py** — CLI evaluation
- **scripts/brain_cli.py** — Unified brain CLI

## Configuration

Environment variables:
\`\`\`
VAULT_PATH          # Vault root directory
INDEX_PATH          # FAISS index directory
BRAIN_CHECKPOINT    # Brain model checkpoint
LOG_LEVEL           # Logging verbosity (DEBUG, INFO, WARNING)
VAULT_DRY_RUN       # Dry-run mode (true/false)
\`\`\`

## Testing

\`\`\`bash
# Run all tests
python -m pytest brain/tests/ vault/tests/ -v

# Run brain tests only
python -m pytest brain/tests/ -v

# Run vault tests only
python -m pytest vault/tests/ -v
\`\`\`
```

#### `docs/USAGE.md`

```markdown
# Usage Guide

## Voice Mode

\`\`\`bash
# Start with brain understanding + vault writing
python transcribe.py --brain --vault

# With interactive corrections
python transcribe.py --brain --vault --interactive

# Verbose mode (shows brain reasoning)
python transcribe.py --brain --vault --verbose
\`\`\`

## CLI Mode

\`\`\`bash
# Quick inference
python scripts/brain_cli.py understand "Meeting with Sarah about the project"

# JSON output
python scripts/brain_cli.py understand "TODO: fix the VAD threshold" --json

# Interactive correction mode
python scripts/brain_cli.py interactive

# Batch processing
echo '["text1", "text2"]' > input.json
python scripts/brain_cli.py batch input.json --output results.json
\`\`\`

## Training

\`\`\`bash
# Generate synthetic data and train
python scripts/train.py --synthetic 15000 --epochs 20

# Quick test run
python scripts/train.py --quick

# Train from vault data
python scripts/train.py --vault vault/

# Export to ONNX
python scripts/train.py --export-onnx
\`\`\`

## Evaluation

\`\`\`bash
# Evaluate on test set
python scripts/evaluate.py

# With confusion matrix
python scripts/evaluate.py --confusion --per-type

# Save report
python scripts/evaluate.py --output reports/eval.json
\`\`\`

## Search

\`\`\`bash
# From Python
from vault.pipeline import VaultPipeline
pipeline = VaultPipeline()
results = pipeline.query("What projects am I working on?")

# From CLI (if available)
python -c "from vault.pipeline import VaultPipeline; print(VaultPipeline().health_check())"
\`\`\`
```

#### `docs/ARCHITECTURE.md`

```markdown
# Architecture

## Data Flow

\`\`\`
1. Audio Input
   ↓
2. Speech Recognition (Whisper)
   ↓
3. Brain Understanding (Neural Network)
   - Classification (29 note types)
   - Entity Extraction (BIO tagging)
   - Priority/Urgency/Sentiment
   - Topic Assignment
   - Confidence Scoring
   ↓
4. Decision: Confident Enough?
   - Yes → Use NN understanding
   - No → Route to LLM (Gemma 3 1B)
   ↓
5. Vault Processing
   - Write markdown note
   - Update search index
   - Update knowledge graph
   - Auto-link to related notes
   ↓
6. Optional: User Correction
   - Log correction for retraining
   - Update note with corrections
\`\`\`

## Model Architecture

\`\`\`
Input: token_ids [batch, 512]
  ↓
Embedding(30000 → 256) + PositionalEncoding(512, 256)
  ↓
TransformerBlock × 4 (pre-norm, 4 heads, FF=1024)
  ↓
├── AttentionPool → context [256]
└── Token outputs → tokens [512, 256]
      ↓
context → classifier, priority, urgency, sentiment, action, temporal, topics
tokens  → entity_tagger, summary
\`\`\`

## Loss Weights

| Head | Weight | Loss |
|------|--------|------|
| Classifier | 1.0 | CrossEntropy |
| Entity | 2.0 | CrossEntropy (BIO) |
| Priority | 1.0 | MSE |
| Urgency | 1.0 | CrossEntropy |
| Sentiment | 1.0 | CrossEntropy |
| Action | 1.0 | CrossEntropy |
| Summary | 0.5 | BinaryCE |
| Temporal | 1.0 | BinaryCE |
| Topics | 0.5 | BinaryCE (multi-label) |
```

---

### Step 6: Add integration to `transcribe.py`

**Instance B adds the vault wiring:**

In `transcribe.py`, add imports at top:

```python
from vault.pipeline import VaultPipeline
from vault.config import VAULT_ROOT
```

In `run_pipeline()`, initialize vault:

```python
# Initialize vault pipeline (if enabled)
vault_pipeline = None
if args.vault:
    try:
        vault_pipeline = VaultPipeline(vault_root=args.vault_path)
        if args.verbose:
            print(f"[Debug] Vault initialized at {args.vault_path}")
    except Exception as e:
        print(f"[Warning] Vault initialization failed: {e}", file=sys.stderr)
```

After brain understanding, write to vault:

```python
# State 4.6: Vault writing
if vault_pipeline and understanding:
    try:
        file_path = vault_pipeline.process_safe(understanding)
        if file_path:
            print(f"[Vault] Saved: {file_path}")
        else:
            print("[Vault] Write failed (see logs)")
    except Exception as e:
        if args.verbose:
            print(f"[Debug] Vault error: {e}", file=sys.stderr)
```

---

### Step 7: Tests

#### `vault/tests/test_pipeline_enhanced.py`

```python
class TestVaultPipelineEnhanced:
    def test_process_safe_success(self, tmp_vault):
        """process_safe returns path on success."""

    def test_process_safe_failure(self, tmp_vault):
        """process_safe returns None on failure, doesn't crash."""

    def test_dry_run(self, tmp_vault):
        """dry_run returns plan without writing."""

    def test_health_check(self, tmp_vault):
        """health_check returns vault stats."""

    def test_backup(self, tmp_vault):
        """backup creates backup before overwrite."""

    def test_process_batch_safe(self, tmp_vault):
        """batch processing with mixed success/failure."""
```

#### `tests/test_vault_integration.py`

```python
class TestVaultIntegration:
    def test_read_write_roundtrip(self, tmp_vault):
        """Write a note, read it back, verify contents."""

    def test_search_after_write(self, tmp_vault):
        """Write a note, search for it, find it."""

    def test_graph_after_write(self, tmp_vault):
        """Write a note with entities, verify graph updated."""

    def test_pipeline_end_to_end(self, tmp_vault):
        """Full pipeline: Understanding → write → index → search."""

    def test_error_handling(self, tmp_vault):
        """Pipeline handles errors gracefully."""
```

---

## What Instance A Needs From You

Instance A will wire the brain side of transcribe.py. They need you to:

1. **Define the pipeline interface** — `vault_pipeline.process(understanding)` works
2. **Define the safe interface** — `vault_pipeline.process_safe(understanding)` never crashes
3. **Define the health check** — `vault_pipeline.health_check()` returns stats

These are already in your existing modules. Just enhance them.

## Commit Messages

```
feat(vault): add integration constants to config
feat(vault): add process_safe, dry_run, health_check to pipeline
feat(vault): add atomic writes and backup to writer
feat(vault): add version and public API exports
docs: add README, USAGE, and ARCHITECTURE documentation
feat(transcribe): wire vault pipeline into transcription flow
feat(vault): add error handling and logging
test(vault): add pipeline and integration tests
```
