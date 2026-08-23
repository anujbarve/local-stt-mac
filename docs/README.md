# Second Brain 🧠

A fully local, voice-driven personal knowledge system.

Speak naturally → the brain understands → the vault remembers.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Transcribe and understand
python transcribe.py --brain --vault

# Or use the brain CLI directly
python scripts/brain_cli.py understand "I need to fix the VAD threshold"
```

## Architecture

```
Audio Input → Transcription → Brain Understanding → Vault Storage
                                    ↓
                              LLM Fallback (if uncertain)
                                    ↓
                              User Correction (optional)
```

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
- **writer.py** — Understanding → vault file (atomic writes, backups)
- **reader.py** — Vault file → structured dict
- **index.py** — FAISS vector index
- **search.py** — Hybrid retrieval
- **graph.py** — Wikilink knowledge graph
- **moc.py** — Map of Content
- **query.py** — NL query parser
- **llm.py** — Gemma 3 1B wrapper
- **answer.py** — Query answering
- **pipeline.py** — End-to-end processor (safe mode, dry-run, health checks)
- **linker.py** — Auto-linking

## Configuration

Environment variables:
```
VAULT_PATH          # Vault root directory
INDEX_PATH          # FAISS index directory
BRAIN_CHECKPOINT    # Brain model checkpoint
LOG_LEVEL           # Logging verbosity (DEBUG, INFO, WARNING)
VAULT_DRY_RUN       # Dry-run mode (true/false)
```

## Testing

```bash
# Run all tests
python -m pytest brain/tests/ vault/tests/ -v

# Run vault tests only
python -m pytest vault/tests/ -v

# Run integration tests
python -m pytest tests/test_vault_integration.py -v
```
