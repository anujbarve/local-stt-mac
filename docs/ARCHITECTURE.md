# Architecture

## Data Flow

```
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
   - Write markdown note (atomic writes)
   - Update search index (FAISS)
   - Update knowledge graph (wikilinks)
   - Auto-link to related notes
   ↓
6. Optional: User Correction
   - Log correction for retraining
   - Update note with corrections
```

## Model Architecture

```
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
```

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

## Vault Pipeline

The vault pipeline is the main entry point for storing understood notes:

```python
from vault.pipeline import VaultPipeline

pipeline = VaultPipeline()

# Standard processing
path = pipeline.process(understanding)

# Safe mode (never crashes)
path = pipeline.process_safe(understanding)

# Dry run (no writes)
plan = pipeline.dry_run(understanding)

# Health check
health = pipeline.health_check()
```

### Error Handling

- `process_safe()` catches all exceptions and returns `None` on failure
- Falls back to writing raw text to daily journal if main write fails
- Atomic writes prevent corruption on crash
- Backups created before overwriting existing notes
