# Parallel Development Guide

## How This Works

Two instances work simultaneously on different parts of the codebase. There should be **zero file conflicts** if both follow these rules.

```
Instance A (brain/)     ←→  brain_contract.py  ←→  Instance B (vault/)
     YOUR CODE              SHARED INTERFACE           YOUR CODE
   no vault files         read-only after W1        no brain files
```

## File Ownership

### Instance A owns (ONLY touches these):
```
brain/
brain/__init__.py
brain/config.py
brain/tokenizer.py
brain/backbone.py
brain/heads.py
brain/model.py
brain/embedder.py
brain/entity_linker.py
brain/data.py
brain/train.py
brain/understand.py
brain/correct.py
brain/tests/
checkpoints/brain/          # model checkpoints
```

### Instance B owns (ONLY touches these):
```
vault/
vault/__init__.py
vault/config.py
vault/schema.py
vault/templates.py
vault/writer.py
vault/reader.py
vault/index.py
vault/search.py
vault/graph.py
vault/linker.py
vault/moc.py
vault/query.py
vault/llm.py
vault/answer.py
vault/pipeline.py
vault/tests/
vault_index/                # FAISS index files
checkpoints/vault/          # LLM model files
```

### Shared (BOTH may edit, with coordination):
```
brain_contract.py           # FREEZE after Week 1
requirements.txt            # Add your deps, don't overwrite others
pyproject.toml              # Add your modules, don't overwrite others
SECOND_BRAIN_PLAN.md        # Reference only, don't modify
```

### OFF LIMITS (Neither instance touches):
```
transcribe.py               # Only modified during merge (Week 9)
audio.py
speech.py
wakeword.py
vad.py
tts.py
hotkey.py
train_wakeword.py
tests/                      # existing tests
```

---

## The Contract: brain_contract.py

Both instances import from this file. It defines:

- `Understanding` — the primary data object brain produces, vault consumes
- `Entity` — extracted entity
- `TemporalInfo` — temporal information
- `QueryResult` — search result
- `Correction` — user correction
- All enums: `NoteType`, `EntityType`, `Priority`, `Urgency`, `Sentiment`
- Constants: `NOTE_TYPE_DIRS`, `TOPICS`, `RELATION_TYPES`

**Rules:**
1. Import from it. Don't copy types.
2. Don't modify it after Week 1.
3. If you need a new type, add a GitHub issue and coordinate.

---

## Git Workflow

### Branching Strategy

```bash
# Phase 1 branches (COMPLETE ✅)
git checkout -b feat/brain-core      # Instance A — merged to main
git checkout -b feat/vault-core      # Instance B — merged to main

# Phase 2 branches
git checkout -b feat/brain-training   # Instance A
git checkout -b feat/vault-retrieval  # Instance B

# Merge to main after testing
git checkout main
git merge feat/brain-training
git merge feat/vault-retrieval
```

### What to commit when

**Instance A commits (Phase 1 — COMPLETE):**
- `feat(brain): add tokenizer` ✅
- `feat(brain): add backbone architecture` ✅
- `feat(brain): add all 10 heads` ✅
- `feat(brain): add synthetic data generator` ✅
- `feat(brain): add training pipeline` ✅
- `feat(brain): add inference API` ✅
- `test(brain): add model tests` ✅

**Instance A commits (Phase 2 — NEXT):**
- `feat(brain): add MetricsTracker for per-head accuracy tracking`
- `feat(brain): add LR scheduling and early stopping`
- `feat(brain): add vault data loader`
- `feat(brain): add data augmentation pipeline`
- `feat(brain): add ONNX export`
- `feat(scripts): add CLI training pipeline`
- `feat(scripts): add CLI evaluation script`
- `test(brain): add training and metrics tests`

**Instance B commits (Phase 1 — COMPLETE):**
- `feat(vault): add config, schema, templates, writer` ✅
- `test(vault): add writer, schema, template tests` ✅

**Instance B commits (Phase 2 — NEXT):**
- `feat(vault): add VaultReader for reading notes from disk`
- `feat(vault): add VaultIndex with FAISS vector search`
- `feat(vault): add VaultSearch with hybrid retrieval`
- `feat(vault): add VaultGraph for wikilink knowledge graph`
- `feat(vault): add VaultMOC for Map of Content`
- `feat(vault): add QueryParser for natural language queries`
- `feat(vault): add LocalLLM wrapper for Gemma 3 1B`
- `feat(vault): add VaultAnswerer for query answering`
- `feat(vault): add VaultPipeline for end-to-end processing`
- `feat(vault): add VaultLinker for auto-linking`
- `test(vault): add reader, index, search, graph tests`

### Conflict prevention

Since each instance has its own directory, there should be **no file conflicts**. The only potential conflicts are:

1. `requirements.txt` — both add deps
   - **Fix:** Each adds their deps in separate commits. Don't overwrite the other's additions.
   
2. `pyproject.toml` — both add module names
   - **Fix:** Each adds their modules to `py-modules` list. Don't overwrite.

3. `brain_contract.py` — both import from it
   - **Fix:** Don't modify it after Week 1.

---

## Integration Points

### Week 1: Interface Alignment

Both instances should:
1. Read `brain_contract.py`
2. Verify they can import all types
3. Agree on the `Understanding` dataclass fields
4. Write one smoke test that produces/consumes an `Understanding`

```bash
# Instance A test:
python -c "
from brain_contract import Understanding, NoteType
u = Understanding(text='test', type=NoteType.TASK)
print(u.to_dict())
"

# Instance B test:
python -c "
from brain_contract import Understanding, NoteType
u = Understanding(text='test', type=NoteType.TASK)
# vault writer should accept this
from vault.writer import VaultWriter
w = VaultWriter('/tmp/test_vault')
path = w.write(u)
print(f'Written to: {path}')
"
```

### Week 5: Embedding Alignment

Both instances use `sentence-transformers/all-MiniLM-L6-v2` with 384-dim output. Verify:
- Instance A's `brain/embedder.py` produces 384-dim vectors
- Instance B's `vault/index.py` expects 384-dim vectors

### Week 9: Full Integration (MERGE)

This is when both instances come together:

```python
# transcribe.py (modified during merge)
from brain.understand import BrainEngine
from vault.pipeline import VaultPipeline

# After transcription...
understanding = brain_engine.understand(transcribed_text)

# If brain is confident, write directly
if understanding.confidence > 0.7:
    vault_pipeline.process(understanding)

# If brain is uncertain, ask the LLM
else:
    llm_understanding = local_llm.analyze(transcribed_text)
    vault_pipeline.process(llm_understanding)
```

---

## Parallel Timeline

### Week 1: Setup
```
Instance A: Create brain/ directory, config.py, tokenizer.py
Instance B: Create vault/ directory, config.py, schema.py
Both: Review brain_contract.py, run smoke tests
```

### Week 2-3: Core
```
Instance A: backbone.py, heads.py, model.py
Instance B: writer.py, reader.py, templates.py
```

### Week 4: Data & Index
```
Instance A: data.py (synthetic generator), train.py
Instance B: index.py (FAISS), graph.py (wikilinks)
```

### Week 5: Search & Inference
```
Instance A: understand.py (main API), embedder.py
Instance B: search.py (hybrid retrieval), linker.py
```

### Week 6-7: Advanced
```
Instance A: entity_linker.py, correct.py
Instance B: query.py, llm.py, answer.py, moc.py
```

### Week 8: Testing & Polish
```
Instance A: Full test suite, edge cases
Instance B: Full test suite, edge cases
```

### Week 9: MERGE
```
Both: Wire brain.understand → vault.pipeline
Both: Modify transcribe.py
Both: End-to-end testing
Both: Update requirements.txt, pyproject.toml
```

---

## Communication Protocol

### When you need something from the other instance:

1. **Check if it exists in `brain_contract.py`** — if the type is defined there, use it
2. **Check the other instance's docs** — `INSTANCE_A_BRAIN.md` or `INSTANCE_B_VAULT.md`
3. **If you need a new shared type** — open a GitHub issue, tag both instances
4. **If you find a bug in brain_contract.py** — both instances must coordinate the fix

### When you make a breaking change:

If you change the shape of `Understanding` or any shared type:
1. Stop. Don't push.
2. Open an issue describing the change
3. Both instances agree
4. Both update their code
5. Both run tests
6. Then push

---

## Quick Reference: Who Calls What

```
transcribe.py
    │
    ├──→ brain.understand.BrainEngine.understand(text)
    │    Returns: Understanding
    │
    ├──→ brain.understand.BrainEngine.should_call_llm(understanding)
    │    Returns: bool
    │
    ├──→ vault.llm.LocalLLM.analyze(text)  [if brain uncertain]
    │    Returns: Understanding
    │
    └──→ vault.pipeline.VaultPipeline.process(understanding)
         Returns: file_path
```

---

## Phase 1 Complete ✅

Brain module (12 modules, 66 tests) and Vault foundation (5 modules, 64 tests) are merged to main.
130 tests passing. Zero conflicts.

---

## Phase 2: Enhancements & Retrieval

### New Plan Files

- **`INSTANCE_A_PLAN.md`** — Brain enhancements + CLI scripts
- **`INSTANCE_B_PLAN.md`** — Vault retrieval + search + reasoning

### New File Ownership (Phase 2)

**Instance A adds:**
```
brain/train.py          # ENHANCE (MetricsTracker, LR scheduling, early stopping)
brain/data.py           # ENHANCE (load_vault_data, DataAugmenter)
brain/understand.py     # ENHANCE (batch optimization)
brain/correct.py        # ENHANCE (retrain integration)
scripts/                # NEW DIRECTORY
scripts/train.py        # NEW: CLI training pipeline
scripts/evaluate.py     # NEW: CLI evaluation script
tests/test_train.py     # NEW
tests/test_correct.py   # NEW
```

**Instance B adds:**
```
vault/reader.py         # NEW: vault file → structured dict
vault/index.py          # NEW: FAISS vector index
vault/search.py         # NEW: hybrid retrieval
vault/graph.py          # NEW: wikilink / backlink graph
vault/moc.py            # NEW: Map of Content
vault/query.py          # NEW: NL query parser
vault/llm.py            # NEW: Gemma 3 1B wrapper
vault/answer.py         # NEW: query answering
vault/pipeline.py       # NEW: end-to-end processor
vault/linker.py         # NEW: auto-linking
vault/tests/test_reader.py   # NEW
vault/tests/test_index.py    # NEW
vault/tests/test_search.py   # NEW
vault/tests/test_graph.py    # NEW
```

### Phase 2 Branches

```bash
# Phase 2 branches
feat/brain-training      # Instance A
feat/vault-retrieval     # Instance B
```

### Phase 2 Timeline

```
Week 10 (parallel):
  Instance A: brain/train.py + brain/data.py
  Instance B: vault/reader.py + vault/index.py

Week 11 (parallel):
  Instance A: scripts/train.py + scripts/evaluate.py
  Instance B: vault/search.py + vault/graph.py

Week 12 (parallel):
  Instance A: brain/correct.py + brain/understand.py
  Instance B: vault/moc.py + vault/query.py

Week 13 (parallel):
  Instance A: tests + first training run
  Instance B: vault/llm.py + vault/answer.py

Week 14 (parallel):
  Instance A: evaluate, tune hyperparams
  Instance B: vault/pipeline.py + vault/linker.py

Week 15 (merge):
  Both: Wire transcribe.py → brain → vault
  Both: End-to-end testing
  Both: CLI + error handling
```

### Phase 2 Commit Checklist

Before starting Phase 2, both instances should:

- [ ] Read `INSTANCE_A_PLAN.md` or `INSTANCE_B_PLAN.md` completely
- [ ] Read the existing modules in your directory (understand current code)
- [ ] Create your branch: `feat/brain-training` or `feat/vault-retrieval`
- [ ] Verify all Phase 1 tests pass: `python -m pytest brain/tests/ vault/tests/ -v`
- [ ] Start building per your plan file
- [ ] Commit with conventional commit messages
- [ ] Push to your branch

Then start building.
