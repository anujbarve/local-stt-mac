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
# Each instance works on their own branch
git checkout -b feat/brain-core      # Instance A
git checkout -b feat/vault-core      # Instance B

# Merge to main only after testing
git checkout main
git merge feat/brain-core
git merge feat/vault-core
```

### What to commit when

**Instance A commits:**
- `feat(brain): add tokenizer` 
- `feat(brain): add backbone architecture`
- `feat(brain): add all 10 heads`
- `feat(brain): add synthetic data generator`
- `feat(brain): add training pipeline`
- `feat(brain): add inference API`
- `test(brain): add model tests`

**Instance B commits:**
- `feat(vault): add note writer`
- `feat(vault): add note reader`
- `feat(vault): add FAISS index`
- `feat(vault): add hybrid search`
- `feat(vault): add knowledge graph`
- `feat(vault): add auto-linker`
- `feat(vault): add LLM integration`
- `test(vault): add writer tests`

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

## First Commit Checklist

Before starting work, both instances should:

- [ ] Read `brain_contract.py` completely
- [ ] Read `INSTANCE_A_BRAIN.md` or `INSTANCE_B_VAULT.md` completely
- [ ] Read `SECOND_BRAIN_PLAN.md` for context
- [ ] Create your branch: `feat/brain-core` or `feat/vault-core`
- [ ] Create your directory: `mkdir brain` or `mkdir vault`
- [ ] Write `__init__.py` with module docstring
- [ ] Run the smoke test from your instance doc
- [ ] Commit: `chore: initialize brain/ module` or `chore: initialize vault/ module`
- [ ] Push to your branch

Then start building.
