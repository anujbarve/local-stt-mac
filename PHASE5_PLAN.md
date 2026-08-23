# Phase 5 — Production Hardening & Real-World Deployment

## Current State

After 4 phases, the second brain has:
- **Brain**: 10-task transformer model, inference API, interactive corrections, knowledge distillation, evaluation
- **Vault**: Markdown storage, FAISS search, wikilink graph, auto-linking, MOC generation, LLM fallback, error handling
- **Integration**: Voice transcription → brain understanding → vault writing, unified CLI
- **Tests**: 344 passing across brain, vault, and integration

**What's missing for production**: The system works end-to-end but lacks the operational, security, and performance infrastructure needed for real daily use.

---

## Phase 5 Goals

1. **Performance**: Batch processing, caching, incremental indexing
2. **Observability**: Structured logging, metrics dashboard, health monitoring
3. **Data Safety**: Backup/restore, corruption detection, data migration
4. **API Layer**: REST API for external tool integration (shortcuts, scripts, webhooks)
5. **Security**: Input sanitization, rate limiting, safe file operations
6. **Model Deployment**: ONNX export, model versioning, A/B testing framework

---

## Instance A (Brain) — Phase 5 Tasks

### Task 1: Model Versioning & ONNX Export
**Files**: `brain/model.py`, `brain/train.py`, `scripts/brain_cli.py`

- Add `model_version` field to checkpoints (semantic versioning)
- Wire ONNX export into `brain_cli.py` (`brain export --onnx`)
- Add model metadata: training date, dataset size, accuracy metrics
- Support loading by version: `BrainEngine.load("checkpoints/brain/v1.2.0")`

### Task 2: Confidence Calibration
**Files**: `brain/understand.py`

- Add temperature scaling for better calibrated confidence scores
- Track calibration metrics (ECE, Brier score) in evaluation
- Add `calibrate()` method that learns temperature on validation set

### Task 3: Incremental Learning
**Files**: `brain/train.py`, `brain/correct.py`

- Add `incremental_train()` that fine-tunes on new corrections without full retrain
- Support curriculum learning: easy examples first, then hard
- Add learning rate warmup for incremental updates

### Task 4: A/B Testing Framework
**Files**: `brain/understand.py`, `brain/evaluate.py`

- Add `model_registry` that tracks multiple model versions
- Support running two models side-by-side and comparing outputs
- Log which model produced each understanding for later analysis

---

## Instance B (Vault) — Phase 5 Tasks

### Task 5: Backup & Restore
**Files**: `vault/backup.py` (NEW), `scripts/unified_cli.py`

- Full vault backup to tarball with timestamps
- Incremental backup (only changed files)
- Restore from backup with conflict resolution
- CLI commands: `vault backup`, `vault restore <timestamp>`, `vault backup list`

### Task 6: Structured Logging & Observability
**Files**: `vault/observability.py` (NEW), `vault/pipeline.py`

- Structured JSON logging for all vault operations
- Request tracing (correlation IDs for multi-step operations)
- Performance metrics: query latency, write latency, index rebuild time
- Export to common formats (JSON lines, Prometheus)

### Task 7: REST API Layer
**Files**: `vault/api.py` (NEW), `vault/api_routes.py` (NEW)

- FastAPI-based REST API for external tools
- Endpoints: POST /notes, GET /search, GET /health, POST /query
- API key authentication
- CORS support for web integrations
- OpenAPI/Swagger documentation auto-generated

### Task 8: Data Integrity & Migration
**Files**: `vault/schema.py`, `vault/migration.py` (NEW)

- Schema versioning for vault notes (v1 → v2 migrations)
- Integrity checks: detect corrupted frontmatter, broken links
- `vault validate` command that scans and reports issues
- Auto-fix common issues (missing frontmatter, broken wikilinks)

### Task 9: Performance Optimization
**Files**: `vault/index.py`, `vault/search.py`, `vault/reader.py`

- LRU cache for frequently accessed notes
- Batch index updates (don't rebuild after every write)
- Lazy loading for large vaults (>1000 notes)
- Background index maintenance

### Task 10: Webhook & Notification System
**Files**: `vault/notifications.py` (NEW), `vault/pipeline.py`

- Task reminder notifications via macOS notifications
- Webhook support for external integrations (Slack, email, etc.)
- Scheduled notification checks (daily digest)
- Notification rules: notify on high-priority tasks, deadlines approaching

---

## Shared Tasks

### Task 11: Input Validation & Security
**Files**: `brain_contract.py`, `vault/schema.py`

- Validate all Understanding fields (type, priority ranges, entity names)
- Sanitize text input (strip control characters, limit length)
- Rate limiting for API endpoints
- Safe file path construction (prevent directory traversal)

### Task 12: End-to-End Performance Benchmarks
**Files**: `tests/benchmarks/` (NEW)

- Benchmark: voice → transcription → understanding → vault write latency
- Benchmark: search query latency at different vault sizes
- Benchmark: batch processing throughput
- Memory usage profiling

---

## Build Order

### Week 1: Safety & Observability
1. Task 11 (Input validation) — foundation for everything
2. Task 6 (Structured logging) — visibility into what's happening
3. Task 8 (Data integrity) — catch issues early

### Week 2: Performance
4. Task 9 (Performance optimization) — make it fast
5. Task 1 (Model versioning) — deployable models
6. Task 3 (Incremental learning) — continuous improvement

### Week 3: API & Integration
7. Task 7 (REST API) — external tool access
8. Task 10 (Notifications) — proactive alerts
9. Task 5 (Backup/restore) — data safety net

### Week 4: Polish
10. Task 2 (Confidence calibration) — better trust signals
11. Task 4 (A/B testing) — model comparison
12. Task 12 (Benchmarks) — performance baseline

---

## File Ownership (Conflict-Free)

### Instance A owns:
```
brain/model.py          # ENHANCE: model versioning
brain/train.py          # ENHANCE: incremental learning, ONNX
brain/understand.py     # ENHANCE: confidence calibration, A/B
brain/evaluate.py       # ENHANCE: calibration metrics, A/B
brain/tests/            # NEW tests
scripts/brain_cli.py    # ENHANCE: export command
```

### Instance B owns:
```
vault/backup.py         # NEW: backup & restore
vault/observability.py  # NEW: structured logging
vault/api.py            # NEW: REST API
vault/api_routes.py     # NEW: API routes
vault/migration.py      # NEW: schema migration
vault/notifications.py  # NEW: webhook & notifications
vault/schema.py         # ENHANCE: validation
vault/pipeline.py       # ENHANCE: observability hooks
vault/index.py          # ENHANCE: caching, batch updates
vault/search.py         # ENHANCE: caching
vault/reader.py         # ENHANCE: lazy loading
vault/__init__.py       # ENHANCE: new exports
vault/tests/            # NEW tests
tests/benchmarks/       # NEW: performance benchmarks
```

### Shared (merge time only):
```
brain_contract.py       # ENHANCE: validation helpers
tests/test_api.py       # NEW: API integration tests
```

### OFF LIMITS:
```
transcribe.py           # Modified during merge only
scripts/unified_cli.py  # Instance B owns
```

---

## Success Criteria

- [ ] All 344 existing tests still pass
- [ ] New tests for every new module (target: 450+ total)
- [ ] REST API serves search/health endpoints
- [ ] Backup/restore cycle works end-to-end
- [ ] Structured logs output JSON for all operations
- [ ] ONNX model exports and runs inference
- [ ] Search latency < 100ms for vaults up to 1000 notes
- [ ] Write latency < 200ms including index update

---

## Commit Messages

```
feat(brain): add model versioning and ONNX export
feat(brain): add confidence calibration
feat(brain): add incremental learning support
feat(brain): add A/B testing framework
feat(vault): add backup and restore system
feat(vault): add structured logging and observability
feat(vault): add REST API layer
feat(vault): add data integrity and migration
feat(vault): optimize search and index performance
feat(vault): add webhook and notification system
feat(contract): add input validation helpers
test: add performance benchmarks
docs: update architecture for Phase 5
```
