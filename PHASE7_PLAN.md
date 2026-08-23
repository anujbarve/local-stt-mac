# Phase 7 — Polish, Deployment & Scale

## Current State (End of Phase 6)

- **109 Python files**, 58 source modules, 49 test files
- **519+ tests passing**
- **Brain**: 10-task transformer, inference, corrections, distillation, evaluation, versioning, ONNX export, calibration, incremental learning, A/B testing, temporal reasoning, clustering, insights, recommendations
- **Vault**: Storage, search, graph, linking, MOC, LLM fallback, correction vault, backup/restore, API, observability, migration, notifications, graph intelligence, scheduling, synthesis, plugins, analytics
- **Integration**: Voice → brain → vault, unified CLI, interactive corrections

**What Phase 7 adds**: The system works end-to-end and is intelligent. Phase 7 makes it *deployable*, *resilient*, and *scalable* — Docker packaging, CI/CD, error recovery, load testing, security hardening, and comprehensive documentation.

---

## Phase 7 Goals

1. **Deployment**: Docker containerization, docker-compose, Makefile
2. **CI/CD**: GitHub Actions for tests, linting, type checking
3. **Resilience**: Retry logic, graceful degradation, circuit breakers
4. **Security**: Input sanitization, rate limiting, secrets management
5. **Performance**: Batch optimization, memory profiling, caching at scale
6. **Documentation**: API docs, developer guide, architecture diagrams
7. **Packaging**: PyPI-ready package, CLI entry points, versioning

---

## Instance A (Brain) — Phase 7 Tasks

### Task 1: Model Packaging & Export Pipeline
**Files**: `brain/export.py` (NEW), `scripts/brain_cli.py`

- Export pipeline: train → evaluate → calibrate → export (ONNX + TorchScript)
- Model cards: JSON metadata with accuracy, training data, version
- `brain export --all` exports ONNX + TorchScript + model card
- Validation: test exported model matches PyTorch inference

### Task 2: Inference Caching
**Files**: `brain/understand.py`

- LRU cache for repeated text inputs
- Cache hit rate tracking
- `engine.understand(text, use_cache=True)`
- Cache invalidation on model update

### Task 3: Batch Inference Optimization
**Files**: `brain/understand.py`

- Dynamic batching: group similar-length texts
- Async inference support
- Memory-efficient streaming for large batches
- `engine.understand_batch(texts, batch_size=32, parallel=True)`

### Task 4: Error Recovery & Fallbacks
**Files**: `brain/understand.py`, `brain/model.py`

- Model load retry with exponential backoff
- Graceful degradation: return partial results on partial failure
- Health check endpoint: `engine.health()`
- Model warm-up: pre-load on startup

---

## Instance B (Vault) — Phase 7 Tasks

### Task 5: Docker Containerization
**Files**: `Dockerfile`, `docker-compose.yml`, `Makefile` (NEW)

- Multi-stage Docker build (slim runtime image)
- docker-compose for vault + API + optional LLM sidecar
- Makefile with common commands: `make train`, `make serve`, `make test`
- Volume mounts for vault data persistence

### Task 6: CI/CD Pipeline
**Files**: `.github/workflows/ci.yml` (NEW)

- Run tests on push/PR
- Lint with ruff
- Type check with mypy (optional)
- Build Docker image on main
- Publish to PyPI on tag

### Task 7: Security Hardening
**Files**: `vault/security.py` (NEW), `vault/api.py`

- Input sanitization: strip control chars, limit length, validate types
- Rate limiting: token bucket per IP
- API key authentication for REST API
- Secrets management: env vars, no hardcoded credentials
- Path traversal prevention in file operations

### Task 8: Resilience & Circuit Breakers
**Files**: `vault/resilience.py` (NEW)

- Retry with exponential backoff for external calls
- Circuit breaker for LLM/API calls (open after N failures)
- Graceful degradation: return cached results when service is down
- Health check aggregation: `vault.health_detailed()`

### Task 9: Performance at Scale
**Files**: `vault/index.py`, `vault/search.py`, `vault/reader.py`

- Memory-mapped index for large vaults (>10K notes)
- Sharded search: partition index by note type
- Lazy note loading with on-demand parsing
- Batch write optimization: write multiple notes atomically

### Task 10: Comprehensive Documentation
**Files**: `docs/API.md`, `docs/DEVELOPER.md`, `docs/DEPLOYMENT.md`

- API reference: all endpoints, request/response formats
- Developer guide: architecture, adding modules, testing
- Deployment guide: Docker, manual, cloud
- Architecture diagrams (Mermaid)

---

## Shared Tasks

### Task 11: Package Setup
**Files**: `pyproject.toml`, `setup.py` (NEW)

- pyproject.toml with all dependencies
- CLI entry points: `brain-cli`, `vault-api`, `unified-cli`
- Version management: single source of truth in `vault/config.py`
- `pip install .` works cleanly

### Task 12: End-to-End Deployment Test
**Files**: `tests/test_deployment.py` (NEW)

- Test: Docker build succeeds
- Test: API serves health endpoint
- Test: Brain CLI exports model
- Test: Full pipeline: audio → transcription → brain → vault → search
- Test: Backup and restore cycle

---

## Build Order

### Week 1: Deployment & Packaging
1. Task 11 (Package setup) — foundation for everything
2. Task 5 (Docker) — containerized deployment
3. Task 6 (CI/CD) — automated testing

### Week 2: Security & Resilience
4. Task 7 (Security) — input sanitization, rate limiting
5. Task 8 (Resilience) — retry, circuit breaker, degradation
6. Task 4 (Error recovery) — brain-side resilience

### Week 3: Performance & Scale
7. Task 9 (Performance at scale) — memory optimization
8. Task 2 (Inference caching) — brain-side caching
9. Task 3 (Batch optimization) — parallel inference

### Week 4: Polish & Docs
10. Task 1 (Model packaging) — export pipeline
11. Task 10 (Documentation) — API, developer, deployment guides
12. Task 12 (E2E deployment test) — verify everything works

---

## File Ownership (Conflict-Free)

### Instance A owns:
```
brain/export.py         # NEW: export pipeline
brain/understand.py     # ENHANCE: caching, batch optimization, error recovery
brain/model.py          # ENHANCE: retry, health check
brain/tests/            # NEW tests
scripts/brain_cli.py    # ENHANCE: export --all
```

### Instance B owns:
```
Dockerfile              # NEW: container image
docker-compose.yml      # NEW: orchestration
Makefile                # NEW: common commands
pyproject.toml          # ENHANCE: package config
.github/workflows/ci.yml # NEW: CI/CD pipeline
vault/security.py       # NEW: security hardening
vault/resilience.py     # NEW: circuit breaker, retry
vault/index.py          # ENHANCE: memory optimization
vault/search.py         # ENHANCE: sharded search
vault/reader.py         # ENHANCE: lazy loading
vault/api.py            # ENHANCE: auth, rate limiting
docs/API.md             # NEW: API reference
docs/DEVELOPER.md       # NEW: developer guide
docs/DEPLOYMENT.md      # NEW: deployment guide
vault/tests/            # NEW tests
tests/test_deployment.py # NEW: E2E deployment tests
```

### Shared (merge time only):
```
brain_contract.py       # FROZEN
transcribe.py           # Modified during merge only
```

---

## Success Criteria

- [ ] All 519+ existing tests still pass
- [ ] Docker image builds and runs
- [ ] `pip install .` works
- [ ] CI pipeline runs tests on push
- [ ] API has rate limiting and auth
- [ ] Circuit breaker prevents cascading failures
- [ ] Inference cache hits for repeated inputs
- [ ] Export pipeline produces ONNX + TorchScript + model card
- [ ] Documentation covers API, developer, and deployment
- [ ] E2E deployment test passes
- [ ] Target: 580+ tests total

---

## Commit Messages

```
feat(brain): add export pipeline with model cards
feat(brain): add inference caching
feat(brain): optimize batch inference
feat(brain): add error recovery and health checks
feat(vault): add Docker containerization
feat(vault): add CI/CD pipeline
feat(vault): add security hardening
feat(vault): add resilience and circuit breakers
feat(vault): optimize for scale (sharded index, lazy loading)
docs: add API reference, developer guide, deployment guide
build: add pyproject.toml and Makefile
test: add E2E deployment tests
```
