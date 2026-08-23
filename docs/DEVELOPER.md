# Developer Guide

## Architecture

```
brain_contract.py          ← shared types (FROZEN)
        ↕
brain/                     ← Instance A: neural network, training, inference
vault/                     ← Instance B: storage, search, intelligence
        ↕
transcribe.py              ← voice pipeline (merged at integration)
scripts/                   ← CLI tools
```

## Adding a Vault Module

1. Create `vault/your_module.py` with a class and docstrings
2. Import shared types from `brain_contract.py`
3. Import other vault modules from `vault.*`
4. **Never** import from `brain/`
5. Add tests in `vault/tests/test_your_module.py`
6. Export in `vault/__init__.py`

## Testing

```bash
# Run all vault tests
python -m pytest vault/tests/ -v

# Run with coverage
python -m pytest vault/tests/ --cov=vault

# Run a single test class
python -m pytest vault/tests/test_writer.py::TestVaultWriter -v
```

## Code Style

- Type hints on all public methods
- Docstrings on all public classes and functions
- `from __future__ import annotations` at top of every file
- `logging` module for debug output, never `print()`

## Module Dependency Graph

```
config.py ← (everything imports from here)
schema.py ← templates.py, writer.py, migration.py
templates.py ← writer.py
writer.py ← pipeline.py, backup.py
reader.py ← search.py, scheduler.py, synthesizer, linker.py
graph.py ← search.py, graph_query.py
index.py ← search.py
search.py ← answer.py, pipeline.py
pipeline.py ← api.py, unified_cli.py
corrections.py ← pipeline.py
backup.py ← pipeline.py (optional)
security.py ← api.py (optional)
resilience.py ← pipeline.py (optional)
```

## Commit Convention

```
feat(vault): add new module
fix(vault): fix bug in module
test(vault): add tests for module
docs: update documentation
refactor(vault): improve module
```
