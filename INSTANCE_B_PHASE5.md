# Instance B: Phase 5 — Vault Production Hardening

## Your Mission (Phase 5)

Make the vault production-ready with backup/restore, structured logging, a REST API, data integrity checks, performance optimization, and notifications. The vault becomes a reliable, observable, API-accessible knowledge store.

**You are the vault. You are reliable. You are observable. You are accessible.**

## Rules

1. **DO NOT** touch `brain/` modules — only import from `brain_contract.py`
2. **DO NOT** modify `brain_contract.py` (FROZEN)
3. **DO NOT** modify `transcribe.py` — modified during merge only
4. **DO NOT** modify `scripts/brain_cli.py` — Instance A owns
5. **DO** import from `brain_contract.py` for shared types
6. **DO** import from `vault/` modules (already built)
7. **DO** write tests for every new module
8. **DO** handle errors gracefully
9. **DO** keep all imports within `vault/` — no cross-boundary imports

## File Ownership (STRICT)

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
scripts/unified_cli.py  # ENHANCE: backup/restore/API commands
```

### OFF LIMITS:
```
brain/                    # Instance A owns
scripts/brain_cli.py      # Instance A owns
transcribe.py             # Modified during merge only
brain_contract.py         # FROZEN
```

---

## Build Order

### Step 1: `vault/backup.py` — Backup & Restore

```python
class VaultBackup:
    """Full and incremental backup for the vault."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.backup_dir = os.path.join(vault_root, "_backups")

    def full_backup(self, label: Optional[str] = None) -> str:
        """Create a full tarball backup.

        Returns backup path.
        """
        ...

    def incremental_backup(self) -> str:
        """Backup only files changed since last full backup.

        Returns backup path.
        """
        ...

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata."""
        ...

    def restore(self, backup_path: str, target: Optional[str] = None) -> bool:
        """Restore vault from backup.

        Args:
            backup_path: Path to backup tarball.
            target: Optional target directory (defaults to vault_root).

        Returns True on success.
        """
        ...

    def verify(self, backup_path: str) -> Dict[str, Any]:
        """Verify backup integrity.

        Returns {"valid": True, "file_count": 42, "size_mb": 1.2}
        """
        ...
```

**Storage format:**
```
vault/_backups/
├── full_2026_08_23_143000.tar.gz
├── incr_2026_08_23_150000.tar.gz
└── manifest.json  # index of all backups
```

---

### Step 2: `vault/observability.py` — Structured Logging

```python
import logging
import json
import time
from typing import Any, Dict, Optional
from contextlib import contextmanager


class StructuredLogger:
    """JSON structured logger for vault operations."""

    def __init__(self, name: str = "vault", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

    def _emit(self, level: str, event: str, **kwargs) -> None:
        """Emit a structured log entry."""
        entry = {
            "timestamp": time.time(),
            "level": level,
            "event": event,
            **kwargs,
        }
        getattr(self.logger, level.lower())(json.dumps(entry))

    def info(self, event: str, **kwargs) -> None:
        self._emit("INFO", event, **kwargs)

    def warning(self, event: str, **kwargs) -> None:
        self._emit("WARNING", event, **kwargs)

    def error(self, event: str, **kwargs) -> None:
        self._emit("ERROR", event, **kwargs)

    @contextmanager
    def trace(self, operation: str, **kwargs):
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
            elapsed = time.time() - start
            self.info(operation, duration_ms=round(elapsed * 1000, 2), **kwargs)
        except Exception as e:
            elapsed = time.time() - start
            self.error(
                f"{operation}_failed",
                duration_ms=round(elapsed * 1000, 2),
                error=str(e),
                **kwargs,
            )
            raise


class MetricsCollector:
    """Collect and report operational metrics."""

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._timers: Dict[str, list] = {}

    def increment(self, name: str, value: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + value

    def timer(self, name: str, duration_ms: float) -> None:
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(duration_ms)

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {"counters": dict(self._counters)}
        for name, durations in self._timers.items():
            metrics[f"{name}_p50"] = sorted(durations)[len(durations) // 2] if durations else 0
            metrics[f"{name}_p99"] = sorted(durations)[int(len(durations) * 0.99)] if durations else 0
            metrics[f"{name}_count"] = len(durations)
        return metrics

    def reset(self) -> None:
        self._counters.clear()
        self._timers.clear()
```

---

### Step 3: `vault/api.py` — REST API

```python
"""Vault REST API — FastAPI-based interface for external tools."""

from typing import Any, Dict, List, Optional

# Lazy imports to avoid hard dependency
_app = None


def create_app(vault_root: str = None):
    """Create FastAPI app instance."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from vault.pipeline import VaultPipeline
    from vault.config import VAULT_ROOT

    app = FastAPI(
        title="Second Brain API",
        version="0.3.0",
        description="REST API for the second brain knowledge system",
    )

    pipeline = VaultPipeline(vault_root=vault_root or VAULT_ROOT)

    class QueryRequest(BaseModel):
        text: str
        max_results: int = 10

    class NoteRequest(BaseModel):
        text: str
        type: str = "observation"
        priority: float = 0.5

    @app.get("/health")
    def health():
        return pipeline.health_check()

    @app.get("/stats")
    def stats():
        return pipeline.stats()

    @app.post("/search")
    def search(req: QueryRequest):
        return pipeline.query(req.text)

    @app.post("/notes")
    def create_note(req: NoteRequest):
        from brain_contract import Understanding, NoteType
        u = Understanding(text=req.text, type=NoteType(req.type), priority=req.priority)
        path = pipeline.process_safe(u)
        if path:
            return {"file_path": path, "status": "created"}
        raise HTTPException(status_code=500, detail="Failed to create note")

    return app


def run_server(host: str = "0.0.0.0", port: int = 8000, vault_root: str = None):
    """Run the API server."""
    import uvicorn
    app = create_app(vault_root=vault_root)
    uvicorn.run(app, host=host, port=port)
```

---

### Step 4: `vault/migration.py` — Schema Migration

```python
class SchemaMigrator:
    """Detect and fix schema issues in vault notes."""

    SCHEMA_VERSION = 2

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root

    def validate_all(self) -> Dict[str, Any]:
        """Scan all notes and report issues.

        Returns:
            {
                "total": 100,
                "valid": 95,
                "issues": [
                    {"file": "tasks/fix-vad.md", "issue": "missing_type_field"},
                    ...
                ]
            }
        """
        ...

    def auto_fix(self, dry_run: bool = True) -> Dict[str, Any]:
        """Automatically fix common issues.

        Returns:
            {"fixed": 5, "skipped": 0, "errors": 0}
        """
        ...

    def migrate_v1_to_v2(self, file_path: str) -> bool:
        """Migrate a single note from v1 to v2 schema."""
        ...
```

---

### Step 5: `vault/notifications.py` — Notifications

```python
class NotificationManager:
    """Manage task reminders and webhook notifications."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.rules: List[Dict] = []

    def add_rule(self, rule: Dict) -> None:
        """Add a notification rule.

        Rule format:
            {"type": "deadline", "hours_before": 24, "channel": "macos"}
            {"type": "priority", "min_priority": 0.9, "channel": "webhook"}
        """
        ...

    def check_reminders(self) -> List[Dict]:
        """Scan vault for upcoming deadlines and high-priority items."""
        ...

    def send_macos_notification(self, title: str, message: str) -> bool:
        """Send macOS notification via osascript."""
        ...

    def send_webhook(self, url: str, payload: Dict) -> bool:
        """Send webhook POST request."""
        ...

    def daily_digest(self) -> Dict[str, Any]:
        """Generate daily digest of tasks and reminders."""
        ...
```

---

### Step 6: Performance Optimization

#### `vault/index.py` — Add batch updates:

```python
class VaultIndex:
    # ... existing ...

    def add_batch(self, entries: List[Tuple[str, str, List[float]]]) -> int:
        """Add multiple entries at once (faster than individual adds).

        Returns number of entries added.
        """
        ...
```

#### `vault/search.py` — Add LRU cache:

```python
from functools import lru_cache

class VaultSearch:
    # ... existing ...

    @lru_cache(maxsize=128)
    def _cached_embedding_search(self, query_hash: str, k: int):
        """Cache frequent search queries."""
        ...
```

---

### Step 7: Tests

#### `vault/tests/test_backup.py`
```python
class TestVaultBackup:
    def test_full_backup_creates_tarball(self): ...
    def test_list_backups(self): ...
    def test_restore_from_backup(self): ...
    def test_verify_backup_integrity(self): ...
    def test_incremental_backup(self): ...
```

#### `vault/tests/test_observability.py`
```python
class TestStructuredLogger:
    def test_info_emits_json(self): ...
    def test_error_includes_exception(self): ...

class TestMetricsCollector:
    def test_increment(self): ...
    def test_timer_stats(self): ...
    def test_get_metrics(self): ...
```

#### `vault/tests/test_api.py`
```python
class TestVaultAPI:
    def test_health_endpoint(self): ...
    def test_stats_endpoint(self): ...
    def test_search_endpoint(self): ...
    def test_create_note(self): ...
```

#### `vault/tests/test_migration.py`
```python
class TestSchemaMigrator:
    def test_validate_all(self): ...
    def test_auto_fix(self): ...
    def test_migrate_v1_to_v2(self): ...
```

#### `vault/tests/test_notifications.py`
```python
class TestNotificationManager:
    def test_add_rule(self): ...
    def test_check_reminders(self): ...
    def test_daily_digest(self): ...
```

---

## Commit Messages

```
feat(vault): add backup and restore system
feat(vault): add structured logging and observability
feat(vault): add REST API layer
feat(vault): add schema migration and data integrity
feat(vault): add webhook and notification system
feat(vault): optimize index with batch updates and caching
feat(vault): add lazy loading for large vaults
test(vault): add backup, observability, API, migration, notification tests
docs: update API documentation
```
