# Instance B: Phase 4 — Correction Vault & Unified CLI

## Your Mission (Phase 4)

Store corrections in the vault, add correction-aware pipeline processing, and build a unified CLI that wraps brain + vault operations. When the brain makes mistakes, corrections get logged and stored for retraining.

**You are the vault. You store corrections. You learn from feedback.**

## Rules

1. **DO NOT** touch `brain/` modules — only import from `brain_contract.py`
2. **DO NOT** modify `brain_contract.py` (FROZEN)
3. **DO NOT** modify `transcribe.py` — modified during merge only
4. **DO NOT** modify `scripts/brain_cli.py` — Instance A owns
5. **DO** import from `brain_contract.py` for shared types
6. **DO** import from `vault/` modules (already built)
7. **DO** write tests for every new module
8. **DO** handle errors gracefully
9. **DO** keep all imports within `vault/` — no cross-boundary imports into `brain/`

## File Ownership (STRICT)

### Instance B owns (ONLY touches these):
```
vault/corrections.py      # NEW: correction vault storage
vault/pipeline.py         # ENHANCE: add correction support (NO brain imports)
vault/__init__.py         # ENHANCE: add correction exports
vault/tests/              # NEW tests
tests/test_correction_integration.py  # NEW
scripts/unified_cli.py    # NEW: thin wrapper around brain_cli + vault
```

### OFF LIMITS:
```
brain/                    # Instance A owns
scripts/brain_cli.py      # Instance A owns
transcribe.py             # Modified during merge only
brain_contract.py         # FROZEN
```

---

## Key Design Principle: No Cross-Boundary Imports

**Instance B NEVER imports from `brain/`.** Instead:

- `vault/corrections.py` stores correction data using `brain_contract` types only
- `vault/pipeline.py` adds `process_with_correction()` that accepts pre-corrected data
- `scripts/unified_cli.py` is a thin CLI wrapper that calls `brain_cli.py` as a subprocess for brain operations

This means zero merge conflicts — Instance A touches `brain/`, Instance B touches `vault/`.

---

## Build Order

### Step 1: `vault/corrections.py` — Correction Vault Storage

**Purpose:** Store corrections in the vault as JSONL buffer + optional markdown notes.

**Interface:**

```python
class CorrectionVault:
    """Store and manage user corrections in the vault."""

    def __init__(self, vault_root: str = VAULT_ROOT):
        self.vault_root = vault_root
        self.corrections_dir = os.path.join(vault_root, "_corrections")
        self.buffer_path = os.path.join(self.corrections_dir, "corrections.jsonl")
        self._ensure_dir()

    def log_correction(
        self,
        text: str,
        predicted: Understanding,
        corrected: Understanding,
    ) -> str:
        """
        Log a correction to the buffer.

        Returns the correction ID.
        """
        ...

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get all unapplied corrections."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        """Correction statistics.

        Returns:
            {
                "total": 45,
                "pending": 28,
                "applied": 17,
                "by_field": {"type": 12, "priority": 8, ...},
                "accuracy_trend": [0.72, 0.75, 0.78, 0.81],
            }
        """
        ...

    def export_training_data(self, output_path: str) -> int:
        """Export corrections as JSONL for brain retraining.

        Returns number of corrections exported.
        """
        ...

    def mark_applied(self, correction_ids: List[str]) -> None:
        """Mark corrections as used for retraining."""
        ...

    def save_correction_note(self, correction: Dict[str, Any]) -> str:
        """Save a correction as a vault note for human review.

        Returns file path.
        """
        ...

    def _ensure_dir(self) -> None:
        """Create corrections directory if needed."""
        ...

    def _generate_id(self) -> str:
        """Generate unique correction ID."""
        ...
```

**Storage format:**

```
vault/_corrections/
├── corrections.jsonl           # Machine-readable buffer
├── corrections_2026_08.md      # Monthly correction notes
└── accuracy_log.json           # Accuracy trend over time
```

**JSONL line format:**
```json
{
    "id": "corr_2026_08_23_001",
    "timestamp": "2026-08-23T14:30:00Z",
    "text": "Sarah told me her birthday is March 15th",
    "predicted": {"type": "observation", "priority": 0.3, ...},
    "corrected": {"type": "person_info", "priority": 0.5, ...},
    "changed_fields": ["type", "priority"],
    "applied": false
}
```

---

### Step 2: Enhance `vault/pipeline.py`

**Add correction-aware processing (NO brain imports):**

```python
class VaultPipeline:
    # ... existing methods ...

    def __init__(self, vault_root=VAULT_ROOT, index_dir=INDEX_DIR):
        # ... existing init ...
        self.correction_vault = CorrectionVault(vault_root)

    def process_with_correction(
        self,
        understanding: Understanding,
        corrected: Optional[Understanding] = None,
    ) -> Tuple[Optional[str], bool]:
        """
        Process with optional pre-corrected understanding.

        Args:
            understanding: Original brain output
            corrected: Pre-corrected version (if user already corrected)

        Returns:
            (file_path, was_corrected)
        """
        if corrected is not None:
            # Log the correction
            self.correction_vault.log_correction(
                understanding.text, understanding, corrected
            )
            understanding = corrected

        path = self.process_safe(understanding)
        return path, corrected is not None

    def get_correction_stats(self) -> Dict[str, Any]:
        """Get correction statistics."""
        return self.correction_vault.get_stats()

    def export_corrections(self, output_path: str) -> int:
        """Export corrections for brain retraining."""
        return self.correction_vault.export_training_data(output_path)
```

**Key:** `process_with_correction()` accepts a pre-corrected `Understanding` — it does NOT import from `brain/`. The correction happens externally (in `transcribe.py` or `unified_cli.py`) and is passed in.

---

### Step 3: `scripts/unified_cli.py` — Unified Brain + Vault CLI

**Purpose:** Single CLI entry point that wraps brain_cli.py (subprocess) + vault operations.

**Key design:** Brain commands call `scripts/brain_cli.py` as a subprocess. No direct brain imports.

```python
#!/usr/bin/env python3
"""Unified CLI for brain + vault operations.

Brain commands delegate to scripts/brain_cli.py via subprocess.
Vault commands use vault modules directly.
"""

import argparse
import json
import subprocess
import sys

from vault.pipeline import VaultPipeline
from vault.corrections import CorrectionVault


def _run_brain_cli(*args):
    """Run brain_cli.py as subprocess."""
    cmd = [sys.executable, "scripts/brain_cli.py"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0 and result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def cmd_brain_understand(args):
    """Brain inference — delegates to brain_cli.py."""
    brain_args = ["understand", args.text]
    if args.json:
        brain_args.append("--json")
    if args.checkpoint:
        brain_args.extend(["--checkpoint", args.checkpoint])
    sys.exit(_run_brain_cli(*brain_args))


def cmd_brain_interactive(args):
    """Interactive brain mode — delegates to brain_cli.py."""
    brain_args = ["interactive"]
    if args.checkpoint:
        brain_args.extend(["--checkpoint", args.checkpoint])
    sys.exit(_run_brain_cli(*brain_args))


def cmd_brain_evaluate(args):
    """Brain evaluation — delegates to brain_cli.py."""
    brain_args = ["evaluate", "--test-data", args.test_data]
    if args.output:
        brain_args.extend(["--output", args.output])
    sys.exit(_run_brain_cli(*brain_args))


def cmd_brain_retrain(args):
    """Brain retrain — delegates to brain_cli.py."""
    brain_args = ["retrain", "--epochs", str(args.epochs)]
    sys.exit(_run_brain_cli(*brain_args))


def cmd_vault_stats(args):
    """Vault statistics."""
    pipeline = VaultPipeline(vault_root=args.vault_path)
    stats = pipeline.stats()
    print(json.dumps(stats, indent=2))


def cmd_vault_search(args):
    """Search the vault."""
    pipeline = VaultPipeline(vault_root=args.vault_path)
    result = pipeline.query(args.query)
    print(f"Answer: {result.get('answer', 'No answer')}")
    print(f"Sources: {len(result.get('sources', []))} notes")


def cmd_vault_health(args):
    """Vault health check."""
    pipeline = VaultPipeline(vault_root=args.vault_path)
    health = pipeline.health_check()
    print(json.dumps(health, indent=2))


def cmd_corrections_stats(args):
    """Correction statistics."""
    vault = CorrectionVault(vault_root=args.vault_path)
    stats = vault.get_stats()
    print(json.dumps(stats, indent=2))


def cmd_corrections_export(args):
    """Export corrections for retraining."""
    vault = CorrectionVault(vault_root=args.vault_path)
    count = vault.export_training_data(args.output)
    print(f"Exported {count} corrections → {args.output}")


def cmd_pipeline(args):
    """End-to-end pipeline."""
    from brain_contract import Understanding, NoteType
    pipeline = VaultPipeline(vault_root=args.vault_path)
    u = Understanding(text=args.text, type=NoteType.OBSERVATION)
    path = pipeline.process_safe(u)
    if path:
        print(f"[Vault] Saved: {path}")
    else:
        print("[Vault] Failed to save")


def main():
    parser = argparse.ArgumentParser(description="Unified Brain + Vault CLI")

    # Global args
    parser.add_argument("--vault-path", default="vault")

    sub = parser.add_subparsers(dest="command")

    # ── Brain subcommands ──
    brain_sub = sub.add_parser("brain", help="Brain operations (delegates to brain_cli.py)")
    brain_cmds = brain_sub.add_subparsers(dest="brain_command")

    p = brain_cmds.add_parser("understand")
    p.add_argument("text")
    p.add_argument("--json", action="store_true")
    p.add_argument("--checkpoint", default="checkpoints/brain")

    brain_cmds.add_parser("interactive")

    p = brain_cmds.add_parser("evaluate")
    p.add_argument("--test-data", required=True)
    p.add_argument("--output")

    p = brain_cmds.add_parser("retrain")
    p.add_argument("--epochs", type=int, default=5)

    # ── Vault subcommands ──
    vault_sub = sub.add_parser("vault", help="Vault operations")
    vault_cmds = vault_sub.add_subparsers(dest="vault_command")

    vault_cmds.add_parser("stats")
    vault_cmds.add_parser("health")

    p = vault_cmds.add_parser("search")
    p.add_argument("query")

    p = vault_cmds.add_parser("process")
    p.add_argument("text")

    # ── Corrections subcommands ──
    corr_sub = sub.add_parser("corrections", help="Correction management")
    corr_cmds = corr_sub.add_subparsers(dest="corr_command")

    corr_cmds.add_parser("stats")

    p = corr_cmds.add_parser("export")
    p.add_argument("--output", default="corrections.jsonl")

    # ── Pipeline subcommands ──
    pipe_sub = sub.add_parser("pipeline", help="End-to-end pipeline")
    pipe_sub.add_argument("text")

    args = parser.parse_args()

    # Dispatch
    dispatch = {
        ("brain", "understand"): cmd_brain_understand,
        ("brain", "interactive"): cmd_brain_interactive,
        ("brain", "evaluate"): cmd_brain_evaluate,
        ("brain", "retrain"): cmd_brain_retrain,
        ("vault", "stats"): cmd_vault_stats,
        ("vault", "search"): cmd_vault_search,
        ("vault", "health"): cmd_vault_health,
        ("vault", "process"): cmd_vault_process,
        ("corrections", "stats"): cmd_corrections_stats,
        ("corrections", "export"): cmd_corrections_export,
        ("pipeline", None): cmd_pipeline,
    }

    key = (args.command, getattr(args, "brain_command", None) or getattr(args, "vault_command", None) or getattr(args, "corr_command", None))
    handler = dispatch.get(key)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

---

### Step 4: Enhance `vault/__init__.py`

```python
from vault.config import VAULT_ROOT, INDEX_DIR, EMBEDDING_DIM, VAULT_VERSION
from vault.pipeline import VaultPipeline
from vault.reader import VaultReader
from vault.search import VaultSearch
from vault.corrections import CorrectionVault

__version__ = VAULT_VERSION
__all__ = [
    "VAULT_ROOT", "INDEX_DIR", "EMBEDDING_DIM", "VAULT_VERSION",
    "VaultPipeline", "VaultReader", "VaultSearch", "CorrectionVault",
]
```

---

### Step 5: Tests

#### `vault/tests/test_corrections.py`
```python
class TestCorrectionVault:
    def test_log_correction(self): ...
    def test_get_pending(self): ...
    def test_get_stats(self): ...
    def test_export_training_data(self): ...
    def test_mark_applied(self): ...
    def test_save_correction_note(self): ...
```

#### `tests/test_correction_integration.py`
```python
class TestCorrectionIntegration:
    def test_correction_roundtrip(self): ...
    def test_pipeline_with_correction(self): ...
    def test_correction_stats_after_pipeline(self): ...
    def test_export_import_cycle(self): ...
```

---

## Integration Point

At merge time, `transcribe.py` wires the interactive mode. Instance B does NOT touch transcribe.py — the merge handler combines both instances' work:

```python
# transcribe.py — wired during merge
if args.interactive and brain_engine and understanding:
    # Brain side: interactive correction (Instance A)
    from brain.interactive import InteractiveCorrector
    corrector = InteractiveCorrector(correction_logger)
    corrected, modified = corrector.present_and_correct(transcription, understanding)

    # Vault side: store correction + write note (Instance B)
    if vault_pipeline:
        path, was_corrected = vault_pipeline.process_with_correction(
            understanding, corrected=corrected if modified else None
        )
```

---

## Commit Messages

```
feat(vault): add correction vault storage
feat(vault): enhance pipeline with correction support
feat(vault): add unified brain + vault CLI
feat(vault): add correction exports and stats
test(vault): add correction and integration tests
```
