# Instance B: Phase 4 — Interactive Correction & Unified CLI

## Your Mission (Phase 4)

Wire the correction system into the vault pipeline. When the brain makes mistakes, the user corrects them, and those corrections get logged for retraining. You also build the unified CLI that combines brain + vault.

**You are the vault. You store corrections. You learn from feedback.**

## Rules

1. **DO NOT** touch `brain/` modules — only import from `brain_contract.py`
2. **DO NOT** modify `brain_contract.py`
3. **DO** import from `brain_contract.py` for shared types
4. **DO** import from `vault/` modules (already built)
5. **DO** write tests for every new module
6. **DO** handle errors gracefully

## File Ownership

### Instance B owns (ONLY touches these):
```
vault/corrections.py      # NEW: correction vault storage
vault/pipeline.py         # ENHANCE: add correction support
vault/__init__.py         # ENHANCE: add correction exports
vault/tests/              # NEW tests
tests/test_correction_integration.py  # NEW
transcribe.py             # ENHANCE: add interactive mode
scripts/unified_cli.py    # NEW: unified brain + vault CLI
```

### Shared (both instances touch during integration):
```
brain_contract.py         # FROZEN
requirements.txt          # add new deps
```

### OFF LIMITS:
```
brain/                    # Instance A owns
scripts/brain_cli.py      # Instance A owns
```

---

## Build Order

### Step 1: `vault/corrections.py` — Correction Vault Storage

**Purpose:** Store corrections in the vault as markdown notes + JSONL buffer.

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

**Add correction-aware processing:**

```python
class VaultPipeline:
    # ... existing methods ...

    def process_with_correction(
        self,
        understanding: Understanding,
        interactive: bool = False,
    ) -> Tuple[Optional[str], bool]:
        """
        Process with optional interactive correction.

        Returns:
            (file_path, was_corrected)
        """
        if interactive:
            from brain.interactive import InteractiveCorrector
            from brain.correct import CorrectionLogger

            corrector = InteractiveCorrector(CorrectionLogger())
            corrected, modified = corrector.present_and_correct(
                understanding.text, understanding
            )
            if modified:
                # Log correction
                self.correction_vault.log_correction(
                    understanding.text, understanding, corrected
                )
                understanding = corrected

        path = self.process_safe(understanding)
        return path, False  # was_corrected flag

    def get_correction_stats(self) -> Dict[str, Any]:
        """Get correction statistics."""
        return self.correction_vault.get_stats()

    def export_corrections(self, output_path: str) -> int:
        """Export corrections for brain retraining."""
        return self.correction_vault.export_training_data(output_path)
```

---

### Step 3: Enhance `transcribe.py` — Interactive Mode

**Add `--interactive` flag:**

```python
# In parse_arguments():
parser.add_argument(
    "--interactive",
    action="store_true",
    help="Enable interactive correction after each transcription",
)

# In run_pipeline() — after transcription:
if args.interactive and brain_engine and vault_pipeline:
    corrected, modified = vault_pipeline.process_with_correction(
        understanding, interactive=True
    )
    if modified:
        print("[Vault] Correction logged for retraining")
```

---

### Step 4: `scripts/unified_cli.py` — Unified Brain + Vault CLI

**Purpose:** Single CLI entry point that combines brain and vault operations.

**Interface:**

```bash
# Brain inference
python scripts/unified_cli.py brain understand "Fix the VAD threshold"

# Brain interactive mode
python scripts/unified_cli.py brain interactive

# Brain batch processing
python scripts/unified_cli.py brain batch input.json --output results.json

# Brain evaluation
python scripts/unified_cli.py brain evaluate --test-data data/test.json

# Brain retrain
python scripts/unified_cli.py brain retrain --epochs 5

# Vault operations
python scripts/unified_cli.py vault stats
python scripts/unified_cli.py vault search "what projects am I working on?"
python scripts/unified_cli.py vault health
python scripts/unified_cli.py vault process "I need to fix the VAD"

# Correction management
python scripts/unified_cli.py corrections stats
python scripts/unified_cli.py corrections export --output corrections.jsonl
python scripts/unified_cli.py corrections history

# End-to-end pipeline
python scripts/unified_cli.py pipeline "Meeting with Sarah about the project"
python scripts/unified_cli.py pipeline --interactive "Fix the bug by Friday"
```

**Implementation:**

```python
#!/usr/bin/env python3
"""Unified CLI for brain + vault operations."""

import argparse
import json
import sys

from brain.understand import BrainEngine
from brain.correct import CorrectionLogger
from brain.interactive import InteractiveCorrector
from vault.pipeline import VaultPipeline
from vault.corrections import CorrectionVault


def cmd_brain_understand(args):
    """Brain inference."""
    engine = BrainEngine.load(args.checkpoint)
    u = engine.understand(args.text)
    if args.json:
        print(json.dumps(u.to_dict(), indent=2))
    else:
        print(f"Type:      {u.type.value}")
        print(f"Priority:  {u.priority:.2f}")
        print(f"Action:    {u.action or 'none'}")
        print(f"Entities:  {[(e.name, e.entity_type.value) for e in u.entities]}")
        print(f"Topics:    {u.topics}")


def cmd_brain_interactive(args):
    """Interactive brain mode."""
    engine = BrainEngine.load(args.checkpoint)
    logger = CorrectionLogger()
    corrector = InteractiveCorrector(logger)

    print("Brain Interactive Mode (type 'quit' to exit)")
    while True:
        text = input("\n> ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        u = engine.understand(text)
        corrected, modified = corrector.present_and_correct(text, u)
        if modified:
            print(f"[Corrected] → {corrected.type.value}")


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
    pipeline = VaultPipeline(vault_root=args.vault_path)
    from brain_contract import Understanding, NoteType
    u = Understanding(text=args.text, type=NoteType.OBSERVATION)

    if args.interactive:
        path, corrected = pipeline.process_with_correction(u, interactive=True)
    else:
        path = pipeline.process_safe(u)

    if path:
        print(f"[Vault] Saved: {path}")
    else:
        print("[Vault] Failed to save")


def main():
    parser = argparse.ArgumentParser(description="Unified Brain + Vault CLI")
    sub = parser.add_subparsers(dest="command")

    # Common args
    parser.add_argument("--vault-path", default="vault")
    parser.add_argument("--checkpoint", default="checkpoints/brain")

    # Brain subcommands
    brain_sub = sub.add_parser("brain")
    brain_cmds = brain_sub.add_subparsers(dest="brain_command")

    p = brain_cmds.add_parser("understand")
    p.add_argument("text")
    p.add_argument("--json", action="store_true")

    p = brain_cmds.add_parser("interactive")

    p = brain_cmds.add_parser("batch")
    p.add_argument("input")
    p.add_argument("--output", default="results.json")

    p = brain_cmds.add_parser("evaluate")
    p.add_argument("--test-data", required=True)
    p.add_argument("--output")

    p = brain_cmds.add_parser("retrain")
    p.add_argument("--epochs", type=int, default=5)

    # Vault subcommands
    vault_sub = sub.add_parser("vault")
    vault_cmds = vault_sub.add_subparsers(dest="vault_command")

    vault_cmds.add_parser("stats")
    vault_cmds.add_parser("health")

    p = vault_cmds.add_parser("search")
    p.add_argument("query")

    p = vault_cmds.add_parser("process")
    p.add_argument("text")

    # Corrections subcommands
    corr_sub = sub.add_parser("corrections")
    corr_cmds = corr_sub.add_subparsers(dest="corr_command")

    corr_cmds.add_parser("stats")

    p = corr_cmds.add_parser("export")
    p.add_argument("--output", default="corrections.jsonl")

    # Pipeline subcommands
    pipe_sub = sub.add_parser("pipeline")
    pipe_sub.add_argument("text")
    pipe_sub.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    # Dispatch
    if args.command == "brain":
        {
            "understand": cmd_brain_understand,
            "interactive": cmd_brain_interactive,
        }[args.brain_command](args)
    elif args.command == "vault":
        {
            "stats": cmd_vault_stats,
            "search": cmd_vault_search,
            "health": cmd_vault_health,
        }[args.vault_command](args)
    elif args.command == "corrections":
        {
            "stats": cmd_corrections_stats,
            "export": cmd_corrections_export,
        }[args.corr_command](args)
    elif args.command == "pipeline":
        cmd_pipeline(args)


if __name__ == "__main__":
    main()
```

---

### Step 5: Enhance `vault/__init__.py`

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

### Step 6: Tests

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

At merge time, `transcribe.py` wires:

```python
# transcribe.py — interactive mode
if args.interactive and brain_engine and vault_pipeline:
    corrected, modified = vault_pipeline.process_with_correction(
        understanding, interactive=True
    )
    if modified:
        print("[Vault] Correction logged")
```

And the unified CLI becomes the single entry point:

```bash
# One CLI to rule them all
python scripts/unified_cli.py brain understand "Fix the VAD"
python scripts/unified_cli.py vault search "what projects?"
python scripts/unified_cli.py pipeline --interactive "Meeting with Sarah"
python scripts/unified_cli.py corrections stats
```

---

## Commit Messages

```
feat(vault): add correction vault storage
feat(vault): enhance pipeline with correction support
feat(vault): add interactive mode to transcribe.py
feat(vault): add unified brain + vault CLI
feat(vault): add correction exports and stats
test(vault): add correction and integration tests
```
