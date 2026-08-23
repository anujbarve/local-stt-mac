# Instance A: Phase 3 — Integration & Polish

## Your Mission (Phase 3)

Wire the brain into the main pipeline, add the correction interface, build a unified CLI, and handle the LLM fallback. This is the final phase — make the system work end-to-end.

**You are the brain. You integrate. You correct. You polish.**

## Rules

1. **DO NOT** modify `vault/` modules directly — only import from them
2. **DO NOT** modify `brain_contract.py`
3. **DO** import from `brain_contract.py` for shared types
4. **DO** import from `vault.pipeline` for the integration wiring
5. **DO** write tests for every new module
6. **DO** handle errors gracefully — the system should never crash

## File Ownership

### Instance A owns (ONLY touches these):
```
brain/understand.py      # ENHANCE: add correction UI hook
brain/correct.py         # ENHANCE: add interactive correction mode
brain/config.py          # ENHANCE: add new config constants
brain/tests/             # NEW tests
scripts/brain_cli.py     # NEW: unified brain CLI
tests/test_integration.py # NEW: end-to-end integration tests
```

### Shared (both instances touch during integration):
```
transcribe.py            # BOTH modify: add brain + vault pipeline
requirements.txt         # add new deps
```

### OFF LIMITS:
```
vault/                   # Instance B owns
```

---

## Build Order

### Step 1: Enhance `brain/config.py`

Add integration constants:

```python
# Integration
DEFAULT_VAULT_PATH = "vault/"
DEFAULT_INDEX_PATH = "vault_index/"
DEFAULT_CHECKPOINT_DIR = "checkpoints/brain"

# LLM fallback
LLM_FALLBACK_ENABLED = True
LLM_MIN_CONFIDENCE = 0.50     # below this, ALWAYS use LLM
LLM_MAX_CONFIDENCE = 0.90     # above this, NEVER use LLM

# Correction
CORRECTION_BUFFER_PATH = "checkpoints/brain/corrections.jsonl"
CORRECTION_MIN_FOR_RETRAIN = 20
CORRECTION_HEAD_ONLY = True   # retrain only affected head

# Pipeline
PIPELINE_WRITE_ENABLED = True
PIPELINE_INDEX_ENABLED = True
PIPELINE_LINK_ENABLED = True
```

---

### Step 2: Enhance `brain/understand.py`

Add correction support and confidence calibration:

```python
class BrainEngine:
    # ... existing methods ...

    def suggest_correction(
        self,
        text: str,
        understanding: Understanding,
    ) -> Dict[str, Any]:
        """Present understanding to user and ask for corrections.

        Returns dict with:
            - "accepted": bool (user accepted the understanding)
            - "corrections": dict of fields the user changed
            - "corrected": Understanding (the corrected version)
        """

    def calibrate_confidence(
        self,
        understanding: Understanding,
        raw_outputs: Dict,
    ) -> float:
        """Calibrate model confidence using temperature scaling.

        Returns a better-calibrated confidence score.
        """

    def explain(self, understanding: Understanding) -> str:
        """Generate a human-readable explanation of the understanding.

        Example output:
            "I classified this as a TASK with HIGH priority.
             I detected 2 entities: VAD (tool), Friday (date).
             The action is 'fix VAD threshold'.
             Confidence: 87%"
        """
```

**Implementation notes:**
- `suggest_correction` is called by transcribe.py after understanding
- It prints the understanding to the user and asks for confirmation
- Returns corrected Understanding if user makes changes
- `explain` is used for verbose mode output

---

### Step 3: Enhance `brain/correct.py`

Add interactive correction mode:

```python
class CorrectionLogger:
    # ... existing methods ...

    def interactive_correct(
        self,
        text: str,
        understanding: Understanding,
    ) -> Optional[Understanding]:
        """Interactive correction flow.

        Shows understanding to user, lets them correct fields:
            1. Type (task/idea/observation/...)
            2. Priority (high/medium/low)
            3. Entities (add/remove/edit)
            4. Topics (add/remove)

        Returns corrected Understanding or None if user skips.
        """

    def batch_correct(
        self,
        corrections: List[Dict],
    ) -> int:
        """Process a batch of corrections from a JSON file.

        Input format:
            [{"text": "...", "type": "task", "priority": "high", ...}, ...]

        Returns number of corrections applied.
        """
```

---

### Step 4: Create `scripts/brain_cli.py`

Unified CLI for brain operations:

```bash
# Quick inference
python scripts/brain_cli.py understand "I need to fix the VAD threshold by Friday"

# Interactive mode
python scripts/brain_cli.py interactive

# Batch processing
python scripts/brain_cli.py batch input.json --output results.json

# Training
python scripts/brain_cli.py train --synthetic 15000 --epochs 20

# Evaluation
python scripts/brain_cli.py evaluate --confusion --per-type

# Stats
python scripts/brain_cli.py stats
```

**Implementation:**

```python
"""
scripts/brain_cli.py — Unified brain CLI.

Commands:
    understand TEXT      Quick inference on text
    interactive         Interactive correction mode
    batch INPUT OUTPUT  Batch process text file
    train [OPTIONS]     Train the model
    evaluate [OPTIONS]  Evaluate the model
    stats               Show model stats and correction buffer info
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Brain CLI")
    subparsers = parser.add_subparsers(dest="command")

    # understand
    p_understand = subparsers.add_parser("understand")
    p_understand.add_argument("text")
    p_understand.add_argument("--explain", action="store_true")
    p_understand.add_argument("--json", action="store_true")

    # interactive
    p_interactive = subparsers.add_parser("interactive")
    p_interactive.add_argument("--checkpoint", default="checkpoints/brain")

    # batch
    p_batch = subparsers.add_parser("batch")
    p_batch.add_argument("input")
    p_batch.add_argument("--output", default="output.json")

    # train
    p_train = subparsers.add_parser("train")
    p_train.add_argument("--synthetic", type=int, default=15000)
    p_train.add_argument("--epochs", type=int, default=20)
    p_train.add_argument("--quick", action="store_true")

    # evaluate
    p_evaluate = subparsers.add_parser("evaluate")
    p_evaluate.add_argument("--confusion", action="store_true")
    p_evaluate.add_argument("--per-type", action="store_true")

    # stats
    subparsers.add_parser("stats")

    args = parser.parse_args()

    if args.command == "understand":
        cmd_understand(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "stats":
        cmd_stats(args)
    else:
        parser.print_help()


def cmd_understand(args):
    from brain.understand import BrainEngine
    engine = BrainEngine.load("checkpoints/brain")
    understanding = engine.understand(args.text)

    if args.json:
        import json
        print(json.dumps(understanding.to_dict(), indent=2))
    elif args.explain:
        print(engine.explain(understanding))
    else:
        print(f"Type:       {understanding.type.value}")
        print(f"Priority:   {understanding.priority:.2f}")
        print(f"Urgency:    {understanding.urgency.value}")
        print(f"Sentiment:  {understanding.sentiment.value}")
        print(f"Confidence: {understanding.confidence:.2f}")
        if understanding.entities:
            print(f"Entities:   {[(e.name, e.entity_type.value) for e in understanding.entities]}")
        if understanding.action:
            print(f"Action:     {understanding.action}")
        if understanding.topics:
            print(f"Topics:     {understanding.topics}")


def cmd_interactive(args):
    from brain.understand import BrainEngine
    from brain.correct import CorrectionLogger

    engine = BrainEngine.load(args.checkpoint)
    logger = CorrectionLogger()

    print("Brain Interactive Mode (Ctrl+C to exit)")
    print("=" * 50)

    try:
        while True:
            text = input("\nYou: ").strip()
            if not text:
                continue

            understanding = engine.understand(text)
            print(f"\nBrain: {engine.explain(understanding)}")

            response = input("\nCorrect? [Y/n/edit]: ").strip().lower()
            if response == "n":
                print("Skipped.")
                continue
            elif response == "edit":
                corrected = logger.interactive_correct(text, understanding)
                if corrected:
                    print(f"Corrected: {corrected.type.value}, priority={corrected.priority:.2f}")
            else:
                print("Accepted.")

    except KeyboardInterrupt:
        print("\nExiting.")
        print(f"Buffer: {logger.stats()}")


def cmd_batch(args):
    import json
    from brain.understand import BrainEngine

    engine = BrainEngine.load("checkpoints/brain")

    with open(args.input) as f:
        texts = json.load(f)

    results = []
    for text in texts:
        u = engine.understand(text)
        results.append(u.to_dict())

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} texts → {args.output}")


def cmd_train(args):
    import subprocess
    cmd = [sys.executable, "scripts/train.py"]
    if args.synthetic:
        cmd.extend(["--synthetic", str(args.synthetic)])
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.quick:
        cmd.append("--quick")
    subprocess.run(cmd)


def cmd_evaluate(args):
    import subprocess
    cmd = [sys.executable, "scripts/evaluate.py"]
    if args.confusion:
        cmd.append("--confusion")
    if args.per_type:
        cmd.append("--per-type")
    subprocess.run(cmd)


def cmd_stats():
    from brain.correct import CorrectionLogger
    logger = CorrectionLogger()
    stats = logger.stats()

    print("Brain Stats")
    print("=" * 40)
    print(f"Corrections:  {stats['total']} total, {stats['unapplied']} unapplied")
    print(f"Model:        checkpoints/brain/model.pt")
    print(f"Tokenizer:    checkpoints/brain/tokenizer.json")
```

---

### Step 5: Add integration to `transcribe.py`

**This is the critical integration point.** Instance A adds the brain wiring:

In `transcribe.py`, after transcription (State 4), add:

```python
# State 4.5: Brain understanding
if brain_engine:
    understanding = brain_engine.understand(transcription)

    if args.verbose:
        print(f"[Brain] {brain_engine.explain(understanding)}")

    # If brain is uncertain, route to LLM
    if brain_engine.should_call_llm(understanding) and vault_pipeline:
        print("[Brain] Low confidence — routing to LLM...")
        if vault_llm:
            understanding = vault_llm.analyze(transcription)

    # Write to vault
    if vault_pipeline and understanding:
        file_path = vault_pipeline.process(understanding)
        print(f"[Vault] Saved to: {file_path}")

    # Offer correction
    if args.interactive and understanding:
        corrected = correction_logger.interactive_correct(transcription, understanding)
        if corrected:
            # Re-write with corrected understanding
            if vault_pipeline:
                vault_pipeline.process(corrected)

    # Print understanding
    if not args.verbose:
        print(f"  Type: {understanding.type.value} | Priority: {understanding.priority:.2f}")
```

**New CLI flags for transcribe.py:**
```python
parser.add_argument("--brain", action="store_true", help="Enable brain understanding")
parser.add_argument("--vault", action="store_true", help="Enable vault writing")
parser.add_argument("--interactive", action="store_true", help="Enable interactive correction")
parser.add_argument("--brain-checkpoint", default="checkpoints/brain", help="Brain model checkpoint")
parser.add_argument("--vault-path", default="vault/", help="Vault root path")
```

---

### Step 6: Error Handling & Logging

Add to `brain/config.py`:

```python
# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Error handling
MAX_RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0
```

Add a simple logger to brain modules:

```python
import logging

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for a brain module."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    return logger
```

---

### Step 7: Tests

#### `brain/tests/test_integration.py`

```python
class TestBrainVaultIntegration:
    def test_understand_then_write(self, tmp_vault):
        """Full pipeline: text → Understanding → vault write."""

    def test_should_call_llm_routes_correctly(self):
        """Low confidence triggers LLM path."""

    def test_correction_flow(self):
        """Text → Understanding → correction → corrected Understanding."""

    def test_batch_processing(self):
        """Process multiple texts end-to-end."""

    def test_explain_output(self):
        """explain() produces human-readable output."""
```

---

## What Instance B Needs From You

Instance B will wire the vault side of transcribe.py. They need you to:

1. **Define the interface** — `brain_engine.understand(text)` returns `Understanding`
2. **Define the LLM fallback** — `brain_engine.should_call_llm(understanding)` returns `bool`
3. **Define the correction interface** — `correction_logger.interactive_correct(text, understanding)` returns `Optional[Understanding]`

These are already defined in your existing modules. Just make sure they work.

## Commit Messages

```
feat(brain): add integration constants to config
feat(brain): add explain() and suggest_correction() to BrainEngine
feat(brain): add interactive correction mode to CorrectionLogger
feat(scripts): add unified brain CLI
feat(transcribe): wire brain understanding into pipeline
feat(transcribe): add --brain, --vault, --interactive flags
feat(brain): add logging and error handling
test(brain): add integration tests
```
