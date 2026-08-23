# Instance A: Phase 4 — Interactive Learning & Brain CLI

## Your Mission (Phase 4)

Build the interactive correction system and knowledge distillation pipeline. The brain learns from user feedback and improves over time. You also build a standalone CLI so Instance B's scripts can use the brain directly.

**You are the brain. You learn. You improve. You get smarter every day.**

## Rules

1. **DO NOT** touch `vault/`, `transcribe.py`, or any voiceassist module
2. **DO NOT** modify `brain_contract.py`
3. **DO** import from `brain_contract.py` for shared types
4. **DO** write tests for every new module
5. **DO** handle errors gracefully — corrections should never crash the system

## File Ownership

### Instance B owns (ONLY touches these):
```
brain/interactive.py      # NEW: interactive correction loop
brain/distill.py          # NEW: knowledge distillation (LLM → NN)
brain/correct.py          # ENHANCE: correction buffer management
brain/evaluate.py         # NEW: model evaluation metrics
brain/tests/              # NEW tests
scripts/brain_cli.py      # NEW: standalone brain CLI
```

### Shared (both instances touch during integration):
```
brain_contract.py         # FROZEN
requirements.txt          # add new deps
```

### OFF LIMITS:
```
vault/                    # Instance B owns
transcribe.py             # Modified during merge only
brain_contract.py         # FROZEN
```

---

## Build Order

### Step 1: `brain/interactive.py` — Interactive Correction Loop

**Purpose:** After brain produces an `Understanding`, show it to the user for review/correction.

**Interface:**

```python
class InteractiveCorrector:
    """Interactive correction loop for brain output."""

    def __init__(self, correction_logger: CorrectionLogger):
        self.logger = correction_logger

    def present_and_correct(
        self,
        text: str,
        understanding: Understanding,
    ) -> Tuple[Understanding, bool]:
        """
        Show the understanding to the user and collect corrections.

        Returns:
            (corrected_understanding, was_modified)

        Display format:
        ┌─────────────────────────────────────────────────┐
        │ 📝 Transcription:                                │
        │ "Sarah told me her birthday is March 15th"      │
        │                                                  │
        │ 🧠 Understood:                                   │
        │ Type: person_info       [✓] [change]           │
        │ Person: Sarah           [✓] [change]           │
        │ Attribute: birthday     [✓] [change]           │
        │ Value: March 15         [✓] [change]           │
        │ Priority: low           [✓] [change]           │
        │                                                  │
        │ 💾 Save to vault? [Yes] [No] [Edit]            │
        └─────────────────────────────────────────────────┘
        """
        ...

    def _format_summary(self, understanding: Understanding) -> str:
        """Format understanding for display."""
        ...

    def _collect_corrections(
        self,
        original: Understanding,
    ) -> Optional[Understanding]:
        """Prompt user for corrections. Returns corrected version or None."""
        ...

    def _log_correction(
        self,
        text: str,
        predicted: Understanding,
        corrected: Understanding,
    ) -> None:
        """Log the correction to the buffer."""
        ...
```

**Display format (plain text for CLI):**

```
📝 "I need to fix the VAD threshold by Friday"

🧠 Type:        task           [✓] [c] [s]
   Priority:    0.82 (high)    [✓] [c] [s]
   Urgency:     soon           [✓] [c] [s]
   Action:      fix VAD threshold
   Entities:    VAD (tool), Friday (date)
   Topics:      technology, engineering

   [✓] Accept  [c] Change field  [s] Skip  [q] Quit
```

**Key:** `[c]` prompts which field to change. `[s]` skips without saving. `[q]` discards.

---

### Step 2: `brain/distill.py` — Knowledge Distillation

**Purpose:** The LLM (Gemma 3 1B) produces high-quality labels. The NN learns from them.

**Interface:**

```python
class DistillationPipeline:
    """LLM teaches the neural network through knowledge distillation."""

    def __init__(
        self,
        brain_engine: Any,       # BrainEngine from understand.py
        llm: Any,                # LocalLLM from vault/llm.py
        correction_logger: CorrectionLogger,
    ):
        self.brain = brain_engine
        self.llm = llm
        self.logger = correction_logger

    def process_transcription(self, text: str) -> Understanding:
        """
        Run both NN and LLM, compare, and log mismatches.

        Steps:
            1. NN predicts: nn_output = brain.understand(text)
            2. LLM predicts: llm_output = llm.analyze(text)
            3. Compare outputs
            4. If mismatch: log as correction (LLM is teacher)
            5. Return LLM output (higher quality)
        """
        ...

    def compare_outputs(
        self,
        nn_output: Understanding,
        llm_output: Understanding,
    ) -> Dict[str, Any]:
        """
        Compare NN and LLM outputs.

        Returns:
            {
                "match": False,
                "differences": [
                    {"field": "type", "nn": "observation", "llm": "task"},
                    {"field": "priority", "nn": 0.3, "llm": 0.8},
                ],
                "nn_confidence": 0.45,
            }
        """
        ...

    def should_retrain(self) -> bool:
        """Check if enough corrections accumulated for retraining."""
        return self.logger.should_retrain()

    def retrain(self, epochs: int = 5) -> Dict[str, float]:
        """
        Retrain the model on accumulated corrections.

        Returns training metrics.
        """
        ...

    def run_distillation_loop(
        self,
        texts: List[str],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a batch of texts through distillation.

        Returns:
            {
                "total": 100,
                "nn_correct": 72,
                "llm_corrections": 28,
                "corrections_logged": 28,
            }
        """
        ...
```

**Distillation strategy:**

```
Text: "I need to fix the VAD threshold by Friday"

NN prediction:
  type: observation (conf: 0.45) ← LOW CONFIDENCE
  priority: 0.3
  entities: []

LLM prediction:
  type: task (conf: 0.90)
  priority: 0.82
  entities: [VAD (tool), Friday (date)]
  action: "fix VAD threshold"

→ MISMATCH on type and priority
→ Log as correction: {text, nn_output, llm_output}
→ Use LLM output for vault writing
→ Retrain classifier head when 20+ corrections accumulated
```

---

### Step 3: `brain/evaluate.py` — Model Evaluation

**Purpose:** Measure model accuracy on test data.

**Interface:**

```python
class BrainEvaluator:
    """Evaluate brain model on test data."""

    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate(
        self,
        test_data: List[Dict],
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Returns:
            {
                "accuracy": {
                    "type": 0.82,
                    "priority_mae": 0.12,
                    "urgency": 0.75,
                    "sentiment": 0.88,
                    "entity_f1": 0.71,
                    "topic_f1": 0.65,
                },
                "confusion_matrix": {...},  # for type classification
                "per_type_accuracy": {...},
                "total_samples": 1000,
                "avg_confidence": 0.78,
            }
        """
        ...

    def confusion_matrix(
        self,
        test_data: List[Dict],
    ) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix for type classification."""
        ...

    def per_type_accuracy(
        self,
        test_data: List[Dict],
    ) -> Dict[str, float]:
        """Accuracy broken down by note type."""
        ...

    def error_analysis(
        self,
        test_data: List[Dict],
    ) -> List[Dict[str, Any]]:
        """
        Find the worst errors for debugging.

        Returns list of {text, predicted, actual, confidence} sorted by confidence.
        """
        ...

    def generate_report(
        self,
        test_data: List[Dict],
        output_path: str = "reports/eval.json",
    ) -> str:
        """Generate full evaluation report and save to JSON."""
        ...
```

---

### Step 4: `brain/correct.py` — Enhancement

**Add to existing CorrectionLogger:**

```python
class CorrectionLogger:
    # ... existing methods ...

    def get_stats(self) -> Dict[str, Any]:
        """Detailed correction statistics.

        Returns:
            {
                "total_corrections": 45,
                "pending": 28,
                "applied": 17,
                "by_field": {"type": 12, "priority": 8, "entities": 15, ...},
                "accuracy_before": 0.72,
                "accuracy_after": 0.81,
                "last_retrained": "2026-08-23T14:00:00Z",
            }
        """
        ...

    def export_training_data(self, path: str) -> int:
        """Export corrections as JSONL for external training."""
        ...

    def import_training_data(self, path: str) -> int:
        """Import corrections from JSONL."""
        ...
```

---

### Step 5: `scripts/brain_cli.py` — Standalone Brain CLI

**Purpose:** CLI tool for Instance B's scripts and direct brain usage.

**Interface:**

```bash
# Quick inference
python scripts/brain_cli.py understand "I need to fix the VAD threshold"

# JSON output
python scripts/brain_cli.py understand "TODO: fix the VAD" --json

# Interactive correction mode
python scripts/brain_cli.py interactive

# Batch processing
python scripts/brain_cli.py batch input.json --output results.json

# Evaluate model
python scripts/brain_cli.py evaluate --test-data data/test.json

# Show correction stats
python scripts/brain_cli.py corrections --stats

# Export corrections
python scripts/brain_cli.py corrections --export corrections.jsonl

# Retrain from corrections
python scripts/brain_cli.py retrain --epochs 5
```

**Implementation:**

```python
#!/usr/bin/env python3
"""Standalone brain CLI for inference, correction, and evaluation."""

import argparse
import json
import sys

from brain.understand import BrainEngine
from brain.correct import CorrectionLogger
from brain.interactive import InteractiveCorrector
from brain.evaluate import BrainEvaluator


def cmd_understand(args):
    """Run brain inference on text."""
    engine = BrainEngine.load(args.checkpoint)
    u = engine.understand(args.text)
    if args.json:
        print(json.dumps(u.to_dict(), indent=2))
    else:
        print(f"Type:        {u.type.value}")
        print(f"Priority:    {u.priority:.2f}")
        print(f"Urgency:     {u.urgency.value}")
        print(f"Sentiment:   {u.sentiment.value}")
        print(f"Action:      {u.action or 'none'}")
        print(f"Entities:    {[(e.name, e.entity_type.value) for e in u.entities]}")
        print(f"Topics:      {u.topics}")
        print(f"Confidence:  {u.confidence:.2f}")


def cmd_interactive(args):
    """Interactive correction loop."""
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
            print(f"[Saved] {corrected.type.value}: {corrected.action or corrected.text[:60]}")
        else:
            print("[Skipped]")


def cmd_batch(args):
    """Batch processing."""
    engine = BrainEngine.load(args.checkpoint)
    with open(args.input) as f:
        texts = json.load(f)

    results = []
    for text in texts:
        u = engine.understand(text)
        results.append(u.to_dict())

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Processed {len(results)} items → {args.output}")


def cmd_evaluate(args):
    """Evaluate model."""
    engine = BrainEngine.load(args.checkpoint)
    evaluator = BrainEvaluator(engine.model, engine.tokenizer)

    with open(args.test_data) as f:
        test_data = json.load(f)

    report = evaluator.evaluate(test_data)
    print(json.dumps(report, indent=2))

    if args.output:
        evaluator.generate_report(test_data, args.output)
        print(f"Report saved to {args.output}")


def cmd_corrections(args):
    """Manage corrections."""
    logger = CorrectionLogger()
    if args.stats:
        stats = logger.get_stats()
        print(json.dumps(stats, indent=2))
    elif args.export:
        count = logger.export_training_data(args.export)
        print(f"Exported {count} corrections")
    elif args.import_file:
        count = logger.import_training_data(args.import_file)
        print(f"Imported {count} corrections")


def cmd_retrain(args):
    """Retrain from corrections."""
    from brain.distill import DistillationPipeline
    engine = BrainEngine.load(args.checkpoint)
    logger = CorrectionLogger()
    # Minimal distillation for retraining
    distiller = DistillationPipeline(engine, llm=None, correction_logger=logger)
    if distiller.should_retrain():
        metrics = distiller.retrain(epochs=args.epochs)
        print(json.dumps(metrics, indent=2))
        engine.save(args.checkpoint)
        print("Model retrained and saved.")
    else:
        print("Not enough corrections for retraining.")


def main():
    parser = argparse.ArgumentParser(description="Brain CLI")
    sub = parser.add_subparsers(dest="command")

    # understand
    p = sub.add_parser("understand")
    p.add_argument("text")
    p.add_argument("--json", action="store_true")
    p.add_argument("--checkpoint", default="checkpoints/brain")

    # interactive
    p = sub.add_parser("interactive")
    p.add_argument("--checkpoint", default="checkpoints/brain")

    # batch
    p = sub.add_parser("batch")
    p.add_argument("input")
    p.add_argument("--output", default="results.json")
    p.add_argument("--checkpoint", default="checkpoints/brain")

    # evaluate
    p = sub.add_parser("evaluate")
    p.add_argument("--test-data", required=True)
    p.add_argument("--output")
    p.add_argument("--checkpoint", default="checkpoints/brain")

    # corrections
    p = sub.add_parser("corrections")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--export")
    p.add_argument("--import-file", dest="import_file")

    # retrain
    p = sub.add_parser("retrain")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--checkpoint", default="checkpoints/brain")

    args = parser.parse_args()
    {
        "understand": cmd_understand,
        "interactive": cmd_interactive,
        "batch": cmd_batch,
        "evaluate": cmd_evaluate,
        "corrections": cmd_corrections,
        "retrain": cmd_retrain,
    }[args.command](args)


if __name__ == "__main__":
    main()
```

---

### Step 6: Tests

#### `brain/tests/test_interactive.py`
```python
class TestInteractiveCorrector:
    def test_present_format(self): ...
    def test_log_correction(self): ...
    def test_no_correction_returns_original(self): ...
```

#### `brain/tests/test_distill.py`
```python
class TestDistillationPipeline:
    def test_compare_outputs_same(self): ...
    def test_compare_outputs_different(self): ...
    def test_process_transcription(self): ...
    def test_should_retrain(self): ...
```

#### `brain/tests/test_evaluate.py`
```python
class TestBrainEvaluator:
    def test_evaluate_returns_metrics(self): ...
    def test_confusion_matrix(self): ...
    def test_per_type_accuracy(self): ...
    def test_error_analysis(self): ...
    def test_generate_report(self): ...
```

---

## Integration Point

Instance B's `transcribe.py` will call:

```python
# In transcribe.py — interactive mode
from brain.interactive import InteractiveCorrector
from brain.correct import CorrectionLogger

corrector = InteractiveCorrector(CorrectionLogger())
corrected, modified = corrector.present_and_correct(text, understanding)
if modified:
    # Log correction for retraining
    pass
```

And Instance B's `scripts/brain_cli.py` will be the unified CLI that both instances use.

---

## Commit Messages

```
feat(brain): add interactive correction loop
feat(brain): add knowledge distillation pipeline
feat(brain): add model evaluation metrics
feat(brain): enhance correction buffer with stats and export
feat(brain): add standalone brain CLI
test(brain): add interactive, distillation, and evaluation tests
```
