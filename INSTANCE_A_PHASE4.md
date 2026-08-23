# Instance A: Phase 4 — Interactive Learning & Brain CLI

## Your Mission (Phase 4)

Build the interactive correction system and knowledge distillation pipeline. The brain learns from user feedback and improves over time. Enhance the brain CLI with new commands.

**You are the brain. You learn. You improve. You get smarter every day.**

## Rules

1. **DO NOT** touch `vault/`, `transcribe.py`, `scripts/unified_cli.py`, or any voiceassist module
2. **DO NOT** modify `brain_contract.py` (FROZEN)
3. **DO** import from `brain_contract.py` for shared types
4. **DO** write tests for every new module
5. **DO** handle errors gracefully — corrections should never crash the system
6. **DO** keep all imports within `brain/` — no cross-boundary imports

## File Ownership (STRICT)

### Instance A owns (ONLY touches these):
```
brain/interactive.py      # NEW: interactive correction loop
brain/distill.py          # NEW: knowledge distillation (LLM → NN)
brain/evaluate.py         # NEW: model evaluation metrics
brain/correct.py          # ENHANCE: add get_stats(), export_training_data(), import_training_data()
brain/tests/              # NEW tests
scripts/brain_cli.py      # ENHANCE: add evaluate, corrections, retrain commands
```

### OFF LIMITS:
```
vault/                    # Instance B owns
transcribe.py             # Modified during merge only
brain_contract.py         # FROZEN
scripts/unified_cli.py    # Instance B owns
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
                "confusion_matrix": {...},
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

### Step 4: Enhance `brain/correct.py`

**Add to existing CorrectionLogger:**

```python
class CorrectionLogger:
    # ... existing methods (log, get_training_data, should_retrain, etc.) ...

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
        """Export corrections as JSONL for external training.
        Returns number of corrections exported."""
        ...

    def import_training_data(self, path: str) -> int:
        """Import corrections from JSONL.
        Returns number of corrections imported."""
        ...
```

---

### Step 5: Enhance `scripts/brain_cli.py`

**Add new commands to existing CLI:**

```bash
# Evaluate model
python scripts/brain_cli.py evaluate --test-data data/test.json

# Show correction stats
python scripts/brain_cli.py corrections --stats

# Export corrections
python scripts/brain_cli.py corrections --export corrections.jsonl

# Retrain from corrections
python scripts/brain_cli.py retrain --epochs 5
```

**Implementation — add these handler functions:**

```python
def cmd_evaluate(args):
    """Evaluate model on test data."""
    from brain.understand import BrainEngine
    from brain.evaluate import BrainEvaluator

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
    from brain.correct import CorrectionLogger
    logger = CorrectionLogger()
    if args.stats:
        stats = logger.get_stats()
        print(json.dumps(stats, indent=2))
    elif args.export:
        count = logger.export_training_data(args.export)
        print(f"Exported {count} corrections")


def cmd_retrain(args):
    """Retrain from corrections."""
    from brain.understand import BrainEngine
    from brain.distill import DistillationPipeline
    from brain.correct import CorrectionLogger

    engine = BrainEngine.load(args.checkpoint)
    logger = CorrectionLogger()
    distiller = DistillationPipeline(engine, llm=None, correction_logger=logger)
    if distiller.should_retrain():
        metrics = distiller.retrain(epochs=args.epochs)
        print(json.dumps(metrics, indent=2))
        engine.save(args.checkpoint)
        print("Model retrained and saved.")
    else:
        print("Not enough corrections for retraining.")
```

**Add subparsers in main():**

```python
# evaluate
p_evaluate = subparsers.add_parser("evaluate")
p_evaluate.add_argument("--test-data", required=True)
p_evaluate.add_argument("--output")
p_evaluate.add_argument("--checkpoint", default="checkpoints/brain")

# corrections
p_corrections = subparsers.add_parser("corrections")
p_corrections.add_argument("--stats", action="store_true")
p_corrections.add_argument("--export")

# retrain
p_retrain = subparsers.add_parser("retrain")
p_retrain.add_argument("--epochs", type=int, default=5)
p_retrain.add_argument("--checkpoint", default="checkpoints/brain")
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

## Commit Messages

```
feat(brain): add interactive correction loop
feat(brain): add knowledge distillation pipeline
feat(brain): add model evaluation metrics
feat(brain): enhance correction buffer with stats and export
feat(brain): add evaluate, corrections, retrain commands to brain CLI
test(brain): add interactive, distillation, and evaluation tests
```
