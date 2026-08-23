# Instance A: Brain Enhancements & Training CLI

## Your Mission (Phase 2)

Enhance the brain module with real training capabilities, evaluation metrics, and CLI tools. You own the intelligence layer — making the model actually learn and improve.

**You are the brain. You train. You evaluate. You improve.**

## Rules

1. **DO NOT** touch `vault/`, `transcribe.py`, or any vault module
2. **DO NOT** modify `brain_contract.py`
3. **DO** import types from `brain_contract.py`
4. **DO** keep your modules self-contained
5. **DO** write tests for every new module
6. **DO** create `scripts/` directory for CLI tools

## File Ownership

### Instance A owns (ONLY touches these):
```
brain/train.py          # ENHANCE (add MetricsTracker, LR scheduling, early stopping)
brain/data.py           # ENHANCE (add load_vault_data, DataAugmenter)
brain/understand.py     # ENHANCE (batch optimization, confidence calibration)
brain/correct.py        # ENHANCE (correction→retrain loop)
scripts/                # NEW DIRECTORY
scripts/train.py        # NEW: CLI training pipeline
scripts/evaluate.py     # NEW: CLI evaluation script
tests/test_train.py     # NEW: training tests
tests/test_correct.py   # NEW: correction tests
```

### OFF LIMITS (do not touch):
```
vault/                  # Instance B owns this
transcribe.py           # Only during merge (Week 9)
brain_contract.py       # Frozen
```

---

## Build Order

### Step 1: `brain/train.py` — Enhance Training Pipeline

Add these classes/functions to the existing `brain/train.py`:

#### 1a. `MetricsTracker` class

```python
class MetricsTracker:
    """Track per-head accuracy, F1, and loss across epochs."""

    def __init__(self): ...

    def update(self, outputs: Dict, batch: Dict, losses: Dict) -> None:
        """Accumulate predictions and labels for one batch."""

    def compute(self) -> Dict[str, float]:
        """Compute epoch-level metrics.
        Returns dict with: classifier_acc, priority_mae, urgency_acc,
        sentiment_acc, entity_f1, entity_precision, entity_recall,
        topic_f1, topic_precision, topic_recall
        """

    def reset(self) -> None:
        """Clear accumulated predictions for next epoch."""
```

**Metric formulas:**
- `classifier_acc` = `argmax(pred) == label`, averaged over batch
- `priority_mae` = `abs(pred - label).mean()`
- `entity_f1` = flatten BIO tags, compute micro-averaged F1 ignoring O-tag
- `topic_f1` = sigmoid threshold at 0.5, compute micro-averaged F1

#### 1b. Learning Rate Scheduling

Add to `BrainTrainer.__init__`:
```python
def __init__(self, ..., scheduler_type: str = "cosine", patience: int = 5):
    # scheduler_type: "cosine", "plateau", "step", "none"
    # patience: early stopping patience
```

- `"cosine"` → `CosineAnnealingLR(optimizer, T_max=epochs)`
- `"plateau"` → `ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)`
- `"step"` → `StepLR(optimizer, step_size=5, gamma=0.5)`

#### 1c. Early Stopping

In `train()` loop:
```python
early_stop_counter = 0
best_val_loss = float("inf")

for epoch in range(1, epochs + 1):
    # ... train and evaluate ...

    if val_losses["total"] < best_val_loss:
        best_val_loss = val_losses["total"]
        early_stop_counter = 0
        # save best checkpoint
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

#### 1d. Data Loading from Vault

```python
def load_vault_data(vault_root: str) -> List[Dict]:
    """Read notes from vault/ and convert to training format.

    Maps frontmatter fields to training labels:
        type → "type" (note type string)
        priority → "priority" (0-1 from Priority.to_score())
        urgency → "urgency" (string)
        sentiment → "sentiment" (string)
        people/tools/projects → "entities" (list of {name, type})
        tags → "topics" (list of strings)
        body text → "text" (string)
    """
```

#### 1e. Data Augmentation

```python
class DataAugmenter:
    def __init__(self, seed: int = 42): ...
    def augment(self, data: List[Dict], factor: int = 2) -> List[Dict]: ...
    def _synonym_replace(self, text: str, n: int = 1) -> str: ...
    def _random_delete(self, text: str, p: float = 0.1) -> str: ...
    def _random_swap(self, text: str, n: int = 1) -> str: ...
```

Use a simple synonym dict (no NLTK dependency):
```python
SYNONYMS = {
    "fix": ["repair", "resolve", "patch"],
    "update": ["revise", "modify", "refresh"],
    "build": ["create", "construct", "develop"],
    "test": ["verify", "validate", "check"],
    "deploy": ["ship", "release", "launch"],
    # ... 30+ word groups
}
```

#### 1f. ONNX Export

```python
def export_onnx(
    model: BrainModel,
    tokenizer: Tokenizer,
    output_path: str = "checkpoints/brain/model.onnx",
) -> None:
    """Export model to ONNX for faster CPU inference."""
```

#### 1g. History Export

```python
def save_history(history: List[Dict], path: str) -> None: ...
def load_history(path: str) -> List[Dict]: ...
```

---

### Step 2: `scripts/train.py` — CLI Training Pipeline

Create `scripts/` directory and `scripts/train.py`.

```bash
# Usage:
python scripts/train.py                          # full pipeline
python scripts/train.py --synthetic 15000        # custom sample count
python scripts/train.py --data data.json          # load from file
python scripts/train.py --vault vault/            # load from vault
python scripts/train.py --epochs 20 --lr 1e-4     # custom hyperparams
python scripts/train.py --resume model_epoch10.pt # resume training
python scripts/train.py --scheduler cosine        # LR schedule
python scripts/train.py --patience 5              # early stopping
python scripts/train.py --augment                 # enable augmentation
python scripts/train.py --quick                   # test mode (1000 samples, 5 epochs)
python scripts/train.py --export-onnx             # export after training
python scripts/train.py --verbose                 # detailed output
```

**Pipeline steps:**
1. Generate or load data
2. Split train/val (85/15)
3. Train tokenizer on corpus
4. Create BrainModel + BrainTrainer
5. Train with metrics tracking
6. Save checkpoint + tokenizer + history
7. Print summary

**Expected output:**
```
============================================================
Brain Training Pipeline
============================================================

[1/5] Preparing data...
  Generating 15000 synthetic samples...

[2/5] Splitting data (15000 samples)...
  Train: 12750, Val: 2250

[3/5] Training tokenizer...
  Vocab size: 4523

[4/5] Training model (20 epochs)...
  Epoch  1/20 | train_loss=5.4321 | val_loss=5.1234 | acc=0.32
  Epoch  2/20 | train_loss=4.8765 | val_loss=4.6543 | acc=0.41
  ...
  Epoch 20/20 | train_loss=1.2345 | val_loss=1.3456 | acc=0.78

[5/5] Saving...
  Done: checkpoints/brain/model.pt

============================================================
Training Complete
============================================================
  Time: 45.2s
  Epochs: 20
  Final train loss: 1.2345
  Final val loss: 1.3456
  Best val accuracy: 78.3%
  Checkpoints: checkpoints/brain/
```

---

### Step 3: `scripts/evaluate.py` — CLI Evaluation Script

```bash
# Usage:
python scripts/evaluate.py                            # evaluate on test set
python scripts/evaluate.py --data test.json           # custom test data
python scripts/evaluate.py --vault vault/             # evaluate against vault
python scripts/evaluate.py --output reports/eval.json # save report
python scripts/evaluate.py --confusion                # print confusion matrix
python scripts/evaluate.py --per-type                 # per-type accuracy
python scripts/evaluate.py --quick                    # smoke test
```

**Metrics to compute:**
- Type accuracy (top-1)
- Priority MAE
- Urgency accuracy
- Sentiment accuracy
- Entity F1 (micro-averaged)
- Topic F1 (micro-averaged)
- Confusion matrix (text-based)
- Per-type accuracy breakdown

**Expected output:**
```
============================================================
Brain Evaluation
============================================================

[1/3] Loading model...
[2/3] Loading test data... 1000 test samples
[3/3] Evaluating...

============================================================
Results
============================================================
  Type accuracy:      78.3%
  Priority MAE:       0.142
  Urgency accuracy:   65.1%
  Sentiment accuracy: 72.8%
  Entity F1:          54.2%
  Topic F1:           43.7%

Per-Type Accuracy:
  task                 94.7% (142/150)
  meeting              90.0% (135/150)
  reminder             93.3% (140/150)
  person_info          92.0% (138/150)
  preference           94.0% (141/150)
  learning             87.3% (131/150)
  observation          84.7% (127/150)
  idea                 85.3% (128/150)
  decision             78.0% (78/100)
  ...
```

---

### Step 4: Enhance `brain/correct.py`

Add retrain integration:

```python
class CorrectionLogger:
    # ... existing methods ...

    def maybe_retrain(
        self,
        trainer: "BrainTrainer",
        min_corrections: int = 20,
        head_only: bool = True,
    ) -> bool:
        """Check if enough corrections accumulated, then retrain.

        Args:
            trainer: BrainTrainer instance
            min_corrections: threshold to trigger retraining
            head_only: if True, retrain only the affected head (fast)

        Returns True if retraining was triggered.
        """
```

---

### Step 5: Enhance `brain/understand.py`

Add batch optimization:

```python
class BrainEngine:
    # ... existing methods ...

    def understand_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[Understanding]:
        """Batch inference with proper batching (not sequential).

        Groups texts into batches, runs single forward pass per batch.
        Much faster than calling understand() in a loop.
        """
```

---

### Step 6: Tests

#### `tests/test_train.py`

```python
class TestMetricsTracker:
    def test_update_and_compute(self): ...
    def test_reset_clears_state(self): ...
    def test_classifier_accuracy(self): ...
    def test_priority_mae(self): ...

class TestBrainTrainerEnhanced:
    def test_cosine_scheduler(self): ...
    def test_plateau_scheduler(self): ...
    def test_early_stopping(self): ...
    def test_save_and_load_history(self): ...

class TestDataAugmenter:
    def test_augment_increases_size(self): ...
    def test_synonym_replace(self): ...
    def test_random_delete(self): ...
    def test_random_swap(self): ...

class TestLoadVaultData:
    def test_load_from_vault(self, tmp_vault): ...
    def test_empty_vault(self, tmp_path): ...
```

#### `tests/test_correct.py`

```python
class TestCorrectionLoggerEnhanced:
    def test_maybe_retrain_triggered(self): ...
    def test_maybe_retrain_not_triggered(self): ...
```

---

## Dependencies

All already in the project:
```
torch>=2.0.0
numpy>=1.24.0
```

Optional (for ONNX):
```
onnx>=1.16.0
onnxruntime>=1.18.0
```

---

## Checkpoint Format

After training, `checkpoints/brain/` should contain:
```
checkpoints/brain/
├── model.pt              # model state dict + config
├── model_best.pt         # best validation checkpoint
├── tokenizer.json        # tokenizer vocabulary
├── history.json          # training history
└── model.onnx            # ONNX export (if --export-onnx)
```

---

## Commit Messages

Follow the convention:
```
feat(brain): add MetricsTracker for per-head accuracy tracking
feat(brain): add LR scheduling and early stopping to trainer
feat(brain): add vault data loader for real training data
feat(brain): add data augmentation pipeline
feat(brain): add ONNX export for model deployment
feat(scripts): add CLI training pipeline
feat(scripts): add CLI evaluation script
test(brain): add training and metrics tests
test(brain): add correction retrain tests
```

---

## Integration Point

At merge time (Week 15), your work connects to Instance B's work:

```python
# scripts/train.py can load vault data:
from brain.train import load_vault_data
data = load_vault_data("vault/")

# scripts/evaluate.py can evaluate against vault:
# (Instance B builds vault/search.py which you'll use later)
```

You do NOT import from `vault/` during parallel build. Only at merge time.
