# Instance A: Phase 5 — Brain Production Hardening

## Your Mission (Phase 5)

Make the brain model deployable, calibrate confidence, support incremental learning, and enable A/B testing. The brain becomes a production-grade inference engine.

**You are the brain. You get precise. You get deployable. You get measurable.**

## Rules

1. **DO NOT** touch `vault/`, `transcribe.py`, `scripts/unified_cli.py`, or any voiceassist module
2. **DO NOT** modify `brain_contract.py` (FROZEN unless validation helpers needed)
3. **DO** import from `brain_contract.py` for shared types
4. **DO** write tests for every new module
5. **DO** handle errors gracefully
6. **DO** keep all imports within `brain/`

## File Ownership (STRICT)

```
brain/model.py          # ENHANCE: model versioning
brain/train.py          # ENHANCE: incremental learning, ONNX
brain/understand.py     # ENHANCE: confidence calibration, A/B
brain/evaluate.py       # ENHANCE: calibration metrics, A/B
brain/tests/            # NEW tests
scripts/brain_cli.py    # ENHANCE: export command
```

---

## Build Order

### Step 1: Model Versioning in `brain/model.py`

Add version tracking to checkpoints:

```python
class BrainModel:
    # ... existing ...

    def save(self, path: str, version: str = "0.1.0", metadata: Dict = None) -> None:
        """Save model with version metadata."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "version": version,
            "metadata": metadata or {},
            "d_model": self.d_model,
            "num_heads": self.num_heads,
        }, path)

    @classmethod
    def load(cls, path: str) -> "BrainModel":
        """Load model, extracting version info."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            d_model=checkpoint.get("d_model", 256),
            num_heads=checkpoint.get("num_heads", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    @classmethod
    def load_versions(cls, checkpoint_dir: str) -> List[Dict[str, Any]]:
        """List all model versions in a directory."""
        versions = []
        for fname in os.listdir(checkpoint_dir):
            if fname.endswith(".pt"):
                path = os.path.join(checkpoint_dir, fname)
                try:
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    versions.append({
                        "file": fname,
                        "version": ckpt.get("version", "unknown"),
                        "metadata": ckpt.get("metadata", {}),
                    })
                except Exception:
                    continue
        return sorted(versions, key=lambda x: x["version"])
```

---

### Step 2: ONNX Export in `scripts/brain_cli.py`

Add export command:

```bash
python scripts/brain_cli.py export --onnx --output checkpoints/brain/model.onnx
python scripts/brain_cli.py export --versions  # list available versions
```

Implementation:

```python
def cmd_export(args):
    """Export model to ONNX or list versions."""
    from brain.understand import BrainEngine
    from brain.train import export_onnx

    if args.versions:
        from brain.model import BrainModel
        versions = BrainModel.load_versions(args.checkpoint)
        print(json.dumps(versions, indent=2))
        return

    engine = BrainEngine.load(args.checkpoint)
    output = args.output or os.path.join(args.checkpoint, "model.onnx")
    export_onnx(engine.model, engine.tokenizer, output)
    print(f"Model exported to {output}")
```

---

### Step 3: Confidence Calibration in `brain/understand.py`

Add temperature scaling for better calibrated confidence:

```python
class BrainEngine:
    # ... existing ...

    def __init__(self, ...):
        # ... existing init ...
        self.temperature = 1.0  # learned temperature

    def calibrate(self, val_data: List[Dict], tokenizer: Tokenizer) -> float:
        """Learn temperature on validation set.

        Args:
            val_data: List of {text, type} dicts.
            tokenizer: Tokenizer instance.

        Returns:
            Learned temperature value.
        """
        # Collect raw logits and true labels
        logits_list = []
        labels_list = []

        for sample in val_data:
            token_ids = tokenizer.encode(sample["text"], add_special=True)
            token_tensor = torch.tensor([token_ids], dtype=torch.long)
            with torch.no_grad():
                outputs = self.model(token_tensor)
            logits_list.append(outputs["classifier"][0])
            labels_list.append(TYPE_TO_IDX.get(sample["type"], 0))

        all_logits = torch.stack(logits_list)
        all_labels = torch.tensor(labels_list)

        # Grid search for temperature
        best_temp = 1.0
        best_nll = float("inf")
        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            scaled = all_logits / temp
            nll = F.cross_entropy(scaled, all_labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp

        self.temperature = best_temp
        return best_temp

    def _compute_confidence(self, outputs: Dict[str, torch.Tensor]) -> float:
        """Compute confidence with temperature scaling."""
        type_logits = outputs["classifier"][0] / self.temperature
        type_probs = torch.softmax(type_logits, dim=0)
        raw_conf = type_probs.max().item()

        # Combine with other heads
        confidences = [raw_conf]
        # ... (rest same as before)
        return sum(confidences) / len(confidences)
```

---

### Step 4: Incremental Learning in `brain/train.py`

Add `incremental_train()` for fine-tuning on corrections:

```python
class BrainTrainer:
    # ... existing ...

    def incremental_train(
        self,
        new_data: List[Dict],
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 1e-5,  # lower LR for fine-tuning
    ) -> Dict[str, float]:
        """Fine-tune on new correction data (freezes backbone).

        Args:
            new_data: New correction samples.
            epochs: Training epochs (fewer than full train).
            batch_size: Batch size.
            lr: Learning rate (lower for fine-tuning).

        Returns:
            Training metrics.
        """
        # Freeze backbone, only train heads
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Create optimizer for heads only
        head_params = []
        for name, param in self.model.named_parameters():
            if "backbone" not in name and param.requires_grad:
                head_params.append(param)

        optimizer = torch.optim.AdamW(head_params, lr=lr)
        loader = self._make_loader(new_data, batch_size, shuffle=True)

        metrics = MetricsTracker()
        for epoch in range(epochs):
            metrics.reset()
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(batch["token_ids"])
                losses = compute_loss(outputs, batch)
                losses["total"].backward()
                optimizer.step()
                metrics.update(outputs, batch)

        # Unfreeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = True

        return metrics.compute()
```

---

### Step 5: A/B Testing in `brain/understand.py`

Add model registry for comparing versions:

```python
class ModelRegistry:
    """Track and compare multiple model versions."""

    def __init__(self):
        self.models: Dict[str, BrainEngine] = {}
        self.results: Dict[str, List[Dict]] = {}

    def register(self, name: str, engine: BrainEngine) -> None:
        """Register a model version."""
        self.models[name] = engine
        self.results[name] = []

    def compare(
        self,
        texts: List[str],
    ) -> Dict[str, Any]:
        """Run all registered models on texts and compare.

        Returns:
            {
                "model_a": {"type_accuracy": 0.85, "avg_confidence": 0.78},
                "model_b": {"type_accuracy": 0.82, "avg_confidence": 0.81},
                "agreement_rate": 0.92,
            }
        """
        all_predictions = {}
        for name, engine in self.models.items():
            predictions = []
            for text in texts:
                u = engine.understand(text)
                predictions.append({
                    "type": u.type.value,
                    "confidence": u.confidence,
                })
            all_predictions[name] = predictions

        # Compute agreement rate
        model_names = list(all_predictions.keys())
        if len(model_names) >= 2:
            agree = sum(
                1 for i in range(len(texts))
                if all_predictions[model_names[0]][i]["type"]
                == all_predictions[model_names[1]][i]["type"]
            )
            agreement_rate = agree / len(texts)
        else:
            agreement_rate = 1.0

        # Compute per-model metrics
        result = {}
        for name, preds in all_predictions.items():
            result[name] = {
                "avg_confidence": sum(p["confidence"] for p in preds) / len(preds),
                "predictions": preds,
            }
        result["agreement_rate"] = agreement_rate

        return result
```

---

### Step 6: Tests

#### `brain/tests/test_model_versioning.py`
```python
class TestModelVersioning:
    def test_save_with_version(self): ...
    def test_load_preserves_version(self): ...
    def test_load_versions ListsAll(self): ...
```

#### `brain/tests/test_calibration.py`
```python
class TestConfidenceCalibration:
    def test_calibrate_returns_temperature(self): ...
    def test_calibrated_confidence_in_range(self): ...
    def test_temperature_affects_confidence(self): ...
```

#### `brain/tests/test_incremental.py`
```python
class TestIncrementalLearning:
    def test_incremental_train_returns_metrics(self): ...
    def test_backbone_frozen_during_incremental(self): ...
    def test_backbone_unfrozen_after(self): ...
```

#### `brain/tests/test_ab_testing.py`
```python
class TestABTesting:
    def test_register_model(self): ...
    def test_compare_two_models(self): ...
    def test_agreement_rate(self): ...
```

---

## Commit Messages

```
feat(brain): add model versioning to checkpoints
feat(brain): add ONNX export to brain CLI
feat(brain): add confidence calibration with temperature scaling
feat(brain): add incremental learning for corrections
feat(brain): add A/B testing framework with model registry
test(brain): add versioning, calibration, incremental, A/B tests
```
