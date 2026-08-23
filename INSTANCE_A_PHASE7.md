# Instance A: Phase 7 — Brain Deployment & Performance

## Your Mission (Phase 7)

Package the brain for production — export pipeline, inference caching, batch optimization, and error recovery. The brain becomes a deployable, performant, resilient inference engine.

**You are the brain. You export. You cache. You optimize. You recover.**

## Rules

1. **DO NOT** touch `vault/`, `transcribe.py`, `scripts/unified_cli.py`
2. **DO NOT** modify `brain_contract.py` (FROZEN)
3. **DO** write tests for every new module
4. **DO** handle errors gracefully
5. **DO** keep all imports within `brain/`

## File Ownership

```
brain/export.py         # NEW: export pipeline
brain/understand.py     # ENHANCE: caching, batch optimization, error recovery
brain/model.py          # ENHANCE: retry, health check
brain/tests/            # NEW tests
scripts/brain_cli.py    # ENHANCE: export --all
```

---

## Build Order

### Step 1: `brain/export.py` — Export Pipeline

```python
class ModelExporter:
    """End-to-end export pipeline: train → evaluate → calibrate → export."""

    def __init__(self, engine: BrainEngine):
        self.engine = engine

    def export_all(self, output_dir: str) -> Dict[str, str]:
        """Export ONNX + TorchScript + model card.

        Returns:
            {
                "onnx": "path/to/model.onnx",
                "torchscript": "path/to/model.pt",
                "model_card": "path/to/model_card.json",
            }
        """
        ...

    def export_onnx(self, path: str) -> str:
        """Export to ONNX format."""
        ...

    def export_torchscript(self, path: str) -> str:
        """Export to TorchScript format."""
        ...

    def generate_model_card(self, path: str, metrics: Dict = None) -> str:
        """Generate model card JSON with metadata.

        Model card includes:
            - version, training_date, dataset_size
            - accuracy metrics (type, priority, urgency, etc.)
            - architecture (d_model, num_heads, num_layers)
            - export formats
        """
        ...

    def validate_export(self, onnx_path: str, test_texts: List[str]) -> Dict:
        """Verify exported model matches PyTorch inference.

        Returns:
            {"match_rate": 0.98, "max_diff": 0.001, "tested": 100}
        """
        ...
```

### Step 2: Inference Caching in `brain/understand.py`

```python
from functools import lru_cache

class BrainEngine:
    # ... existing ...

    def __init__(self, ...):
        # ... existing ...
        self._cache_enabled = True
        self._cache_hits = 0
        self._cache_misses = 0

    def understand(self, text: str, use_cache: bool = True) -> Understanding:
        """Understand with optional caching."""
        if use_cache and self._cache_enabled:
            cached = self._get_cached(text)
            if cached is not None:
                self._cache_hits += 1
                return cached
            self._cache_misses += 1

        result = self._understand_uncached(text)

        if use_cache and self._cache_enabled:
            self._set_cache(text, result)

        return result

    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0,
            "enabled": self._cache_enabled,
        }

    def clear_cache(self) -> None:
        """Clear inference cache."""
        ...
```

### Step 3: Batch Optimization in `brain/understand.py`

```python
    def understand_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        parallel: bool = False,
    ) -> List[Understanding]:
        """Optimized batch inference with dynamic batching.

        Groups texts by similar length for efficient padding.
        """
        ...
```

### Step 4: Error Recovery in `brain/model.py`

```python
class BrainModel:
    # ... existing ...

    @classmethod
    def load_with_retry(
        cls,
        path: str,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
    ) -> "BrainModel":
        """Load model with exponential backoff retry."""
        ...
```

### Step 5: Health Check in `brain/understand.py`

```python
    def health(self) -> Dict[str, Any]:
        """Check brain engine health.

        Returns:
            {
                "status": "healthy",
                "model_loaded": True,
                "tokenizer_ready": True,
                "temperature": 1.0,
                "cache_stats": {...},
                "device": "cpu",
            }
        """
        ...
```

---

## Tests

#### `brain/tests/test_export.py`
```python
class TestModelExporter:
    def test_export_all(self): ...
    def test_generate_model_card(self): ...
    def test_validate_export(self): ...
```

#### `brain/tests/test_cache.py`
```python
class TestInferenceCache:
    def test_cache_hit(self): ...
    def test_cache_miss(self): ...
    def test_cache_stats(self): ...
    def test_clear_cache(self): ...
```

#### `brain/tests/test_resilience.py`
```python
class TestBrainResilience:
    def test_load_with_retry(self): ...
    def test_health_check(self): ...
    def test_graceful_degradation(self): ...
```
