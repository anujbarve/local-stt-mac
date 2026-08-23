# Instance B: Phase 7 — Vault Deployment & Security

## Your Mission (Phase 7)

Make the vault deployable, secure, and resilient — Docker packaging, CI/CD, security hardening, circuit breakers, performance at scale, and documentation. The vault becomes production-grade infrastructure.

**You are the vault. You deploy. You secure. You scale. You document.**

## Rules

1. **DO NOT** touch `brain/` modules — only import from `brain_contract.py`
2. **DO NOT** modify `brain_contract.py` (FROZEN)
3. **DO NOT** modify `transcribe.py` — modified during merge only
4. **DO NOT** modify `scripts/brain_cli.py` — Instance A owns
5. **DO** import from `brain_contract.py` for shared types
6. **DO** import from `vault/` modules
7. **DO** write tests for every new module
8. **DO** handle errors gracefully
9. **DO** keep all imports within `vault/`

## File Ownership

```
Dockerfile              # NEW: container image
docker-compose.yml      # NEW: orchestration
Makefile                # NEW: common commands
pyproject.toml          # ENHANCE: package config
.github/workflows/ci.yml # NEW: CI/CD pipeline
vault/security.py       # NEW: security hardening
vault/resilience.py     # NEW: circuit breaker, retry
vault/index.py          # ENHANCE: memory optimization
vault/search.py         # ENHANCE: sharded search
vault/reader.py         # ENHANCE: lazy loading
vault/api.py            # ENHANCE: auth, rate limiting
docs/API.md             # NEW: API reference
docs/DEVELOPER.md       # NEW: developer guide
docs/DEPLOYMENT.md      # NEW: deployment guide
vault/tests/            # NEW tests
tests/test_deployment.py # NEW: E2E deployment tests
scripts/unified_cli.py  # ENHANCE: new commands
```

---

## Build Order

### Step 1: `Dockerfile`

```dockerfile
# Multi-stage build
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["python", "-m", "vault.api", "serve"]
```

### Step 2: `docker-compose.yml`

```yaml
version: "3.8"
services:
  vault:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./vault_data:/app/vault_data
      - ./checkpoints:/app/checkpoints
    environment:
      - VAULT_PATH=/app/vault_data
      - LOG_LEVEL=INFO
```

### Step 3: `Makefile`

```makefile
.PHONY: test train serve build clean

test:
	KMP_DUPLICATE_LIB_OK=TRUE python -m pytest brain/tests/ vault/tests/ -v

train:
	python scripts/train.py --quick

serve:
	python -m vault.api serve

build:
	docker build -t second-brain .

lint:
	ruff check brain/ vault/

typecheck:
	mypy brain/ vault/ --ignore-missing-imports
```

### Step 4: `vault/security.py`

```python
class InputSanitizer:
    """Sanitize user input for safety."""

    MAX_TEXT_LENGTH = 10000

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Strip control chars, limit length."""
        ...

    @staticmethod
    def validate_path(path: str, base_dir: str) -> bool:
        """Prevent directory traversal."""
        ...

    @staticmethod
    def validate_type(value: str, allowed: List[str]) -> bool:
        """Validate against allowed values."""
        ...


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_tokens: int = 100, refill_rate: float = 10.0):
        ...

    def allow(self, key: str, tokens: int = 1) -> bool:
        """Check if request is allowed."""
        ...


class APIKeyAuth:
    """Simple API key authentication."""

    def __init__(self, valid_keys: List[str]):
        ...

    def verify(self, key: Optional[str]) -> bool:
        """Verify API key."""
        ...
```

### Step 5: `vault/resilience.py`

```python
class CircuitBreaker:
    """Circuit breaker for external calls."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.state = "closed"  # closed, open, half-open
        ...

    def call(self, func, *args, **kwargs):
        """Execute with circuit breaker protection."""
        ...


class RetryPolicy:
    """Exponential backoff retry."""

    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        ...

    def execute(self, func, *args, **kwargs):
        """Execute with retry."""
        ...


class HealthAggregator:
    """Aggregate health from multiple components."""

    def check_all(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Check health of all components."""
        ...
```

### Step 6: CI/CD `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: KMP_DUPLICATE_LIB_OK=TRUE pytest brain/tests/ vault/tests/ -v
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install ruff
      - run: ruff check brain/ vault/
```

### Step 7: Documentation

#### `docs/API.md`
Full REST API reference with endpoints, request/response examples.

#### `docs/DEVELOPER.md`
Architecture overview, how to add modules, testing guide, contributing.

#### `docs/DEPLOYMENT.md`
Docker deployment, manual setup, cloud deployment, environment variables.

---

## Tests

#### `vault/tests/test_security.py`
```python
class TestInputSanitizer:
    def test_sanitize_text(self): ...
    def test_validate_path(self): ...
    def test_validate_type(self): ...

class TestRateLimiter:
    def test_allow_within_limit(self): ...
    def test_deny_over_limit(self): ...
    def test_refill(self): ...

class TestAPIKeyAuth:
    def test_valid_key(self): ...
    def test_invalid_key(self): ...
    def test_missing_key(self): ...
```

#### `vault/tests/test_resilience.py`
```python
class TestCircuitBreaker:
    def test_closed_allows_calls(self): ...
    def test_open_blocks_calls(self): ...
    def test_half_open_recovery(self): ...

class TestRetryPolicy:
    def test_retry_succeeds(self): ...
    def test_retry_exhausted(self): ...
```

#### `tests/test_deployment.py`
```python
class TestDeployment:
    def test_dockerfile_exists(self): ...
    def test_makefile_works(self): ...
    def test_api_health_endpoint(self): ...
    def test_brain_cli_export(self): ...
```
