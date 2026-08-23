"""
Vault Security
==============

Input sanitization, rate limiting, and API key authentication.
Protects the vault from malicious input and abuse.
"""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional


class InputSanitizer:
    """Sanitize user input for safety."""

    MAX_TEXT_LENGTH = 10_000
    MAX_TAG_LENGTH = 100
    MAX_TAGS = 50

    @staticmethod
    def sanitize_text(text: str) -> str:
        """Strip control characters and limit length."""
        if not isinstance(text, str):
            return ""
        # Remove control characters (keep newlines and tabs)
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return sanitized[:InputSanitizer.MAX_TEXT_LENGTH]

    @staticmethod
    def sanitize_tag(tag: str) -> str:
        """Sanitize a single tag."""
        if not isinstance(tag, str):
            return ""
        tag = re.sub(r"[^\w\s-]", "", tag.lower().strip())
        tag = re.sub(r"\s+", "-", tag)
        return tag[: InputSanitizer.MAX_TAG_LENGTH]

    @staticmethod
    def sanitize_tags(tags: List[str]) -> List[str]:
        """Sanitize a list of tags."""
        sanitized = [InputSanitizer.sanitize_tag(t) for t in tags]
        return [t for t in sanitized if t][: InputSanitizer.MAX_TAGS]

    @staticmethod
    def validate_path(path: str, base_dir: str) -> bool:
        """Prevent directory traversal attacks.

        Returns True if *path* is safely inside *base_dir*.
        """
        try:
            base = os.path.realpath(base_dir)
            target = os.path.realpath(os.path.join(base_dir, path))
            return target.startswith(base + os.sep) or target == base
        except (ValueError, OSError):
            return False

    @staticmethod
    def validate_type(value: str, allowed: List[str]) -> bool:
        """Validate against a list of allowed values."""
        return value in allowed

    @staticmethod
    def validate_priority(value: float) -> float:
        """Clamp priority to valid range."""
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def validate_note_id(note_id: str) -> bool:
        """Validate note ID format."""
        return bool(re.match(r"^note_\d{4}_\d{2}_\d{2}_\d{3}$", note_id))


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, max_tokens: int = 100, refill_rate: float = 10.0):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate  # tokens per second
        self._buckets: Dict[str, Dict[str, Any]] = {}

    def allow(self, key: str, tokens: int = 1) -> bool:
        """Check if request is allowed under the rate limit."""
        now = time.time()
        bucket = self._buckets.get(key)

        if bucket is None:
            bucket = {"tokens": self.max_tokens, "last_refill": now}
            self._buckets[key] = bucket

        # Refill tokens
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(
            self.max_tokens,
            bucket["tokens"] + elapsed * self.refill_rate,
        )
        bucket["last_refill"] = now

        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            return True
        return False

    def get_remaining(self, key: str) -> float:
        """Get remaining tokens for a key."""
        bucket = self._buckets.get(key)
        if bucket is None:
            return self.max_tokens
        now = time.time()
        elapsed = now - bucket["last_refill"]
        return min(
            self.max_tokens,
            bucket["tokens"] + elapsed * self.refill_rate,
        )

    def reset(self, key: Optional[str] = None) -> None:
        """Reset rate limiter state."""
        if key:
            self._buckets.pop(key, None)
        else:
            self._buckets.clear()


class APIKeyAuth:
    """Simple API key authentication."""

    def __init__(self, valid_keys: Optional[List[str]] = None):
        self.valid_keys = set(valid_keys or [])
        self._env_key = os.getenv("VAULT_API_KEY", "")
        if self._env_key:
            self.valid_keys.add(self._env_key)

    def verify(self, key: Optional[str]) -> bool:
        """Verify an API key.

        Returns True if:
          - No keys are configured (open access)
          - The provided key is in the valid set
        """
        if not self.valid_keys:
            return True  # No keys configured — open access
        if not key:
            return False
        return key in self.valid_keys

    def add_key(self, key: str) -> None:
        """Add a valid API key."""
        self.valid_keys.add(key)

    def remove_key(self, key: str) -> bool:
        """Remove an API key."""
        if key in self.valid_keys:
            self.valid_keys.remove(key)
            return True
        return False

    @property
    def is_configured(self) -> bool:
        """True if at least one API key is configured."""
        return len(self.valid_keys) > 0
