"""Tests for vault.security — InputSanitizer, RateLimiter, APIKeyAuth."""

import os
import tempfile
import time

import pytest

from vault.security import APIKeyAuth, InputSanitizer, RateLimiter


# ---------------------------------------------------------------------------
# InputSanitizer
# ---------------------------------------------------------------------------


class TestSanitizeText:
    def test_normal_text(self):
        assert InputSanitizer.sanitize_text("hello world") == "hello world"

    def test_strips_control_chars(self):
        text = "hello\x00\x01\x02world"
        assert InputSanitizer.sanitize_text(text) == "helloworld"

    def test_keeps_newlines_and_tabs(self):
        text = "line1\nline2\ttab"
        assert InputSanitizer.sanitize_text(text) == "line1\nline2\ttab"

    def test_strips_delete_char(self):
        text = "hello\x7fworld"
        assert InputSanitizer.sanitize_text(text) == "helloworld"

    def test_truncates_long_text(self):
        text = "a" * 20000
        result = InputSanitizer.sanitize_text(text)
        assert len(result) == InputSanitizer.MAX_TEXT_LENGTH

    def test_non_string_returns_empty(self):
        assert InputSanitizer.sanitize_text(123) == ""
        assert InputSanitizer.sanitize_text(None) == ""

    def test_empty_string(self):
        assert InputSanitizer.sanitize_text("") == ""


class TestSanitizeTag:
    def test_normal_tag(self):
        assert InputSanitizer.sanitize_tag("Python") == "python"

    def test_strips_special_chars(self):
        tag = InputSanitizer.sanitize_tag("hello!@#$%world")
        assert tag == "helloworld"

    def test_replaces_spaces_with_dashes(self):
        assert InputSanitizer.sanitize_tag("machine learning") == "machine-learning"

    def test_non_string_returns_empty(self):
        assert InputSanitizer.sanitize_tag(None) == ""
        assert InputSanitizer.sanitize_tag(42) == ""

    def test_truncates_long_tag(self):
        tag = "a" * 200
        result = InputSanitizer.sanitize_tag(tag)
        assert len(result) == InputSanitizer.MAX_TAG_LENGTH


class TestSanitizeTags:
    def test_normal_tags(self):
        tags = InputSanitizer.sanitize_tags(["Python", "ML", "AI"])
        assert tags == ["python", "ml", "ai"]

    def test_filters_empty_tags(self):
        tags = InputSanitizer.sanitize_tags(["ok", "", "!!!", "fine"])
        assert tags == ["ok", "fine"]

    def test_limits_count(self):
        tags = [f"tag{i}" for i in range(100)]
        result = InputSanitizer.sanitize_tags(tags)
        assert len(result) == InputSanitizer.MAX_TAGS


class TestValidatePath:
    def test_safe_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert InputSanitizer.validate_path("notes/file.md", tmp) is True

    def test_traversal_attack(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert InputSanitizer.validate_path("../../etc/passwd", tmp) is False

    def test_nested_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert InputSanitizer.validate_path("a/../../b", tmp) is False

    def test_base_dir_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert InputSanitizer.validate_path("", tmp) is True


class TestValidateType:
    def test_allowed(self):
        assert InputSanitizer.validate_type("task", ["task", "idea"]) is True

    def test_not_allowed(self):
        assert InputSanitizer.validate_type("hack", ["task", "idea"]) is False


class TestValidatePriority:
    def test_normal(self):
        assert InputSanitizer.validate_priority(0.5) == 0.5

    def test_clamp_above(self):
        assert InputSanitizer.validate_priority(2.0) == 1.0

    def test_clamp_below(self):
        assert InputSanitizer.validate_priority(-0.5) == 0.0

    def test_string_number(self):
        assert InputSanitizer.validate_priority("0.7") == 0.7


class TestValidateNoteId:
    def test_valid_id(self):
        assert InputSanitizer.validate_note_id("note_2026_08_23_001") is True

    def test_invalid_id(self):
        assert InputSanitizer.validate_note_id("bad_id") is False
        assert InputSanitizer.validate_note_id("note_2026_08_2") is False
        assert InputSanitizer.validate_note_id("") is False


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_initial_requests(self):
        rl = RateLimiter(max_tokens=5, refill_rate=100)
        assert rl.allow("user1") is True
        assert rl.allow("user1") is True

    def test_blocks_when_exhausted(self):
        rl = RateLimiter(max_tokens=2, refill_rate=0)
        assert rl.allow("u") is True
        assert rl.allow("u") is True
        assert rl.allow("u") is False

    def test_refill_over_time(self):
        rl = RateLimiter(max_tokens=5, refill_rate=1000)
        rl.allow("u")
        rl.allow("u")
        # With high refill rate, tokens should replenish quickly
        time.sleep(0.01)
        assert rl.allow("u") is True

    def test_separate_keys(self):
        rl = RateLimiter(max_tokens=1, refill_rate=0)
        assert rl.allow("a") is True
        assert rl.allow("b") is True
        assert rl.allow("a") is False
        assert rl.allow("b") is False

    def test_get_remaining(self):
        rl = RateLimiter(max_tokens=10, refill_rate=0)
        rl.allow("u")
        remaining = rl.get_remaining("u")
        assert remaining < 10

    def test_reset_single_key(self):
        rl = RateLimiter(max_tokens=1, refill_rate=0)
        rl.allow("u")
        rl.reset("u")
        assert rl.allow("u") is True

    def test_reset_all(self):
        rl = RateLimiter(max_tokens=1, refill_rate=0)
        rl.allow("a")
        rl.allow("b")
        rl.reset()
        assert rl.allow("a") is True
        assert rl.allow("b") is True

    def test_costly_request(self):
        rl = RateLimiter(max_tokens=10, refill_rate=0)
        assert rl.allow("u", tokens=10) is True
        assert rl.allow("u", tokens=1) is False


# ---------------------------------------------------------------------------
# APIKeyAuth
# ---------------------------------------------------------------------------


class TestAPIKeyAuth:
    def test_no_keys_configured(self):
        auth = APIKeyAuth(valid_keys=[])
        assert auth.verify(None) is True
        assert auth.verify("anything") is True

    def test_valid_key(self):
        auth = APIKeyAuth(valid_keys=["secret123"])
        assert auth.verify("secret123") is True

    def test_invalid_key(self):
        auth = APIKeyAuth(valid_keys=["secret123"])
        assert auth.verify("wrong") is False

    def test_no_key_when_required(self):
        auth = APIKeyAuth(valid_keys=["secret123"])
        assert auth.verify(None) is False
        assert auth.verify("") is False

    def test_add_key(self):
        auth = APIKeyAuth(valid_keys=[])
        auth.add_key("newkey")
        assert auth.verify("newkey") is True

    def test_remove_key(self):
        auth = APIKeyAuth(valid_keys=["k1", "k2"])
        assert auth.remove_key("k1") is True
        assert auth.verify("k1") is False
        assert auth.verify("k2") is True

    def test_remove_nonexistent(self):
        auth = APIKeyAuth(valid_keys=["k1"])
        assert auth.remove_key("nope") is False

    def test_is_configured(self):
        auth_empty = APIKeyAuth(valid_keys=[])
        assert auth_empty.is_configured is False

        auth_full = APIKeyAuth(valid_keys=["k1"])
        assert auth_full.is_configured is True

    def test_env_key(self, monkeypatch):
        monkeypatch.setenv("VAULT_API_KEY", "env-key")
        auth = APIKeyAuth(valid_keys=["manual"])
        assert auth.verify("env-key") is True
        assert auth.verify("manual") is True
