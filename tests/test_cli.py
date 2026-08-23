"""Unit tests for CLI argument parsing and flags."""

from unittest.mock import patch
import pytest
from transcribe import parse_arguments


def test_default_arguments():
    with patch("transcribe.load_config", return_value={}):
        args = parse_arguments([])
        assert args.model == "base"
        assert args.wake_word is None
        assert args.hotkey == "alt_r"
        assert args.no_hotkey is False
        assert args.tts is True
        assert args.tts_voice == "rosie"
        assert args.threshold == 0.45
        assert args.silence == 1.5
        assert args.device is None
        assert args.once is False
        assert args.verbose is False
        assert args.list_devices is False
        assert args.list_wake_words is False


def test_enable_wake_word_flag():
    args = parse_arguments(["--enable-wake-word"])
    assert args.wake_word == "hey_atlas"


def test_custom_arguments():
    args = parse_arguments([
        "--model", "small",
        "--wake-word", "hey_atlas",
        "--hotkey", "space",
        "--no-hotkey",
        "--no-tts",
        "--tts-voice", "bella",
        "--threshold", "0.7",
        "--silence", "2.0",
        "--device", "1",
        "--list-wake-words",
        "--once",
        "--verbose",
    ])
    assert args.model == "small"
    assert args.wake_word == "hey_atlas"
    assert args.hotkey == "space"
    assert args.no_hotkey is True
    assert args.no_tts is True
    assert args.tts_voice == "bella"
    assert args.threshold == 0.7
    assert args.silence == 2.0
    assert args.device == "1"
    assert args.list_wake_words is True
    assert args.once is True
    assert args.verbose is True


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("VOICEASSIST_WAKE_WORD", "hey_beemo")
    monkeypatch.setenv("VOICEASSIST_MODEL", "tiny")
    monkeypatch.setenv("VOICEASSIST_HOTKEY", "option_r")
    args = parse_arguments([])
    assert args.wake_word == "hey_beemo"
    assert args.model == "tiny"
    assert args.hotkey == "option_r"
