"""Unit tests for hotkey listener."""

from pynput import keyboard
import pytest
from hotkey import HotkeyListener


def test_hotkey_listener_initialization():
    listener = HotkeyListener(hotkey="alt_r")
    assert listener.hotkey_name == "alt_r"
    assert listener.is_triggered() is False
    assert listener.is_held() is False


def test_hotkey_key_matching():
    listener = HotkeyListener(hotkey="alt_r")
    assert listener._matches_key(keyboard.Key.alt_r) is True
    assert listener._matches_key(keyboard.Key.space) is False

    space_listener = HotkeyListener(hotkey="space")
    assert space_listener._matches_key(keyboard.Key.space) is True


def test_hotkey_trigger_event():
    triggered = False

    def callback():
        nonlocal triggered
        triggered = True

    listener = HotkeyListener(hotkey="alt_r", on_trigger=callback)
    # Simulate press
    listener._on_press(keyboard.Key.alt_r)
    assert listener.is_held() is True
    assert listener.is_triggered() is True
    assert triggered is True

    # Check is_triggered clears flag
    assert listener.is_triggered() is False

    # Simulate release
    listener._on_release(keyboard.Key.alt_r)
    assert listener.is_held() is False
