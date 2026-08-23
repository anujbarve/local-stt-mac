"""Global hotkey listener using pynput for macOS."""

import sys
import threading
from typing import Callable, Optional
from pynput import keyboard


class HotkeyListener:
    """Manages background keyboard listener for push-to-talk or tap-to-record trigger."""

    def __init__(self, hotkey: str = "alt_r", on_trigger: Optional[Callable[[], None]] = None):
        self.hotkey_name = hotkey.lower().replace(" ", "_")
        self.on_trigger = on_trigger
        self._is_pressed = False
        self._triggered = threading.Event()
        self._listener: Optional[keyboard.Listener] = None

    def _matches_key(self, key) -> bool:
        """Check if the pressed key matches the configured hotkey."""
        if self.hotkey_name in ("alt_r", "right_alt", "option_r", "right_option", "alt_gr"):
            return key in (keyboard.Key.alt_r, getattr(keyboard.Key, "alt_gr", None))
        elif self.hotkey_name in ("alt", "option"):
            return key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, getattr(keyboard.Key, "alt_gr", None))
        elif self.hotkey_name in ("ctrl_r", "right_ctrl"):
            return key == getattr(keyboard.Key, "ctrl_r", None)
        elif self.hotkey_name in ("cmd_r", "right_cmd"):
            return key == getattr(keyboard.Key, "cmd_r", None)
        elif self.hotkey_name in ("shift_r", "right_shift"):
            return key == getattr(keyboard.Key, "shift_r", None)
        elif self.hotkey_name == "space":
            return key == keyboard.Key.space
        else:
            # Match by character name
            try:
                return hasattr(key, "char") and key.char == self.hotkey_name
            except Exception:
                return False

    def _on_press(self, key):
        if self._matches_key(key):
            self._is_pressed = True
            self._triggered.set()
            if self.on_trigger:
                try:
                    self.on_trigger()
                except Exception:
                    pass

    def _on_release(self, key):
        if self._matches_key(key):
            self._is_pressed = False

    def start(self) -> None:
        """Start listening for keyboard events."""
        if self._listener is not None and self._listener.is_alive():
            return
        try:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self._listener.daemon = True
            self._listener.start()
        except Exception as e:
            # Fallback if Accessibility permissions are not yet granted
            print(
                f"[Notice] Global hotkey disabled or restricted: {e}\n"
                "Tip: Enable Accessibility in System Settings → Privacy & Security → Accessibility\n"
                "to capture the hotkey globally across all apps.",
                file=sys.stderr
            )

    def stop(self) -> None:
        """Stop listening for keyboard events."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            finally:
                self._listener = None

    def is_triggered(self) -> bool:
        """Check if hotkey was triggered (and clear the event)."""
        if self._triggered.is_set():
            self._triggered.clear()
            return True
        return False

    def is_held(self) -> bool:
        """Check if hotkey is currently held down."""
        return self._is_pressed

    def reset(self) -> None:
        """Reset trigger state."""
        self._triggered.clear()
        self._is_pressed = False
