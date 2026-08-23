"""
Vault Plugins
=============

Extensible plugin system for the vault.  Plugins hook into lifecycle
events (note created, search performed, backup created, etc.) to add
custom behaviour.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional


class BasePlugin:
    """Base class for vault plugins.  Override event methods to add behaviour."""

    name: str = "base"
    version: str = "0.1.0"

    def on_note_created(self, note: Dict[str, Any]) -> None:
        """Called when a new note is created."""
        pass

    def on_note_updated(self, note: Dict[str, Any]) -> None:
        """Called when a note is updated."""
        pass

    def on_search(self, query: str, results: List[Dict[str, Any]]) -> None:
        """Called when a search is performed."""
        pass

    def on_schedule(self, schedule: Dict[str, Any]) -> None:
        """Called when a schedule is generated."""
        pass

    def on_backup(self, backup_path: str) -> None:
        """Called when a backup is created."""
        pass

    def on_correction(self, correction: Dict[str, Any]) -> None:
        """Called when a user correction is logged."""
        pass


class PluginManager:
    """Manage plugin lifecycle and event dispatch."""

    def __init__(self) -> None:
        self.plugins: Dict[str, BasePlugin] = {}
        self.enabled: Dict[str, bool] = {}
        self._hooks: Dict[str, List[Callable]] = {}

    def register(self, plugin: BasePlugin) -> None:
        """Register a plugin."""
        self.plugins[plugin.name] = plugin
        self.enabled[plugin.name] = True

    def unregister(self, name: str) -> bool:
        """Unregister a plugin."""
        if name in self.plugins:
            del self.plugins[name]
            self.enabled.pop(name, None)
            return True
        return False

    def enable(self, name: str) -> None:
        """Enable a plugin."""
        if name in self.plugins:
            self.enabled[name] = True

    def disable(self, name: str) -> None:
        """Disable a plugin."""
        if name in self.plugins:
            self.enabled[name] = False

    def fire(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire an event to all enabled plugins."""
        method_name = f"on_{event}"
        for name, plugin in self.plugins.items():
            if not self.enabled.get(name, False):
                continue
            handler = getattr(plugin, method_name, None)
            if handler and callable(handler):
                try:
                    handler(*args, **kwargs)
                except Exception:
                    pass  # Plugins should not crash the system

    def add_hook(self, event: str, callback: Callable) -> None:
        """Add a direct callback hook for an event."""
        self._hooks.setdefault(event, []).append(callback)

    def fire_hooks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Fire hooks (non-plugin callbacks) for an event."""
        for callback in self._hooks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception:
                pass

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [
            {
                "name": name,
                "version": plugin.version,
                "enabled": self.enabled.get(name, False),
            }
            for name, plugin in self.plugins.items()
        ]

    def load_from_config(self, config_path: str) -> int:
        """Load plugins from JSON config.

        Config format::

            {
                "plugins": [
                    {"name": "my_plugin", "module": "my_plugin", "class": "MyPlugin", "enabled": true}
                ]
            }

        Returns number of plugins loaded.
        """
        if not os.path.exists(config_path):
            return 0

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return 0

        count = 0
        for plugin_spec in config.get("plugins", []):
            try:
                import importlib
                module = importlib.import_module(plugin_spec["module"])
                cls = getattr(module, plugin_spec["class"])
                plugin = cls()
                self.register(plugin)
                if not plugin_spec.get("enabled", True):
                    self.disable(plugin.name)
                count += 1
            except (ImportError, AttributeError, KeyError):
                continue

        return count

    def save_config(self, config_path: str) -> None:
        """Save current plugin configuration."""
        config = {
            "plugins": [
                {
                    "name": name,
                    "enabled": self.enabled.get(name, False),
                }
                for name in self.plugins
            ]
        }
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)
