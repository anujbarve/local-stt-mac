"""Tests for vault/plugins.py"""

import pytest

from vault.plugins import BasePlugin, PluginManager


class MockPlugin(BasePlugin):
    name = "mock"
    version = "1.0.0"

    def __init__(self):
        self.events_received = []

    def on_note_created(self, note):
        self.events_received.append(("note_created", note))

    def on_search(self, query, results):
        self.events_received.append(("search", query))


class TestPluginSystem:
    def test_register_plugin(self):
        pm = PluginManager()
        p = MockPlugin()
        pm.register(p)
        assert "mock" in pm.plugins

    def test_enable_disable(self):
        pm = PluginManager()
        p = MockPlugin()
        pm.register(p)
        pm.disable("mock")
        assert pm.enabled["mock"] is False
        pm.enable("mock")
        assert pm.enabled["mock"] is True

    def test_fire_event(self):
        pm = PluginManager()
        p = MockPlugin()
        pm.register(p)
        pm.fire("note_created", {"id": "n1"})
        assert len(p.events_received) == 1
        assert p.events_received[0] == ("note_created", {"id": "n1"})

    def test_fire_disabled_plugin(self):
        pm = PluginManager()
        p = MockPlugin()
        pm.register(p)
        pm.disable("mock")
        pm.fire("note_created", {"id": "n1"})
        assert len(p.events_received) == 0

    def test_unregister(self):
        pm = PluginManager()
        p = MockPlugin()
        pm.register(p)
        assert pm.unregister("mock")
        assert "mock" not in pm.plugins

    def test_unregister_nonexistent(self):
        pm = PluginManager()
        assert not pm.unregister("nonexistent")

    def test_list_plugins(self):
        pm = PluginManager()
        p = MockPlugin()
        pm.register(p)
        plugins = pm.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "mock"
        assert plugins[0]["enabled"] is True

    def test_add_hook(self):
        pm = PluginManager()
        received = []
        pm.add_hook("note_created", lambda note: received.append(note))
        pm.fire_hooks("note_created", {"id": "n1"})
        assert len(received) == 1

    def test_fire_exception_does_not_crash(self):
        pm = PluginManager()

        class BadPlugin(BasePlugin):
            name = "bad"
            def on_note_created(self, note):
                raise RuntimeError("boom")

        pm.register(BadPlugin())
        # Should not raise
        pm.fire("note_created", {})

    def test_save_and_load_config(self, tmp_path):
        pm = PluginManager()
        p = MockPlugin()
        pm.register(p)
        config_path = str(tmp_path / "plugins.json")
        pm.save_config(config_path)
        assert os.path.exists(config_path)

        pm2 = PluginManager()
        # Note: load_from_config needs actual importable modules, so we just test save
        import json
        with open(config_path) as f:
            config = json.load(f)
        assert len(config["plugins"]) == 1
        assert config["plugins"][0]["name"] == "mock"


import os
