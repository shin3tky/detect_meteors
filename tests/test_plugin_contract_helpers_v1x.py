"""Tests for plugin contract helper utilities."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest


if importlib.util.find_spec("yaml") is None:

    class _YamlError(Exception):
        """Fallback YAML error type for tests."""

    sys.modules["yaml"] = types.SimpleNamespace(  # type: ignore[assignment]
        safe_load=lambda _: {},
        YAMLError=_YamlError,
    )

module_path = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "meteor_core",
    "plugin_contract.py",
)
spec = importlib.util.spec_from_file_location(
    "meteor_core.plugin_contract",
    os.path.abspath(module_path),
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load meteor_core.plugin_contract")
plugin_contract = importlib.util.module_from_spec(spec)
sys.modules["meteor_core.plugin_contract"] = plugin_contract
spec.loader.exec_module(plugin_contract)


class TestPluginContractHelpers(unittest.TestCase):
    """Coverage for plugin contract helper functions."""

    def test_require_plugin_name_returns_valid_name(self) -> None:
        class SamplePlugin:
            plugin_name = "sample"

        self.assertEqual(
            plugin_contract.require_plugin_name(SamplePlugin, kind="inputs"),
            "sample",
        )

    def test_require_plugin_name_rejects_missing_or_empty(self) -> None:
        class MissingPluginName:
            plugin_name = ""

        with self.assertRaises(ValueError):
            plugin_contract.require_plugin_name(MissingPluginName, kind="inputs")

    def test_require_config_type_reads_attribute(self) -> None:
        class SamplePlugin:
            ConfigType = object()

        self.assertIs(
            plugin_contract.require_config_type(SamplePlugin),
            SamplePlugin.ConfigType,
        )

    def test_forbid_unknown_keys_updates_pydantic_model_config(self) -> None:
        if plugin_contract.BaseModel is None:
            with self.assertRaises(ImportError):
                plugin_contract.forbid_unknown_keys(dict)
            return

        class PluginConfig(plugin_contract.BaseModel):
            name: str

            model_config = {"extra": "allow"}

        updated_model = plugin_contract.forbid_unknown_keys(PluginConfig)
        self.assertIs(updated_model, PluginConfig)
        self.assertEqual(PluginConfig.model_config.get("extra"), "forbid")

    def test_forbid_unknown_keys_rejects_non_model(self) -> None:
        if plugin_contract.BaseModel is None:
            with self.assertRaises(ImportError):
                plugin_contract.forbid_unknown_keys(dict)
        else:
            with self.assertRaises(TypeError):
                plugin_contract.forbid_unknown_keys(dict)
