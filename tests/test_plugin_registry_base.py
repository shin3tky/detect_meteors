"""Tests for PluginRegistryBase._coerce_config() variations."""

from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
import types
import unittest


def _load_registry_base():
    """Load PluginRegistryBase without importing the full meteor_core package.

    The real package pulls in heavy optional dependencies (e.g., OpenCV) inside
    ``meteor_core.__init__``. For these focused tests we synthesize a minimal
    package namespace and import only the modules needed to exercise
    ``PluginRegistryBase._coerce_config``.
    """

    project_root = Path(__file__).resolve().parent.parent

    # Create a lightweight meteor_core package placeholder.
    meteor_core_pkg = types.ModuleType("meteor_core")
    meteor_core_pkg.__path__ = [str(project_root / "meteor_core")]
    sys.modules.setdefault("meteor_core", meteor_core_pkg)

    def _load(name: str, filename: str):
        module_name = f"meteor_core.{name}"
        module_path = project_root / "meteor_core" / filename
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(meteor_core_pkg, name, module)
        return module

    _load("plugin_registry", "plugin_registry.py")
    plugin_registry_base = _load("plugin_registry_base", "plugin_registry_base.py")
    return plugin_registry_base.PluginRegistryBase


PluginRegistryBase = _load_registry_base()


class DummyRegistry(PluginRegistryBase[object]):
    """Minimal registry subclass for exercising _coerce_config."""

    _plugin_kind = "dummy"

    @classmethod
    def _discover_internal(cls):  # pragma: no cover - not used
        return {}

    @classmethod
    def _is_valid_plugin(cls, plugin_cls):  # pragma: no cover - not used
        return True


@dataclass
class RequiredConfig:
    required: str


class PluginWithRequiredConfig:
    plugin_name = "with_required"
    ConfigType = RequiredConfig


class PluginWithoutConfigType:
    plugin_name = "no_config_type"


class PlainConfig:
    def __init__(self, value):
        self.value = value


class PluginWithPlainClass:
    plugin_name = "plain_class"
    ConfigType = PlainConfig


class PluginWithDataclass:
    plugin_name = "dataclass_plugin"
    ConfigType = RequiredConfig


class TestPluginRegistryBaseCoerceConfig(unittest.TestCase):
    """Covers input variations to lock down _coerce_config contract."""

    def test_none_with_required_args_raises_typeerror(self):
        """None cannot instantiate ConfigType that requires arguments."""

        with self.assertRaises(TypeError) as ctx:
            DummyRegistry._coerce_config(PluginWithRequiredConfig, None)

        self.assertIn("requires arguments", str(ctx.exception))

    def test_none_without_configtype_returns_none(self):
        """None is returned unchanged when plugin defines no ConfigType."""

        marker = object()
        self.assertIs(
            marker, DummyRegistry._coerce_config(PluginWithoutConfigType, marker)
        )

    def test_dict_to_non_dataclass_non_pydantic_raises_typeerror(self):
        """Dict cannot be coerced when ConfigType lacks dataclass/pydantic hooks."""

        with self.assertRaises(TypeError) as ctx:
            DummyRegistry._coerce_config(PluginWithPlainClass, {"value": 1})

        self.assertIn("Cannot coerce dict", str(ctx.exception))

    def test_unhandled_type_passes_through(self):
        """Non-dict configs are returned as-is even with a ConfigType."""

        config = ("not", "a", "dict")
        self.assertIs(
            config,
            DummyRegistry._coerce_config(PluginWithDataclass, config),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
