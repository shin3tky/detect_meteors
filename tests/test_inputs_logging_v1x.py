"""
Exception and logging coverage for input loader utilities.
"""

import sys
import types
import warnings
from pathlib import Path
import tempfile
import unittest

if "cv2" not in sys.modules:  # pragma: no cover - optional dependency shim
    sys.modules["cv2"] = types.SimpleNamespace()

from meteor_core.exceptions import MeteorValidationError
from meteor_core.inputs.base import BaseInputLoader
from meteor_core.inputs.discovery import _add_loader, _discover_handlers_internal
from meteor_core.inputs.raw import RawLoaderConfig


class DummyValidLoader(BaseInputLoader):
    plugin_name = "dummy"
    ConfigType = type(None)

    def load(self, filepath: str):  # pragma: no cover - minimal implementation
        return filepath


class TestDiscoveryLogging(unittest.TestCase):
    def test_add_loader_warns_for_invalid_class(self):
        class NotALoader:
            plugin_name = "fake_loader"

        registry = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs(
                "meteor_core.inputs.discovery", level="WARNING"
            ) as logs:
                _add_loader(registry, NotALoader, "unit-test origin")

        self.assertEqual(registry, {})
        self.assertTrue(
            any("does not inherit" in str(w.message) for w in caught),
            "Expected warning about missing BaseInputLoader inheritance",
        )
        self.assertTrue(
            any("unit-test origin" in message for message in logs.output),
            "Expected log mentioning the origin of the skipped loader",
        )

    def test_add_loader_warns_for_empty_plugin_name(self):
        class EmptyNameLoader(BaseInputLoader):
            plugin_name = ""
            ConfigType = type(None)

            def load(self, filepath: str):  # pragma: no cover
                return filepath

        registry = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs(
                "meteor_core.inputs.discovery", level="WARNING"
            ) as logs:
                _add_loader(registry, EmptyNameLoader, "unit-test origin")

        self.assertEqual(registry, {})
        self.assertTrue(
            any("missing plugin_name" in str(w.message) for w in caught),
            "Expected warning about missing plugin name",
        )
        self.assertTrue(
            any("missing plugin_name" in message for message in logs.output),
            "Expected log entry describing the missing plugin name",
        )

    def test_add_loader_warns_on_duplicate_plugin_name(self):
        class DuplicateLoader(BaseInputLoader):
            plugin_name = "dummy"
            ConfigType = type(None)

            def load(self, filepath: str):  # pragma: no cover
                return filepath

        registry = {"dummy": DummyValidLoader}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertLogs(
                "meteor_core.inputs.discovery", level="WARNING"
            ) as logs:
                _add_loader(registry, DuplicateLoader, "unit-test origin")

        self.assertEqual(registry["dummy"], DummyValidLoader)
        self.assertTrue(
            any("Duplicate loader name" in str(w.message) for w in caught),
            "Expected warning about duplicate plugin name",
        )
        self.assertTrue(
            any("Duplicate loader name" in message for message in logs.output),
            "Expected duplicate warning to be logged",
        )

    def test_discover_handlers_logs_plugin_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir)
            bad_plugin = plugin_dir / "bad.py"
            bad_plugin.write_text("raise RuntimeError('boom')\n")

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with self.assertLogs(
                    "meteor_core.inputs.discovery", level="WARNING"
                ) as logs:
                    registry = _discover_handlers_internal(plugin_dir)

        self.assertIn("raw", registry)
        self.assertTrue(
            any("Failed to load plugin module" in message for message in logs.output),
            "Expected failure to load plugin module to be logged",
        )


class TestRawLoaderConfigLogging(unittest.TestCase):
    def test_invalid_binning_logs_error(self):
        with self.assertLogs("meteor_core.inputs.raw", level="ERROR") as logs:
            with self.assertRaises(MeteorValidationError):
                RawLoaderConfig(binning=3)

        self.assertTrue(
            any("Invalid binning factor" in message for message in logs.output),
            "Expected invalid binning to be logged as an error",
        )


if __name__ == "__main__":
    unittest.main()
