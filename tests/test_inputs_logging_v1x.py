"""
Exception and logging coverage for input loader utilities.
"""

import sys
import types
import warnings
from pathlib import Path
import tempfile
import unittest
from unittest import mock

if "cv2" not in sys.modules:  # pragma: no cover - optional dependency shim
    sys.modules["cv2"] = types.SimpleNamespace()

from meteor_core.exceptions import MeteorValidationError
from meteor_core.inputs.base import BaseInputLoader
from meteor_core.schema import InputContext
from meteor_core.inputs.discovery import _add_loader, _discover_handlers_internal
from meteor_core.inputs.raw import RawImageLoader, RawLoaderConfig
from meteor_core.inputs.registry import LoaderRegistry


class DummyValidLoader(BaseInputLoader):
    plugin_name = "dummy"
    ConfigType = type(None)

    def load(self, filepath: str):  # pragma: no cover - minimal implementation
        return InputContext(image_data=filepath, filepath=filepath)


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
                return InputContext(image_data=filepath, filepath=filepath)

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
                return InputContext(image_data=filepath, filepath=filepath)

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

    def test_loader_creation_logs_config_errors(self):
        with self.assertLogs("meteor_core.inputs.registry", level="ERROR") as logs:
            with self.assertRaises(MeteorValidationError):
                LoaderRegistry.create("raw", {"binning": 3})

        self.assertTrue(
            any("Failed to coerce config" in message for message in logs.output),
            "Expected config coercion failure to be logged",
        )

    def test_raw_loader_load_logs_errors(self):
        loader = RawImageLoader(RawLoaderConfig())

        with mock.patch(
            "meteor_core.inputs.raw.load_and_bin_raw_fast",
            side_effect=RuntimeError("boom"),
        ):
            with self.assertLogs("meteor_core.inputs.raw", level="ERROR") as logs:
                with self.assertRaises(RuntimeError):
                    loader.load("/tmp/example.raw")

        self.assertTrue(
            any("Failed to load RAW image" in message for message in logs.output),
            "Expected loading failure to be logged",
        )

    def test_extract_metadata_logs_warning_on_failure(self):
        loader = RawImageLoader(RawLoaderConfig())

        with mock.patch(
            "meteor_core.inputs.raw.extract_exif_metadata",
            side_effect=RuntimeError("oops"),
        ):
            with self.assertLogs("meteor_core.inputs.raw", level="WARNING") as logs:
                metadata = loader.extract_metadata("/tmp/example.raw")

        self.assertEqual(metadata, {})
        self.assertTrue(
            any(
                "Failed to extract EXIF metadata" in message for message in logs.output
            ),
            "Expected metadata extraction issues to be logged",
        )


if __name__ == "__main__":
    unittest.main()
