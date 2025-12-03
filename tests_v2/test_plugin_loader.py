import contextlib
import logging
import sys
import tempfile
import textwrap
from importlib.metadata import EntryPoint
from pathlib import Path
from unittest import TestCase, mock

# Stub external dependencies used during module import.
sys.modules.setdefault("cv2", type(sys)("cv2"))
sys.modules.setdefault("rawpy", type(sys)("rawpy"))

from detect_meteors import app, plugin_loader


class FakeEntryPoints(list):
    def select(self, **params):
        if params.get("group") == plugin_loader.PLUGIN_ENTRYPOINT_GROUP:
            return self
        return []


def cleanup_registry():
    for name in [
        "entry_detector",
        "entry_preprocessor",
        "entry_writer",
        "folder_detector",
        "folder_preprocessor",
        "folder_writer",
        "shared_detector",
        "good_detector",
        "good_preprocessor",
        "good_writer",
    ]:
        for unregister in (
            app.unregister_detector,
            app.unregister_preprocessor,
            app.unregister_output_writer,
        ):
            try:
                unregister(name)
            except KeyError:
                pass


@contextlib.contextmanager
def prepended_sys_path(path: Path):
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(path))


class TestPluginLoader(TestCase):
    def tearDown(self):
        cleanup_registry()

    def _write_plugin(self, path: Path, contents: str) -> None:
        path.write_text(textwrap.dedent(contents))

    def test_loads_plugins_from_entry_points_and_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            plugin_folder = tmp_path / "plugins"
            plugin_folder.mkdir()

            entry_module_name = "entrypoint_plugin"
            entry_module_path = tmp_path / f"{entry_module_name}.py"
            self._write_plugin(
                entry_module_path,
                """
                from detect_meteors import app

                class EntryDetector(app.Detector):
                    plugin_info = app.PluginInfo(
                        name="entry_detector", version="0.0.1", capabilities=["detector"]
                    )

                    def detect(self, **kwargs):  # type: ignore[override]
                        return 3

                class EntryPreprocessor:
                    plugin_info = app.PluginInfo(
                        name="entry_preprocessor", version="0.0.1", capabilities=["preprocessor"]
                    )

                    def preprocess(self, target_folder: str) -> str:
                        return f"processed:{target_folder}"

                class EntryWriter:
                    plugin_info = app.PluginInfo(
                        name="entry_writer", version="0.0.1", capabilities=["writer"]
                    )

                    def write(self, detected_count: int, warnings):
                        return {"count": detected_count, "warnings": warnings}

                DETECTORS = {"entry_detector": EntryDetector()}
                PREPROCESSORS = {"entry_preprocessor": EntryPreprocessor()}
                OUTPUT_WRITERS = {"entry_writer": EntryWriter()}
                """,
            )

            folder_plugin = plugin_folder / "folder_plugin.py"
            self._write_plugin(
                folder_plugin,
                """
                from detect_meteors import app

                class FolderDetector(app.Detector):
                    plugin_info = app.PluginInfo(
                        name="folder_detector", version="0.0.2", capabilities=["detector"]
                    )

                    def detect(self, **kwargs):  # type: ignore[override]
                        return 5

                class FolderPreprocessor:
                    plugin_info = app.PluginInfo(
                        name="folder_preprocessor", version="0.0.2", capabilities=["preprocessor"]
                    )

                    def preprocess(self, target_folder: str) -> str:
                        return f"folder:{target_folder}"

                class FolderWriter:
                    plugin_info = app.PluginInfo(
                        name="folder_writer", version="0.0.2", capabilities=["writer"]
                    )

                    def write(self, detected_count: int, warnings):
                        return {"folder": detected_count, "warnings": warnings}

                DETECTORS = {"folder_detector": FolderDetector()}
                PREPROCESSORS = {"folder_preprocessor": FolderPreprocessor()}
                OUTPUT_WRITERS = {"folder_writer": FolderWriter()}
                """,
            )

            entry_point = EntryPoint(
                name="entry-plugin", value=entry_module_name, group=plugin_loader.PLUGIN_ENTRYPOINT_GROUP
            )
            fake_eps = FakeEntryPoints([entry_point])

            with prepended_sys_path(tmp_path), mock.patch.object(
                plugin_loader, "entry_points", return_value=fake_eps
            ):
                plugin_loader.load_plugins(plugin_folder=plugin_folder)

            preprocessor = app.get_preprocessor("entry_preprocessor")
            self.assertEqual(preprocessor.preprocess("target"), "processed:target")

            detector = app.get_detector("folder_detector")
            self.assertEqual(detector.detect(), 5)

            writer = app.get_output_writer("entry_writer")
            self.assertEqual(writer.write(2, ["warn"]), {"count": 2, "warnings": ["warn"]})

            self.assertEqual(app._DETECTOR_REGISTRY["entry_detector"].info.capabilities, ["detector"])
            self.assertEqual(app._OUTPUT_WRITER_REGISTRY["folder_writer"].info.capabilities, ["writer"])

    def test_duplicates_and_import_errors_are_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir, self.assertLogs(
            plugin_loader.LOGGER, level=logging.WARNING
        ) as captured:
            tmp_path = Path(tmpdir)
            plugin_folder = tmp_path / "plugins"
            plugin_folder.mkdir()

            duplicate_entry = tmp_path / "duplicate.py"
            self._write_plugin(
                duplicate_entry,
                """
                from detect_meteors import app

                class DuplicateDetector(app.Detector):
                    plugin_info = app.PluginInfo(
                        name="shared_detector", version="1.0.0", capabilities=["detector"]
                    )

                    def detect(self, **kwargs):  # type: ignore[override]
                        return 10

                DETECTORS = {"shared_detector": DuplicateDetector()}
                PREPROCESSORS = {}
                OUTPUT_WRITERS = {}
                """,
            )

            entry_points_list = FakeEntryPoints(
                [
                    EntryPoint(
                        name="missing-module", value="missing.module", group=plugin_loader.PLUGIN_ENTRYPOINT_GROUP
                    ),
                    EntryPoint(
                        name="duplicate", value="duplicate", group=plugin_loader.PLUGIN_ENTRYPOINT_GROUP
                    ),
                ]
            )

            folder_dup = plugin_folder / "folder_dup.py"
            self._write_plugin(
                folder_dup,
                """
                from detect_meteors import app

                class FolderDuplicate(app.Detector):
                    plugin_info = app.PluginInfo(
                        name="shared_detector", version="2.0.0", capabilities=["detector"]
                    )

                    def detect(self, **kwargs):  # type: ignore[override]
                        return 20

                DETECTORS = {"shared_detector": FolderDuplicate()}
                PREPROCESSORS = {}
                OUTPUT_WRITERS = {}
                """,
            )

            with prepended_sys_path(tmp_path), mock.patch.object(
                plugin_loader, "entry_points", return_value=entry_points_list
            ):
                plugin_loader.load_plugins(plugin_folder=plugin_folder)

            detector = app.get_detector("shared_detector")
            self.assertEqual(detector.detect(), 10)

            self.assertTrue(
                any("Failed to load plugin entry point 'missing-module'" in message for message in captured.output)
            )
            self.assertTrue(
                any("Skipping duplicate detector 'shared_detector'" in message for message in captured.output)
            )

    def test_folder_import_failures_do_not_block_other_plugins(self):
        with tempfile.TemporaryDirectory() as tmpdir, self.assertLogs(
            plugin_loader.LOGGER, level=logging.ERROR
        ) as captured:
            tmp_path = Path(tmpdir)
            plugin_folder = tmp_path / "plugins"
            plugin_folder.mkdir()

            (plugin_folder / "bad_plugin.py").write_text("raise ImportError('boom')\n")
            good_plugin = plugin_folder / "good_plugin.py"
            self._write_plugin(
                good_plugin,
                """
                from detect_meteors import app

                class GoodDetector(app.Detector):
                    plugin_info = app.PluginInfo(
                        name="good_detector", version="1.2.3", capabilities=["detector", "tests"]
                    )

                    def detect(self, **kwargs):  # type: ignore[override]
                        return 42

                class GoodPreprocessor:
                    plugin_info = app.PluginInfo(
                        name="good_preprocessor", version="1.2.3", capabilities=["preprocessor"]
                    )

                    def preprocess(self, target_folder: str) -> str:
                        return f"good:{target_folder}"

                class GoodWriter:
                    plugin_info = app.PluginInfo(
                        name="good_writer", version="1.2.3", capabilities=["writer"]
                    )

                    def write(self, detected_count: int, warnings):
                        return {"detected": detected_count, "warnings": warnings}

                DETECTORS = {"good_detector": GoodDetector()}
                PREPROCESSORS = {"good_preprocessor": GoodPreprocessor()}
                OUTPUT_WRITERS = {"good_writer": GoodWriter()}
                """,
            )

            with prepended_sys_path(tmp_path), mock.patch.object(
                plugin_loader, "entry_points", return_value=FakeEntryPoints()
            ):
                plugin_loader.load_plugins(plugin_folder=plugin_folder)

            detector = app.get_detector("good_detector")
            self.assertEqual(detector.detect(), 42)
            self.assertEqual(app.get_preprocessor("good_preprocessor").preprocess("target"), "good:target")
            self.assertEqual(app.get_output_writer("good_writer").write(1, []), {"detected": 1, "warnings": []})

            with self.assertRaises(KeyError):
                app.get_detector("bad_plugin")

            self.assertTrue(
                any("Failed to import plugin module from" in message for message in captured.output)
            )
