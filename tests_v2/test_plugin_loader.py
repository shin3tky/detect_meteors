import logging
import textwrap
import types
from importlib.metadata import EntryPoint
import sys

import pytest

# Stub external dependencies used during module import.
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("rawpy", types.ModuleType("rawpy"))

from detect_meteors import app, plugin_loader


class FakeEntryPoints(list):
    def select(self, **params):
        if params.get("group") == plugin_loader.PLUGIN_ENTRYPOINT_GROUP:
            return self
        return []


@pytest.fixture(autouse=True)
def cleanup_registry():
    yield
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
        try:
            app.unregister_detector(name)
        except KeyError:
            pass
        try:
            app.unregister_preprocessor(name)
        except KeyError:
            pass
        try:
            app.unregister_output_writer(name)
        except KeyError:
            pass


@pytest.fixture
def plugin_source(tmp_path, monkeypatch):
    plugin_folder = tmp_path / "plugins"
    plugin_folder.mkdir()

    entry_module_name = "entrypoint_plugin"
    entry_module_path = tmp_path / f"{entry_module_name}.py"
    entry_module_path.write_text(
        textwrap.dedent(
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
            """
        )
    )

    folder_plugin = plugin_folder / "folder_plugin.py"
    folder_plugin.write_text(
        textwrap.dedent(
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
            """
        )
    )

    monkeypatch.syspath_prepend(str(tmp_path))

    entry_point = EntryPoint(
        name="entry-plugin", value=entry_module_name, group=plugin_loader.PLUGIN_ENTRYPOINT_GROUP
    )
    fake_eps = FakeEntryPoints([entry_point])
    monkeypatch.setattr(plugin_loader, "entry_points", lambda: fake_eps)

    return plugin_folder


def test_loads_plugins_from_entry_points_and_folder(plugin_source):
    plugin_loader.load_plugins(plugin_folder=plugin_source)

    preprocessor = app.get_preprocessor("entry_preprocessor")
    assert preprocessor.preprocess("target") == "processed:target"

    detector = app.get_detector("folder_detector")
    assert detector.detect() == 5

    writer = app.get_output_writer("entry_writer")
    assert writer.write(2, ["warn"]) == {"count": 2, "warnings": ["warn"]}

    assert app._DETECTOR_REGISTRY["entry_detector"].info.capabilities == ["detector"]
    assert app._OUTPUT_WRITER_REGISTRY["folder_writer"].info.capabilities == ["writer"]


def test_duplicates_and_import_errors_are_logged(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.syspath_prepend(str(tmp_path))

    duplicate_entry = tmp_path / "duplicate.py"
    duplicate_entry.write_text(
        textwrap.dedent(
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
            """
        )
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
    monkeypatch.setattr(plugin_loader, "entry_points", lambda: entry_points_list)

    plugin_folder = tmp_path / "plugins"
    plugin_folder.mkdir()
    (plugin_folder / "folder_dup.py").write_text(
        textwrap.dedent(
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
            """
        )
    )

    plugin_loader.load_plugins(plugin_folder=plugin_folder)

    detector = app.get_detector("shared_detector")
    assert detector.detect() == 10

    assert any("Failed to load plugin entry point 'missing-module'" in message for message in caplog.messages)
    assert any("Skipping duplicate detector 'shared_detector'" in message for message in caplog.messages)


def test_folder_import_failures_do_not_block_other_plugins(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.ERROR)
    plugin_folder = tmp_path / "plugins"
    plugin_folder.mkdir()

    (plugin_folder / "bad_plugin.py").write_text("raise ImportError('boom')\n")
    (plugin_folder / "good_plugin.py").write_text(
        textwrap.dedent(
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
            """
        )
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(plugin_loader, "entry_points", lambda: FakeEntryPoints())

    plugin_loader.load_plugins(plugin_folder=plugin_folder)

    detector = app.get_detector("good_detector")
    assert detector.detect() == 42
    assert app.get_preprocessor("good_preprocessor").preprocess("target") == "good:target"
    assert app.get_output_writer("good_writer").write(1, []) == {"detected": 1, "warnings": []}

    with pytest.raises(KeyError):
        app.get_detector("bad_plugin")

    assert any("Failed to import plugin module from" in message for message in caplog.messages)

