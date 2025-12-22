"""Parity checks for discovery helpers across plugin kinds."""

from __future__ import annotations

import importlib
import sys
import tempfile
import textwrap
import types
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable
import unittest


class _StubEntryPoint:
    """Simple stand-in for importlib metadata entry points."""

    def __init__(self, name: str, loader: Callable[[], type]):
        self.name = name
        self._loader = loader
        self.value = f"stub:{name}"

    def load(self):  # pragma: no cover - trivial
        return self._loader()


def _write_plugin(
    destination: Path, class_name: str, plugin_name: str, base_import: str, body: str
) -> None:
    destination.write_text(
        textwrap.dedent(
            f"""
            {base_import}

            class {class_name}(BaseClass):
                plugin_name = "{plugin_name}"
            {body}
            """
        )
    )


def _loader_methods(_: str) -> Dict[str, Callable]:
    def __init__(self, config=None):
        self.config = config

    def load(self, filepath: str):
        return filepath

    return {"__init__": __init__, "load": load}


def _detector_methods(_: str) -> Dict[str, Callable]:
    def __init__(self, config=None):
        self.config = config

    def detect(self, context):
        return {}

    return {"__init__": __init__, "detect": detect}


def _handler_methods(_: str) -> Dict[str, Callable]:
    def __init__(self, config=None):
        self.config = config

    def save_candidate(self, source_path, filename, debug_image=None, roi_polygon=None):
        return True

    def save_debug_image(self, debug_image, filename, roi_polygon=None):
        return "saved"

    return {
        "__init__": __init__,
        "save_candidate": save_candidate,
        "save_debug_image": save_debug_image,
    }


def _get_discovery_cases():
    """Load discovery modules with a stubbed OpenCV dependency."""

    sys.modules.setdefault("cv2", types.SimpleNamespace())

    input_discovery = importlib.import_module("meteor_core.inputs.discovery")
    BaseInputLoader = importlib.import_module("meteor_core.inputs.base").BaseInputLoader

    detector_discovery = importlib.import_module("meteor_core.detectors.discovery")
    BaseDetector = importlib.import_module("meteor_core.detectors.base").BaseDetector

    output_discovery = importlib.import_module("meteor_core.outputs.discovery")
    BaseOutputHandler = importlib.import_module(
        "meteor_core.outputs.base"
    ).BaseOutputHandler

    return (
        {
            "kind": "input",
            "module": input_discovery,
            "base_cls": BaseInputLoader,
            "base_import": "from meteor_core.inputs.base import BaseInputLoader as BaseClass",
            "method_block": """
                def __init__(self, config=None):
                    self.config = config

                def load(self, filepath: str):
                    return filepath
            """,
            "methods_factory": _loader_methods,
            "noun": "loader",
        },
        {
            "kind": "detector",
            "module": detector_discovery,
            "base_cls": BaseDetector,
            "base_import": "from meteor_core.detectors.base import BaseDetector as BaseClass",
            "method_block": """
                def __init__(self, config=None):
                    self.config = config

                def detect(self, context):
                    return {}
            """,
            "methods_factory": _detector_methods,
            "noun": "detector",
        },
        {
            "kind": "output",
            "module": output_discovery,
            "base_cls": BaseOutputHandler,
            "base_import": "from meteor_core.outputs.base import BaseOutputHandler as BaseClass",
            "method_block": """
                def __init__(self, config=None):
                    self.config = config

                def save_candidate(self, source_path, filename, debug_image=None, roi_polygon=None):
                    return True

                def save_debug_image(self, debug_image, filename, roi_polygon=None):
                    return "saved"
            """,
            "methods_factory": _handler_methods,
            "noun": "handler",
        },
    )


def _make_entry_points(
    base_cls: type, methods_factory: Callable[[str], Dict[str, Callable]]
) -> Iterable[_StubEntryPoint]:
    def _make(plugin_name: str) -> type:
        namespace = {"plugin_name": plugin_name, **methods_factory(plugin_name)}
        return types.new_class(
            plugin_name.title(), (base_cls,), {}, lambda ns: ns.update(namespace)
        )

    return (
        _StubEntryPoint("z_ep", lambda: _make("ep_z")),
        _StubEntryPoint("a_ep", lambda: _make("ep_a")),
    )


class TestDiscoveryParity(unittest.TestCase):
    def test_discovery_order_matches(self):
        """Entry points precede plugin files consistently across modules."""

        for case in _get_discovery_cases():
            with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
                plugin_dir = Path(tmpdir)
                _write_plugin(
                    plugin_dir / "b_plugin.py",
                    "PluginB",
                    "dir_b",
                    case["base_import"],
                    case["method_block"],
                )
                _write_plugin(
                    plugin_dir / "a_plugin.py",
                    "PluginA",
                    "dir_a",
                    case["base_import"],
                    case["method_block"],
                )

                entry_points = list(
                    _make_entry_points(case["base_cls"], case["methods_factory"])
                )

                module = case["module"]
                original_iter_entry_points = module._iter_entry_points
                module._iter_entry_points = lambda: entry_points  # type: ignore[assignment]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        discovered = module._discover_handlers_internal(plugin_dir)
                finally:
                    module._iter_entry_points = original_iter_entry_points  # type: ignore[assignment]
                keys = list(discovered.keys())
                self.assertEqual(
                    keys[-4:], ["ep_a", "ep_z", "dir_a", "dir_b"], case["kind"]
                )

    def test_duplicate_warning_alignment(self):
        """Duplicate plugin_name warnings follow the same format per module."""

        for case in _get_discovery_cases():
            with self.subTest(kind=case["kind"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    plugin_dir = Path(tmpdir)
                    _write_plugin(
                        plugin_dir / "dup_plugin.py",
                        "DuplicatePlugin",
                        "dup_plugin",
                        case["base_import"],
                        case["method_block"],
                    )

                    entry_points = (
                        _StubEntryPoint(
                            "dup_ep",
                            lambda: types.new_class(
                                "DupEntryPoint",
                                (case["base_cls"],),
                                {},
                                lambda ns: ns.update(
                                    {
                                        "plugin_name": "dup_plugin",
                                        **case["methods_factory"]("dup_plugin"),
                                    }
                                ),
                            ),
                        ),
                    )

                    module = case["module"]
                    original_iter_entry_points = module._iter_entry_points
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            module._iter_entry_points = lambda: entry_points  # type: ignore[assignment]
                            module._discover_handlers_internal(plugin_dir)
                    finally:
                        module._iter_entry_points = original_iter_entry_points  # type: ignore[assignment]

                    self.assertTrue(caught, "Expected duplicate warning")
                    message = str(caught[-1].message)
                    self.assertIn(
                        f"Duplicate {case['noun']} name 'dup_plugin' from plugin file",
                        message,
                    )
