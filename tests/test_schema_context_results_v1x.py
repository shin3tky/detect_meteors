"""Tests for schema context/result dataclasses."""

import importlib.machinery
import importlib.util
import sys
import types
import unittest

if importlib.util.find_spec("yaml") is None:
    sys.modules.setdefault(
        "yaml", types.SimpleNamespace(safe_load=lambda *_: {}, YAMLError=Exception)
    )
if importlib.util.find_spec("numpy") is None:
    sys.modules.setdefault(
        "numpy",
        types.SimpleNamespace(
            ndarray=object,
            float32=float,
            uint8=int,
            uint16=int,
            int32=int,
        ),
    )
if importlib.util.find_spec("rawpy") is None:
    sys.modules.setdefault("rawpy", types.SimpleNamespace())
if importlib.util.find_spec("cv2") is None:
    sys.modules.setdefault("cv2", types.SimpleNamespace())
try:
    _PIL_SPEC = importlib.util.find_spec("PIL.Image")
except ModuleNotFoundError:
    _PIL_SPEC = None
if _PIL_SPEC is None:
    pil_module = types.ModuleType("PIL")
    pil_module.__path__ = []
    pil_module.__spec__ = importlib.machinery.ModuleSpec("PIL", loader=None)
    image_module = types.ModuleType("PIL.Image")
    image_module.__spec__ = importlib.machinery.ModuleSpec("PIL.Image", loader=None)
    sys.modules.setdefault("PIL", pil_module)
    sys.modules.setdefault("PIL.Image", image_module)

from meteor_core.schema import (
    INPUT_CONTEXT_SCHEMA_VERSION,
    OUTPUT_RESULT_SCHEMA_VERSION,
    InputContext,
    OutputResult,
)


class TestInputContext(unittest.TestCase):
    """Coverage for InputContext helpers and defaults."""

    def test_to_dict_exposes_metadata_without_image_data(self) -> None:
        context = InputContext(
            image_data="image-bytes",
            filepath="/tmp/test.raw",
            metadata={"camera": "nx"},
            loader_info={"loader": "raw"},
        )

        payload = context.to_dict()

        self.assertEqual(
            payload,
            {
                "filepath": "/tmp/test.raw",
                "metadata": {"camera": "nx"},
                "loader_info": {"loader": "raw"},
                "schema_version": INPUT_CONTEXT_SCHEMA_VERSION,
            },
        )
        self.assertNotIn("image_data", payload)

    def test_default_dicts_are_independent(self) -> None:
        first = InputContext(image_data=None, filepath="/tmp/first.raw")
        second = InputContext(image_data=None, filepath="/tmp/second.raw")

        first.metadata["key"] = "value"
        first.loader_info["loader"] = "first"

        self.assertEqual(second.metadata, {})
        self.assertEqual(second.loader_info, {})


class TestOutputResult(unittest.TestCase):
    """Coverage for OutputResult helpers and defaults."""

    def test_to_dict_includes_paths_and_metrics(self) -> None:
        result = OutputResult(
            saved=True,
            output_path="/tmp/out.fits",
            debug_path=None,
            handler_info={"handler": "file"},
            metrics={"elapsed_ms": 12.5},
        )

        payload = result.to_dict()

        self.assertEqual(
            payload,
            {
                "saved": True,
                "output_path": "/tmp/out.fits",
                "debug_path": None,
                "handler_info": {"handler": "file"},
                "metrics": {"elapsed_ms": 12.5},
                "schema_version": OUTPUT_RESULT_SCHEMA_VERSION,
            },
        )

    def test_default_dicts_are_independent(self) -> None:
        first = OutputResult(saved=False, output_path=None, debug_path=None)
        second = OutputResult(saved=False, output_path=None, debug_path=None)

        first.handler_info["key"] = "value"
        first.metrics["elapsed_ms"] = 10.0

        self.assertEqual(second.handler_info, {})
        self.assertEqual(second.metrics, {})


if __name__ == "__main__":
    unittest.main()
