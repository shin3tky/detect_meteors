"""Tests for output handler base classes."""

from dataclasses import dataclass
import importlib
import importlib.machinery
import importlib.util
import sys
import types
import unittest
from unittest import mock

try:
    _PIL_SPEC = importlib.util.find_spec("PIL.Image")
except ModuleNotFoundError:
    _PIL_SPEC = None
if _PIL_SPEC is None:
    pil_module = types.ModuleType("PIL")
    pil_module.__spec__ = importlib.machinery.ModuleSpec("PIL", loader=None)
    image_module = types.ModuleType("PIL.Image")
    image_module.__spec__ = importlib.machinery.ModuleSpec("PIL.Image", loader=None)
    sys.modules.setdefault("PIL", pil_module)
    sys.modules.setdefault("PIL.Image", image_module)

try:
    _NUMPY_SPEC = importlib.util.find_spec("numpy")
except (ModuleNotFoundError, ValueError):
    _NUMPY_SPEC = None
if _NUMPY_SPEC is None:
    numpy_module = sys.modules.get("numpy")
    if numpy_module is None:
        numpy_module = types.ModuleType("numpy")
        sys.modules["numpy"] = numpy_module
    numpy_module.__spec__ = importlib.machinery.ModuleSpec("numpy", loader=None)
    numpy_module.ndarray = object
    numpy_module.float32 = float
    numpy_module.uint8 = int
    numpy_module.uint16 = int
    numpy_module.int32 = int
    numpy_module.zeros = lambda *_args, **_kwargs: [[0]]

import numpy as np

from meteor_core.outputs.base import (
    BaseOutputHandler,
    DataclassOutputHandler,
    PydanticOutputHandler,
    _is_valid_output_handler,
)
from meteor_core.schema import OutputResult


@dataclass
class ExampleOutputConfig:
    """Dataclass configuration for a sample output handler."""

    output_dir: str = "/tmp"


class ExampleOutputHandler(DataclassOutputHandler[ExampleOutputConfig]):
    """Concrete output handler using dataclass configuration."""

    plugin_name = "example_output"
    ConfigType = ExampleOutputConfig

    def save_candidate(
        self, source_path: str, filename: str, debug_image=None, roi_polygon=None
    ) -> OutputResult:
        return OutputResult(saved=True, output_path=source_path, debug_path=None)

    def save_debug_image(
        self, debug_image: np.ndarray, filename: str, roi_polygon=None
    ) -> str:
        return f"/tmp/{filename}"


class MinimalOutputHandler(BaseOutputHandler):
    """Minimal output handler implementation."""

    plugin_name = "minimal_output"

    def save_candidate(
        self, source_path: str, filename: str, debug_image=None, roi_polygon=None
    ) -> OutputResult:
        return OutputResult(saved=False, output_path=source_path, debug_path=None)

    def save_debug_image(
        self, debug_image: np.ndarray, filename: str, roi_polygon=None
    ) -> str:
        return f"/tmp/{filename}"


class TestOutputHandlerBases(unittest.TestCase):
    """Coverage for output handler base behaviors."""

    def test_base_output_handler_methods_are_callable(self) -> None:
        handler = MinimalOutputHandler()
        self.assertIsNone(BaseOutputHandler.save_candidate(handler, "src", "name"))
        self.assertIsNone(
            BaseOutputHandler.save_debug_image(handler, np.zeros((1, 1)), "dbg")
        )
        handler.on_candidate_detected("file", True, score=1.0, aspect_ratio=1.2)
        handler.on_batch_complete(1, 1, 1)
        handler.on_pipeline_complete(1, 1, 0.5)

    def test_output_handler_get_info(self) -> None:
        handler = MinimalOutputHandler()
        info = handler.get_info()
        self.assertEqual(info["plugin_name"], "minimal_output")
        self.assertEqual(info["class"], "MinimalOutputHandler")

    def test_dataclass_output_handler_validation(self) -> None:
        with self.assertRaises(TypeError):

            class BadConfig:
                pass

            class BadHandler(DataclassOutputHandler[BadConfig]):
                plugin_name = "bad_output"
                ConfigType = BadConfig

                def save_candidate(
                    self,
                    source_path: str,
                    filename: str,
                    debug_image=None,
                    roi_polygon=None,
                ) -> OutputResult:
                    return OutputResult(
                        saved=True, output_path=source_path, debug_path=None
                    )

                def save_debug_image(
                    self, debug_image: np.ndarray, filename: str, roi_polygon=None
                ) -> str:
                    return "/tmp/bad.png"

            BadHandler(BadConfig())

        class GoodHandler(DataclassOutputHandler[ExampleOutputConfig]):
            plugin_name = "good_output"
            ConfigType = ExampleOutputConfig

            def save_candidate(
                self,
                source_path: str,
                filename: str,
                debug_image=None,
                roi_polygon=None,
            ) -> OutputResult:
                return OutputResult(
                    saved=True, output_path=source_path, debug_path=None
                )

            def save_debug_image(
                self, debug_image: np.ndarray, filename: str, roi_polygon=None
            ) -> str:
                return "/tmp/good.png"

        with self.assertRaises(TypeError):
            GoodHandler(config="not-a-config")

        handler = GoodHandler(ExampleOutputConfig(output_dir="/data"))
        self.assertEqual(handler.config.output_dir, "/data")

    def test_is_valid_output_handler_variants(self) -> None:
        class NotHandler:
            plugin_name = "not_handler"

        class EmptyNameHandler(BaseOutputHandler):
            plugin_name = ""

            def save_candidate(
                self,
                source_path: str,
                filename: str,
                debug_image=None,
                roi_polygon=None,
            ) -> OutputResult:
                return OutputResult(
                    saved=True, output_path=source_path, debug_path=None
                )

            def save_debug_image(
                self, debug_image: np.ndarray, filename: str, roi_polygon=None
            ) -> str:
                return "/tmp/empty.png"

        class ValidHandler(MinimalOutputHandler):
            plugin_name = "valid_output"

        self.assertFalse(_is_valid_output_handler(123))
        self.assertFalse(_is_valid_output_handler(NotHandler))
        self.assertFalse(_is_valid_output_handler(EmptyNameHandler))
        self.assertTrue(_is_valid_output_handler(ValidHandler))

    def test_pydantic_output_handler_import_error_without_basemodel(self) -> None:
        with mock.patch("meteor_core.outputs.base.BaseModel", None):

            class NoPydanticHandler(PydanticOutputHandler[object]):
                plugin_name = "no_pydantic"
                ConfigType = object

                def save_candidate(
                    self,
                    source_path: str,
                    filename: str,
                    debug_image=None,
                    roi_polygon=None,
                ) -> OutputResult:
                    return OutputResult(
                        saved=True, output_path=source_path, debug_path=None
                    )

                def save_debug_image(
                    self, debug_image: np.ndarray, filename: str, roi_polygon=None
                ) -> str:
                    return "/tmp/nop.png"

            with self.assertRaises(ImportError):
                NoPydanticHandler(object())

    def test_pydantic_output_handler_type_validation_with_stub(self) -> None:
        class DummyBaseModel:
            pass

        class GoodModel(DummyBaseModel):
            pass

        with mock.patch("meteor_core.outputs.base.BaseModel", DummyBaseModel):

            class BadConfig:
                pass

            class BadHandler(PydanticOutputHandler[BadConfig]):
                plugin_name = "bad_config"
                ConfigType = BadConfig

                def save_candidate(
                    self,
                    source_path: str,
                    filename: str,
                    debug_image=None,
                    roi_polygon=None,
                ) -> OutputResult:
                    return OutputResult(
                        saved=True, output_path=source_path, debug_path=None
                    )

                def save_debug_image(
                    self, debug_image: np.ndarray, filename: str, roi_polygon=None
                ) -> str:
                    return "/tmp/bad.png"

            class GoodHandler(PydanticOutputHandler[GoodModel]):
                plugin_name = "good_config"
                ConfigType = GoodModel

                def save_candidate(
                    self,
                    source_path: str,
                    filename: str,
                    debug_image=None,
                    roi_polygon=None,
                ) -> OutputResult:
                    return OutputResult(
                        saved=True, output_path=source_path, debug_path=None
                    )

                def save_debug_image(
                    self, debug_image: np.ndarray, filename: str, roi_polygon=None
                ) -> str:
                    return "/tmp/good.png"

            with self.assertRaises(TypeError):
                BadHandler(BadConfig())

            with self.assertRaises(TypeError):
                GoodHandler(DummyBaseModel())

            config = GoodModel()
            handler = GoodHandler(config)
            self.assertIs(handler.config, config)

    def test_outputs_base_reload_without_pydantic(self) -> None:
        import meteor_core.outputs.base as outputs_base

        with mock.patch("importlib.util.find_spec", return_value=None):
            module_name = "meteor_core.outputs.base_no_pydantic"
            spec = importlib.util.spec_from_file_location(
                module_name,
                outputs_base.__file__,
            )
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(spec.loader)
            spec.loader.exec_module(module)
            self.assertIsNone(module.BaseModel)
