"""Tests for input loader base classes."""

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

from meteor_core.inputs.base import (
    BaseInputLoader,
    BaseMetadataExtractor,
    DataclassInputLoader,
    PydanticInputLoader,
    _is_valid_input_loader,
    supports_metadata_extraction,
)
from meteor_core.schema import InputContext


@dataclass
class ExampleConfig:
    """Dataclass configuration for the sample loader."""

    threshold: int = 5


class ExampleDataclassLoader(DataclassInputLoader[ExampleConfig]):
    """Concrete loader using dataclass configuration."""

    plugin_name = "example_dataclass"
    ConfigType = ExampleConfig

    def load(self, filepath: str) -> InputContext:
        return InputContext(image_data=filepath, filepath=filepath)


_PYDANTIC_SPEC = importlib.util.find_spec("pydantic")
if _PYDANTIC_SPEC:
    from pydantic import BaseModel

    class ExampleModel(BaseModel):
        """Pydantic configuration for the sample loader."""

        threshold: int = 10

    class ExamplePydanticLoader(PydanticInputLoader[ExampleModel]):
        """Concrete loader using pydantic configuration."""

        plugin_name = "example_pydantic"
        ConfigType = ExampleModel

        def load(self, filepath: str) -> InputContext:
            return InputContext(image_data=filepath, filepath=filepath)
else:
    BaseModel = None
    ExampleModel = None
    ExamplePydanticLoader = None


class ExampleMetadataLoader(ExampleDataclassLoader, BaseMetadataExtractor):
    """Loader that supports metadata extraction."""

    plugin_name = "example_metadata"

    def extract_metadata(self, filepath: str) -> dict:
        return {"path": filepath}


class MinimalInputLoader(BaseInputLoader[None]):
    """Minimal concrete loader to exercise base class paths."""

    plugin_name = "minimal_input"

    def load(self, filepath: str) -> InputContext:
        return InputContext(image_data=filepath, filepath=filepath)


class MinimalMetadataExtractor(BaseMetadataExtractor):
    """Minimal metadata extractor implementation."""

    def extract_metadata(self, filepath: str) -> dict:
        return {"filepath": filepath}


class TestInputLoaderBases(unittest.TestCase):
    """Coverage for dataclass and pydantic loader bases."""

    def test_dataclass_loader_accepts_config(self) -> None:
        config = ExampleConfig(threshold=7)
        loader = ExampleDataclassLoader(config)
        self.assertEqual(loader.config.threshold, 7)
        info = loader.get_info()
        self.assertEqual(info["plugin_name"], "example_dataclass")
        self.assertEqual(info["class"], "ExampleDataclassLoader")

    def test_dataclass_loader_rejects_wrong_config_type(self) -> None:
        with self.assertRaises(TypeError):
            ExampleDataclassLoader(config="not-a-config")

    def test_dataclass_loader_rejects_non_dataclass_config_type(self) -> None:
        class BadConfig:
            pass

        class BadLoader(DataclassInputLoader[BadConfig]):
            plugin_name = "bad_loader"
            ConfigType = BadConfig

            def load(self, filepath: str) -> InputContext:
                return InputContext(image_data=filepath, filepath=filepath)

        with self.assertRaises(TypeError):
            BadLoader(BadConfig())

    @unittest.skipIf(BaseModel is None, "pydantic is not installed")
    def test_pydantic_loader_accepts_config(self) -> None:
        config = ExampleModel(threshold=12)
        loader = ExamplePydanticLoader(config)
        self.assertEqual(loader.config.threshold, 12)
        info = loader.get_info()
        self.assertEqual(info["plugin_name"], "example_pydantic")

    @unittest.skipIf(BaseModel is None, "pydantic is not installed")
    def test_pydantic_loader_rejects_wrong_config_type(self) -> None:
        with self.assertRaises(TypeError):
            ExamplePydanticLoader(config="not-a-model")

    @unittest.skipIf(BaseModel is None, "pydantic is not installed")
    def test_pydantic_loader_rejects_non_pydantic_config_type(self) -> None:
        class BadConfig:
            pass

        class BadLoader(PydanticInputLoader[BadConfig]):
            plugin_name = "bad_pydantic"
            ConfigType = BadConfig

            def load(self, filepath: str) -> InputContext:
                return InputContext(image_data=filepath, filepath=filepath)

        with self.assertRaises(TypeError):
            BadLoader(BadConfig())

    def test_supports_metadata_extraction(self) -> None:
        loader = ExampleMetadataLoader(ExampleConfig())
        self.assertTrue(supports_metadata_extraction(loader))
        self.assertFalse(
            supports_metadata_extraction(ExampleDataclassLoader(ExampleConfig()))
        )

    def test_base_methods_are_callable(self) -> None:
        loader = MinimalInputLoader()
        extractor = MinimalMetadataExtractor()
        self.assertIsNone(BaseInputLoader.load(loader, "sample.fits"))
        self.assertIsNone(
            BaseMetadataExtractor.extract_metadata(extractor, "sample.fits")
        )

    def test_is_valid_input_loader_variants(self) -> None:
        class NotLoader:
            plugin_name = "not_loader"

        class EmptyNameLoader(BaseInputLoader[None]):
            plugin_name = ""

            def load(self, filepath: str) -> InputContext:
                return InputContext(image_data=filepath, filepath=filepath)

        class ValidLoader(BaseInputLoader[None]):
            plugin_name = "valid_loader"

            def load(self, filepath: str) -> InputContext:
                return InputContext(image_data=filepath, filepath=filepath)

        self.assertFalse(_is_valid_input_loader(123))
        self.assertFalse(_is_valid_input_loader(NotLoader))
        self.assertFalse(_is_valid_input_loader(EmptyNameLoader))
        self.assertTrue(_is_valid_input_loader(ValidLoader))

    def test_pydantic_loader_import_error_without_basemodel(self) -> None:
        with mock.patch("meteor_core.inputs.base.BaseModel", None):

            class NoPydanticLoader(PydanticInputLoader[object]):
                plugin_name = "no_pydantic"
                ConfigType = object

                def load(self, filepath: str) -> InputContext:
                    return InputContext(image_data=filepath, filepath=filepath)

            with self.assertRaises(ImportError):
                NoPydanticLoader(object())

    def test_pydantic_loader_type_validation_with_stub(self) -> None:
        class DummyBaseModel:
            pass

        class GoodModel(DummyBaseModel):
            pass

        with mock.patch("meteor_core.inputs.base.BaseModel", DummyBaseModel):

            class BadConfig:
                pass

            class BadLoader(PydanticInputLoader[BadConfig]):
                plugin_name = "bad_config"
                ConfigType = BadConfig

                def load(self, filepath: str) -> InputContext:
                    return InputContext(image_data=filepath, filepath=filepath)

            class GoodLoader(PydanticInputLoader[GoodModel]):
                plugin_name = "good_config"
                ConfigType = GoodModel

                def load(self, filepath: str) -> InputContext:
                    return InputContext(image_data=filepath, filepath=filepath)

            with self.assertRaises(TypeError):
                BadLoader(BadConfig())

            with self.assertRaises(TypeError):
                GoodLoader(DummyBaseModel())

            config = GoodModel()
            loader = GoodLoader(config)
            self.assertIs(loader.config, config)

    def test_inputs_base_reload_without_pydantic(self) -> None:
        import meteor_core.inputs.base as inputs_base

        with mock.patch("importlib.util.find_spec", return_value=None):
            module_name = "meteor_core.inputs.base_no_pydantic"
            spec = importlib.util.spec_from_file_location(
                module_name,
                inputs_base.__file__,
            )
            self.assertIsNotNone(spec)
            module = importlib.util.module_from_spec(spec)
            self.assertIsNotNone(spec.loader)
            spec.loader.exec_module(module)
            self.assertIsNone(module.BaseModel)
