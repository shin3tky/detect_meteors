"""Tests for input loader base classes."""

from dataclasses import dataclass
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

from meteor_core.inputs.base import (
    BaseMetadataExtractor,
    DataclassInputLoader,
    PydanticInputLoader,
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
