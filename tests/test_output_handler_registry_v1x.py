#
# Detect Meteors CLI - Output Handler Registry Tests
# © 2025 Shinichi Morita (shin3tky)
#

"""
Unit tests for OutputHandlerRegistry.
"""

import importlib.util
import tempfile
import unittest
import warnings
from tempfile import TemporaryDirectory
from pathlib import Path

from meteor_core.outputs.registry import OutputHandlerRegistry
from meteor_core.outputs.base import (
    BaseOutputHandler,
    PydanticOutputHandler,
    _is_valid_output_handler,
    forbid_unknown_keys,
)
from meteor_core.outputs.file_handler import FileOutputHandler, FileOutputConfig
from meteor_core.exceptions import MeteorConfigError
from meteor_core.schema import (
    DEFAULT_DEBUG_FOLDER,
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_OUTPUT_HANDLER_NAME,
)

_PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None
if _PYDANTIC_AVAILABLE:
    from pydantic import BaseModel, ValidationError


class TestOutputHandlerRegistryDiscovery(unittest.TestCase):
    """Tests for OutputHandlerRegistry discovery functionality."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_discover_returns_dict(self):
        """discover() returns a dictionary."""
        handlers = OutputHandlerRegistry.discover()
        self.assertIsInstance(handlers, dict)

    def test_discover_includes_builtin_file(self):
        """Built-in file handler is always discovered."""
        handlers = OutputHandlerRegistry.discover()
        self.assertIn("file", handlers)

    def test_discover_caches_result(self):
        """discover() caches results for subsequent calls."""
        handlers1 = OutputHandlerRegistry.discover()
        handlers2 = OutputHandlerRegistry.discover()
        self.assertIs(handlers1, handlers2)

    def test_discover_force_refreshes_cache(self):
        """discover(force=True) refreshes the cache."""
        handlers1 = OutputHandlerRegistry.discover()
        handlers2 = OutputHandlerRegistry.discover(force=True)
        # Should be equal content but different object
        self.assertEqual(handlers1.keys(), handlers2.keys())


class TestOutputHandlerRegistryGet(unittest.TestCase):
    """Tests for OutputHandlerRegistry.get() method."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_get_existing_handler(self):
        """get() returns handler class for existing name."""
        handler_cls = OutputHandlerRegistry.get("file")
        self.assertEqual(handler_cls.plugin_name, "file")

    def test_get_case_insensitive(self):
        """get() is case-insensitive."""
        handler_cls1 = OutputHandlerRegistry.get("file")
        handler_cls2 = OutputHandlerRegistry.get("FILE")
        handler_cls3 = OutputHandlerRegistry.get("File")
        self.assertIs(handler_cls1, handler_cls2)
        self.assertIs(handler_cls2, handler_cls3)

    def test_get_nonexistent_raises_keyerror(self):
        """get() raises KeyError for unknown handler name."""
        with self.assertRaises(KeyError) as ctx:
            OutputHandlerRegistry.get("nonexistent_handler")

        # Error message should include available handlers
        error_msg = str(ctx.exception)
        self.assertIn("nonexistent_handler", error_msg)
        self.assertIn("Available", error_msg)

    def test_get_custom_takes_priority(self):
        """Runtime-registered handler takes priority over discovered."""

        class CustomFileHandler(BaseOutputHandler):
            plugin_name = "file"

            def __init__(self, config=None):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/custom/path"

        OutputHandlerRegistry.register(CustomFileHandler)
        handler_cls = OutputHandlerRegistry.get("file")
        self.assertIs(handler_cls, CustomFileHandler)


class TestOutputHandlerRegistryRegister(unittest.TestCase):
    """Tests for OutputHandlerRegistry.register() method."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_register_valid_handler(self):
        """register() accepts valid handler class."""

        class MockHandler(BaseOutputHandler):
            plugin_name = "mock"

            def __init__(self, config=None):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/mock/path"

        OutputHandlerRegistry.register(MockHandler)
        self.assertIn("mock", OutputHandlerRegistry.list_available())

    def test_register_invalid_handler_raises_valueerror(self):
        """register() raises ValueError for invalid handler."""

        class NotAHandler:
            plugin_name = "invalid"

        with self.assertRaises(ValueError):
            OutputHandlerRegistry.register(NotAHandler)

    def test_register_empty_plugin_name_raises_valueerror(self):
        """register() raises ValueError for empty plugin_name."""

        class EmptyNameHandler(BaseOutputHandler):
            plugin_name = ""

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        with self.assertRaises(ValueError):
            OutputHandlerRegistry.register(EmptyNameHandler)

    def test_register_duplicate_warns(self):
        """register() warns when overwriting existing handler."""

        class MockHandler1(BaseOutputHandler):
            plugin_name = "mock_dup"

            def __init__(self, config=None):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        class MockHandler2(BaseOutputHandler):
            plugin_name = "mock_dup"

            def __init__(self, config=None):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return False

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path2"

        OutputHandlerRegistry.register(MockHandler1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            OutputHandlerRegistry.register(MockHandler2)
            self.assertEqual(len(w), 1)
            self.assertIn("Overwriting", str(w[0].message))


class TestOutputHandlerRegistryUnregister(unittest.TestCase):
    """Tests for OutputHandlerRegistry.unregister() method."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_unregister_custom_handler(self):
        """unregister() removes runtime-registered handler."""

        class MockHandler(BaseOutputHandler):
            plugin_name = "mock_unreg"

            def __init__(self, config=None):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        OutputHandlerRegistry.register(MockHandler)
        self.assertIn("mock_unreg", OutputHandlerRegistry.list_available())

        result = OutputHandlerRegistry.unregister("mock_unreg")
        self.assertTrue(result)
        self.assertNotIn("mock_unreg", OutputHandlerRegistry.list_available())

    def test_unregister_nonexistent_returns_false(self):
        """unregister() returns False for nonexistent handler."""
        result = OutputHandlerRegistry.unregister("nonexistent")
        self.assertFalse(result)

    def test_unregister_discovered_returns_false(self):
        """unregister() returns False for discovered (not custom) handlers."""
        # file handler is discovered, not custom
        result = OutputHandlerRegistry.unregister("file")
        self.assertFalse(result)
        # Still available via discovery
        self.assertIn("file", OutputHandlerRegistry.list_available())


class TestOutputHandlerRegistryListAvailable(unittest.TestCase):
    """Tests for OutputHandlerRegistry.list_available() method."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_list_available_returns_sorted_list(self):
        """list_available() returns a sorted list of names."""
        available = OutputHandlerRegistry.list_available()
        self.assertIsInstance(available, list)
        self.assertEqual(available, sorted(available))

    def test_list_available_includes_file(self):
        """list_available() includes built-in file handler."""
        available = OutputHandlerRegistry.list_available()
        self.assertIn("file", available)

    def test_list_available_includes_custom(self):
        """list_available() includes runtime-registered handlers."""

        class CustomHandler(BaseOutputHandler):
            plugin_name = "custom_list"

            def __init__(self, config=None):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        OutputHandlerRegistry.register(CustomHandler)
        available = OutputHandlerRegistry.list_available()
        self.assertIn("custom_list", available)


class TestOutputHandlerRegistryCreate(unittest.TestCase):
    """Tests for OutputHandlerRegistry.create() method."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_create_with_config_dict(self):
        """create() accepts config as dictionary."""
        handler = OutputHandlerRegistry.create(
            "file",
            {
                "output_folder": "/tmp/output",
                "debug_folder": "/tmp/debug",
            },
        )
        self.assertIsInstance(handler, FileOutputHandler)
        self.assertEqual(handler.config.output_folder, "/tmp/output")
        self.assertEqual(handler.config.debug_folder, "/tmp/debug")

    def test_create_with_config_instance(self):
        """create() accepts config as dataclass instance."""
        config = FileOutputConfig(
            output_folder="/tmp/output2",
            debug_folder="/tmp/debug2",
            output_overwrite=True,
        )
        handler = OutputHandlerRegistry.create("file", config)
        self.assertIsInstance(handler, FileOutputHandler)
        self.assertEqual(handler.config.output_folder, "/tmp/output2")
        self.assertTrue(handler.config.output_overwrite)

    def test_create_nonexistent_raises_keyerror(self):
        """create() raises KeyError for unknown handler name."""
        with self.assertRaises(KeyError):
            OutputHandlerRegistry.create("nonexistent", {})

    def test_create_with_invalid_config_raises_typeerror(self):
        """create() raises TypeError for invalid config structure."""
        with self.assertRaises(TypeError):
            OutputHandlerRegistry.create("file", {"invalid_key": "value"})


class TestOutputHandlerRegistryCreateDefault(unittest.TestCase):
    """Tests for OutputHandlerRegistry.create_default() method."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_create_default_returns_file_handler(self):
        """create_default() returns FileOutputHandler instance."""
        with (
            TemporaryDirectory() as output_folder,
            TemporaryDirectory() as debug_folder,
        ):
            handler = OutputHandlerRegistry.create_default(
                output_folder=output_folder,
                debug_folder=debug_folder,
            )
            self.assertIsInstance(handler, FileOutputHandler)
            self.assertEqual(handler.plugin_name, DEFAULT_OUTPUT_HANDLER_NAME)
            self.assertEqual(handler.config.output_folder, output_folder)
            self.assertEqual(handler.config.debug_folder, debug_folder)

    def test_create_default_respects_overwrite_flag(self):
        """create_default() passes output_overwrite to config."""
        with (
            TemporaryDirectory() as output_folder,
            TemporaryDirectory() as debug_folder,
        ):
            handler = OutputHandlerRegistry.create_default(
                output_folder=output_folder,
                debug_folder=debug_folder,
                output_overwrite=True,
            )
            self.assertTrue(handler.config.output_overwrite)

    def test_create_default_uses_config_defaults_when_not_overridden(self):
        """create_default() seeds config from ConfigType defaults."""
        with (
            TemporaryDirectory() as output_folder,
            TemporaryDirectory() as debug_folder,
        ):
            handler = OutputHandlerRegistry.create_default(
                output_folder=output_folder,
                debug_folder=debug_folder,
            )
            self.assertFalse(handler.config.output_overwrite)


class TestOutputHandlerRegistryCoerceConfig(unittest.TestCase):
    """Tests for OutputHandlerRegistry._coerce_config() method."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_coerce_dict_to_dataclass(self):
        """_coerce_config() converts dict to dataclass."""
        config = OutputHandlerRegistry._coerce_config(
            FileOutputHandler,
            {"output_folder": "/tmp/o", "debug_folder": "/tmp/d"},
        )
        self.assertIsInstance(config, FileOutputConfig)
        self.assertEqual(config.output_folder, "/tmp/o")

    def test_coerce_none_uses_defaults(self):
        """_coerce_config() creates default config when ConfigType has defaults."""
        config = OutputHandlerRegistry._coerce_config(FileOutputHandler, None)
        self.assertIsInstance(config, FileOutputConfig)
        self.assertEqual(config.output_folder, DEFAULT_OUTPUT_FOLDER)
        self.assertEqual(config.debug_folder, DEFAULT_DEBUG_FOLDER)

    def test_coerce_passthrough_correct_type(self):
        """_coerce_config() passes through correct type unchanged."""
        original = FileOutputConfig(
            output_folder="/tmp/o",
            debug_folder="/tmp/d",
        )
        result = OutputHandlerRegistry._coerce_config(FileOutputHandler, original)
        self.assertIs(result, original)


@unittest.skipUnless(_PYDANTIC_AVAILABLE, "pydantic not installed")
class TestPydanticOutputHandlerSupport(unittest.TestCase):
    """Tests for PydanticOutputHandler and config utilities."""

    def setUp(self):
        OutputHandlerRegistry._reset()

    def tearDown(self):
        OutputHandlerRegistry._reset()

    def test_coerce_dict_to_pydantic_model(self):
        """_coerce_config() converts dict to Pydantic model for handlers."""

        class PydanticConfig(BaseModel):
            output_folder: str
            debug_folder: str

        class PydanticHandler(PydanticOutputHandler[PydanticConfig]):
            plugin_name = "pydantic_handler"
            ConfigType = PydanticConfig

            def __init__(self, config: PydanticConfig):
                super().__init__(config)

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        coerced = OutputHandlerRegistry._coerce_config(
            PydanticHandler,
            {"output_folder": "/tmp/o", "debug_folder": "/tmp/d"},
        )
        self.assertIsInstance(coerced, PydanticConfig)
        handler = PydanticHandler(coerced)
        self.assertIsInstance(handler.config, PydanticConfig)

    def test_forbid_unknown_keys_rejects_extra_fields(self):
        """forbid_unknown_keys() enforces extra=forbid on Pydantic models."""

        class Config(BaseModel):
            output_folder: str

        StrictConfig = forbid_unknown_keys(Config)

        with self.assertRaises(ValidationError):
            if hasattr(StrictConfig, "model_validate"):
                StrictConfig.model_validate(
                    {"output_folder": "/tmp/o", "unknown": "value"}
                )
            else:
                StrictConfig.parse_obj({"output_folder": "/tmp/o", "unknown": "value"})


class TestOutputHandlerRegistryReset(unittest.TestCase):
    """Tests for OutputHandlerRegistry._reset() method."""

    def test_reset_clears_custom_handlers(self):
        """_reset() clears runtime-registered handlers."""

        class TempHandler(BaseOutputHandler):
            plugin_name = "temp_reset"

            def __init__(self, config=None):
                self.config = config

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        OutputHandlerRegistry.register(TempHandler)
        self.assertIn("temp_reset", OutputHandlerRegistry.list_available())

        OutputHandlerRegistry._reset()
        self.assertNotIn("temp_reset", OutputHandlerRegistry.list_available())

    def test_reset_clears_discovered_cache(self):
        """_reset() clears the discovered handlers cache."""
        OutputHandlerRegistry.discover()  # Populate cache
        self.assertIsNotNone(OutputHandlerRegistry._discovered)

        OutputHandlerRegistry._reset()
        self.assertIsNone(OutputHandlerRegistry._discovered)


class TestDiscoverHandlersDeprecation(unittest.TestCase):
    """Tests for deprecated discover_handlers() function."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()

    def test_discover_handlers_warns_deprecation(self):
        """discover_handlers() emits DeprecationWarning."""
        from meteor_core.outputs.discovery import discover_handlers

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            discover_handlers()
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, DeprecationWarning))
            self.assertIn("deprecated", str(w[0].message).lower())


class TestIsValidOutputHandler(unittest.TestCase):
    """Tests for _is_valid_output_handler() function."""

    def test_valid_handler_returns_true(self):
        """_is_valid_output_handler() returns True for valid handler."""

        class ValidHandler(BaseOutputHandler):
            plugin_name = "valid"

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        self.assertTrue(_is_valid_output_handler(ValidHandler))

    def test_empty_plugin_name_returns_false(self):
        """_is_valid_output_handler() returns False for empty plugin_name."""

        class EmptyNameHandler(BaseOutputHandler):
            plugin_name = ""

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        self.assertFalse(_is_valid_output_handler(EmptyNameHandler))

    def test_non_subclass_returns_false(self):
        """_is_valid_output_handler() returns False for non-subclass."""

        class NotAHandler:
            plugin_name = "not_handler"

        self.assertFalse(_is_valid_output_handler(NotAHandler))

    def test_non_type_returns_false(self):
        """_is_valid_output_handler() returns False for non-type objects."""
        self.assertFalse(_is_valid_output_handler("not a type"))
        self.assertFalse(_is_valid_output_handler(123))
        self.assertFalse(_is_valid_output_handler(None))


class TestFileOutputHandler(unittest.TestCase):
    """Tests for FileOutputHandler class."""

    def test_file_output_handler_plugin_name(self):
        """FileOutputHandler has correct plugin_name."""
        self.assertEqual(FileOutputHandler.plugin_name, "file")

    def test_file_output_handler_get_info(self):
        """FileOutputHandler.get_info() returns correct metadata."""
        config = FileOutputConfig(
            output_folder="/tmp/o",
            debug_folder="/tmp/d",
        )
        handler = FileOutputHandler(config)
        info = handler.get_info()

        self.assertEqual(info["plugin_name"], "file")
        self.assertEqual(info["name"], "File Output Handler")
        self.assertIn("version", info)
        self.assertEqual(info["class"], "FileOutputHandler")


class TestBaseOutputHandlerHooks(unittest.TestCase):
    """Tests for BaseOutputHandler notification hooks."""

    def test_hooks_are_callable_noop(self):
        """Default hook implementations are callable and do nothing."""

        class MinimalHandler(BaseOutputHandler):
            plugin_name = "minimal"

            def __init__(self):
                pass

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

        handler = MinimalHandler()

        # These should not raise
        handler.on_candidate_detected("test.CR2", True, 100.0, 5.0)
        handler.on_batch_complete(10, 2, 5)
        handler.on_pipeline_complete(100, 10, 60.5)

    def test_hooks_can_be_overridden(self):
        """Hooks can be overridden to provide custom behavior."""
        callback_log = []

        class LoggingHandler(BaseOutputHandler):
            plugin_name = "logging"

            def __init__(self):
                pass

            def save_candidate(
                self, source_path, filename, debug_image=None, roi_polygon=None
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return "/path"

            def on_candidate_detected(
                self, filename, saved, score=0.0, aspect_ratio=0.0
            ):
                callback_log.append(("candidate", filename, saved, score))

            def on_batch_complete(self, processed_count, detected_count, batch_size):
                callback_log.append(("batch", processed_count, detected_count))

            def on_pipeline_complete(
                self, total_processed, total_detected, elapsed_seconds
            ):
                callback_log.append(("complete", total_processed, total_detected))

        handler = LoggingHandler()
        handler.on_candidate_detected("test.CR2", True, 150.0, 5.0)
        handler.on_batch_complete(10, 2, 5)
        handler.on_pipeline_complete(100, 10, 60.0)

        self.assertEqual(len(callback_log), 3)
        self.assertEqual(callback_log[0], ("candidate", "test.CR2", True, 150.0))
        self.assertEqual(callback_log[1], ("batch", 10, 2))
        self.assertEqual(callback_log[2], ("complete", 100, 10))


class TestResolveOutputHandler(unittest.TestCase):
    """Tests for _resolve_output_handler function."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()
        self._tempdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self._tempdir.name)

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()
        self._tempdir.cleanup()

    def _temp_path(self, name: str) -> str:
        """Return a namespaced temporary path for the given name."""
        return str(self.base_path / name)

    def test_resolve_with_explicit_instance(self):
        """Explicit handler instance is returned as-is."""
        from meteor_core.pipeline import _resolve_output_handler

        handler = FileOutputHandler(
            FileOutputConfig(
                output_folder=self._temp_path("a"),
                debug_folder=self._temp_path("b"),
            )
        )
        resolved = _resolve_output_handler(output_handler=handler)
        self.assertIs(resolved, handler)

    def test_resolve_with_handler_name(self):
        """Handler is created from registry by name."""
        from meteor_core.pipeline import _resolve_output_handler

        resolved = _resolve_output_handler(
            handler_name="file",
            handler_config={
                "output_folder": self._temp_path("x"),
                "debug_folder": self._temp_path("y"),
            },
        )
        self.assertIsInstance(resolved, FileOutputHandler)
        self.assertEqual(resolved.config.output_folder, self._temp_path("x"))
        self.assertEqual(resolved.config.debug_folder, self._temp_path("y"))

    def test_resolve_with_fallback_config(self):
        """Default handler is created with fallback config."""
        from meteor_core.pipeline import _resolve_output_handler

        resolved = _resolve_output_handler(
            fallback_output_folder=self._temp_path("out"),
            fallback_debug_folder=self._temp_path("debug"),
            fallback_output_overwrite=True,
        )
        self.assertIsInstance(resolved, FileOutputHandler)
        self.assertEqual(resolved.config.output_folder, self._temp_path("out"))
        self.assertEqual(resolved.config.debug_folder, self._temp_path("debug"))
        self.assertTrue(resolved.config.output_overwrite)

    def test_resolve_without_config_raises_error(self):
        """MeteorConfigError raised when no config provided."""
        from meteor_core.pipeline import _resolve_output_handler

        with self.assertRaises(MeteorConfigError) as context:
            _resolve_output_handler()
        self.assertIn("Cannot resolve output handler", str(context.exception))

    def test_resolve_priority_explicit_over_name(self):
        """Explicit instance takes priority over handler_name."""
        from meteor_core.pipeline import _resolve_output_handler

        explicit_handler = FileOutputHandler(
            FileOutputConfig(
                output_folder=self._temp_path("explicit"),
                debug_folder=self._temp_path("explicit_debug"),
            )
        )
        resolved = _resolve_output_handler(
            output_handler=explicit_handler,
            handler_name="file",
            handler_config={
                "output_folder": self._temp_path("named"),
                "debug_folder": self._temp_path("named_debug"),
            },
        )
        self.assertIs(resolved, explicit_handler)

    def test_resolve_priority_name_over_fallback(self):
        """Handler name takes priority over fallback config."""
        from meteor_core.pipeline import _resolve_output_handler

        resolved = _resolve_output_handler(
            handler_name="file",
            handler_config={
                "output_folder": self._temp_path("named"),
                "debug_folder": self._temp_path("named_debug"),
            },
            fallback_output_folder=self._temp_path("fallback"),
            fallback_debug_folder=self._temp_path("fallback_debug"),
        )
        self.assertEqual(resolved.config.output_folder, self._temp_path("named"))

    def test_resolve_invalid_handler_name_raises_config_error(self):
        """MeteorConfigError raised for unknown handler name."""
        from meteor_core.pipeline import _resolve_output_handler

        with self.assertRaises(MeteorConfigError) as ctx:
            _resolve_output_handler(handler_name="nonexistent")

        err = ctx.exception
        self.assertIn("nonexistent", str(err))
        self.assertEqual(err.plugin_name, "nonexistent")


class TestPipelineOutputHandlerIntegration(unittest.TestCase):
    """Tests for pipeline integration with output handler resolution."""

    def setUp(self):
        """Reset registry before each test."""
        OutputHandlerRegistry._reset()
        self._tempdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self._tempdir.name)

    def tearDown(self):
        """Reset registry after each test."""
        OutputHandlerRegistry._reset()
        self._tempdir.cleanup()

    def _temp_path(self, name: str) -> str:
        """Return a namespaced temporary path for the given name."""
        return str(self.base_path / name)

    def test_pipeline_with_output_handler_name(self):
        """Pipeline uses output_handler_name from config."""
        from meteor_core.pipeline import MeteorDetectionPipeline
        from meteor_core.schema import PipelineConfig, DetectionParams

        config = PipelineConfig(
            target_folder=self._temp_path("raw"),
            output_folder=self._temp_path("out"),
            debug_folder=self._temp_path("debug"),
            params=DetectionParams(),
            output_handler_name="file",
            output_handler_config={
                "output_folder": self._temp_path("custom_out"),
                "debug_folder": self._temp_path("custom_debug"),
            },
        )
        pipeline = MeteorDetectionPipeline(config)
        self.assertIsInstance(pipeline.output_handler, FileOutputHandler)
        self.assertEqual(
            pipeline.output_handler.config.output_folder, self._temp_path("custom_out")
        )

    def test_pipeline_fallback_to_config_folders(self):
        """Pipeline falls back to config folders when no handler specified."""
        from meteor_core.pipeline import MeteorDetectionPipeline
        from meteor_core.schema import PipelineConfig, DetectionParams

        config = PipelineConfig(
            target_folder=self._temp_path("raw"),
            output_folder=self._temp_path("fallback_out"),
            debug_folder=self._temp_path("fallback_debug"),
            params=DetectionParams(),
        )
        pipeline = MeteorDetectionPipeline(config)
        self.assertIsInstance(pipeline.output_handler, FileOutputHandler)
        self.assertEqual(
            pipeline.output_handler.config.output_folder,
            self._temp_path("fallback_out"),
        )

    def test_pipeline_with_explicit_handler(self):
        """Pipeline accepts explicit output_handler parameter."""
        from meteor_core.pipeline import MeteorDetectionPipeline
        from meteor_core.schema import PipelineConfig, DetectionParams

        explicit_handler = FileOutputHandler(
            FileOutputConfig(
                output_folder=self._temp_path("explicit"),
                debug_folder=self._temp_path("explicit_debug"),
            )
        )
        config = PipelineConfig(
            target_folder=self._temp_path("raw"),
            output_folder=self._temp_path("out"),
            debug_folder=self._temp_path("debug"),
            params=DetectionParams(),
        )
        pipeline = MeteorDetectionPipeline(config, output_handler=explicit_handler)
        self.assertIs(pipeline.output_handler, explicit_handler)


if __name__ == "__main__":
    unittest.main()
