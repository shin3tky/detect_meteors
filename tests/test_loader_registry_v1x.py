#!/usr/bin/env python
#
# Detect Meteors CLI - Loader Registry Tests
# © 2025 Shinichi Morita (shin3tky)
#

"""
Unit tests for LoaderRegistry.
"""

import unittest
import warnings
from typing import Any

from meteor_core.inputs.registry import LoaderRegistry
from meteor_core.inputs.base import BaseInputLoader
from meteor_core.schema import DEFAULT_LOADER_NAME


class TestLoaderRegistryDiscovery(unittest.TestCase):
    """Tests for LoaderRegistry discovery functionality."""

    def setUp(self):
        """Reset registry before each test."""
        LoaderRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        LoaderRegistry._reset()

    def test_discover_returns_dict(self):
        """discover() returns a dictionary."""
        loaders = LoaderRegistry.discover()
        self.assertIsInstance(loaders, dict)

    def test_discover_includes_builtin_raw(self):
        """Built-in raw loader is always discovered."""
        loaders = LoaderRegistry.discover()
        self.assertIn("raw", loaders)

    def test_discover_caches_result(self):
        """discover() caches results for subsequent calls."""
        loaders1 = LoaderRegistry.discover()
        loaders2 = LoaderRegistry.discover()
        self.assertIs(loaders1, loaders2)

    def test_discover_force_refreshes_cache(self):
        """discover(force=True) refreshes the cache."""
        loaders1 = LoaderRegistry.discover()
        loaders2 = LoaderRegistry.discover(force=True)
        # Should be equal content but different object
        self.assertEqual(loaders1.keys(), loaders2.keys())


class TestLoaderRegistryGet(unittest.TestCase):
    """Tests for LoaderRegistry.get() method."""

    def setUp(self):
        """Reset registry before each test."""
        LoaderRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        LoaderRegistry._reset()

    def test_get_existing_loader(self):
        """get() returns loader class for existing name."""
        loader_cls = LoaderRegistry.get("raw")
        self.assertEqual(loader_cls.plugin_name, "raw")

    def test_get_nonexistent_raises_keyerror(self):
        """get() raises KeyError for unknown loader name."""
        with self.assertRaises(KeyError) as ctx:
            LoaderRegistry.get("nonexistent_loader")

        # Error message should include available loaders
        error_msg = str(ctx.exception)
        self.assertIn("nonexistent_loader", error_msg)
        self.assertIn("Available", error_msg)

    def test_get_custom_takes_priority(self):
        """Runtime-registered loader takes priority over discovered."""

        class CustomRawLoader(BaseInputLoader):
            plugin_name = "raw"

            def __init__(self, config=None):
                self.config = config

            def load(self, filepath: str) -> Any:
                return "custom_result"

        LoaderRegistry.register(CustomRawLoader)
        loader_cls = LoaderRegistry.get("raw")
        self.assertIs(loader_cls, CustomRawLoader)


class TestLoaderRegistryRegister(unittest.TestCase):
    """Tests for LoaderRegistry.register() method."""

    def setUp(self):
        """Reset registry before each test."""
        LoaderRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        LoaderRegistry._reset()

    def test_register_valid_loader(self):
        """register() adds valid loader to registry."""

        class MockLoader(BaseInputLoader):
            plugin_name = "mock"

            def __init__(self, config=None):
                self.config = config

            def load(self, filepath: str) -> Any:
                return None

        LoaderRegistry.register(MockLoader)
        loader_cls = LoaderRegistry.get("mock")
        self.assertIs(loader_cls, MockLoader)

    def test_register_invalid_class_raises(self):
        """register() raises ValueError for invalid class."""

        class NotALoader:
            plugin_name = "invalid"

        with self.assertRaises(ValueError) as ctx:
            LoaderRegistry.register(NotALoader)  # type: ignore

        self.assertIn("Invalid loader class", str(ctx.exception))

    def test_register_empty_plugin_name_raises(self):
        """register() raises ValueError for empty plugin_name."""

        class EmptyNameLoader(BaseInputLoader):
            plugin_name = ""

            def __init__(self, config=None):
                self.config = config

            def load(self, filepath: str) -> Any:
                return None

        with self.assertRaises(ValueError) as ctx:
            LoaderRegistry.register(EmptyNameLoader)

        self.assertIn("non-empty plugin_name", str(ctx.exception))

    def test_register_overwrite_warns(self):
        """register() warns when overwriting existing loader."""

        class MockLoader1(BaseInputLoader):
            plugin_name = "mock"

            def __init__(self, config=None):
                self.config = config

            def load(self, filepath: str) -> Any:
                return "v1"

        class MockLoader2(BaseInputLoader):
            plugin_name = "mock"

            def __init__(self, config=None):
                self.config = config

            def load(self, filepath: str) -> Any:
                return "v2"

        LoaderRegistry.register(MockLoader1)

        with self.assertWarns(UserWarning) as ctx:
            LoaderRegistry.register(MockLoader2)

        self.assertIn("Overwriting", str(ctx.warning))


class TestLoaderRegistryUnregister(unittest.TestCase):
    """Tests for LoaderRegistry.unregister() method."""

    def setUp(self):
        """Reset registry before each test."""
        LoaderRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        LoaderRegistry._reset()

    def test_unregister_custom_loader(self):
        """unregister() removes runtime-registered loader."""

        class MockLoader(BaseInputLoader):
            plugin_name = "mock"

            def __init__(self, config=None):
                self.config = config

            def load(self, filepath: str) -> Any:
                return None

        LoaderRegistry.register(MockLoader)
        self.assertTrue(LoaderRegistry.unregister("mock"))

        with self.assertRaises(KeyError):
            LoaderRegistry.get("mock")

    def test_unregister_nonexistent_returns_false(self):
        """unregister() returns False for non-registered loader."""
        result = LoaderRegistry.unregister("never_registered")
        self.assertFalse(result)

    def test_unregister_discovered_returns_false(self):
        """unregister() cannot remove discovered loaders."""
        # "raw" is discovered, not runtime-registered
        result = LoaderRegistry.unregister("raw")
        self.assertFalse(result)

        # "raw" should still be available
        loader_cls = LoaderRegistry.get("raw")
        self.assertEqual(loader_cls.plugin_name, "raw")


class TestLoaderRegistryListAvailable(unittest.TestCase):
    """Tests for LoaderRegistry.list_available() method."""

    def setUp(self):
        """Reset registry before each test."""
        LoaderRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        LoaderRegistry._reset()

    def test_list_available_includes_discovered(self):
        """list_available() includes discovered loaders."""
        available = LoaderRegistry.list_available()
        self.assertIn("raw", available)

    def test_list_available_includes_custom(self):
        """list_available() includes runtime-registered loaders."""

        class MockLoader(BaseInputLoader):
            plugin_name = "mock"

            def __init__(self, config=None):
                self.config = config

            def load(self, filepath: str) -> Any:
                return None

        LoaderRegistry.register(MockLoader)
        available = LoaderRegistry.list_available()

        self.assertIn("raw", available)
        self.assertIn("mock", available)

    def test_list_available_is_sorted(self):
        """list_available() returns sorted list."""
        available = LoaderRegistry.list_available()
        self.assertEqual(available, sorted(available))


class TestLoaderRegistryCreate(unittest.TestCase):
    """Tests for LoaderRegistry.create() method."""

    def setUp(self):
        """Reset registry before each test."""
        LoaderRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        LoaderRegistry._reset()

    def test_create_with_default_config(self):
        """create() with None config uses default."""
        loader = LoaderRegistry.create("raw")
        self.assertEqual(loader.plugin_name, "raw")
        # Default config should have binning=2
        self.assertEqual(loader.config.binning, 2)

    def test_create_with_dict_config(self):
        """create() coerces dict to ConfigType."""
        loader = LoaderRegistry.create("raw", {"binning": 2, "normalize": True})
        self.assertTrue(loader.config.normalize)

    def test_create_with_config_instance(self):
        """create() accepts ConfigType instance directly."""
        from meteor_core.inputs.raw import RawLoaderConfig

        config = RawLoaderConfig(normalize=True)
        loader = LoaderRegistry.create("raw", config)
        self.assertTrue(loader.config.normalize)

    def test_create_nonexistent_raises(self):
        """create() raises KeyError for unknown loader."""
        with self.assertRaises(KeyError):
            LoaderRegistry.create("nonexistent_loader")


class TestLoaderRegistryCreateDefault(unittest.TestCase):
    """Tests for LoaderRegistry.create_default() method."""

    def setUp(self):
        """Reset registry before each test."""
        LoaderRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        LoaderRegistry._reset()

    def test_create_default_returns_raw_loader(self):
        """create_default() returns raw loader with default config."""
        loader = LoaderRegistry.create_default()
        self.assertEqual(loader.plugin_name, DEFAULT_LOADER_NAME)

    def test_create_default_uses_default_config(self):
        """create_default() uses default configuration."""
        loader = LoaderRegistry.create_default()
        self.assertEqual(loader.config.binning, 2)
        self.assertFalse(loader.config.normalize)


class TestLoaderRegistryCoerceConfig(unittest.TestCase):
    """Tests for LoaderRegistry._coerce_config() method."""

    def test_coerce_none_to_default(self):
        """_coerce_config() creates default instance from None."""
        from meteor_core.inputs.raw import RawImageLoader

        config = LoaderRegistry._coerce_config(RawImageLoader, None)
        self.assertIsNotNone(config)
        self.assertEqual(config.binning, 2)

    def test_coerce_dict_to_dataclass(self):
        """_coerce_config() converts dict to dataclass."""
        from meteor_core.inputs.raw import RawImageLoader

        config = LoaderRegistry._coerce_config(
            RawImageLoader, {"binning": 2, "normalize": True}
        )
        self.assertTrue(config.normalize)

    def test_coerce_passthrough_correct_type(self):
        """_coerce_config() passes through correct ConfigType."""
        from meteor_core.inputs.raw import RawImageLoader, RawLoaderConfig

        original = RawLoaderConfig(normalize=True)
        config = LoaderRegistry._coerce_config(RawImageLoader, original)
        self.assertIs(config, original)


class TestLoaderRegistryReset(unittest.TestCase):
    """Tests for LoaderRegistry._reset() method."""

    def test_reset_clears_discovered_cache(self):
        """_reset() clears discovered loader cache."""
        # Trigger discovery
        LoaderRegistry.discover()
        self.assertIsNotNone(LoaderRegistry._discovered)

        LoaderRegistry._reset()
        self.assertIsNone(LoaderRegistry._discovered)

    def test_reset_clears_custom_loaders(self):
        """_reset() clears runtime-registered loaders."""

        class MockLoader(BaseInputLoader):
            plugin_name = "mock"

            def __init__(self, config=None):
                self.config = config

            def load(self, filepath: str) -> Any:
                return None

        LoaderRegistry.register(MockLoader)
        self.assertEqual(len(LoaderRegistry._custom), 1)

        LoaderRegistry._reset()
        self.assertEqual(len(LoaderRegistry._custom), 0)


class TestDiscoverInputLoadersDeprecation(unittest.TestCase):
    """Tests for deprecated discover_input_loaders function."""

    def setUp(self):
        """Reset registry before each test."""
        LoaderRegistry._reset()

    def tearDown(self):
        """Reset registry after each test."""
        LoaderRegistry._reset()

    def test_discover_input_loaders_warns(self):
        """discover_input_loaders() emits DeprecationWarning."""
        from meteor_core.inputs.discovery import discover_input_loaders

        with self.assertWarns(DeprecationWarning) as ctx:
            discover_input_loaders()

        self.assertIn("deprecated", str(ctx.warning))
        self.assertIn("LoaderRegistry", str(ctx.warning))

    def test_discover_input_loaders_returns_same_result(self):
        """discover_input_loaders() returns same loaders as registry."""
        from meteor_core.inputs.discovery import discover_input_loaders

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            deprecated_loaders = discover_input_loaders()

        registry_loaders = LoaderRegistry.discover()

        self.assertEqual(set(deprecated_loaders.keys()), set(registry_loaders.keys()))


if __name__ == "__main__":
    unittest.main()
