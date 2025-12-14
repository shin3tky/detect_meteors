#!/usr/bin/env python
#
# Detect Meteors CLI - Input Loader Registry
# © 2025 Shinichi Morita (shin3tky)
#

"""
Registry for input loader plugins with discovery, registration, and instantiation.

This module provides a centralized registry for managing input loaders,
supporting both automatic discovery (via entry points and plugin directory)
and runtime registration for testing and dynamic plugins.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

from ..plugin_registry_base import PluginRegistryBase
from ..schema import DEFAULT_LOADER_NAME
from .base import BaseInputLoader, _is_valid_input_loader


class LoaderRegistry(PluginRegistryBase[BaseInputLoader]):
    """Input loader registry with discovery, registration, and instantiation.

    This registry provides:
    - Lazy discovery of loaders from entry points and plugin directory
    - Runtime registration/unregistration for testing and dynamic plugins
    - Config coercion (dict -> dataclass/Pydantic model)
    - Factory method for creating loader instances

    The registry uses class-level state for caching discovered loaders
    and storing runtime-registered loaders.

    Example:
        >>> # Get a loader class
        >>> loader_cls = LoaderRegistry.get("raw")
        >>> print(loader_cls.plugin_name)
        'raw'

        >>> # Create an instance with config
        >>> loader = LoaderRegistry.create("raw", {"normalize": True})
        >>> image = loader.load("photo.CR2")

        >>> # Register a custom loader for testing
        >>> LoaderRegistry.register(MyCustomLoader)
        >>> LoaderRegistry.unregister("my_custom")
    """

    _plugin_kind = "loader"

    @classmethod
    def _discover_internal(cls) -> Dict[str, Type[BaseInputLoader]]:
        # Import here to avoid circular dependency
        # Keep plugin directory path in sync with meteor_core.inputs.discovery.PLUGIN_DIR
        from .discovery import _discover_loaders_internal

        return _discover_loaders_internal()

    @classmethod
    def _is_valid_plugin(cls, loader_cls: Type[BaseInputLoader]) -> bool:
        return _is_valid_input_loader(loader_cls)

    # ========================================
    # Instantiation
    # ========================================

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> BaseInputLoader:
        """Create a loader instance with config coercion.

        Args:
            name: Loader plugin_name.
            config: Configuration for the loader. Can be:
                - None: Uses default config (ConfigType() if available)
                - Dict: Coerced to ConfigType (dataclass or Pydantic model)
                - ConfigType instance: Used as-is

        Returns:
            Configured loader instance.

        Raises:
            KeyError: If loader not found.
            TypeError: If config type is incompatible.
            ValueError: If config validation fails.

        Example:
            >>> loader = LoaderRegistry.create("raw", {"normalize": True})
            >>> loader = LoaderRegistry.create("raw")  # default config
        """
        loader_cls = cls.get(name)
        coerced_config = cls._coerce_config(loader_cls, config)
        return loader_cls(coerced_config)

    @classmethod
    def create_default(cls) -> BaseInputLoader:
        """Create the default loader with default config.

        This constructs the default loader's ConfigType instance with its own
        defaults before instantiating the loader.
        """

        loader_cls = cls.get(DEFAULT_LOADER_NAME)
        config_type = getattr(loader_cls, "ConfigType", None)
        config = config_type() if config_type else None
        return loader_cls(config)


__all__ = ["LoaderRegistry"]
