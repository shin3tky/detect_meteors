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

Developer guidance
------------------
``create`` should be used when a caller supplies explicit configuration and
wants registry-level coercion (e.g., converting a dict into ``ConfigType``).
``create_default`` exists to build the default loader from its zero-argument
``ConfigType`` constructor; if a loader does not expose such defaults, the
method raises to avoid silently constructing an incomplete instance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, Union

from ..plugin_registry import _PLUGIN_KIND_INPUT
from ..plugin_registry_base import PluginRegistryBase
from ..schema import DEFAULT_LOADER_NAME
from .base import BaseInputLoader, _is_valid_input_loader

# Module-level logger for registry operations
logger = logging.getLogger(__name__)


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

    _plugin_kind = _PLUGIN_KIND_INPUT

    @classmethod
    def _discover_internal(cls) -> Dict[str, Type[BaseInputLoader]]:
        # Import here to avoid circular dependency
        # Keep plugin directory path in sync with meteor_core.inputs.discovery.PLUGIN_DIR
        from .discovery import _discover_handlers_internal

        return _discover_handlers_internal()

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

        Prefer this method when the caller needs to control configuration
        values explicitly; the registry will coerce dictionaries into
        ``ConfigType`` instances where available.

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
        logger.debug("Creating input loader '%s' with config %r", name, config)
        loader_cls = cls.get(name)
        coerced_config = cls._coerce_config(loader_cls, config)
        instance = loader_cls(coerced_config)
        logger.debug(
            "Created input loader '%s' (%s) with config type %s",
            name,
            loader_cls.__name__,
            type(coerced_config).__name__ if coerced_config is not None else None,
        )
        return instance

    @classmethod
    def create_default(
        cls, config: Optional[Union[Dict[str, Any], Any]] = None
    ) -> BaseInputLoader:
        """Create the default loader with default config.

        The default loader must expose a zero-argument ``ConfigType`` that
        yields a complete configuration. Callers can optionally supply a config
        object or mapping, which will be coerced using the same rules as
        :meth:`create`.
        """
        # Input loaders do not expose registry-level overrides; their defaults are
        # defined solely by ConfigType, whereas output handlers also allow path
        # overrides for CLI compatibility.
        logger.debug(
            "Creating default input loader '%s' with override config %r",
            DEFAULT_LOADER_NAME,
            config,
        )
        loader_cls = cls.get(DEFAULT_LOADER_NAME)
        coerced_config = cls._coerce_config(loader_cls, config)
        if coerced_config is None:
            raise TypeError(
                "Default loader does not define ConfigType; cannot create default."
            )

        instance = loader_cls(coerced_config)
        logger.debug(
            "Created default input loader '%s' (%s) with config type %s",
            DEFAULT_LOADER_NAME,
            loader_cls.__name__,
            type(coerced_config).__name__,
        )
        return instance


__all__ = ["LoaderRegistry"]
