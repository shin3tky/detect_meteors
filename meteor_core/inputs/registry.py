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

import warnings
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Type, Union

from ..schema import DEFAULT_LOADER_NAME
from .base import BaseInputLoader, _is_valid_input_loader


class LoaderRegistry:
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

    # Discovered loaders (from entry points + plugin dir), lazily initialized
    _discovered: Optional[Dict[str, Type[BaseInputLoader]]] = None

    # Runtime-registered loaders (for testing/dynamic plugins)
    _custom: Dict[str, Type[BaseInputLoader]] = {}

    # ========================================
    # Discovery & Registration
    # ========================================

    @classmethod
    def discover(cls, force: bool = False) -> Dict[str, Type[BaseInputLoader]]:
        """Discover available loaders (cached, lazy).

        Discovers loaders from:
        1. Built-in loaders (e.g., RawImageLoader)
        2. Entry points (detect_meteors.input group)
        3. Plugin directory (~/.detect_meteors/input_plugins)

        Args:
            force: If True, re-discover even if already cached.

        Returns:
            Dict mapping plugin_name to loader class.
            Does not include runtime-registered loaders; use get() or
            list_available() for the complete list.
        """
        if cls._discovered is None or force:
            # Import here to avoid circular dependency
            # Keep plugin directory path in sync with meteor_core.inputs.discovery.PLUGIN_DIR
            from .discovery import _discover_loaders_internal

            cls._discovered = _discover_loaders_internal()
        return cls._discovered

    @classmethod
    def get(cls, name: str) -> Type[BaseInputLoader]:
        """Get loader class by name.

        Lookup priority:
        1. Runtime-registered loaders (_custom)
        2. Discovered loaders (_discovered)

        Args:
            name: Loader plugin_name (case-insensitive).

        Returns:
            Loader class.

        Raises:
            KeyError: If loader not found.
        """
        # Normalize name to lowercase for case-insensitive lookup
        name_lower = name.lower()

        # 1. Custom (runtime-registered) takes priority
        if name_lower in cls._custom:
            return cls._custom[name_lower]

        # 2. Discovered loaders
        discovered = cls.discover()
        if name_lower in discovered:
            return discovered[name_lower]

        # 3. Not found - provide helpful error message
        available = cls.list_available()
        available_str = ", ".join(sorted(available)) if available else "none"
        raise KeyError(f"Unknown loader '{name}'. Available: {available_str}")

    @classmethod
    def register(cls, loader_cls: Type[BaseInputLoader]) -> None:
        """Register a loader class at runtime.

        Runtime-registered loaders take priority over discovered loaders
        with the same name. This is useful for testing and dynamic plugins.

        Args:
            loader_cls: Loader class with plugin_name attribute.

        Raises:
            ValueError: If plugin_name is empty or class is invalid.

        Example:
            >>> class MockLoader(BaseInputLoader):
            ...     plugin_name = "mock"
            ...     def load(self, filepath): return None
            >>> LoaderRegistry.register(MockLoader)
        """
        if not _is_valid_input_loader(loader_cls):
            raise ValueError(
                f"Invalid loader class: {loader_cls}. "
                "Must inherit from BaseInputLoader and have non-empty plugin_name."
            )

        name = loader_cls.plugin_name
        if not name:
            raise ValueError("Loader must have non-empty plugin_name")

        # Normalize to lowercase
        name_lower = name.lower()

        # Warn on overwrite (but allow it for testing purposes)
        if name_lower in cls._custom:
            warnings.warn(
                f"Overwriting existing runtime-registered loader '{name}'",
                stacklevel=2,
            )

        cls._custom[name_lower] = loader_cls

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a loader by name.

        Only removes from runtime-registered loaders. Discovered loaders
        cannot be unregistered (they will be re-discovered).

        Args:
            name: Loader plugin_name to remove (case-insensitive).

        Returns:
            True if removed, False if not found in runtime registry.
        """
        name_lower = name.lower()
        if name_lower in cls._custom:
            del cls._custom[name_lower]
            return True
        return False

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available loader names.

        Returns:
            Sorted list of available loader plugin_names,
            including both discovered and runtime-registered loaders.
        """
        discovered = cls.discover()
        all_names = set(discovered.keys()) | set(cls._custom.keys())
        return sorted(all_names)

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

        Returns:
            Default loader instance (currently "raw" loader).
        """
        return cls.create(DEFAULT_LOADER_NAME)

    # ========================================
    # Internal Methods
    # ========================================

    @classmethod
    def _coerce_config(
        cls,
        loader_cls: Type[BaseInputLoader],
        config: Optional[Union[Dict[str, Any], Any]],
    ) -> Any:
        """Coerce config to the loader's expected ConfigType.

        Handles:
        - None -> Default ConfigType instance (if available)
        - Dict -> Dataclass or Pydantic model
        - ConfigType instance -> Pass through

        Args:
            loader_cls: Loader class to get ConfigType from.
            config: Configuration to coerce.

        Returns:
            Coerced configuration suitable for loader initialization.

        Raises:
            TypeError: If config cannot be coerced to the expected type.
            ValueError: If config validation fails (e.g., invalid field values).
        """
        config_type = getattr(loader_cls, "ConfigType", None)

        # Case 1: No config provided
        if config is None:
            if config_type is not None:
                try:
                    return config_type()  # Default instance
                except TypeError as e:
                    raise TypeError(
                        f"Failed to create default config for loader "
                        f"'{loader_cls.plugin_name}': {config_type.__name__} "
                        f"requires arguments. Provide a config dict or instance."
                    ) from e
                except Exception as e:
                    raise ValueError(
                        f"Failed to create default config for loader "
                        f"'{loader_cls.plugin_name}': {e}"
                    ) from e
            return None

        # Case 2: No ConfigType defined on loader
        if config_type is None:
            return config

        # Case 3: Already correct type
        if isinstance(config, config_type):
            return config

        # Case 4: Dict -> Dataclass
        if is_dataclass(config_type) and isinstance(config, dict):
            try:
                return config_type(**config)
            except TypeError as e:
                raise TypeError(
                    f"Invalid config for loader '{loader_cls.plugin_name}': {e}"
                ) from e

        # Case 5: Dict -> Pydantic v2 (model_validate)
        if hasattr(config_type, "model_validate") and isinstance(config, dict):
            try:
                return config_type.model_validate(config)
            except Exception as e:
                raise ValueError(
                    f"Config validation failed for loader "
                    f"'{loader_cls.plugin_name}': {e}"
                ) from e

        # Case 6: Dict -> Pydantic v1 (parse_obj)
        if hasattr(config_type, "parse_obj") and isinstance(config, dict):
            try:
                return config_type.parse_obj(config)
            except Exception as e:
                raise ValueError(
                    f"Config validation failed for loader "
                    f"'{loader_cls.plugin_name}': {e}"
                ) from e

        # Case 7: Incompatible type
        if isinstance(config, dict):
            raise TypeError(
                f"Cannot coerce dict to {config_type.__name__} for loader "
                f"'{loader_cls.plugin_name}': ConfigType is neither a "
                f"dataclass nor a Pydantic model."
            )

        # Fallback: return as-is (may fail later in loader __init__)
        return config

    @classmethod
    def _reset(cls) -> None:
        """Reset registry state.

        This method is intended for testing only. It clears both
        the discovered cache and runtime-registered loaders.
        """
        cls._discovered = None
        cls._custom = {}


__all__ = ["LoaderRegistry"]
