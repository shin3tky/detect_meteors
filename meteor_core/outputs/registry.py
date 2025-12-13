#!/usr/bin/env python
#
# Detect Meteors CLI - Output Handler Registry
# © 2025 Shinichi Morita (shin3tky)
#

"""
Registry for output handler plugins with discovery, registration, and instantiation.

This module provides a centralized registry for managing output handlers,
supporting both automatic discovery (via entry points and plugin directory)
and runtime registration for testing and dynamic plugins.
"""

from __future__ import annotations

import warnings
from dataclasses import is_dataclass
from typing import Any, Dict, List, Optional, Type, Union

from ..schema import DEFAULT_OUTPUT_HANDLER_NAME
from .base import BaseOutputHandler, _is_valid_output_handler


class OutputHandlerRegistry:
    """Output handler registry with discovery, registration, and instantiation.

    This registry provides:
    - Lazy discovery of handlers from entry points and plugin directory
    - Runtime registration/unregistration for testing and dynamic plugins
    - Config coercion (dict -> dataclass/Pydantic model)
    - Factory method for creating handler instances

    The registry uses class-level state for caching discovered handlers
    and storing runtime-registered handlers.

    Example:
        >>> # Get a handler class
        >>> handler_cls = OutputHandlerRegistry.get("file")
        >>> print(handler_cls.plugin_name)
        'file'

        >>> # Create an instance with config
        >>> handler = OutputHandlerRegistry.create("file", {
        ...     "output_folder": "./candidates",
        ...     "debug_folder": "./debug",
        ... })
        >>> saved = handler.save_candidate("/path/to/source.CR2", "source.CR2")

        >>> # Register a custom handler for testing
        >>> OutputHandlerRegistry.register(MyCustomHandler)
        >>> OutputHandlerRegistry.unregister("my_custom")
    """

    # Discovered handlers (from entry points + plugin dir), lazily initialized
    _discovered: Optional[Dict[str, Type[BaseOutputHandler]]] = None

    # Runtime-registered handlers (for testing/dynamic plugins)
    _custom: Dict[str, Type[BaseOutputHandler]] = {}

    # ========================================
    # Discovery & Registration
    # ========================================

    @classmethod
    def discover(cls, force: bool = False) -> Dict[str, Type[BaseOutputHandler]]:
        """Discover available handlers (cached, lazy).

        Discovers handlers from:
        1. Built-in handlers (e.g., FileOutputHandler)
        2. Entry points (detect_meteors.output group)
        3. Plugin directory (~/.detect_meteors/output_plugins)

        Args:
            force: If True, re-discover even if already cached.

        Returns:
            Dict mapping plugin_name to handler class.
            Does not include runtime-registered handlers; use get() or
            list_available() for the complete list.
        """
        if cls._discovered is None or force:
            # Import here to avoid circular dependency
            from .discovery import _discover_handlers_internal

            cls._discovered = _discover_handlers_internal()
        return cls._discovered

    @classmethod
    def get(cls, name: str) -> Type[BaseOutputHandler]:
        """Get handler class by name.

        Lookup priority:
        1. Runtime-registered handlers (_custom)
        2. Discovered handlers (_discovered)

        Args:
            name: Handler plugin_name (case-insensitive).

        Returns:
            Handler class.

        Raises:
            KeyError: If handler not found.
        """
        # Normalize name to lowercase for case-insensitive lookup
        name_lower = name.lower()

        # 1. Custom (runtime-registered) takes priority
        if name_lower in cls._custom:
            return cls._custom[name_lower]

        # 2. Discovered handlers
        discovered = cls.discover()
        if name_lower in discovered:
            return discovered[name_lower]

        # 3. Not found - provide helpful error message
        available = cls.list_available()
        available_str = ", ".join(sorted(available)) if available else "none"
        raise KeyError(f"Unknown output handler '{name}'. Available: {available_str}")

    @classmethod
    def register(cls, handler_cls: Type[BaseOutputHandler]) -> None:
        """Register a handler class at runtime.

        Runtime-registered handlers take priority over discovered handlers
        with the same name. This is useful for testing and dynamic plugins.

        Args:
            handler_cls: Handler class with plugin_name attribute.

        Raises:
            ValueError: If plugin_name is empty or class is invalid.

        Example:
            >>> class MockHandler(BaseOutputHandler):
            ...     plugin_name = "mock"
            ...     def save_candidate(self, ...): return True
            ...     def save_debug_image(self, ...): return "/path"
            >>> OutputHandlerRegistry.register(MockHandler)
        """
        if not _is_valid_output_handler(handler_cls):
            raise ValueError(
                f"Invalid handler class: {handler_cls}. "
                "Must inherit from BaseOutputHandler and have non-empty plugin_name."
            )

        name = handler_cls.plugin_name
        if not name:
            raise ValueError("Handler must have non-empty plugin_name")

        # Normalize to lowercase
        name_lower = name.lower()

        # Warn on overwrite (but allow it for testing purposes)
        if name_lower in cls._custom:
            warnings.warn(
                f"Overwriting existing runtime-registered handler '{name}'",
                stacklevel=2,
            )

        cls._custom[name_lower] = handler_cls

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a handler by name.

        Only removes from runtime-registered handlers. Discovered handlers
        cannot be unregistered (they will be re-discovered).

        Args:
            name: Handler plugin_name to remove (case-insensitive).

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
        """List all available handler names.

        Returns:
            Sorted list of available handler plugin_names,
            including both discovered and runtime-registered handlers.
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
    ) -> BaseOutputHandler:
        """Create a handler instance with config coercion.

        Args:
            name: Handler plugin_name.
            config: Configuration for the handler. Can be:
                - None: Uses default config (ConfigType() if available)
                - Dict: Coerced to ConfigType (dataclass or Pydantic model)
                - ConfigType instance: Used as-is

        Returns:
            Configured handler instance.

        Raises:
            KeyError: If handler not found.
            TypeError: If config type is incompatible.
            ValueError: If config validation fails.

        Example:
            >>> handler = OutputHandlerRegistry.create("file", {
            ...     "output_folder": "./candidates",
            ...     "debug_folder": "./debug",
            ... })
            >>> handler = OutputHandlerRegistry.create("file", config_instance)
        """
        handler_cls = cls.get(name)
        coerced_config = cls._coerce_config(handler_cls, config)
        return handler_cls(coerced_config)

    @classmethod
    def create_default(
        cls,
        output_folder: str,
        debug_folder: str,
        output_overwrite: bool = False,
    ) -> BaseOutputHandler:
        """Create the default output handler with file-based configuration.

        Args:
            output_folder: Directory for candidate files.
            debug_folder: Directory for debug images.
            output_overwrite: Whether to overwrite existing files.

        Returns:
            Default handler instance (FileOutputHandler).
        """
        from .file_handler import FileOutputConfig

        config = FileOutputConfig(
            output_folder=output_folder,
            debug_folder=debug_folder,
            output_overwrite=output_overwrite,
        )
        return cls.create(DEFAULT_OUTPUT_HANDLER_NAME, config)

    # ========================================
    # Internal Methods
    # ========================================

    @classmethod
    def _coerce_config(
        cls,
        handler_cls: Type[BaseOutputHandler],
        config: Optional[Union[Dict[str, Any], Any]],
    ) -> Any:
        """Coerce config to the handler's expected ConfigType.

        Handles:
        - None -> Default ConfigType instance (if available)
        - Dict -> Dataclass or Pydantic model
        - ConfigType instance -> Pass through

        Args:
            handler_cls: Handler class to get ConfigType from.
            config: Configuration to coerce.

        Returns:
            Coerced configuration suitable for handler initialization.

        Raises:
            TypeError: If config cannot be coerced to the expected type.
            ValueError: If config validation fails (e.g., invalid field values).
        """
        config_type = getattr(handler_cls, "ConfigType", None)

        # Case 1: No config provided
        if config is None:
            if config_type is not None:
                try:
                    return config_type()  # Default instance
                except TypeError as e:
                    raise TypeError(
                        f"Failed to create default config for handler "
                        f"'{handler_cls.plugin_name}': {config_type.__name__} "
                        f"requires arguments. Provide a config dict or instance."
                    ) from e
                except Exception as e:
                    raise ValueError(
                        f"Failed to create default config for handler "
                        f"'{handler_cls.plugin_name}': {e}"
                    ) from e
            return None

        # Case 2: No ConfigType defined on handler
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
                    f"Invalid config for handler '{handler_cls.plugin_name}': {e}"
                ) from e

        # Case 5: Dict -> Pydantic v2 (model_validate)
        if hasattr(config_type, "model_validate") and isinstance(config, dict):
            try:
                return config_type.model_validate(config)
            except Exception as e:
                raise ValueError(
                    f"Config validation failed for handler "
                    f"'{handler_cls.plugin_name}': {e}"
                ) from e

        # Case 6: Dict -> Pydantic v1 (parse_obj)
        if hasattr(config_type, "parse_obj") and isinstance(config, dict):
            try:
                return config_type.parse_obj(config)
            except Exception as e:
                raise ValueError(
                    f"Config validation failed for handler "
                    f"'{handler_cls.plugin_name}': {e}"
                ) from e

        # Case 7: Incompatible type
        if isinstance(config, dict):
            raise TypeError(
                f"Cannot coerce dict to {config_type.__name__} for handler "
                f"'{handler_cls.plugin_name}': ConfigType is neither a "
                f"dataclass nor a Pydantic model."
            )

        # Fallback: return as-is (may fail later in handler __init__)
        return config

    @classmethod
    def _reset(cls) -> None:
        """Reset registry state.

        This method is intended for testing only. It clears both
        the discovered cache and runtime-registered handlers.
        """
        cls._discovered = None
        cls._custom = {}


__all__ = ["OutputHandlerRegistry"]
