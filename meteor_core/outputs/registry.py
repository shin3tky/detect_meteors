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

from typing import Any, Dict, Optional, Type, Union

from ..plugin_registry import _PLUGIN_KIND_OUTPUT
from ..plugin_registry_base import PluginRegistryBase
from ..schema import DEFAULT_OUTPUT_HANDLER_NAME
from .base import BaseOutputHandler, _is_valid_output_handler


class OutputHandlerRegistry(PluginRegistryBase[BaseOutputHandler]):
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

    _plugin_kind = _PLUGIN_KIND_OUTPUT

    @classmethod
    def _discover_internal(cls) -> Dict[str, Type[BaseOutputHandler]]:
        # Import here to avoid circular dependency
        # Keep plugin directory path in sync with meteor_core.outputs.discovery.PLUGIN_DIR
        from .discovery import _discover_output_handlers_internal

        return _discover_output_handlers_internal()

    @classmethod
    def _is_valid_plugin(cls, handler_cls: Type[BaseOutputHandler]) -> bool:
        return _is_valid_output_handler(handler_cls)

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
            >>> handler = OutputHandlerRegistry.create("file", {"output_folder": "./candidates"})
            >>> handler = OutputHandlerRegistry.create("file")  # default config
        """
        handler_cls = cls.get(name)
        coerced_config = cls._coerce_config(handler_cls, config)
        return handler_cls(coerced_config)

    @classmethod
    def create_default(
        cls,
        *,
        output_folder: Optional[str] = None,
        debug_folder: Optional[str] = None,
        output_overwrite: Optional[bool] = None,
    ) -> BaseOutputHandler:
        """Create the default handler using its ConfigType defaults.

        The default configuration is built by calling ``ConfigType()`` and then
        overriding any provided parameters. Folders can be customized without
        losing other defaults defined on the ConfigType.
        """

        handler_cls = cls.get(DEFAULT_OUTPUT_HANDLER_NAME)
        config_type = getattr(handler_cls, "ConfigType", None)
        config = config_type() if config_type else None
        if config is None:
            raise TypeError(
                "Default output handler does not define ConfigType; cannot create default."
            )

        if output_folder is not None:
            config.output_folder = output_folder
        if debug_folder is not None:
            config.debug_folder = debug_folder
        if output_overwrite is not None:
            config.output_overwrite = output_overwrite

        return handler_cls(config)


__all__ = ["OutputHandlerRegistry"]
