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

Developer guidance
------------------
Call ``create`` when overriding configuration explicitly so that the registry
can coerce dictionaries into the handler's ``ConfigType``. ``create_default``
is reserved for the built-in default handler and assumes its ``ConfigType``
constructor yields a fully populated configuration; it raises if no such
defaults exist to avoid silently skipping required values.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, Union

from ..plugin_registry import _PLUGIN_KIND_OUTPUT
from ..plugin_registry_base import PluginRegistryBase
from ..schema import DEFAULT_OUTPUT_HANDLER_NAME
from .base import BaseOutputHandler, _is_valid_output_handler

# Module-level logger for registry operations
logger = logging.getLogger(__name__)


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
        >>> result = handler.save_candidate("/path/to/source.CR2", "source.CR2")

        >>> # Register a custom handler for testing
        >>> OutputHandlerRegistry.register(MyCustomHandler)
        >>> OutputHandlerRegistry.unregister("my_custom")
    """

    _plugin_kind = _PLUGIN_KIND_OUTPUT

    @classmethod
    def _discover_internal(cls) -> Dict[str, Type[BaseOutputHandler]]:
        # Import here to avoid circular dependency
        # Keep plugin directory path in sync with meteor_core.outputs.discovery.PLUGIN_DIR
        from .discovery import _discover_handlers_internal

        return _discover_handlers_internal()

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

        Use this when you need to supply custom settings. The registry will
        coerce dictionaries into ``ConfigType`` instances when supported by the
        handler, ensuring type-appropriate initialization.

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
        logger.debug("Creating output handler '%s' with config %r", name, config)
        handler_cls = cls.get(name)
        try:
            coerced_config = cls._coerce_config(handler_cls, config)
        except Exception as exc:
            logger.error(
                "Failed to coerce config for output handler '%s': %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
            raise

        try:
            instance = handler_cls(coerced_config)
        except Exception as exc:
            logger.error(
                "Failed to instantiate output handler '%s' (%s): %s: %s",
                name,
                handler_cls.__name__,
                type(exc).__name__,
                exc,
            )
            raise
        logger.debug(
            "Created output handler '%s' (%s) with config type %s",
            name,
            handler_cls.__name__,
            type(coerced_config).__name__ if coerced_config is not None else None,
        )
        return instance

    @classmethod
    def create_default(
        cls,
        config: Optional[Union[Dict[str, Any], Any]] = None,
        *,
        output_folder: Optional[str] = None,
        debug_folder: Optional[str] = None,
        output_overwrite: Optional[bool] = None,
    ) -> BaseOutputHandler:
        """Create the default handler using its canonical defaults.

        The default handler must expose a zero-argument ``ConfigType`` that
        yields a fully populated configuration. Callers can optionally supply a
        config object or mapping, which will be coerced via ``_coerce_config``
        to honor the same semantics as :meth:`create`. Path-related overrides
        remain supported for compatibility; they are applied after config
        coercion when the underlying ``ConfigType`` exposes matching fields.
        """
        logger.debug(
            "Creating default output handler '%s' with override config %r, "
            "output_folder=%s, debug_folder=%s, output_overwrite=%s",
            DEFAULT_OUTPUT_HANDLER_NAME,
            config,
            output_folder,
            debug_folder,
            output_overwrite,
        )

        handler_cls = cls.get(DEFAULT_OUTPUT_HANDLER_NAME)
        try:
            coerced_config = cls._coerce_config(handler_cls, config)
        except Exception as exc:
            logger.error(
                "Failed to coerce config for default output handler '%s': %s: %s",
                DEFAULT_OUTPUT_HANDLER_NAME,
                type(exc).__name__,
                exc,
            )
            raise

        if coerced_config is None:
            logger.error(
                "Default output handler '%s' does not define ConfigType; "
                "cannot create default instance",
                DEFAULT_OUTPUT_HANDLER_NAME,
            )
            raise TypeError(
                "Default output handler does not define ConfigType; cannot create default."
            )

        cls._apply_path_overrides(
            coerced_config,
            output_folder=output_folder,
            debug_folder=debug_folder,
            output_overwrite=output_overwrite,
        )

        try:
            instance = handler_cls(coerced_config)
        except Exception as exc:
            logger.error(
                "Failed to instantiate default output handler '%s' (%s): %s: %s",
                DEFAULT_OUTPUT_HANDLER_NAME,
                handler_cls.__name__,
                type(exc).__name__,
                exc,
            )
            raise
        logger.debug(
            "Created default output handler '%s' (%s) with config type %s",
            DEFAULT_OUTPUT_HANDLER_NAME,
            handler_cls.__name__,
            type(coerced_config).__name__,
        )
        return instance

    @classmethod
    def create_default_with_paths(
        cls,
        *,
        output_folder: Optional[str] = None,
        debug_folder: Optional[str] = None,
        output_overwrite: Optional[bool] = None,
    ) -> BaseOutputHandler:
        """Create the default handler while overriding common path settings."""
        logger.debug(
            "create_default_with_paths: output_folder=%s, debug_folder=%s, "
            "output_overwrite=%s",
            output_folder,
            debug_folder,
            output_overwrite,
        )
        return cls.create_default(
            None,
            output_folder=output_folder,
            debug_folder=debug_folder,
            output_overwrite=output_overwrite,
        )

    @staticmethod
    def _apply_path_overrides(
        config: Any,
        *,
        output_folder: Optional[str],
        debug_folder: Optional[str],
        output_overwrite: Optional[bool],
    ) -> None:
        """Apply folder/overwrite overrides when the config exposes the fields."""

        if output_folder is not None:
            if not hasattr(config, "output_folder"):
                logger.error(
                    "_apply_path_overrides: ConfigType %s is missing 'output_folder' field",
                    type(config).__name__,
                )
                raise AttributeError("ConfigType is missing 'output_folder' field")
            logger.debug(
                "_apply_path_overrides: setting output_folder=%s",
                output_folder,
            )
            config.output_folder = output_folder
        if debug_folder is not None:
            if not hasattr(config, "debug_folder"):
                logger.error(
                    "_apply_path_overrides: ConfigType %s is missing 'debug_folder' field",
                    type(config).__name__,
                )
                raise AttributeError("ConfigType is missing 'debug_folder' field")
            logger.debug(
                "_apply_path_overrides: setting debug_folder=%s",
                debug_folder,
            )
            config.debug_folder = debug_folder
        if output_overwrite is not None:
            if not hasattr(config, "output_overwrite"):
                logger.error(
                    "_apply_path_overrides: ConfigType %s is missing 'output_overwrite' field",
                    type(config).__name__,
                )
                raise AttributeError("ConfigType is missing 'output_overwrite' field")
            logger.debug(
                "_apply_path_overrides: setting output_overwrite=%s",
                output_overwrite,
            )
            config.output_overwrite = output_overwrite


__all__ = ["OutputHandlerRegistry"]
