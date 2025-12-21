#!/usr/bin/env python
#
# Detect Meteors CLI - Plugin Registry Base
# © 2025 Shinichi Morita (shin3tky)
#

"""Shared registry utilities for plugin-based components."""

from __future__ import annotations

import logging
import warnings
from dataclasses import is_dataclass
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from .plugin_registry import _PLUGIN_KIND_GENERIC

# Module-level logger for registry operations
logger = logging.getLogger(__name__)

PluginType = TypeVar("PluginType")


class PluginRegistryBase(Generic[PluginType]):
    """Base class providing common registry behaviors.

    Subclasses must implement `_discover_internal` and `_is_valid_plugin` to
    supply discovery logic and validation for their specific plugin type.

    Registry consumers follow a shared convention for configuration objects:
    plugins expose a ``ConfigType`` attribute, and its constructor accepts all
    required fields while optional settings carry defaults. Helpers such as
    ``create_default`` are expected to build configurations via ``ConfigType()``
    (or to coerce a provided mapping/object) and then instantiate the plugin.
    ``create_default`` assumes that ``ConfigType`` can be instantiated without
    arguments to produce a complete default configuration; implementations
    raise clear errors when that contract is not satisfied so callers are not
    left with partially defined defaults.

    See :doc:`PLUGIN_AUTHOR_GUIDE` for a broader overview of plugin lifecycle
    (discovery order, config coercion rules, and available hooks).
    """

    _plugin_kind: str = _PLUGIN_KIND_GENERIC

    # Discovered plugins (from entry points + plugin dir), lazily initialized
    _discovered: Optional[Dict[str, Type[PluginType]]] = None

    # Runtime-registered plugins (for testing/dynamic plugins)
    _custom: Dict[str, Type[PluginType]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Ensure each subclass gets its own registry state
        cls._discovered = None
        cls._custom = {}

    # ========================================
    # Discovery & Registration
    # ========================================
    @classmethod
    def _discover_internal(cls) -> Dict[str, Type[PluginType]]:
        raise NotImplementedError

    @classmethod
    def _is_valid_plugin(cls, plugin_cls: Type[PluginType]) -> bool:
        raise NotImplementedError

    @classmethod
    def discover(cls, force: bool = False) -> Dict[str, Type[PluginType]]:
        """Discover available plugins (cached, lazy)."""

        if cls._discovered is None or force:
            logger.debug(
                "%s.discover: starting discovery (force=%s)",
                cls.__name__,
                force,
            )
            cls._discovered = cls._discover_internal()
            logger.debug(
                "%s.discover: completed, found %d %s(s): %s",
                cls.__name__,
                len(cls._discovered),
                cls._plugin_kind,
                ", ".join(sorted(cls._discovered.keys())) or "none",
            )
        return cls._discovered

    @classmethod
    def get(cls, name: str) -> Type[PluginType]:
        """Get plugin class by name."""

        name_lower = name.lower()

        # 1. Custom (runtime-registered) takes priority
        if name_lower in cls._custom:
            logger.debug(
                "%s.get('%s'): found in custom registry (%s.%s)",
                cls.__name__,
                name,
                cls._custom[name_lower].__module__,
                cls._custom[name_lower].__name__,
            )
            return cls._custom[name_lower]

        # 2. Discovered plugins
        discovered = cls.discover()
        if name_lower in discovered:
            logger.debug(
                "%s.get('%s'): found in discovered registry (%s.%s)",
                cls.__name__,
                name,
                discovered[name_lower].__module__,
                discovered[name_lower].__name__,
            )
            return discovered[name_lower]

        # 3. Not found - provide helpful error message
        available = cls.list_available()
        available_str = ", ".join(sorted(available)) if available else "none"
        logger.warning(
            "%s.get('%s'): %s not found. Available: %s",
            cls.__name__,
            name,
            cls._plugin_kind,
            available_str,
        )
        raise KeyError(
            f"Unknown {cls._plugin_kind} '{name}'. Available: {available_str}"
        )

    @classmethod
    def register(cls, plugin_cls: Type[PluginType]) -> None:
        """Register a plugin class at runtime."""

        if not cls._is_valid_plugin(plugin_cls):
            logger.error(
                "%s.register: invalid %s class %s (must inherit from proper base "
                "and have non-empty plugin_name)",
                cls.__name__,
                cls._plugin_kind,
                plugin_cls,
            )
            raise ValueError(
                f"Invalid {cls._plugin_kind} class: {plugin_cls}. "
                f"Must inherit from the proper base and have non-empty plugin_name."
            )

        name = getattr(plugin_cls, "plugin_name", "")
        if not name:
            logger.error(
                "%s.register: %s %s.%s has empty plugin_name",
                cls.__name__,
                cls._plugin_kind,
                plugin_cls.__module__,
                plugin_cls.__name__,
            )
            raise ValueError(
                f"{cls._plugin_kind.capitalize()} must have non-empty plugin_name"
            )

        # Normalize to lowercase
        name_lower = name.lower()

        # Warn on overwrite (but allow it for testing purposes)
        if name_lower in cls._custom:
            logger.warning(
                "%s.register: overwriting existing runtime-registered %s '%s' "
                "(%s.%s -> %s.%s)",
                cls.__name__,
                cls._plugin_kind,
                name,
                cls._custom[name_lower].__module__,
                cls._custom[name_lower].__name__,
                plugin_cls.__module__,
                plugin_cls.__name__,
            )
            warnings.warn(
                f"Overwriting existing runtime-registered {cls._plugin_kind} '{name}'",
                stacklevel=2,
            )

        cls._custom[name_lower] = plugin_cls
        logger.info(
            "%s.register: registered %s '%s' (%s.%s)",
            cls.__name__,
            cls._plugin_kind,
            name,
            plugin_cls.__module__,
            plugin_cls.__name__,
        )

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a plugin by name."""

        name_lower = name.lower()
        if name_lower in cls._custom:
            removed_cls = cls._custom[name_lower]
            del cls._custom[name_lower]
            logger.info(
                "%s.unregister: removed %s '%s' (%s.%s)",
                cls.__name__,
                cls._plugin_kind,
                name,
                removed_cls.__module__,
                removed_cls.__name__,
            )
            return True
        logger.debug(
            "%s.unregister: %s '%s' not found in custom registry",
            cls.__name__,
            cls._plugin_kind,
            name,
        )
        return False

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available plugin names."""

        discovered = cls.discover()
        all_names = set(discovered.keys()) | set(cls._custom.keys())
        return sorted(all_names)

    # ========================================
    # Internal Methods
    # ========================================
    @classmethod
    def _coerce_config(
        cls,
        plugin_cls: Type[PluginType],
        config: Optional[Union[Dict[str, Any], Any]],
    ) -> Any:
        """Coerce config to the plugin's expected ConfigType."""

        config_type = getattr(plugin_cls, "ConfigType", None)
        plugin_name = getattr(plugin_cls, "plugin_name", "unknown")

        # Case 1: No config provided
        if config is None:
            if config_type is not None:
                try:
                    result = config_type()  # Default instance
                    logger.debug(
                        "_coerce_config(%s): created default %s instance",
                        plugin_name,
                        config_type.__name__,
                    )
                    return result
                except TypeError as exc:
                    logger.error(
                        "_coerce_config(%s): failed to create default %s - %s",
                        plugin_name,
                        config_type.__name__,
                        exc,
                    )
                    raise TypeError(
                        f"Failed to create default config for {cls._plugin_kind} "
                        f"'{plugin_name}': {config_type.__name__} "
                        f"requires arguments. Provide a config dict or instance."
                    ) from exc
                except Exception as exc:
                    logger.error(
                        "_coerce_config(%s): unexpected error creating default %s - "
                        "%s: %s",
                        plugin_name,
                        config_type.__name__,
                        type(exc).__name__,
                        exc,
                    )
                    raise ValueError(
                        f"Failed to create default config for {cls._plugin_kind} "
                        f"'{plugin_name}': {exc}"
                    ) from exc
            logger.debug(
                "_coerce_config(%s): no ConfigType defined, returning None",
                plugin_name,
            )
            return None

        # Case 2: No ConfigType defined on plugin
        if config_type is None:
            logger.debug(
                "_coerce_config(%s): no ConfigType defined, passing config as-is "
                "(type=%s)",
                plugin_name,
                type(config).__name__,
            )
            return config

        # Case 3: Already correct type
        if isinstance(config, config_type):
            logger.debug(
                "_coerce_config(%s): config already correct type (%s)",
                plugin_name,
                config_type.__name__,
            )
            return config

        # Case 4: Dict -> Dataclass
        if is_dataclass(config_type) and isinstance(config, dict):
            try:
                result = config_type(**config)
                logger.debug(
                    "_coerce_config(%s): coerced dict to dataclass %s",
                    plugin_name,
                    config_type.__name__,
                )
                return result
            except TypeError as exc:
                logger.error(
                    "_coerce_config(%s): failed to coerce dict to dataclass %s - %s",
                    plugin_name,
                    config_type.__name__,
                    exc,
                )
                raise TypeError(
                    f"Invalid config for {cls._plugin_kind} '{plugin_name}': {exc}"
                ) from exc

        # Case 5: Dict -> Pydantic v2 (model_validate)
        if hasattr(config_type, "model_validate") and isinstance(config, dict):
            try:
                result = config_type.model_validate(config)
                logger.debug(
                    "_coerce_config(%s): coerced dict to Pydantic v2 model %s",
                    plugin_name,
                    config_type.__name__,
                )
                return result
            except Exception as exc:  # pragma: no cover - pydantic error
                logger.error(
                    "_coerce_config(%s): Pydantic v2 validation failed for %s - %s: %s",
                    plugin_name,
                    config_type.__name__,
                    type(exc).__name__,
                    exc,
                )
                raise ValueError(
                    f"Config validation failed for {cls._plugin_kind} "
                    f"'{plugin_name}': {exc}"
                ) from exc

        # Case 6: Dict -> Pydantic v1 (parse_obj)
        if hasattr(config_type, "parse_obj") and isinstance(config, dict):
            try:
                result = config_type.parse_obj(config)
                logger.debug(
                    "_coerce_config(%s): coerced dict to Pydantic v1 model %s",
                    plugin_name,
                    config_type.__name__,
                )
                return result
            except Exception as exc:  # pragma: no cover - pydantic error
                logger.error(
                    "_coerce_config(%s): Pydantic v1 validation failed for %s - %s: %s",
                    plugin_name,
                    config_type.__name__,
                    type(exc).__name__,
                    exc,
                )
                raise ValueError(
                    f"Config validation failed for {cls._plugin_kind} "
                    f"'{plugin_name}': {exc}"
                ) from exc

        # Case 7: Incompatible type
        if isinstance(config, dict):
            logger.error(
                "_coerce_config(%s): cannot coerce dict to %s (not a dataclass "
                "or Pydantic model)",
                plugin_name,
                config_type.__name__,
            )
            raise TypeError(
                f"Cannot coerce dict to {config_type.__name__} for {cls._plugin_kind} "
                f"'{plugin_name}': ConfigType is neither a "
                f"dataclass nor a Pydantic model."
            )

        # Fallback: return as-is (may fail later in plugin __init__)
        logger.debug(
            "_coerce_config(%s): passing config as-is (type=%s, expected=%s)",
            plugin_name,
            type(config).__name__,
            config_type.__name__,
        )
        return config

    @classmethod
    def _reset(cls) -> None:
        """Reset registry state (for tests)."""

        cls._discovered = None
        cls._custom = {}
        logger.debug("%s._reset: cleared registry state", cls.__name__)


__all__ = ["PluginRegistryBase"]
