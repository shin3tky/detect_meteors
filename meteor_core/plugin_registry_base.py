#!/usr/bin/env python
#
# Detect Meteors CLI - Plugin Registry Base
# © 2025 Shinichi Morita (shin3tky)
#

"""Shared registry utilities for plugin-based components."""

from __future__ import annotations

import warnings
from dataclasses import is_dataclass
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from .plugin_registry import _PLUGIN_KIND_GENERIC

PluginType = TypeVar("PluginType")


class PluginRegistryBase(Generic[PluginType]):
    """Base class providing common registry behaviors.

    Subclasses must implement `_discover_internal` and `_is_valid_plugin` to
    supply discovery logic and validation for their specific plugin type.

    Registry consumers follow a shared convention for configuration objects:
    plugins expose a ``ConfigType`` attribute, and its constructor accepts all
    required fields while optional settings carry defaults. Helpers such as
    ``create_default`` are expected to build configurations via ``ConfigType()``
    and then override parameters as needed. ``create_default`` assumes that
    ``ConfigType`` can be instantiated without arguments to produce a complete
    default configuration; implementations raise clear errors when that
    contract is not satisfied so callers are not left with partially defined
    defaults.
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
            cls._discovered = cls._discover_internal()
        return cls._discovered

    @classmethod
    def get(cls, name: str) -> Type[PluginType]:
        """Get plugin class by name."""

        name_lower = name.lower()

        # 1. Custom (runtime-registered) takes priority
        if name_lower in cls._custom:
            return cls._custom[name_lower]

        # 2. Discovered plugins
        discovered = cls.discover()
        if name_lower in discovered:
            return discovered[name_lower]

        # 3. Not found - provide helpful error message
        available = cls.list_available()
        available_str = ", ".join(sorted(available)) if available else "none"
        raise KeyError(
            f"Unknown {cls._plugin_kind} '{name}'. Available: {available_str}"
        )

    @classmethod
    def register(cls, plugin_cls: Type[PluginType]) -> None:
        """Register a plugin class at runtime."""

        if not cls._is_valid_plugin(plugin_cls):
            raise ValueError(
                f"Invalid {cls._plugin_kind} class: {plugin_cls}. "
                f"Must inherit from the proper base and have non-empty plugin_name."
            )

        name = getattr(plugin_cls, "plugin_name", "")
        if not name:
            raise ValueError(
                f"{cls._plugin_kind.capitalize()} must have non-empty plugin_name"
            )

        # Normalize to lowercase
        name_lower = name.lower()

        # Warn on overwrite (but allow it for testing purposes)
        if name_lower in cls._custom:
            warnings.warn(
                f"Overwriting existing runtime-registered {cls._plugin_kind} '{name}'",
                stacklevel=2,
            )

        cls._custom[name_lower] = plugin_cls

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a plugin by name."""

        name_lower = name.lower()
        if name_lower in cls._custom:
            del cls._custom[name_lower]
            return True
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

        # Case 1: No config provided
        if config is None:
            if config_type is not None:
                try:
                    return config_type()  # Default instance
                except TypeError as exc:
                    raise TypeError(
                        f"Failed to create default config for {cls._plugin_kind} "
                        f"'{plugin_cls.plugin_name}': {config_type.__name__} "
                        f"requires arguments. Provide a config dict or instance."
                    ) from exc
                except Exception as exc:
                    raise ValueError(
                        f"Failed to create default config for {cls._plugin_kind} "
                        f"'{plugin_cls.plugin_name}': {exc}"
                    ) from exc
            return None

        # Case 2: No ConfigType defined on plugin
        if config_type is None:
            return config

        # Case 3: Already correct type
        if isinstance(config, config_type):
            return config

        # Case 4: Dict -> Dataclass
        if is_dataclass(config_type) and isinstance(config, dict):
            try:
                return config_type(**config)
            except TypeError as exc:
                raise TypeError(
                    f"Invalid config for {cls._plugin_kind} "
                    f"'{plugin_cls.plugin_name}': {exc}"
                ) from exc

        # Case 5: Dict -> Pydantic v2 (model_validate)
        if hasattr(config_type, "model_validate") and isinstance(config, dict):
            try:
                return config_type.model_validate(config)
            except Exception as exc:  # pragma: no cover - pydantic error
                raise ValueError(
                    f"Config validation failed for {cls._plugin_kind} "
                    f"'{plugin_cls.plugin_name}': {exc}"
                ) from exc

        # Case 6: Dict -> Pydantic v1 (parse_obj)
        if hasattr(config_type, "parse_obj") and isinstance(config, dict):
            try:
                return config_type.parse_obj(config)
            except Exception as exc:  # pragma: no cover - pydantic error
                raise ValueError(
                    f"Config validation failed for {cls._plugin_kind} "
                    f"'{plugin_cls.plugin_name}': {exc}"
                ) from exc

        # Case 7: Incompatible type
        if isinstance(config, dict):
            raise TypeError(
                f"Cannot coerce dict to {config_type.__name__} for {cls._plugin_kind} "
                f"'{plugin_cls.plugin_name}': ConfigType is neither a "
                f"dataclass nor a Pydantic model."
            )

        # Fallback: return as-is (may fail later in plugin __init__)
        return config

    @classmethod
    def _reset(cls) -> None:
        """Reset registry state (for tests)."""

        cls._discovered = None
        cls._custom = {}


__all__ = ["PluginRegistryBase"]
