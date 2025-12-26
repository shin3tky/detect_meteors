"""Registry for hook plugins."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type, Union

from ..plugin_registry import _PLUGIN_KIND_HOOK
from ..plugin_registry_base import PluginRegistryBase
from .base import BaseHook, _is_valid_hook

logger = logging.getLogger(__name__)


class HookRegistry(PluginRegistryBase[BaseHook]):
    """Registry for pipeline hook plugins."""

    _plugin_kind = _PLUGIN_KIND_HOOK

    @classmethod
    def _discover_internal(cls) -> Dict[str, Type[BaseHook]]:
        return {}

    @classmethod
    def _is_valid_plugin(cls, plugin_cls: Type[BaseHook]) -> bool:
        return _is_valid_hook(plugin_cls)

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> BaseHook:
        """Create a hook instance with config coercion."""
        logger.debug("Creating hook '%s' with config %r", name, config)
        hook_cls = cls.get(name)
        try:
            coerced_config = cls._coerce_config(hook_cls, config)
        except Exception as exc:
            logger.error(
                "Failed to coerce config for hook '%s': %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
            raise

        try:
            instance = hook_cls(coerced_config)
        except Exception as exc:
            logger.error(
                "Failed to instantiate hook '%s' (%s): %s: %s",
                name,
                hook_cls.__name__,
                type(exc).__name__,
                exc,
            )
            raise
        logger.debug(
            "Created hook '%s' (%s) with config type %s",
            name,
            hook_cls.__name__,
            type(coerced_config).__name__ if coerced_config is not None else None,
        )
        return instance

    @classmethod
    def create_all(cls) -> List[BaseHook]:
        """Instantiate all registered hooks."""
        instances: List[BaseHook] = []
        for name in cls.list_available():
            try:
                instances.append(cls.create(name))
            except Exception as exc:
                logger.warning(
                    "Failed to instantiate hook '%s' (%s): %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
        return instances


__all__ = ["HookRegistry"]
