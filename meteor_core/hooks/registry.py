"""Registry for hook plugins."""

from __future__ import annotations

import inspect
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
        # Keep plugin directory path in sync with meteor_core.hooks.discovery.PLUGIN_DIR
        from .discovery import _discover_handlers_internal

        return _discover_handlers_internal()

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
            if coerced_config is None and _can_instantiate_without_config(hook_cls):
                instance = hook_cls()
            else:
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
    def create_all(cls, skip_on_error: bool = False) -> List[BaseHook]:
        """Instantiate all registered hooks.

        Args:
            skip_on_error: When True, log and skip hooks that fail to create.

        Raises:
            Exception: Propagates hook creation failures to the caller.
        """
        instances: List[BaseHook] = []
        for name in cls.list_available():
            try:
                instances.append(cls.create(name))
            except Exception as exc:
                if not skip_on_error:
                    raise
                logger.warning(
                    "Skipping hook '%s' after creation failure: %s: %s",
                    name,
                    type(exc).__name__,
                    exc,
                )
        return instances


__all__ = ["HookRegistry"]


def _can_instantiate_without_config(hook_cls: Type[BaseHook]) -> bool:
    signature = inspect.signature(hook_cls.__init__)
    parameters = list(signature.parameters.values())
    if parameters:
        parameters = parameters[1:]
    for param in parameters:
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return True
        if param.default is inspect.Parameter.empty and param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return False
    return True
