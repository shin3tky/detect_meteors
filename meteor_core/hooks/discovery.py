#!/usr/bin/env python
#
# Detect Meteors CLI - Hook Discovery
# © 2025 Shinichi Morita (shin3tky)
#

"""Discovery utilities for hook plugins."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, Type, cast

try:  # pragma: no cover - fallback for older Python
    from importlib import metadata
except ImportError:  # pragma: no cover
    import importlib_metadata as metadata  # type: ignore

from .base import BaseHook, _is_valid_hook
from .file_found import AllowAllFilesFoundHook

# Module-level logger for discovery diagnostics
logger = logging.getLogger(__name__)

PLUGIN_GROUP = "detect_meteors.hook"
PLUGIN_DIR = Path.home() / ".detect_meteors" / "hook_plugins"

# Classes to skip during discovery (base classes).
# Keep this list aligned with abstract bases to avoid registering helpers.
_SKIP_CLASSES = frozenset(
    {
        "BaseHook",
        "DataclassHook",
        "PydanticHook",
        "ABC",
        "Generic",
    }
)


def _iter_entry_points() -> Iterable[metadata.EntryPoint]:
    """Iterate over entry points for the plugin group."""
    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=PLUGIN_GROUP)
    return eps.get(PLUGIN_GROUP, [])  # type: ignore[return-value]


def _add_hook(
    registry: Dict[str, Type[BaseHook]],
    hook_cls: type,
    origin: str,
) -> None:
    """Add a hook class to the registry if valid.

    Args:
        registry: The registry dictionary to add to.
        hook_cls: The hook class to potentially add.
        origin: Description of where this hook came from (for warnings).
    """
    # Skip non-class objects (functions, modules, etc.)
    if not inspect.isclass(hook_cls):
        return

    # Skip base classes
    if hook_cls.__name__ in _SKIP_CLASSES:
        logger.debug(
            "Skipping base class %s from %s",
            hook_cls.__name__,
            origin,
        )
        return

    # Validate hook class
    if not _is_valid_hook(hook_cls):
        # Only warn for classes that look like they might be intended hooks
        class_name_lower = hook_cls.__name__.lower()
        if "hook" in class_name_lower:
            logger.warning(
                "Skipping hook from %s: %s.%s does not inherit from "
                "BaseHook (or missing plugin_name)",
                origin,
                hook_cls.__module__,
                hook_cls.__name__,
            )
            warnings.warn(
                f"Skipping hook from {origin}: {hook_cls.__module__}.{hook_cls.__name__} "
                "does not inherit from BaseHook (or missing plugin_name).",
                stacklevel=3,
            )
        return

    hook_cls = cast(Type[BaseHook], hook_cls)

    # Get plugin_name from the class
    plugin_name = getattr(hook_cls, "plugin_name", "")
    if not plugin_name:
        logger.warning(
            "Skipping hook from %s: %s has empty plugin_name",
            origin,
            hook_cls.__name__,
        )
        warnings.warn(
            f"Skipping hook from {origin}: {hook_cls.__name__} has empty plugin_name",
            stacklevel=3,
        )
        return

    # Normalize to lowercase for case-insensitive lookup
    plugin_name_lower = plugin_name.lower()

    # Check for duplicates and allow override by later discoveries
    if plugin_name_lower in registry:
        existing = registry[plugin_name_lower]
        logger.info(
            "Duplicate hook name '%s' from %s; overriding %s.%s",
            plugin_name,
            origin,
            existing.__module__,
            existing.__name__,
        )

    logger.debug(
        "Registered hook '%s' from %s (%s.%s)",
        plugin_name,
        origin,
        hook_cls.__module__,
        hook_cls.__name__,
    )
    registry[plugin_name_lower] = hook_cls


def _load_module_from_file(filepath: Path):
    """Load a Python module from a file path.

    Args:
        filepath: Path to the Python file to load.

    Returns:
        The loaded module object.

    Raises:
        ImportError: If the module cannot be loaded.
    """
    spec = importlib.util.spec_from_file_location(filepath.stem, filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise ImportError(f"Could not load spec for {filepath}")


def discover_hooks(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseHook]]:
    """Discover available :class:`BaseHook` implementations.

    .. deprecated::
        Use :meth:`HookRegistry.discover` instead. This function will be
        removed in a future version.

    Discovery order is deterministic:

    1. Built-in hooks are registered first
    2. Entry points sorted by entry-point name
    3. Plugin files in the local plugin directory sorted alphabetically

    Later discoveries with duplicate ``plugin_name`` values override
    earlier registrations so that the most recent hook is used.

    Args:
        plugin_dir: Optional custom plugin directory. Defaults to
            ``~/.detect_meteors/hook_plugins``.

    Returns:
        Dictionary mapping plugin_name to hook class.
    """
    warnings.warn(
        "discover_hooks() is deprecated. Use HookRegistry.discover() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _discover_handlers_internal(plugin_dir)


def _discover_hooks_internal(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseHook]]:
    """Compatibility wrapper retained for legacy imports."""

    return _discover_handlers_internal(plugin_dir)


def _discover_handlers_internal(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseHook]]:
    """Internal discovery function used by HookRegistry.

    This is the core discovery implementation. It discovers hooks from:
    1. Built-in hooks (e.g., AllowAllFilesFoundHook)
    2. Entry points (detect_meteors.hook group)
    3. Plugin directory

    Args:
        plugin_dir: Optional custom plugin directory. If None, uses PLUGIN_DIR.

    Returns:
        Dictionary mapping plugin_name to hook class.
    """
    directory = plugin_dir if plugin_dir is not None else PLUGIN_DIR

    logger.info("Starting hook discovery")
    logger.debug("Plugin directory: %s", directory)

    registry: Dict[str, Type[BaseHook]] = {}

    # 1. Register built-in hooks first
    logger.debug("Phase 1: Registering built-in hooks")
    _add_hook(registry, AllowAllFilesFoundHook, "built-in AllowAllFilesFoundHook")

    # 2. Register hooks from entry points (sorted for determinism)
    logger.debug(
        "Phase 2: Discovering hooks from entry points (group=%s)", PLUGIN_GROUP
    )
    entry_point_count = 0
    for ep in sorted(_iter_entry_points(), key=lambda e: e.name):
        entry_point_count += 1
        logger.debug("Loading entry point: %s from %s", ep.name, ep.value)
        try:
            hook_cls = ep.load()
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "Failed to load hook entry point '%s' from %s: %s: %s",
                ep.name,
                ep.value,
                type(exc).__name__,
                exc,
            )
            warnings.warn(
                f"Failed to load hook entry point '{ep.name}' from {ep.value}: {exc}",
                stacklevel=2,
            )
            continue
        _add_hook(registry, hook_cls, f"entry point {ep.name}")

    if entry_point_count == 0:
        logger.debug("No entry points found for group %s", PLUGIN_GROUP)
    else:
        logger.debug("Processed %d entry points", entry_point_count)

    # 3. Register hooks from plugin directory
    logger.debug("Phase 3: Discovering hooks from plugin directory")
    if directory.exists() and directory.is_dir():
        plugin_files = sorted(directory.glob("*.py"))
        logger.debug("Found %d plugin files in %s", len(plugin_files), directory)

        for path in plugin_files:
            logger.debug("Loading plugin module: %s", path)
            try:
                module = _load_module_from_file(path)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to load hook plugin module %s: %s: %s",
                    path,
                    type(exc).__name__,
                    exc,
                )
                warnings.warn(
                    f"Failed to load hook plugin module {path}: {exc}",
                    stacklevel=2,
                )
                continue

            # Inspect all classes defined in the module
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # Only consider classes defined in this module (not imports)
                if obj.__module__ == module.__name__:
                    _add_hook(registry, obj, f"plugin file {path}")
    else:
        logger.debug(
            "Plugin directory does not exist or is not a directory: %s", directory
        )

    logger.info(
        "Hook discovery completed: %d hooks registered (%s)",
        len(registry),
        ", ".join(sorted(registry.keys())),
    )

    return registry


__all__ = ["discover_hooks", "PLUGIN_DIR", "PLUGIN_GROUP"]
