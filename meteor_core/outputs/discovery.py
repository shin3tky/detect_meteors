#!/usr/bin/env python
#
# Detect Meteors CLI - Output Handler Discovery
# © 2025 Shinichi Morita (shin3tky)
#

"""Discovery utilities for output handler plugins."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import warnings
from pathlib import Path
from typing import Dict, Iterable, Type

try:  # pragma: no cover - fallback for older Python
    from importlib import metadata
except ImportError:  # pragma: no cover
    import importlib_metadata as metadata  # type: ignore

from .base import BaseOutputHandler, _is_valid_output_handler
from .file_handler import FileOutputHandler

PLUGIN_GROUP = "detect_meteors.output"
PLUGIN_DIR = Path.home() / ".detect_meteors" / "output_plugins"

# Classes to skip during discovery (base classes).
# Keep this list aligned with abstract bases to avoid registering helpers.
_SKIP_CLASSES = frozenset(
    {
        "BaseOutputHandler",
        "DataclassOutputHandler",
        "PydanticOutputHandler",
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


def _add_handler(
    registry: Dict[str, Type[BaseOutputHandler]],
    handler_cls: Type[BaseOutputHandler],
    origin: str,
) -> None:
    """Add a handler class to the registry if valid.

    Args:
        registry: The registry dictionary to add to.
        handler_cls: The handler class to potentially add.
        origin: Description of where this handler came from (for warnings).
    """
    # Skip non-class objects (functions, modules, etc.)
    if not inspect.isclass(handler_cls):
        return

    # Skip base classes
    if handler_cls.__name__ in _SKIP_CLASSES:
        return

    # Validate handler class
    if not _is_valid_output_handler(handler_cls):
        # Only warn for classes that look like they might be intended handlers
        class_name_lower = handler_cls.__name__.lower()
        if "handler" in class_name_lower or "output" in class_name_lower:
            warnings.warn(
                f"Skipping handler from {origin}: {handler_cls.__module__}.{handler_cls.__name__} "
                "does not inherit from BaseOutputHandler (or missing plugin_name).",
                stacklevel=3,
            )
        return

    # Get plugin_name from the class
    plugin_name = getattr(handler_cls, "plugin_name", "")
    if not plugin_name:
        warnings.warn(
            f"Skipping handler from {origin}: {handler_cls.__name__} has empty plugin_name",
            stacklevel=3,
        )
        return

    # Normalize to lowercase for case-insensitive lookup
    plugin_name_lower = plugin_name.lower()

    # Check for duplicates
    if plugin_name_lower in registry:
        existing = registry[plugin_name_lower]
        warnings.warn(
            f"Duplicate handler name '{plugin_name}' from {origin}; "
            f"keeping {existing.__module__}.{existing.__name__}",
            stacklevel=3,
        )
        return

    registry[plugin_name_lower] = handler_cls


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


def discover_handlers(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseOutputHandler]]:
    """Discover available :class:`BaseOutputHandler` implementations.

    .. deprecated::
        Use :meth:`OutputHandlerRegistry.discover` instead. This function will be
        removed in a future version.

    Discovery order is deterministic:

    1. Built-in handlers are registered first
    2. Entry points sorted by entry-point name
    3. Plugin files in the local plugin directory sorted alphabetically

    Later discoveries with duplicate ``plugin_name`` values are ignored
    with a warning so that resolution is predictable.

    Args:
        plugin_dir: Optional custom plugin directory. Defaults to
            ``~/.detect_meteors/output_plugins``.

    Returns:
        Dictionary mapping plugin_name to handler class.

    Example:
        >>> handlers = discover_handlers()
        >>> print(handlers.keys())
        dict_keys(['file', ...])
        >>> FileHandler = handlers['file']
    """
    warnings.warn(
        "discover_handlers() is deprecated. "
        "Use OutputHandlerRegistry.discover() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _discover_handlers_internal(plugin_dir)


# NOTE:
# ``OutputHandlerRegistry`` imports ``_discover_handlers_internal`` for its
# discovery step. The wrapper below remains for legacy imports while keeping
# a single implementation path.
def _discover_output_handlers_internal(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseOutputHandler]]:
    """Compatibility wrapper used by :class:`OutputHandlerRegistry`.

    Historically the registry imported ``_discover_output_handlers_internal``.
    The core implementation lives in ``_discover_handlers_internal``; this
    wrapper simply delegates to it to avoid import errors while keeping a
    single discovery code path.
    """

    return _discover_handlers_internal(plugin_dir)


def _discover_handlers_internal(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseOutputHandler]]:
    """Internal discovery function used by OutputHandlerRegistry.

    This is the core discovery implementation. It discovers handlers from:
    1. Built-in handlers (e.g., FileOutputHandler)
    2. Entry points (detect_meteors.output group)
    3. Plugin directory

    Args:
        plugin_dir: Optional custom plugin directory. If None, uses PLUGIN_DIR.

    Returns:
        Dictionary mapping plugin_name to handler class.
    """
    directory = plugin_dir if plugin_dir is not None else PLUGIN_DIR

    registry: Dict[str, Type[BaseOutputHandler]] = {}

    # 1. Register built-in handlers first
    _add_handler(registry, FileOutputHandler, "built-in FileOutputHandler")

    # 2. Register handlers from entry points (sorted for determinism)
    for ep in sorted(_iter_entry_points(), key=lambda e: e.name):
        try:
            handler_cls = ep.load()
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load output handler entry point '{ep.name}' "
                f"from {ep.value}: {exc}",
                stacklevel=2,
            )
            continue
        _add_handler(registry, handler_cls, f"entry point {ep.name}")

    # 3. Register handlers from plugin directory
    if directory.exists() and directory.is_dir():
        for path in sorted(directory.glob("*.py")):
            try:
                module = _load_module_from_file(path)
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"Failed to load output handler plugin module {path}: {exc}",
                    stacklevel=2,
                )
                continue

            # Inspect all classes defined in the module
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # Only consider classes defined in this module (not imports)
                if obj.__module__ == module.__name__:
                    _add_handler(registry, obj, f"plugin file {path}")

    return registry


__all__ = [
    "discover_handlers",
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
]
