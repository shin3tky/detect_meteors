#!/usr/bin/env python
#
# Detect Meteors CLI - Input Loader Discovery
# © 2025 Shinichi Morita (shin3tky)
#

"""Discovery utilities for input loader plugins."""

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

from .base import BaseInputLoader, _is_valid_input_loader
from .raw import RawImageLoader

PLUGIN_GROUP = "detect_meteors.input"
PLUGIN_DIR = Path.home() / ".detect_meteors" / "input_plugins"

# Classes to skip during discovery (base classes).
# Keep this list aligned with abstract bases to avoid registering helpers.
_SKIP_CLASSES = frozenset(
    {
        "BaseInputLoader",
        "BaseMetadataExtractor",
        "DataclassInputLoader",
        "PydanticInputLoader",
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


def _add_loader(
    registry: Dict[str, Type[BaseInputLoader]],
    loader_cls: Type[BaseInputLoader],
    origin: str,
) -> None:
    """Add a loader class to the registry if valid.

    Args:
        registry: The registry dictionary to add to.
        loader_cls: The loader class to potentially add.
        origin: Description of where this loader came from (for warnings).
    """
    # Skip non-class objects (functions, modules, etc.)
    if not inspect.isclass(loader_cls):
        return

    # Skip base classes and protocols themselves
    if loader_cls.__name__ in _SKIP_CLASSES:
        return

    # Use structural check instead of issubclass for Protocol
    # This is more reliable than isinstance/issubclass with Protocol
    if not _is_valid_input_loader(loader_cls):
        # Only warn for classes that look like they might be intended loaders
        class_name_lower = loader_cls.__name__.lower()
        if "loader" in class_name_lower or "input" in class_name_lower:
            warnings.warn(
                f"Skipping loader from {origin}: {loader_cls.__module__}.{loader_cls.__name__} "
                "does not inherit from BaseInputLoader (or missing plugin_name).",
                stacklevel=3,
            )
        return

    # Get plugin_name from the class
    plugin_name = getattr(loader_cls, "plugin_name", "")
    if not plugin_name:
        warnings.warn(
            f"Skipping loader from {origin}: {loader_cls.__name__} has empty plugin_name",
            stacklevel=3,
        )
        return

    # Normalize to lowercase for case-insensitive lookup
    plugin_name_lower = plugin_name.lower()

    # Check for duplicates
    if plugin_name_lower in registry:
        existing = registry[plugin_name_lower]
        warnings.warn(
            f"Duplicate loader name '{plugin_name}' from {origin}; "
            f"keeping {existing.__module__}.{existing.__name__}",
            stacklevel=3,
        )
        return

    registry[plugin_name_lower] = loader_cls


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


def discover_loaders(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseInputLoader]]:
    """Discover available :class:`BaseInputLoader` implementations.

    .. deprecated::
        Use :meth:`LoaderRegistry.discover` instead. This function will be
        removed in a future version.

    Discovery order is deterministic:

    1. Built-in loaders are registered first
    2. Entry points sorted by entry-point name
    3. Plugin files in the local plugin directory sorted alphabetically

    Later discoveries with duplicate ``plugin_name`` values are ignored
    with a warning so that resolution is predictable.

    Args:
        plugin_dir: Optional custom plugin directory. Defaults to
            ``~/.detect_meteors/input_plugins``.

    Returns:
        Dictionary mapping plugin_name to loader class.

    Example:
        >>> loaders = discover_loaders()
        >>> print(loaders.keys())
        dict_keys(['raw', ...])
        >>> RawLoader = loaders['raw']
    """
    warnings.warn(
        "discover_loaders() is deprecated. " "Use LoaderRegistry.discover() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _discover_handlers_internal(plugin_dir)


def _discover_loaders_internal(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseInputLoader]]:
    """Compatibility wrapper retained for legacy imports.

    Historically ``LoaderRegistry`` imported ``_discover_loaders_internal``.
    The shared implementation now lives in ``_discover_handlers_internal``;
    this wrapper delegates to preserve backward compatibility while
    centralizing discovery logic under a single name.
    """

    return _discover_handlers_internal(plugin_dir)


def _discover_handlers_internal(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseInputLoader]]:
    """Internal discovery function used by LoaderRegistry.

    This is the core discovery implementation. It discovers loaders from:
    1. Built-in loaders (e.g., RawImageLoader)
    2. Entry points (detect_meteors.input group)
    3. Plugin directory

    Args:
        plugin_dir: Optional custom plugin directory. If None, uses PLUGIN_DIR.

    Returns:
        Dictionary mapping plugin_name to loader class.
    """
    directory = plugin_dir if plugin_dir is not None else PLUGIN_DIR

    registry: Dict[str, Type[BaseInputLoader]] = {}

    # 1. Register built-in loaders first
    _add_loader(registry, RawImageLoader, "built-in RawImageLoader")

    # 2. Register loaders from entry points (sorted for determinism)
    for ep in sorted(_iter_entry_points(), key=lambda e: e.name):
        try:
            loader_cls = ep.load()
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load input loader entry point '{ep.name}' "
                f"from {ep.value}: {exc}",
                stacklevel=2,
            )
            continue
        _add_loader(registry, loader_cls, f"entry point {ep.name}")

    # 3. Register loaders from plugin directory
    if directory.exists() and directory.is_dir():
        for path in sorted(directory.glob("*.py")):
            try:
                module = _load_module_from_file(path)
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"Failed to load plugin module {path}: {exc}",
                    stacklevel=2,
                )
                continue

            # Inspect all classes defined in the module
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # Only consider classes defined in this module (not imports)
                if obj.__module__ == module.__name__:
                    _add_loader(registry, obj, f"plugin file {path}")

    return registry


__all__ = ["discover_loaders", "PLUGIN_DIR", "PLUGIN_GROUP"]
