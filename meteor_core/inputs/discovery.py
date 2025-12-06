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

from .base import InputLoader, _is_valid_input_loader
from .raw import RawImageLoader

PLUGIN_GROUP = "detect_meteors.input"
PLUGIN_DIR = Path.home() / ".detect_meteors" / "plugins"

# Classes to skip during discovery (base classes and protocols)
_SKIP_CLASSES = frozenset(
    {
        "InputLoader",
        "MetadataExtractor",
        "DataclassInputLoader",
        "PydanticInputLoader",
        "Protocol",
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
    registry: Dict[str, Type[InputLoader]], loader_cls: Type[InputLoader], origin: str
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
                "does not implement InputLoader protocol (missing plugin_name or load method).",
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

    # Check for duplicates
    if plugin_name in registry:
        existing = registry[plugin_name]
        warnings.warn(
            f"Duplicate loader name '{plugin_name}' from {origin}; "
            f"keeping {existing.__module__}.{existing.__name__}",
            stacklevel=3,
        )
        return

    registry[plugin_name] = loader_cls


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


def discover_input_loaders(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[InputLoader]]:
    """Discover available :class:`InputLoader` implementations.

    Discovery order is deterministic:

    1. Built-in loaders are registered first
    2. Entry points sorted by entry-point name
    3. Plugin files in the local plugin directory sorted alphabetically

    Later discoveries with duplicate ``plugin_name`` values are ignored
    with a warning so that resolution is predictable.

    Args:
        plugin_dir: Optional custom plugin directory. Defaults to
            ``~/.detect_meteors/plugins``.

    Returns:
        Dictionary mapping plugin_name to loader class.

    Example:
        >>> loaders = discover_input_loaders()
        >>> print(loaders.keys())
        dict_keys(['raw', ...])
        >>> RawLoader = loaders['raw']
    """

    registry: Dict[str, Type[InputLoader]] = {}

    # 1. Register built-in loaders first
    _add_loader(registry, RawImageLoader, "built-in RawImageLoader")

    # 2. Register loaders from entry points (sorted for determinism)
    for ep in sorted(_iter_entry_points(), key=lambda e: e.name):
        try:
            loader_cls = ep.load()
        except Exception as exc:  # pragma: no cover - import-time failure handling
            warnings.warn(
                f"Failed to load input loader entry point '{ep.name}' from {ep.value}: {exc}",
                stacklevel=2,
            )
            continue
        _add_loader(registry, loader_cls, f"entry point {ep.name}")

    # 3. Register loaders from plugin directory (sorted for determinism)
    directory = Path(plugin_dir) if plugin_dir is not None else PLUGIN_DIR
    if directory.exists() and directory.is_dir():
        for path in sorted(directory.glob("*.py")):
            try:
                module = _load_module_from_file(path)
            except Exception as exc:  # pragma: no cover - import-time failure handling
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


__all__ = ["discover_input_loaders", "PLUGIN_DIR", "PLUGIN_GROUP"]
