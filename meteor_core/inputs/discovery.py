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

from .base import InputLoader, _require_plugin_name
from .raw import RawImageLoader

PLUGIN_GROUP = "detect_meteors.input"
PLUGIN_DIR = Path.home() / ".detect_meteors" / "plugins"


def _iter_entry_points() -> Iterable[metadata.EntryPoint]:
    eps = metadata.entry_points()
    if hasattr(eps, "select"):
        return eps.select(group=PLUGIN_GROUP)
    return eps.get(PLUGIN_GROUP, [])  # type: ignore[return-value]


def _add_loader(
    registry: Dict[str, Type[InputLoader]], loader_cls: Type[InputLoader], origin: str
) -> None:
    if not inspect.isclass(loader_cls):
        warnings.warn(f"Skipping non-class loader from {origin}: {loader_cls!r}")
        return
    if not issubclass(loader_cls, InputLoader):
        warnings.warn(
            f"Skipping loader from {origin}: {loader_cls.__module__}.{loader_cls.__name__} "
            "does not implement InputLoader."
        )
        return
    try:
        plugin_name = _require_plugin_name(loader_cls)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"Skipping loader from {origin}: {exc}")
        return

    if plugin_name in registry:
        existing = registry[plugin_name]
        warnings.warn(
            f"Duplicate loader name '{plugin_name}' from {origin}; "
            f"keeping {existing.__module__}.{existing.__name__}",
            stacklevel=2,
        )
        return

    registry[plugin_name] = loader_cls


def _load_module_from_file(filepath: Path):
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
    built-in loaders are registered first, followed by entry points sorted by
    entry-point name, and finally plugin files in the local plugin directory
    sorted alphabetically. Later discoveries with duplicate ``plugin_name``
    values are ignored with a warning so that resolution is predictable.
    """

    registry: Dict[str, Type[InputLoader]] = {}

    _add_loader(registry, RawImageLoader, "built-in RawImageLoader")

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

            for _, obj in inspect.getmembers(module, inspect.isclass):
                _add_loader(registry, obj, f"plugin file {path}")

    return registry


__all__ = ["discover_input_loaders", "PLUGIN_DIR", "PLUGIN_GROUP"]
