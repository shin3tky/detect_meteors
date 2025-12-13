#!/usr/bin/env python
#
# Detect Meteors CLI - Detector Discovery
# © 2025 Shinichi Morita (shin3tky)
#

"""Discovery utilities for detector plugins."""

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

from .base import BaseDetector, _is_valid_detector
from .hough_default import HoughDetector
from .simple_threshold import SimpleThresholdDetector

PLUGIN_GROUP = "detect_meteors.detector"
PLUGIN_DIR = Path.home() / ".detect_meteors" / "detector_plugins"

# Classes to skip during discovery (base classes).
# Keep this list aligned with concrete abstract bases to avoid registering
# helper classes in favor of real detector implementations.
_SKIP_CLASSES = frozenset(
    {
        "BaseDetector",
        "DataclassDetector",
        "PydanticDetector",
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


def _add_detector(
    registry: Dict[str, Type[BaseDetector]],
    detector_cls: Type[BaseDetector],
    origin: str,
) -> None:
    """Add a detector class to the registry if valid.

    Args:
        registry: The registry dictionary to add to.
        detector_cls: The detector class to potentially add.
        origin: Description of where this detector came from (for warnings).
    """
    # Skip non-class objects (functions, modules, etc.)
    if not inspect.isclass(detector_cls):
        return

    # Skip base classes
    if detector_cls.__name__ in _SKIP_CLASSES:
        return

    # Validate detector class
    if not _is_valid_detector(detector_cls):
        # Only warn for classes that look like they might be intended detectors
        class_name_lower = detector_cls.__name__.lower()
        if "detector" in class_name_lower:
            warnings.warn(
                f"Skipping detector from {origin}: {detector_cls.__module__}.{detector_cls.__name__} "
                "does not inherit from BaseDetector (or missing plugin_name).",
                stacklevel=3,
            )
        return

    # Get plugin_name from the class
    plugin_name = getattr(detector_cls, "plugin_name", "")
    if not plugin_name:
        warnings.warn(
            f"Skipping detector from {origin}: {detector_cls.__name__} has empty plugin_name",
            stacklevel=3,
        )
        return

    # Normalize to lowercase for case-insensitive lookup
    plugin_name_lower = plugin_name.lower()

    # Check for duplicates
    if plugin_name_lower in registry:
        existing = registry[plugin_name_lower]
        warnings.warn(
            f"Duplicate detector name '{plugin_name}' from {origin}; "
            f"keeping {existing.__module__}.{existing.__name__}",
            stacklevel=3,
        )
        return

    registry[plugin_name_lower] = detector_cls


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


def discover_detectors(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseDetector]]:
    """Discover available :class:`BaseDetector` implementations.

    .. deprecated::
        Use :meth:`DetectorRegistry.discover` instead. This function will be
        removed in a future version.

    Discovery order is deterministic:

    1. Built-in detectors are registered first
    2. Entry points sorted by entry-point name
    3. Plugin files in the local plugin directory sorted alphabetically

    Later discoveries with duplicate ``plugin_name`` values are ignored
    with a warning so that resolution is predictable.

    Args:
        plugin_dir: Optional custom plugin directory. Defaults to
            ``~/.detect_meteors/detector_plugins``.

    Returns:
        Dictionary mapping plugin_name to detector class.

    Example:
        >>> detectors = discover_detectors()
        >>> print(detectors.keys())
        dict_keys(['hough', ...])
        >>> HoughDetector = detectors['hough']
    """
    warnings.warn(
        "discover_detectors() is deprecated. "
        "Use DetectorRegistry.discover() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _discover_detectors_internal(plugin_dir)


def _discover_detectors_internal(
    plugin_dir: Path | None = None,
) -> Dict[str, Type[BaseDetector]]:
    """Internal discovery function used by DetectorRegistry.

    This is the core discovery implementation. It discovers detectors from:
    1. Built-in detectors (e.g., HoughDetector)
    2. Entry points (detect_meteors.detector group)
    3. Plugin directory

    Args:
        plugin_dir: Optional custom plugin directory. If None, uses PLUGIN_DIR.

    Returns:
        Dictionary mapping plugin_name to detector class.
    """
    directory = plugin_dir if plugin_dir is not None else PLUGIN_DIR

    registry: Dict[str, Type[BaseDetector]] = {}

    # 1. Register built-in detectors first
    _add_detector(registry, HoughDetector, "built-in HoughDetector")
    _add_detector(
        registry,
        SimpleThresholdDetector,
        "built-in SimpleThresholdDetector",
    )

    # 2. Register detectors from entry points (sorted for determinism)
    for ep in sorted(_iter_entry_points(), key=lambda e: e.name):
        try:
            detector_cls = ep.load()
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                f"Failed to load detector entry point '{ep.name}' "
                f"from {ep.value}: {exc}",
                stacklevel=2,
            )
            continue
        _add_detector(registry, detector_cls, f"entry point {ep.name}")

    # 3. Register detectors from plugin directory
    if directory.exists() and directory.is_dir():
        for path in sorted(directory.glob("*.py")):
            try:
                module = _load_module_from_file(path)
            except Exception as exc:  # pragma: no cover
                warnings.warn(
                    f"Failed to load detector plugin module {path}: {exc}",
                    stacklevel=2,
                )
                continue

            # Inspect all classes defined in the module
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # Only consider classes defined in this module (not imports)
                if obj.__module__ == module.__name__:
                    _add_detector(registry, obj, f"plugin file {path}")

    return registry


__all__ = ["discover_detectors", "PLUGIN_DIR", "PLUGIN_GROUP"]
