"""Plugin discovery and registration utilities."""

import logging
from importlib import util
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Optional

from detect_meteors import app

LOGGER = logging.getLogger(__name__)
PLUGIN_ENTRYPOINT_GROUP = "detect_meteors.plugins"
_DEFAULT_PLUGIN_FOLDER = "plugins"


def load_plugins(plugin_folder: Optional[Path | str] = None) -> None:
    """Discover and register plugins from entry points and an optional folder."""

    seen_detectors = set(app._DETECTOR_REGISTRY)  # type: ignore[attr-defined]
    seen_preprocessors = set(app._PREPROCESSOR_REGISTRY)  # type: ignore[attr-defined]
    seen_output_writers = set(app._OUTPUT_WRITER_REGISTRY)  # type: ignore[attr-defined]

    for module in _iter_entry_point_modules():
        _register_plugin_module(
            module,
            seen_detectors=seen_detectors,
            seen_preprocessors=seen_preprocessors,
            seen_output_writers=seen_output_writers,
        )

    for module in _iter_folder_modules(plugin_folder):
        _register_plugin_module(
            module,
            seen_detectors=seen_detectors,
            seen_preprocessors=seen_preprocessors,
            seen_output_writers=seen_output_writers,
        )


def _iter_entry_point_modules() -> Iterable[ModuleType]:
    plugin_entry_points = entry_points().select(group=PLUGIN_ENTRYPOINT_GROUP)
    for entry_point in plugin_entry_points:
        try:
            module = entry_point.load()
        except Exception:
            LOGGER.exception("Failed to load plugin entry point '%s'", entry_point.name)
            continue

        if isinstance(module, ModuleType):
            yield module
        else:
            LOGGER.error(
                "Entry point '%s' did not resolve to a module (got %r)",
                entry_point.name,
                module,
            )


def _iter_folder_modules(plugin_folder: Optional[Path | str]) -> Iterable[ModuleType]:
    folder = Path(plugin_folder) if plugin_folder else Path.cwd() / _DEFAULT_PLUGIN_FOLDER
    if not folder.exists():
        return []

    modules = []
    for plugin_file in folder.glob("*.py"):
        if plugin_file.name.startswith("__"):
            continue

        spec = util.spec_from_file_location(plugin_file.stem, plugin_file)
        if not spec or not spec.loader:
            LOGGER.error("Could not create spec for plugin module '%s'", plugin_file)
            continue

        try:
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        except Exception:
            LOGGER.exception("Failed to import plugin module from '%s'", plugin_file)
            continue

    return modules


def _register_plugin_module(
    module: ModuleType,
    *,
    seen_detectors: set[str],
    seen_preprocessors: set[str],
    seen_output_writers: set[str],
) -> None:
    detectors = _get_mapping(module, "DETECTORS")
    preprocessors = _get_mapping(module, "PREPROCESSORS")
    output_writers = _get_mapping(module, "OUTPUT_WRITERS")

    _register_items(
        detectors,
        app.register_detector,
        seen_detectors,
        item_type="detector",
        source=module.__name__,
    )
    _register_items(
        preprocessors,
        app.register_preprocessor,
        seen_preprocessors,
        item_type="preprocessor",
        source=module.__name__,
    )
    _register_items(
        output_writers,
        app.register_output_writer,
        seen_output_writers,
        item_type="output writer",
        source=module.__name__,
    )


def _get_mapping(module: ModuleType, attribute: str) -> Dict[str, object]:
    mapping = getattr(module, attribute, {})
    if mapping is None:
        return {}

    if not isinstance(mapping, dict):
        LOGGER.error(
            "Plugin module '%s' attribute '%s' is not a dict (got %r)",
            module.__name__,
            attribute,
            type(mapping),
        )
        return {}

    return mapping


def _register_items(
    items: Dict[str, object],
    register_func,
    seen: set[str],
    *,
    item_type: str,
    source: str,
) -> None:
    for name, implementation in items.items():
        if name in seen:
            LOGGER.warning("Skipping duplicate %s '%s' from '%s'", item_type, name, source)
            continue

        try:
            register_func(name, implementation)
        except ValueError as exc:
            LOGGER.error(
                "Failed to register %s '%s' from '%s': %s", item_type, name, source, exc
            )
        except Exception:
            LOGGER.exception(
                "Unexpected error registering %s '%s' from '%s'", item_type, name, source
            )
        else:
            seen.add(name)

