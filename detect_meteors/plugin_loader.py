"""Plugin discovery and registration utilities."""

import logging
from importlib import util
from importlib.metadata import entry_points
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, TypedDict

from detect_meteors import app

LOGGER = logging.getLogger(__name__)
PLUGIN_ENTRYPOINT_GROUP = "detect_meteors.plugins"
_DEFAULT_PLUGIN_FOLDER = "plugins"


class SkippedPlugin(TypedDict):
    name: Optional[str]
    item_type: Optional[str]
    source: str
    reason: str


class PluginLoadResult(TypedDict):
    loaded: Dict[str, List[str]]
    skipped: List[SkippedPlugin]


def load_plugins(
    plugin_folder: Optional[Path | str] = None,
    *,
    entrypoint_group: Optional[str] = None,
) -> PluginLoadResult:
    """Discover and register plugins from entry points and an optional folder."""

    seen_detectors = set(app._DETECTOR_REGISTRY)  # type: ignore[attr-defined]
    seen_preprocessors = set(app._PREPROCESSOR_REGISTRY)  # type: ignore[attr-defined]
    seen_output_writers = set(app._OUTPUT_WRITER_REGISTRY)  # type: ignore[attr-defined]

    load_result: PluginLoadResult = {
        "loaded": {"detectors": [], "preprocessors": [], "output_writers": []},
        "skipped": [],
    }

    group = entrypoint_group or PLUGIN_ENTRYPOINT_GROUP

    modules = list(_iter_entry_point_modules(group, load_result["skipped"]))
    modules.extend(_iter_folder_modules(plugin_folder, load_result["skipped"]))

    for module in modules:
        _register_plugin_module(
            module,
            seen_detectors=seen_detectors,
            seen_preprocessors=seen_preprocessors,
            seen_output_writers=seen_output_writers,
            load_result=load_result,
        )

    if not app._DETECTOR_REGISTRY and not app._PREPROCESSOR_REGISTRY and not app._OUTPUT_WRITER_REGISTRY:  # type: ignore[attr-defined]
        _load_builtin_plugins(
            seen_detectors=seen_detectors,
            seen_preprocessors=seen_preprocessors,
            seen_output_writers=seen_output_writers,
            load_result=load_result,
        )

    return load_result


def _iter_entry_point_modules(group: str, skipped: List[SkippedPlugin]) -> Iterable[ModuleType]:
    plugin_entry_points = entry_points().select(group=group)
    for entry_point in plugin_entry_points:
        try:
            module = entry_point.load()
        except Exception as exc:
            reason = f"failed to load entry point '{entry_point.name}': {exc}"
            LOGGER.exception("Failed to load plugin entry point '%s'", entry_point.name)
            skipped.append(
                {
                    "name": entry_point.name,
                    "item_type": None,
                    "source": f"entrypoint:{group}",
                    "reason": reason,
                }
            )
            continue

        if isinstance(module, ModuleType):
            yield module
        else:
            LOGGER.error(
                "Entry point '%s' did not resolve to a module (got %r)",
                entry_point.name,
                module,
            )
            skipped.append(
                {
                    "name": entry_point.name,
                    "item_type": None,
                    "source": f"entrypoint:{group}",
                    "reason": f"entry point '{entry_point.name}' did not resolve to module",
                }
            )


def _iter_folder_modules(
    plugin_folder: Optional[Path | str], skipped: List[SkippedPlugin]
) -> Iterable[ModuleType]:
    folder = Path(plugin_folder) if plugin_folder else Path.cwd() / _DEFAULT_PLUGIN_FOLDER
    if not folder.exists():
        return []

    modules = []
    skipped_entries: list[str] = []
    for plugin_file in folder.glob("*.py"):
        if plugin_file.name.startswith("__"):
            skipped.append(
                {
                    "name": None,
                    "item_type": None,
                    "source": str(plugin_file),
                    "reason": "filename starts with '__'",
                }
            )
            skipped_entries.append(f"{plugin_file.name}: starts with '__'")
            continue

        spec = util.spec_from_file_location(plugin_file.stem, plugin_file)
        if not spec or not spec.loader:
            LOGGER.error("Could not create spec for plugin module '%s'", plugin_file)
            skipped.append(
                {
                    "name": None,
                    "item_type": None,
                    "source": str(plugin_file),
                    "reason": "could not create import spec",
                }
            )
            skipped_entries.append(f"{plugin_file.name}: no import spec")
            continue

        try:
            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        except Exception as exc:
            LOGGER.exception("Failed to import plugin module from '%s'", plugin_file)
            skipped.append(
                {
                    "name": None,
                    "item_type": None,
                    "source": str(plugin_file),
                    "reason": f"import error: {exc}",
                }
            )
            skipped_entries.append(f"{plugin_file.name}: import error ({exc})")
            continue

    if skipped_entries:
        LOGGER.warning(
            "Skipped plugin files in %s: %s", folder, "; ".join(skipped_entries)
        )

    return modules


def _load_builtin_plugins(
    *,
    seen_detectors: set[str],
    seen_preprocessors: set[str],
    seen_output_writers: set[str],
    load_result: PluginLoadResult,
) -> None:
    try:
        from detect_meteors import builtin_plugins
    except Exception:
        LOGGER.exception("Failed to import built-in plugins module")
        return

    LOGGER.info("Falling back to built-in detect_meteors plugins")
    _register_plugin_module(
        builtin_plugins,
        seen_detectors=seen_detectors,
        seen_preprocessors=seen_preprocessors,
        seen_output_writers=seen_output_writers,
        load_result=load_result,
    )


def _register_plugin_module(
    module: ModuleType,
    *,
    seen_detectors: set[str],
    seen_preprocessors: set[str],
    seen_output_writers: set[str],
    load_result: PluginLoadResult,
) -> None:
    detectors = _get_mapping(module, "DETECTORS")
    preprocessors = _get_mapping(module, "PREPROCESSORS")
    output_writers = _get_mapping(module, "OUTPUT_WRITERS")

    _register_items(
        detectors,
        app.register_detector,
        seen_detectors,
        item_type="detector",
        registry_key="detectors",
        source=module.__name__,
        load_result=load_result,
    )
    _register_items(
        preprocessors,
        app.register_preprocessor,
        seen_preprocessors,
        item_type="preprocessor",
        registry_key="preprocessors",
        source=module.__name__,
        load_result=load_result,
    )
    _register_items(
        output_writers,
        app.register_output_writer,
        seen_output_writers,
        item_type="output writer",
        registry_key="output_writers",
        source=module.__name__,
        load_result=load_result,
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
    registry_key: str,
    source: str,
    load_result: PluginLoadResult,
) -> None:
    for name, implementation in items.items():
        if name in seen:
            reason = "duplicate name already registered"
            LOGGER.warning("Skipping duplicate %s '%s' from '%s'", item_type, name, source)
            load_result["skipped"].append(
                {
                    "name": name,
                    "item_type": item_type,
                    "source": source,
                    "reason": reason,
                }
            )
            continue

        try:
            register_func(name, implementation)
        except ValueError as exc:
            LOGGER.error(
                "Failed to register %s '%s' from '%s': %s", item_type, name, source, exc
            )
            load_result["skipped"].append(
                {
                    "name": name,
                    "item_type": item_type,
                    "source": source,
                    "reason": str(exc),
                }
            )
        except Exception:
            LOGGER.exception(
                "Unexpected error registering %s '%s' from '%s'", item_type, name, source
            )
            load_result["skipped"].append(
                {
                    "name": name,
                    "item_type": item_type,
                    "source": source,
                    "reason": "unexpected error during registration",
                }
            )
        else:
            seen.add(name)
            load_result["loaded"][registry_key].append(name)

