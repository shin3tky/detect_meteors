# Plugin Architecture

`detect_meteors` supports pluggable detectors, preprocessors, and output writers. Plugins can be supplied either through Python package entry points or by placing Python modules inside a `plugins/` directory next to the running process.

## Interfaces and metadata

Plugins supply concrete implementations for one or more of the following categories:

- **Detector**: subclass `detect_meteors.app.Detector` and implement `detect(**kwargs) -> int`.
- **Preprocessor**: provide a `preprocess(target_folder: str) -> str` method returning the folder to analyze.
- **Output writer**: implement `write(detected_count: int, warnings) -> Any` to emit results.

Each implementation must expose a `plugin_info` attribute containing a `detect_meteors.app.PluginInfo` with:

- `name`: registry key used for selection.
- `version`: plugin version string.
- `capabilities`: list of capability tags (e.g., `detector`, `preprocessor`, `writer`, `cli_output_writer`).

During registration, lifecycle hooks named `initialize()` and `shutdown()` are invoked if present on the plugin object.

## Module exports and naming

A plugin module should define one or more of these dictionaries:

```python
DETECTORS = {"custom_detector": CustomDetector()}
PREPROCESSORS = {"custom_pre": CustomPreprocessor()}
OUTPUT_WRITERS = {"custom_writer": CustomWriter()}
```

Dictionary keys become registry names and must match the `plugin_info.name` field of the corresponding implementation. Duplicate names are ignored with a warning when discovered in later modules, preserving the first loaded implementation.

## Entry points

Packaged plugins can be discovered via the `detect_meteors.plugins` entry-point group. Each entry point must resolve to a module object exporting the dictionaries above. For example, in `pyproject.toml` or `setup.cfg`:

```toml
[project.entry-points."detect_meteors.plugins"]
my-meteors-plugin = "my_package.plugin_module"
```

## Local plugin folder

When a `plugins/` directory exists in the current working directory (or a custom path is provided to `detect_meteors.plugin_loader.load_plugins`), every `*.py` file within it is treated as a plugin module unless it starts with `__`. Modules that fail to import are logged and skipped without interrupting discovery.

## Capability metadata

Registry lookups preserve the attached `PluginInfo`, allowing callers to inspect names, versions, and declared capabilities (used by the `--list-plugins` CLI output). Ensure capability tags accurately describe provided behavior so users can select appropriate plugins.

## Safety guidelines

- Keep plugin dependencies isolated to avoid interfering with the main application environment.
- Handle external resources defensively (validate paths, sanitize inputs, and close files or network handles).
- Avoid long-running work inside module import; perform costly initialization in `initialize()`.
- Fail gracefully: raise descriptive exceptions during `detect`, `preprocess`, or `write` so errors can be surfaced to users.
- Respect the registry contract: do not mutate registries directly—use `register_*` helpers for overrides and clean shutdowns.
