# Plugin Author Guide

This guide documents how plugin configuration objects are interpreted by
`PluginRegistryBase._coerce_config()` and offers implementation tips for new
plugins. The behavior is shared by loader, detector, and output handler
registries.

## Supported configuration inputs

`_coerce_config()` converts the configuration passed to a registry `create()`
method into the plugin's `ConfigType`. The conversion attempts run in order:

1. **`None` with `ConfigType`** → Instantiate `ConfigType()` using defaults.
   - Raises `TypeError` when the constructor requires positional arguments.
   - Raises `ValueError` if the constructor raises any other exception.
2. **`None` without `ConfigType`** → Returns `None` unchanged.
3. **Already an instance of `ConfigType`** → Returned unchanged.
4. **`dict` + dataclass `ConfigType`** → Instantiates the dataclass via keyword
   arguments; `TypeError` is raised for missing/unknown fields.
5. **`dict` + Pydantic v2 `ConfigType`** (has `model_validate`) → Validated via
   `model_validate()`; validation errors propagate as `ValueError`.
6. **`dict` + Pydantic v1 `ConfigType`** (has `parse_obj`) → Validated via
   `parse_obj()`; validation errors propagate as `ValueError`.
7. **`dict` + other `ConfigType`** → `TypeError` because the registry cannot
   coerce the mapping to an unsupported class.
8. **Any other input** → Returned as-is; plugins are responsible for type
   checking inside their constructors if needed.

## Recommended ConfigType patterns

- Prefer `@dataclass` with type hints and sensible defaults; this keeps the
  default `ConfigType()` path working and makes dict coercion straightforward.
- For richer validation, use a **Pydantic model** (v1 or v2). Registries call
  `model_validate()` or `parse_obj()` automatically when given a dict.
- Keep `ConfigType` constructors free of required positional-only arguments so
  `ConfigType()` can build a default instance when no config is supplied.

### Dataclass example

```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    output_folder: str = "./candidates"
    debug_folder: str = "./debug"

class MyHandler(BaseOutputHandler):
    plugin_name = "my_handler"
    ConfigType = MyConfig
    def __init__(self, config: MyConfig):
        super().__init__(config)
```

- `create("my_handler", None)` → `MyConfig()` using defaults.
- `create("my_handler", {"output_folder": "/tmp"})` → `MyConfig` built from
  the dict.

### Pydantic example

```python
from pydantic import BaseModel

class MyModel(BaseModel):
    source: str
    threshold: float = 0.5

class MyDetector(BaseDetector):
    plugin_name = "my_detector"
    ConfigType = MyModel
    def __init__(self, config: MyModel):
        super().__init__(config)
```

- `create("my_detector", {"source": "cam0"})` → validated via
  `model_validate()` (Pydantic v2) or `parse_obj()` (v1).
- Passing extra keys in the dict triggers a validation error when the model is
  configured with `extra="forbid"`.

## Plugin lifecycle and hooks

The three plugin kinds share a common lifecycle to keep authoring consistent:

1. **Discovery order** — Registries register built-ins first, then entry points
   (sorted by entry-point name), then files in the user plugin directory
   (alphabetical). Duplicate `plugin_name` values are warned and skipped,
   preserving deterministic resolution.
2. **Config coercion** — Registry `create()`/`create_default()` methods convert
   `dict` inputs into `ConfigType` (dataclass or Pydantic) and call the plugin
   constructor. Supplying `None` defaults to `ConfigType()`; missing
   `ConfigType` values cause `create_default()` to raise for clarity.
3. **Default construction** — All registries expose a consistent
   `create_default(config: ConfigType | dict | None = None)` signature to build
   the default handler with optional overrides via normal coercion rules. The
   output registry also offers `create_default_with_paths()` for the common
   folder override use case while keeping `create_default()` symmetric.

Output handlers provide additional optional progress hooks that other plugin
kinds do not implement:

- `on_candidate_detected(filename, saved, score=0.0, aspect_ratio=0.0)`
- `on_batch_complete(processed_count, detected_count, batch_size)`
- `on_pipeline_complete(total_processed, total_detected, elapsed_seconds)`

These hooks are no-ops by default; override them to integrate notifications or
pipeline metrics without affecting the primary save logic.
