# Plugin Author Guide

> ⚠️ **Experimental**: The plugin architecture is under active development and **may undergo breaking changes before the v2.0 stable release**.
>
> **Current status (v1.6.4)**:
>
> - ✅ Registry system and base classes are stable
> - ✅ Input Loaders, Detectors, Output Handlers work as documented
> - ✅ `on_detection_result` and `on_candidate_detected` hooks are invoked (v1.6.4)
> - ⚠️ Detector/runtime parameter contracts may still evolve
> - ⚠️ `on_batch_complete` and `on_pipeline_complete` hooks reserved for future releases

This guide provides comprehensive instructions for developing custom plugins for Detect Meteors CLI.

---

## Architecture Overview

Before diving into the details, it helps to understand the overall architecture. The plugin system is designed with **loose coupling** between three distinct layers, each with clearly defined responsibilities and data contracts.

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Plugin Architecture                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐          │
│  │   Input Layer     │   │  Detection Layer  │   │   Output Layer    │          │
│  │  (Input Loaders)  │──▶│    (Detectors)    │──▶│ (Output Handlers) │          │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘          │
│          │                        │                        │                    │
│          ▼                        ▼                        ▼                    │
│   ┌─────────────┐         ┌───────────────┐        ┌─────────────┐              │
│   │InputContext │         │DetectionResult│        │OutputResult │              │
│   └─────────────┘         └───────────────┘        └─────────────┘              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

The pipeline processes image pairs (current frame, previous frame) to detect meteor candidates:

```
filepath ──▶ Input Loader ──▶ InputContext
                                    │
                                    ▼
                            ┌────────────────┐
                            │DetectionContext│  (current + previous InputContext + ROI + params)
                            └────────────────┘
                                    │
                                    ▼
                              Detector
                                    │
                                    ▼
                            ┌────────────────┐
                            │DetectionResult │  (is_candidate, score, lines, debug_image, ...)
                            └────────────────┘
                                    │
                                    ▼
                            Output Handler
                                    │
                                    ▼
                            ┌───────────────┐
                            │ OutputResult  │  (saved, output_path, debug_path, ...)
                            └───────────────┘
```

### Layer Responsibilities

| Layer | Base Class | Input | Output | Responsibility |
|-------|------------|-------|--------|----------------|
| **Input** | `BaseInputLoader` | `filepath` | `InputContext` | Load images from various formats (CR2, ARW, DNG, TIFF, FITS, ...), extract metadata |
| **Detection** | `BaseDetector` | `DetectionContext` | `DetectionResult` | Analyze image pairs to detect meteor candidates (frame differencing, Hough transform, ML, ...) |
| **Output** | `BaseOutputHandler` | `DetectionResult` | `OutputResult` | Save results, generate reports, send notifications (file, cloud, Slack, database, ...) |

### Benefits of Loose Coupling

This architecture provides significant flexibility:

- **Input Layer**: Detectors never need to know about file formats or image loading libraries. Whether you use rawpy, OpenCV, or a custom FITS reader, the detector simply receives a normalized `InputContext`.

- **Detection Layer**: The entire detection algorithm (preprocessing → analysis → scoring) is encapsulated. You can replace the built-in Hough transform approach with deep learning, morphological analysis, or video-based detection—the input and output layers remain unchanged.

- **Output Layer**: Storage destinations can be changed from local disk to cloud storage (S3, GCS), databases, or notification services (Slack, Discord) without affecting detection logic.

Each layer communicates only through well-defined dataclasses (`InputContext`, `DetectionContext`, `DetectionResult`, `OutputResult`), ensuring that internal implementation changes don't break the pipeline.

---

## Table of Contents

0. [Architecture Overview](#architecture-overview)
1. [Application Lifecycle](#1-application-lifecycle)
2. [Extension Points](#2-extension-points)
3. [Plugin Architecture](#3-plugin-architecture)
4. [Data Contracts Reference](#4-data-contracts-reference)
5. [Sample Code](#5-sample-code)
6. [Best Practices](#6-best-practices)
7. [Step-by-Step Tutorial](#7-step-by-step-tutorial)

---

## 1. Application Lifecycle

Understanding the detection pipeline lifecycle is essential for effective plugin development.

### 1.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Detection Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ 1. Initialize│───▶│ 2. Collect   │───▶│ 3. ROI Selection     │   │
│  │    Pipeline  │    │    Files     │    │    (if enabled)      │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                      │              │
│                                                      ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ 6. Finalize  │◀───│ 5. Save      │◀───│ 4. Process Batches   │   │
│  │    & Report  │    │    Results   │    │    (parallel/seq)    │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Processing Flow Detail

```
For each image pair (current, previous):

    ┌─────────────────────────────────────────────────────────────┐
    │                    Input Loader                             │
    │  • Load current image  ───▶  InputContext                   │
    │  • Load previous image ───▶  InputContext                   │
    │  • Extract metadata (optional)                              │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      Detector                               │
    │  • Compute frame difference                                 │
    │  • Apply ROI mask                                           │
    │  • Detect meteor candidates                                 │
    │  • Generate debug visualization                             │
    │                                                             │
    │  Returns: DetectionResult                                   │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   Output Handler                            │
    │  • save_candidate() ── Save detected meteor image           │
    │  • save_debug_image() ── Save debug visualization           │
    │                                                             │
    │  Returns: OutputResult                                      │
    │  Lifecycle Hooks (Output Handlers):                         │
    │  • on_detection_result(context)                             │
    │  • on_candidate_detected()                                  │
    │  • on_batch_complete()                                      │
    │  • on_pipeline_complete()                                   │
    └─────────────────────────────────────────────────────────────┘
```

### 1.3 Lifecycle Events (Output Handler Only)

Output Handlers define lifecycle hooks. The pipeline now invokes per-frame hooks
(`on_detection_result`, `on_candidate_detected`), while batch/pipeline hooks are
reserved for future releases. The `on_detection_result` hook receives a
serialized context payload (from `DetectionContext.to_dict()`), not raw image
arrays.

| Event | Current Status | Intended Use Case |
|-------|----------------|-------------------|
| `on_detection_result` | Invoked | Per-frame inspection, logging, telemetry |
| `on_candidate_detected` | Invoked | Real-time notifications (Slack, webhook) |
| `on_batch_complete` | Not invoked | Progress reporting, metrics collection |
| `on_pipeline_complete` | Not invoked | Final summary, cleanup, reporting |

**Important**: Input Loaders and Detectors do **not** receive lifecycle events.

---

## 2. Extension Points

The plugin system provides three extension points:

### 2.1 Input Loaders

**Purpose**: Load images from various file formats

**When to create**:
- Support a new image format (TIFF, FITS, etc.)
- Apply pre-processing during load (debayer, normalize)
- Extract custom metadata

**Required methods**:
| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `(filepath: str) -> InputContext` | Load image data plus metadata and loader info |

**InputContext type** (return value of `load`):
```python
ImageLike = Union[np.ndarray, "torch.Tensor", "PIL.Image.Image"]

@dataclass
class InputContext:
    """Input bundle for loader execution."""

    image_data: ImageLike
    filepath: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    loader_info: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1                             # INPUT_CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging (excludes image_data)."""
        ...
```

**Fields**:
- `image_data`: Loaded image pixels. This is the payload used by the detector.
- `filepath`: Original path of the loaded image.
- `metadata`: Loader-provided metadata (EXIF, timestamps, camera info, etc.).
- `loader_info`: Loader identity details (from `BaseInputLoader.get_info()`).
- `schema_version`: Contract version for future migrations (current: `1`).

**Schema versioning**: The `schema_version` field enables future migration of loader plugins without breaking changes. When the schema evolves (e.g., new required fields), the version increments, allowing loaders to handle different versions gracefully. Current version is `1`.

**Normalization point**: The pipeline calls `meteor_core.schema.normalize_input_context` immediately after `load` returns. If `schema_version` is older, the pipeline uses any converter registered via `meteor_core.schema.register_input_context_converter`; otherwise it rejects the input with a configuration error.

**Serialization**: Use `context.to_dict()` to get a JSON-serializable representation (excludes `image_data` to avoid large binary data in logs).

**Optional features**:
- Implement `BaseMetadataExtractor` for EXIF-like metadata extraction
- Define `name`, `version` attributes for plugin info

**BaseMetadataExtractor mixin**:

Implement this optional interface to extract metadata (EXIF, timestamps, camera info) from image files:

```python
class MyLoader(DataclassInputLoader[MyConfig], BaseMetadataExtractor):
    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        # Return metadata dictionary
        return {"timestamp": ..., "camera": ..., "exposure": ...}
```

**When is `extract_metadata` called?**
- The pipeline calls it for each image pair; if your loader does **not** implement
  `BaseMetadataExtractor`, the pipeline falls back to `meteor_core.image_io.extract_exif_metadata`.
- Detectors receive metadata as `context.metadata = {"current": ..., "previous": ...}`.
- You can also call it manually for custom processing.
- If extraction fails, return `{}` to keep the pipeline moving.

### 2.2 Detectors

**Purpose**: Implement meteor detection algorithms

**When to create**:
- Use different detection approach (ML-based, morphological)
- Optimize for specific conditions (bright meteors, fireball detection)
- Add custom scoring logic

**Required methods**:
| Method | Signature | Description |
|--------|-----------|-------------|
| `detect` | `(context: DetectionContext) -> DetectionResult` | Main detection (see below) |
| `compute_line_score` | `(mask, hough_params) -> Tuple[float, List]` | Line scoring (called internally by `detect`) |

**DetectionContext type** (input to `detect`):
```python
ImageLike = Union[np.ndarray, "torch.Tensor", "PIL.Image.Image"]

@dataclass
class DetectionContext:
    """Input bundle for detector execution."""

    current_image: ImageLike
    previous_image: ImageLike
    roi_mask: Any                                       # Typically np.ndarray (uint8 mask)
    runtime_params: Union["RuntimeParams", Dict[str, Any]]
    metadata: Dict[str, Any]                            # {"current": {...}, "previous": {...}} in pipeline
    schema_version: int = 1                             # DETECTION_CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging (excludes current_image/previous_image/roi_mask)."""
        ...
```

**Schema versioning**: The `schema_version` field enables future migration of detector plugins without breaking changes. When the schema evolves (e.g., new required fields), the version increments, allowing detectors to handle different versions gracefully. Current version is `1`.

**Normalization point**: The pipeline normalizes `DetectionContext` internally before passing it to your `detect` method. This normalization is handled by a pipeline-internal function (`_normalize_detection_context`), so plugin authors do not need to call it directly. Unlike `InputContext`, `DetectionResult`, and `OutputResult`, there is no public `normalize_detection_context` function or converter registration API exposed by `meteor_core.schema`.

`current_image` and `previous_image` are typically `numpy.ndarray` today, but can
also be provided as `torch.Tensor` or `PIL.Image.Image` for ML-based detectors.
If you rely on specific array operations, normalize these inputs at the start
of your detector implementation. The helper `meteor_core.utils.ensure_numpy`
converts `numpy.ndarray`, `torch.Tensor`, and `PIL.Image.Image` into a
`numpy.ndarray`. If you prefer working with PyTorch, `meteor_core.utils.ensure_tensor`
performs the same normalization into a `torch.Tensor`.

**DetectionResult type** (return value of `detect`):
```python
@dataclass
class DetectionResult:
    """Result returned by detectors.

    Standard diagnostics belong in ``metrics`` (e.g. ``duration_ms``,
    ``num_contours``, ``mask_area``, ``hough_votes``). Use ``extras`` for
    detector-specific or auxiliary data that should not be part of the
    normalized comparison surface.
    """

    is_candidate: bool
    score: float
    lines: List[Tuple[int, int, int, int]]
    aspect_ratio: float
    debug_image: Optional[Any]                          # Typically np.ndarray (BGR)
    extras: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1                             # DETECTION_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for JSON/logging (excludes debug_image)."""
        ...
```

**Schema versioning**: Like `DetectionContext`, the `schema_version` enables forward-compatible result handling. Downstream consumers can check the version before processing.

**Normalization point**: The pipeline calls `meteor_core.schema.normalize_detection_result` right after your `detect` implementation returns. If `schema_version` is older, it applies converters registered with `meteor_core.schema.register_detection_result_converter`; if no converter exists, the pipeline rejects the result.

**Normalized vs detector-specific outputs**

`lines` remains the normalized, line-segment-centric output for downstream
consumers. Detectors that do not produce line segments should still return
`lines=[]` and place non-linear detections into `extras`.

Recommended `extras` keys for non-linear detections:
- `bounding_boxes`: List of rects, e.g. `[{x1, y1, x2, y2}, ...]`
- `polygons`: List of polygons, e.g. `[[[x, y], [x, y], ...], ...]`
- `masks`: Detector-specific masks (numpy arrays or references/paths)

**Standard diagnostics** (`DetectionResult.metrics`):

Use `metrics` to emit stable, comparable diagnostics across detectors. These
entries are intended for downstream analysis and visualization tools, while
`extras` should hold detector-specific or auxiliary data.

Recommended keys:
- `duration_ms`: Total detection wall time for the call.
- `num_contours`: Number of contours found in the binary mask.
- `mask_area`: Non-zero pixel count in the mask used for line/contour analysis.
- `hough_votes`: Hough line evidence count (e.g., number of detected lines).

**Serialization**: Use `result.to_dict()` to get a JSON-serializable representation (excludes `debug_image` to avoid large binary data in logs).

**Runtime parameters** (`context.runtime_params`):

Runtime parameters are carried as a `RuntimeParams` dataclass
(`meteor_core.schema.RuntimeParams`):

```python
@dataclass
class RuntimeParams:
    """Runtime parameters passed into detector execution."""

    schema_version: int = 1                             # RUNTIME_PARAMS_SCHEMA_VERSION
    global_params: Dict[str, Any] = field(default_factory=dict)
    detector: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self, include_schema_version: bool = True) -> Dict[str, Any]:
        """Serialize to dict for JSON/logging."""
        ...
```

The serialized shape (via `to_dict()`) is:

```python
{
    "schema_version": 1,
    "global": { ... },        # Pipeline-wide params
    "detector": {
        "<plugin_name>": { ... }  # Detector-specific overrides
    },
}
```

**Versioning policy**:
- `schema_version` increments only when the structure above changes in a
  backward-incompatible way.
- New optional keys may be added without bumping the version as long as
  existing keys remain valid.

**Compatibility rules**:
- `context.runtime_params` may be either a `RuntimeParams` instance or a plain
  dict with the same keys.
- Legacy detectors may still receive a flat dict; prefer reading from the
  namespaced structure when available.

`BaseDetector` provides helpers to make this easy:
- `split_runtime_params(runtime_params)` → `(global_params, detector_params)`
- `build_runtime_params(flat_params)` → `RuntimeParams`
- `detect_legacy(current_image, previous_image, roi_mask, params)` → adapter for
  the old signature

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `diff_threshold` | `int` | `8` | Frame difference threshold (scale matches input dtype) |
| `min_area` | `int` | `10` | Minimum contour area in pixels |
| `min_line_score` | `float` | `30.0` | Minimum score to classify as candidate |
| `min_aspect_ratio` | `float` | `2.0` | Minimum contour aspect ratio |
| `hough_threshold` | `int` | `50` | Hough transform vote threshold |
| `hough_min_line_length` | `int` | `50` | Minimum line length in pixels |
| `hough_max_line_gap` | `int` | `10` | Maximum gap between line segments |

**Note**: `compute_line_score` is a helper method typically called within your `detect` implementation. The pipeline calls only `detect`.

### 2.3 Output Handlers

**Purpose**: Save results and send notifications

**When to create**:
- Upload to cloud storage (S3, GCS)
- Send notifications (Slack, Discord, email)
- Store in database
- Generate custom reports

**Required methods**:
| Method | Signature | Description |
|--------|-----------|-------------|
| `save_candidate` | `(source_path, filename, ...) -> OutputResult` | Save meteor candidate |
| `save_debug_image` | `(debug_image, filename, ...) -> str` | Save debug image |

**OutputResult type** (return value of `save_candidate`):
```python
@dataclass
class OutputResult:
    """Result returned by output handlers."""

    saved: bool
    output_path: Optional[str]
    debug_path: Optional[str]
    handler_info: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1                             # OUTPUT_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for JSON/logging."""
        ...
```

**Fields**:
- `saved`: True if the handler persisted the candidate successfully.
- `output_path`: Location of the persisted candidate (if any).
- `debug_path`: Location of the persisted debug image (if any).
- `handler_info`: Handler identity details (from `BaseOutputHandler.get_info()`).
- `metrics`: Stable diagnostics (duration, bytes written, upload timings, etc.).
- `schema_version`: Contract version for future migrations (current: `1`).

**Schema versioning**: The `schema_version` field enables future migration of handler plugins without breaking changes. When the schema evolves (e.g., new required fields), the version increments, allowing handlers to handle different versions gracefully. Current version is `1`.

**Normalization point**: The pipeline calls `meteor_core.schema.normalize_output_result` immediately after `save_candidate` returns. If `schema_version` is older, it uses converters registered via `meteor_core.schema.register_output_result_converter`; without a converter, the pipeline rejects the result.

**Serialization**: Use `result.to_dict()` to get a JSON-serializable representation for logging and debugging.

**Lifecycle hooks (optional)**:
| Hook | Signature |
|------|-----------|
| `on_detection_result` | `(context, result, filepath) -> None` |
| `on_candidate_detected` | `(filename, saved, score, aspect_ratio) -> None` |
| `on_batch_complete` | `(processed_count, detected_count, batch_size) -> None` |
| `on_pipeline_complete` | `(total_processed, total_detected, elapsed_seconds) -> None` |

**Invocation order (per frame)**:
1. `on_detection_result()` — called immediately after the detector returns and the pipeline normalizes `DetectionResult`. The `context` parameter is the result of `DetectionContext.to_dict()` and excludes image/ROI arrays.
2. `save_candidate()` — only if `result.is_candidate` is `True`.
3. `on_candidate_detected()` — called after `save_candidate()` returns (with `saved` reflecting the output decision).

---

## 3. Plugin Architecture

### 3.1 Registry System

Each plugin type has its own registry:

```python
from meteor_core.inputs import LoaderRegistry
from meteor_core.detectors import DetectorRegistry
from meteor_core.outputs import OutputHandlerRegistry
```

**Registry Operations**:
```python
# Register a plugin class
LoaderRegistry.register(MyLoader)

# Get plugin class by name (case-insensitive)
loader_cls = LoaderRegistry.get("my_loader")
loader_cls = LoaderRegistry.get("MY_LOADER")  # Same result

# Create configured instance
loader = LoaderRegistry.create("my_loader", {"option": "value"})

# Create default instances (requires ConfigType with zero-arg defaults)
detector = DetectorRegistry.create_default()
handler = OutputHandlerRegistry.create_default(
    output_folder="./candidates",
    debug_folder="./debug_masks",
)

# List available plugins
names = LoaderRegistry.list_available()  # ["raw", "my_loader", ...]

# Trigger discovery (automatic on first use)
LoaderRegistry.discover()
```

### 3.2 Base Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input Loaders                              │
├─────────────────────────────────────────────────────────────────┤
│  BaseInputLoader (ABC)                                          │
│  ├── DataclassInputLoader[ConfigType] ── Dataclass config       │
│  │   └── RawImageLoader ── Built-in RAW loader                  │
│  └── PydanticInputLoader[ConfigType] ── Pydantic config         │
│                                                                 │
│  BaseMetadataExtractor (ABC) ── Optional mixin for metadata     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Detectors                                │
├─────────────────────────────────────────────────────────────────┤
│  BaseDetector (ABC)                                             │
│  ├── DataclassDetector[ConfigType] ── Dataclass config          │
│  │   ├── HoughDetector ── Built-in Hough detector               │
│  │   └── SimpleThresholdDetector ── Built-in threshold detector │
│  └── PydanticDetector[ConfigType] ── Pydantic config            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Output Handlers                             │
├─────────────────────────────────────────────────────────────────┤
│  BaseOutputHandler (ABC) ── Includes lifecycle hooks            │
│  ├── DataclassOutputHandler[ConfigType] ── Dataclass config     │
│  │   └── FileOutputHandler ── Built-in file handler             │
│  └── PydanticOutputHandler[ConfigType] ── Pydantic config       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Configuration Management (ConfigType)

Plugins can define a `ConfigType` for typed configuration. See [5.1 Choosing ConfigType](#51-choosing-configtype) for guidance on when to use dataclass vs Pydantic.

**Coercion Rules**:
| Input | ConfigType | Result |
|-------|------------|--------|
| `None` | Defined | `ConfigType()` with defaults |
| `None` | Not defined | `None` |
| ConfigType instance | — | Used as-is |
| `dict` | Dataclass | `ConfigType(**dict)` |
| `dict` | Pydantic v2 | `ConfigType.model_validate(dict)` |
| `dict` | Pydantic v1 | `ConfigType.parse_obj(dict)` |
| Other | — | Passed as-is |

**Error Handling**:
- `TypeError`: Missing required fields, wrong type
- `ValueError`: Validation failed (Pydantic)

**Default instance requirement**:
`create_default()` for the built-in loader/detector/handler assumes your
`ConfigType()` constructor yields a complete default configuration. If it does
not, `create_default()` raises a `TypeError` to avoid silently creating a
misconfigured plugin.

### 3.4 Plugin Discovery

Plugins are discovered in this order (duplicates warn but don't overwrite):

1. **Built-in plugins** (RawImageLoader, HoughDetector, SimpleThresholdDetector, FileOutputHandler)
2. **Entry points** (sorted alphabetically by name)
3. **Plugin directories** (sorted alphabetically by filename)
4. **Runtime registrations** via `Registry.register()` (overrides discovered entries)

**Plugin Directories**:
| Plugin Type | Directory |
|-------------|-----------|
| Input Loaders | `~/.detect_meteors/input_plugins/` |
| Detectors | `~/.detect_meteors/detector_plugins/` |
| Output Handlers | `~/.detect_meteors/output_plugins/` |

**How plugin directory discovery works**:
1. Place your `.py` file in the appropriate directory (create it if needed).
2. The registry loads all `.py` files alphabetically on first access.
3. Any class defined in the module that subclasses the correct base and has
   `plugin_name` is auto-registered (you do **not** need to call
   `Registry.register()`).
4. No special naming convention required, but descriptive names help organization.

Example file structure:
```
~/.detect_meteors/
└── input_plugins/
    ├── fits_loader.py      # Defines: class FitsLoader(DataclassInputLoader)
    └── tiff_loader.py      # Defines: class TiffLoader(DataclassInputLoader)
```

**Entry Points** (in `pyproject.toml`):
```toml
[project.entry-points."detect_meteors.input"]
my_loader = "my_package.loaders:MyLoader"

[project.entry-points."detect_meteors.detector"]
my_detector = "my_package.detectors:MyDetector"

[project.entry-points."detect_meteors.output"]
my_handler = "my_package.handlers:MyHandler"
```

---

## 4. Data Contracts Reference

This section provides a comprehensive reference for the dataclasses used in the plugin system. All dataclasses are imported from `meteor_core.schema`.

```python
from meteor_core.schema import (
    InputContext,
    DetectionContext,
    DetectionResult,
    OutputResult,
    RuntimeParams,
)
```

### 4.1 InputContext

`InputContext` bundles the loaded image data with metadata for downstream processing.

**Import**: `from meteor_core.schema import InputContext`

```python
@dataclass
class InputContext:
    """Input bundle for loader execution."""

    image_data: ImageLike                               # Loaded image pixels
    filepath: str                                       # Original file path
    metadata: Dict[str, Any] = field(default_factory=dict)
    loader_info: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = INPUT_CONTEXT_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...
```

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image_data` | `ImageLike` | — | Loaded image pixels. Accepts `np.ndarray`, `torch.Tensor`, or `PIL.Image.Image`. |
| `filepath` | `str` | — | Original path of the loaded image file. |
| `metadata` | `Dict[str, Any]` | `{}` | Loader-provided metadata (EXIF, timestamps, camera info, etc.). |
| `loader_info` | `Dict[str, Any]` | `{}` | Loader identity details from `BaseInputLoader.get_info()`. |
| `schema_version` | `int` | `1` | Contract version for future migrations. |

**Usage Example**:

```python
from meteor_core.schema import InputContext
from meteor_core.inputs import DataclassInputLoader

class MyLoader(DataclassInputLoader[MyConfig]):
    def load(self, filepath: str) -> InputContext:
        image = self._load_image(filepath)
        return InputContext(
            image_data=image,
            filepath=filepath,
            metadata={"camera": "Canon EOS R5", "iso": 6400},
            loader_info=self.get_info(),
        )
```

**Serialization**: `context.to_dict()` returns a JSON-serializable dict (excludes `image_data` to avoid large binary data in logs).

### 4.2 DetectionContext

`DetectionContext` bundles all inputs required for detector execution.

**Import**: `from meteor_core.schema import DetectionContext`

```python
@dataclass
class DetectionContext:
    """Input bundle for detector execution."""

    current_image: ImageLike                            # Current frame
    previous_image: ImageLike                           # Previous frame for differencing
    roi_mask: Any                                       # ROI mask (typically np.ndarray uint8)
    runtime_params: Union[RuntimeParams, Dict[str, Any]]
    metadata: Dict[str, Any]                            # {"current": {...}, "previous": {...}}
    schema_version: int = DETECTION_CONTEXT_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...
```

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `current_image` | `ImageLike` | — | Current frame to analyze. |
| `previous_image` | `ImageLike` | — | Previous frame for frame differencing. |
| `roi_mask` | `Any` | — | ROI mask (typically `np.ndarray` with dtype `uint8`). |
| `runtime_params` | `RuntimeParams \| Dict` | — | Runtime parameters (see below). |
| `metadata` | `Dict[str, Any]` | — | Metadata dict with keys `"current"` and `"previous"` containing per-frame metadata. |
| `schema_version` | `int` | `1` | Contract version for future migrations. |

**RuntimeParams Structure**:

```python
@dataclass
class RuntimeParams:
    schema_version: int = 1
    global_params: Dict[str, Any] = field(default_factory=dict)  # Pipeline-wide params
    detector: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Per-detector overrides
```

Serialized form via `to_dict()`:
```python
{
    "schema_version": 1,
    "global": {"diff_threshold": 8, "min_area": 10, ...},
    "detector": {"hough": {"hough_threshold": 50}, ...},
}
```

**Usage Example**:

```python
from meteor_core.schema import DetectionContext, DetectionResult
from meteor_core.detectors import DataclassDetector
from meteor_core.utils import ensure_numpy

class MyDetector(DataclassDetector[MyConfig]):
    def detect(self, context: DetectionContext) -> DetectionResult:
        # Normalize images to numpy arrays
        current = ensure_numpy(context.current_image)
        previous = ensure_numpy(context.previous_image)
        roi_mask = ensure_numpy(context.roi_mask)

        # Extract runtime params
        global_params, detector_params = self.split_runtime_params(
            context.runtime_params
        )
        params = {**global_params, **detector_params}

        # Access per-frame metadata
        current_meta = context.metadata.get("current", {})
        previous_meta = context.metadata.get("previous", {})

        # Perform detection...
        return DetectionResult(...)
```

**Serialization**: `context.to_dict()` returns a JSON-serializable dict (excludes `current_image`, `previous_image`, and `roi_mask`). The pipeline uses this payload when invoking `on_detection_result()` to avoid transferring large image arrays.

**Normalization**: Unlike `InputContext` and `OutputResult`, the normalization of `DetectionContext` is handled internally by the pipeline (via `_normalize_detection_context` in `meteor_core.pipeline`). Plugin authors do not need to call any normalization function for `DetectionContext`, and there is no public `normalize_detection_context()` or `register_detection_context_converter()` function exposed by `meteor_core.schema`.

### 4.3 DetectionResult

`DetectionResult` encapsulates the output of a detector's `detect()` method.

**Import**: `from meteor_core.schema import DetectionResult`

```python
@dataclass
class DetectionResult:
    """Result returned by detectors.

    Standard diagnostics belong in ``metrics`` (e.g. ``duration_ms``,
    ``num_contours``, ``mask_area``, ``hough_votes``). Use ``extras`` for
    detector-specific or auxiliary data that should not be part of the
    normalized comparison surface.
    """

    is_candidate: bool                                  # Detection decision
    score: float                                        # Detection confidence score
    lines: List[Tuple[int, int, int, int]]              # Detected line segments
    aspect_ratio: float                                 # Max contour aspect ratio
    debug_image: Optional[Any]                          # Debug visualization (BGR)
    extras: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = DETECTION_RESULT_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...
```

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `is_candidate` | `bool` | — | `True` if the frame contains a meteor candidate. |
| `score` | `float` | — | Detection confidence score (higher = more confident). |
| `lines` | `List[Tuple[int, int, int, int]]` | — | Detected line segments as `(x1, y1, x2, y2)` tuples. |
| `aspect_ratio` | `float` | — | Maximum aspect ratio of detected contours. |
| `debug_image` | `Optional[Any]` | — | Debug visualization image (typically BGR `np.ndarray`). |
| `extras` | `Dict[str, Any]` | `{}` | Detector-specific auxiliary data (see below). |
| `metrics` | `Dict[str, Any]` | `{}` | Standard diagnostics for analysis tools. |
| `schema_version` | `int` | `1` | Contract version for future migrations. |

**metrics vs extras**:

| Dictionary | Purpose | Recommended Keys |
|------------|---------|------------------|
| `metrics` | Stable diagnostics for downstream analysis tools | `duration_ms`, `num_contours`, `mask_area`, `hough_votes` |
| `extras` | Detector-specific or auxiliary data | `bounding_boxes`, `polygons`, `masks`, custom keys |

**Usage Example**:

```python
from meteor_core.schema import DetectionResult

def detect(self, context: DetectionContext) -> DetectionResult:
    start_time = time.perf_counter()

    # ... detection logic ...

    duration_ms = (time.perf_counter() - start_time) * 1000

    return DetectionResult(
        is_candidate=score >= threshold,
        score=score,
        lines=[(x1, y1, x2, y2) for line in detected_lines],
        aspect_ratio=max_aspect_ratio,
        debug_image=debug_visualization,
        extras={
            "bounding_boxes": [{"x1": 10, "y1": 20, "x2": 100, "y2": 50}],
            "algorithm_variant": "adaptive",
        },
        metrics={
            "duration_ms": duration_ms,
            "num_contours": len(contours),
            "mask_area": int(np.count_nonzero(binary_mask)),
            "hough_votes": len(hough_lines),
        },
    )
```

**Serialization**: `result.to_dict()` returns a JSON-serializable dict (excludes `debug_image`).

### 4.4 OutputResult

`OutputResult` encapsulates the result of an output handler's `save_candidate()` method.

**Import**: `from meteor_core.schema import OutputResult`

```python
@dataclass
class OutputResult:
    """Result returned by output handlers."""

    saved: bool                                         # Whether save succeeded
    output_path: Optional[str]                          # Path to saved candidate
    debug_path: Optional[str]                           # Path to saved debug image
    handler_info: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = OUTPUT_RESULT_SCHEMA_VERSION  # Currently 1

    def to_dict(self) -> Dict[str, Any]: ...
```

**Fields**:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `saved` | `bool` | — | `True` if the handler persisted the candidate successfully. |
| `output_path` | `Optional[str]` | — | Location of the persisted candidate file (if any). |
| `debug_path` | `Optional[str]` | — | Location of the persisted debug image (if any). |
| `handler_info` | `Dict[str, Any]` | `{}` | Handler identity details from `BaseOutputHandler.get_info()`. |
| `metrics` | `Dict[str, Any]` | `{}` | Stable diagnostics (duration, bytes written, etc.). |
| `schema_version` | `int` | `1` | Contract version for future migrations. |

**Usage Example**:

```python
from meteor_core.schema import OutputResult

def save_candidate(
    self,
    source_path: str,
    filename: str,
    debug_image: Optional[np.ndarray] = None,
    roi_polygon: Optional[List[List[int]]] = None,
) -> OutputResult:
    start_time = time.perf_counter()
    dest_path = os.path.join(self.config.output_folder, filename)

    try:
        shutil.copy2(source_path, dest_path)
        bytes_written = os.path.getsize(dest_path)

        debug_path = None
        if debug_image is not None:
            debug_path = self.save_debug_image(debug_image, filename, roi_polygon)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return OutputResult(
            saved=True,
            output_path=dest_path,
            debug_path=debug_path,
            handler_info=self.get_info(),
            metrics={
                "duration_ms": duration_ms,
                "bytes_written": bytes_written,
            },
        )
    except OSError as e:
        return OutputResult(
            saved=False,
            output_path=dest_path,
            debug_path=None,
            handler_info=self.get_info(),
            metrics={"error": str(e)},
        )
```

**Serialization**: `result.to_dict()` returns a JSON-serializable dict.

### 4.5 Schema Versioning and Normalization

All four dataclasses include a `schema_version` field (currently `1`) to enable forward-compatible evolution of the plugin system.

**Available normalize/converter functions**:

| Dataclass | normalize function | converter registration |
|-----------|-------------------|------------------------|
| `InputContext` | `meteor_core.schema.normalize_input_context()` | `register_input_context_converter()` |
| `DetectionContext` | Pipeline-internal only | Not exposed publicly |
| `DetectionResult` | `meteor_core.schema.normalize_detection_result()` | `register_detection_result_converter()` |
| `OutputResult` | `meteor_core.schema.normalize_output_result()` | `register_output_result_converter()` |

> **Note**: `DetectionContext` normalization is handled internally by the pipeline (`_normalize_detection_context` in `meteor_core.pipeline`). Plugin authors do not need to call it directly, and there is no public converter registration function for `DetectionContext`.

**How it works**:
1. Plugins return dataclass instances with their implemented `schema_version`.
2. The pipeline calls `normalize_*()` functions immediately after plugin methods return.
3. If `schema_version` matches the current version, the instance passes through unchanged.
4. If `schema_version` is older, registered converters upgrade the instance.
5. If no converter exists for an older version, the pipeline raises `ValueError` (or `MeteorConfigError` for `DetectionContext`).

**Registering converters** (for backward compatibility):

```python
from meteor_core.schema import (
    register_input_context_converter,
    register_detection_result_converter,
    register_output_result_converter,
)

def upgrade_detection_result_v0_to_v1(result: DetectionResult) -> DetectionResult:
    """Convert v0 DetectionResult to v1 format."""
    return DetectionResult(
        is_candidate=result.is_candidate,
        score=result.score,
        lines=result.lines,
        aspect_ratio=result.aspect_ratio,
        debug_image=result.debug_image,
        extras=result.extras,
        metrics=result.metrics if hasattr(result, 'metrics') else {},
        schema_version=1,
    )

register_detection_result_converter(0, upgrade_detection_result_v0_to_v1)
```

**Versioning policy**:
- `schema_version` increments only for backward-incompatible structural changes.
- New optional fields can be added without version bump.
- Current version for all dataclasses is `1`.

---

## 5. Sample Code

### 5.1 Input Loader (Complete Example)

```python
"""Custom TIFF image loader with metadata extraction, logging, and exceptions."""
import logging
import os
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from meteor_core.inputs import (
    DataclassInputLoader,
    BaseMetadataExtractor,
    LoaderRegistry,
)
from meteor_core.schema import InputContext
from meteor_core.exceptions import (
    MeteorLoadError,
    MeteorUnsupportedFormatError,
)

# Set up logger for this plugin
logger = logging.getLogger("meteor_core.inputs.tiff_loader")


@dataclass
class TiffLoaderConfig:
    """Configuration for TIFF loader."""
    normalize: bool = False
    bit_depth: int = 16


class TiffImageLoader(DataclassInputLoader[TiffLoaderConfig], BaseMetadataExtractor):
    """Load TIFF images with optional normalization."""

    plugin_name = "tiff"           # Required: unique identifier
    name = "TIFF Image Loader"     # Optional: human-readable name
    version = "1.0.0"              # Optional: version string
    ConfigType = TiffLoaderConfig  # Optional: configuration class

    def __init__(self, config: TiffLoaderConfig = None):
        super().__init__(config)
        logger.debug(f"TiffImageLoader initialized with config: {self.config}")

    def load(self, filepath: str) -> InputContext:
        """Load a TIFF image file.

        Args:
            filepath: Path to the TIFF file.

        Returns:
            InputContext with the loaded image data.

        Raises:
            MeteorUnsupportedFormatError: If file is not a TIFF.
            MeteorLoadError: If file cannot be loaded.
        """
        logger.debug(f"Loading TIFF file: {filepath}")

        # Validate file extension
        if not filepath.lower().endswith((".tiff", ".tif")):
            logger.warning(f"Unsupported file extension: {filepath}")
            raise MeteorUnsupportedFormatError(
                f"Unsupported format: {filepath}",
                filepath=filepath,
                context={"supported_formats": [".tiff", ".tif"]},
            )

        # Check file existence
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise MeteorLoadError(
                f"File not found: {filepath}",
                filepath=filepath,
            )

        try:
            import tifffile

            image = tifffile.imread(filepath)
            logger.debug(f"Raw image shape: {image.shape}, dtype: {image.dtype}")

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                logger.debug("Converting RGB to grayscale")
                image = np.mean(image, axis=2)

            # Normalize if configured
            if self.config.normalize:
                max_val = 2 ** self.config.bit_depth - 1
                image = (image / max_val * 65535).astype(np.uint16)
                logger.debug(f"Normalized to uint16 (bit_depth={self.config.bit_depth})")
            else:
                image = image.astype(np.uint16)

            logger.debug(f"Final image shape: {image.shape}, dtype: {image.dtype}")
            return InputContext(
                image_data=image,
                filepath=filepath,
                metadata=self.extract_metadata(filepath),
                loader_info=self.get_info(),
            )

        except ImportError as e:
            logger.error("tifffile package not installed")
            raise MeteorLoadError(
                "tifffile package is required for TIFF support",
                filepath=filepath,
                original_error=e,
                context={"install_hint": "pip install tifffile"},
            )
        except Exception as e:
            logger.exception(f"Failed to load TIFF file: {filepath}")
            raise MeteorLoadError(
                f"Failed to load TIFF file: {e}",
                filepath=filepath,
                original_error=e,
                context={"loader": self.plugin_name},
            )

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from TIFF file.

        This method should NOT raise exceptions - return empty dict on failure.

        Args:
            filepath: Path to the TIFF file.

        Returns:
            Dictionary with metadata, or empty dict on failure.
        """
        logger.debug(f"Extracting metadata from: {filepath}")

        try:
            import tifffile

            with tifffile.TiffFile(filepath) as tif:
                page = tif.pages[0]
                metadata = {
                    "width": page.shape[1] if len(page.shape) > 1 else page.shape[0],
                    "height": page.shape[0],
                    "dtype": str(page.dtype),
                    "compression": page.compression.name if page.compression else None,
                }
                logger.debug(f"Extracted metadata: {list(metadata.keys())}")
                return metadata

        except Exception as e:
            # Metadata extraction is optional - don't fail the pipeline
            logger.warning(f"Could not extract metadata from {filepath}: {e}")
            return {}


# Register the plugin
LoaderRegistry.register(TiffImageLoader)
```

### 5.2 Detector (Complete Example)

```python
"""Simple threshold-based detector for bright meteors with logging."""
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np

from meteor_core.detectors import DataclassDetector, DetectorRegistry
from meteor_core.schema import DetectionContext, DetectionResult
from meteor_core.utils import ensure_numpy

# Set up logger for this plugin
logger = logging.getLogger("meteor_core.detectors.threshold")


@dataclass
class ThresholdDetectorConfig:
    """Configuration for threshold detector."""
    brightness_multiplier: float = 1.5
    min_contour_area: int = 50


class ThresholdDetector(DataclassDetector[ThresholdDetectorConfig]):
    """Detect bright meteors using simple thresholding."""

    plugin_name = "threshold"
    name = "Threshold Detector"
    version = "1.0.0"
    ConfigType = ThresholdDetectorConfig

    def __init__(self, config: ThresholdDetectorConfig = None):
        super().__init__(config)
        logger.debug(
            f"ThresholdDetector initialized: "
            f"brightness_multiplier={self.config.brightness_multiplier}, "
            f"min_contour_area={self.config.min_contour_area}"
        )

    def detect(
        self,
        context: DetectionContext,
    ) -> DetectionResult:
        """Detect meteor candidates using threshold-based approach.

        Raise for invalid inputs/configuration or return a failure result for
        recoverable cases. The pipeline treats exceptions as no-detection results.

        Args:
            context: Input bundle containing frames, ROI, and runtime params.

        Returns:
            DetectionResult with the detection outcome.
        """
        current_image = ensure_numpy(context.current_image)
        previous_image = ensure_numpy(context.previous_image)
        roi_mask = ensure_numpy(context.roi_mask)
        global_params, detector_params = self.split_runtime_params(
            context.runtime_params
        )
        params = {**global_params, **detector_params}
        logger.debug(
            f"Starting detection: image_shape={current_image.shape}, "
            f"diff_threshold={params.get('diff_threshold', 8)}"
        )

        try:
            # Compute absolute difference
            diff = cv2.absdiff(current_image, previous_image)
            logger.debug(f"Computed frame difference: max={diff.max()}, mean={diff.mean():.1f}")

            # Apply ROI mask
            diff = cv2.bitwise_and(diff, diff, mask=roi_mask)

            # Get threshold from params
            threshold = params.get("diff_threshold", 8)
            effective_threshold = int(threshold * self.config.brightness_multiplier)
            logger.debug(f"Applying threshold: {effective_threshold}")

            # Apply threshold
            _, binary = cv2.threshold(
                diff,
                effective_threshold,
                255,
                cv2.THRESH_BINARY
            )
            binary = binary.astype(np.uint8)

            # Find contours
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            logger.debug(f"Found {len(contours)} raw contours")

            # Filter by area
            valid_contours = [
                c for c in contours
                if cv2.contourArea(c) >= self.config.min_contour_area
            ]
            logger.debug(
                f"After area filter (>={self.config.min_contour_area}): "
                f"{len(valid_contours)} contours"
            )

            if not valid_contours:
                logger.debug("No valid contours found - returning no detection")
                return DetectionResult(
                    is_candidate=False,
                    score=0.0,
                    lines=[],
                    aspect_ratio=0.0,
                    debug_image=None,
                    extras={},
                    metrics={
                        "duration_ms": 0.0,
                        "num_contours": 0,
                        "mask_area": 0,
                        "hough_votes": 0,
                    },
                )

            # Compute metrics
            max_area = max(cv2.contourArea(c) for c in valid_contours)
            score = float(max_area) / 100.0

            # Compute aspect ratios
            aspect_ratios = []
            for c in valid_contours:
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratios.append(max(w, h) / max(min(w, h), 1))
            max_aspect_ratio = max(aspect_ratios) if aspect_ratios else 0.0

            logger.debug(f"Metrics: score={score:.1f}, max_aspect_ratio={max_aspect_ratio:.2f}")

            # Create debug image
            debug_image = cv2.cvtColor(
                (current_image // 256).astype(np.uint8), cv2.COLOR_GRAY2BGR
            )
            cv2.drawContours(debug_image, valid_contours, -1, (0, 255, 0), 2)

            # Simple line representation (bounding box diagonal)
            lines = []
            for c in valid_contours:
                x, y, w, h = cv2.boundingRect(c)
                lines.append((x, y, x + w, y + h))

            min_score = params.get("min_line_score", 30.0)
            is_candidate = score >= min_score

            if is_candidate:
                logger.info(f"Candidate detected: score={score:.1f} >= {min_score}")
            else:
                logger.debug(f"Not a candidate: score={score:.1f} < {min_score}")

            return DetectionResult(
                is_candidate=is_candidate,
                score=score,
                lines=lines,
                aspect_ratio=max_aspect_ratio,
                debug_image=debug_image,
                extras={"valid_contours": len(valid_contours)},
                metrics={
                    "duration_ms": 0.0,
                    "num_contours": len(valid_contours),
                    "mask_area": int(np.count_nonzero(binary)),
                    "hough_votes": 0,
                },
            )

        except Exception as e:
            # The pipeline treats exceptions as a failed detection (no candidate).
            # Raise if you want the pipeline to log the error for this file.
            logger.warning(f"Detection failed with error: {e}")
            raise

    def compute_line_score(
        self,
        mask: np.ndarray,
        hough_params: Dict[str, int],
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """Compute line score (simplified for threshold detector).

        Args:
            mask: Binary mask of detected regions.
            hough_params: Hough transform parameters (unused here).

        Returns:
            Tuple of (score, line_segments).
        """
        logger.debug(f"Computing line score for mask shape: {mask.shape}")

        try:
            # Find contours and compute score
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                logger.debug("No contours found in mask")
                return 0.0, []

            total_area = sum(cv2.contourArea(c) for c in contours)
            score = float(total_area) / 100.0
            logger.debug(f"Line score: {score:.1f} (from {len(contours)} contours)")
            return score, []

        except Exception as e:
            logger.warning(f"compute_line_score failed: {e}")
            return 0.0, []


# Register the plugin
DetectorRegistry.register(ThresholdDetector)
```

### 5.3 Output Handler with Lifecycle Hooks (Secondary Handler Example)

Per-frame hooks run during detection, so you can use `DetectionResult.lines` to
inspect line segments and `DetectionResult.extras` to read detector-specific
metadata (e.g., bounding boxes, masks, or algorithm tags). The
`context` argument contains `DetectionContext.to_dict()` output (no
image buffers). Keep `extras` JSON-serializable so it can be logged or emitted
to observability tools.

```python
"""Slack notification handler with full lifecycle support, logging, and exceptions."""
import json
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from meteor_core.i18n import DEFAULT_LOCALE, get_message
from meteor_core.outputs import DataclassOutputHandler, OutputHandlerRegistry
from meteor_core.exceptions import MeteorWriteError
from meteor_core.schema import DetectionResult, OutputResult

# Set up logger for this plugin
logger = logging.getLogger("meteor_core.outputs.slack_handler")


@dataclass
class SlackOutputConfig:
    """Configuration for Slack notification handler."""
    output_folder: str = "./candidates"
    debug_folder: str = "./debug_masks"
    webhook_url: str = ""
    notify_on_detection: bool = True
    notify_on_complete: bool = True
    channel: str = "#meteor-alerts"
    locale: str = DEFAULT_LOCALE


class SlackNotificationHandler(DataclassOutputHandler[SlackOutputConfig]):
    """Output handler with Slack notifications."""

    plugin_name = "slack"
    name = "Slack Notification Handler"
    version = "1.0.0"
    ConfigType = SlackOutputConfig

    def __init__(self, config: SlackOutputConfig):
        super().__init__(config)
        self._detection_count = 0

        logger.debug(
            f"SlackNotificationHandler initialized: "
            f"output={self.config.output_folder}, "
            f"webhook={'configured' if self.config.webhook_url else 'not configured'}"
        )

        # Create output directories
        try:
            os.makedirs(self.config.output_folder, exist_ok=True)
            logger.debug(f"Created output folder: {self.config.output_folder}")
            if self.config.debug_folder:
                os.makedirs(self.config.debug_folder, exist_ok=True)
                logger.debug(f"Created debug folder: {self.config.debug_folder}")
        except OSError as e:
            logger.error(f"Failed to create output directories: {e}")
            # Don't raise - let pipeline continue and fail on actual save

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> OutputResult:
        """Save a meteor candidate file.

        NOTE: This is a secondary (non-critical) handler example.
        It returns False on failure instead of raising exceptions.
        For primary handlers like FileOutputHandler, raise MeteorWriteError instead.

        Args:
            source_path: Path to source RAW file.
            filename: Output filename.
            debug_image: Optional debug visualization.
            roi_polygon: Optional ROI polygon.

        Returns:
            OutputResult with saved flag and paths.
        """
        logger.debug(f"Saving candidate: {filename}")

        dest_path = os.path.join(self.config.output_folder, filename)

        # Skip if exists
        if os.path.exists(dest_path):
            logger.debug(f"File already exists, skipping: {dest_path}")
            return OutputResult(
                saved=False,
                output_path=dest_path,
                debug_path=None,
            )

        try:
            # Copy the file
            shutil.copy2(source_path, dest_path)
            logger.info(f"Saved candidate: {filename}")

            # Save debug image if provided
            if debug_image is not None and self.config.debug_folder:
                debug_filename = os.path.splitext(filename)[0] + "_debug.png"
                self.save_debug_image(debug_image, debug_filename, roi_polygon)

            return OutputResult(
                saved=True,
                output_path=dest_path,
                debug_path=None,
            )

        except OSError as e:
            # Create structured error with context for diagnostics
            error = MeteorWriteError(
                f"Failed to copy candidate file: {e}",
                filepath=source_path,
                destination_path=dest_path,
                operation="copy",
                original_error=e,
                context={"error_category": "copy_failed"},
            )
            # Log error but don't fail the pipeline
            logger.error(str(error))
            return OutputResult(
                saved=False,
                output_path=dest_path,
                debug_path=None,
            )
        except Exception as e:
            logger.exception(f"Unexpected error saving candidate {filename}")
            return OutputResult(
                saved=False,
                output_path=dest_path,
                debug_path=None,
            )

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Save a debug visualization.

        Args:
            debug_image: Debug image (BGR).
            filename: Output filename.
            roi_polygon: Optional ROI polygon to draw.

        Returns:
            Path to saved debug image, or empty string on failure.
        """
        if not self.config.debug_folder:
            return ""

        logger.debug(f"Saving debug image: {filename}")
        path = os.path.join(self.config.debug_folder, filename)

        try:
            # Draw ROI if provided
            if roi_polygon and len(roi_polygon) >= 3:
                pts = np.array(roi_polygon, dtype=np.int32)
                cv2.polylines(debug_image, [pts], True, (255, 0, 0), 2)

            cv2.imwrite(path, debug_image)
            logger.debug(f"Saved debug image: {path}")
            return path

        except OSError as e:
            error = MeteorWriteError(
                f"Failed to save debug image: {e}",
                destination_path=path,
                operation="save_image",
                original_error=e,
                context={"error_category": "image_write_failed"},
            )
            logger.warning(str(error))
            return ""
        except Exception as e:
            logger.warning(f"Failed to save debug image {filename}: {e}")
            return ""

    # ========== LIFECYCLE HOOKS ==========
    # These methods must NEVER raise exceptions!

    def on_detection_result(
        self,
        context: Dict[str, Any],
        result: DetectionResult,
        filepath: str,
    ) -> None:
        """Called immediately after each detection result.

        Use this to inspect detector outputs before saving candidates.
        This method must NOT raise exceptions.
        """
        try:
            basename = os.path.basename(filepath)
            runtime_params = context.get("runtime_params", {})
            line_count = len(result.lines)
            extra_keys = ", ".join(sorted(result.extras.keys()))
            logger.debug(
                "on_detection_result: %s is_candidate=%s lines=%d extras=%s params=%s",
                basename,
                result.is_candidate,
                line_count,
                extra_keys or "(none)",
                runtime_params,
            )
        except Exception as e:
            # NEVER raise in lifecycle hooks
            logger.warning(f"on_detection_result hook failed: {e}")

    def on_candidate_detected(
        self,
        filename: str,
        saved: bool,
        score: float = 0.0,
        aspect_ratio: float = 0.0,
    ) -> None:
        """Called after each meteor detection.

        Use this for real-time notifications.
        This method must NOT raise exceptions.

        Args:
            filename: Detected file name.
            saved: Whether file was saved (False if skipped).
            score: Detection score.
            aspect_ratio: Contour aspect ratio.
        """
        try:
            self._detection_count += 1
            logger.debug(
                f"on_candidate_detected: {filename}, saved={saved}, "
                f"score={score:.1f}, aspect_ratio={aspect_ratio:.2f}"
            )

            if self.config.notify_on_detection and self.config.webhook_url and saved:
                message = get_message(
                    "ui.notification.meteor_detected",
                    locale=self.config.locale,
                    filename=filename,
                    score=f"{score:.1f}",
                    aspect_ratio=f"{aspect_ratio:.2f}",
                )
                self._send_slack(message)

        except Exception as e:
            # NEVER raise in lifecycle hooks
            logger.warning(f"on_candidate_detected hook failed: {e}")

    def on_batch_complete(
        self,
        processed_count: int,
        detected_count: int,
        batch_size: int,
    ) -> None:
        """Called after each batch completes.

        Use for progress tracking.
        This method must NOT raise exceptions.

        Args:
            processed_count: Total processed so far.
            detected_count: Total detected so far.
            batch_size: Files in this batch.
        """
        try:
            logger.debug(
                f"on_batch_complete: processed={processed_count}, "
                f"detected={detected_count}, batch_size={batch_size}"
            )
            # Could implement periodic progress updates here

        except Exception as e:
            # NEVER raise in lifecycle hooks
            logger.warning(f"on_batch_complete hook failed: {e}")

    def on_pipeline_complete(
        self,
        total_processed: int,
        total_detected: int,
        elapsed_seconds: float,
    ) -> None:
        """Called when pipeline finishes.

        Use for final summary notifications.
        This method must NOT raise exceptions.

        Args:
            total_processed: Total files processed.
            total_detected: Total candidates detected.
            elapsed_seconds: Total time in seconds.
        """
        try:
            minutes = elapsed_seconds / 60
            rate = total_processed / elapsed_seconds if elapsed_seconds > 0 else 0

            logger.info(
                f"Pipeline complete: {total_processed} processed, "
                f"{total_detected} detected, {minutes:.1f} min, {rate:.2f} img/s"
            )

            if self.config.notify_on_complete and self.config.webhook_url:
                message = get_message(
                    "ui.notification.detection_complete",
                    locale=self.config.locale,
                    processed=total_processed,
                    detected=total_detected,
                    minutes=f"{minutes:.1f}",
                    rate=f"{rate:.2f}",
                )
                self._send_slack(message)

        except Exception as e:
            # NEVER raise in lifecycle hooks
            logger.warning(f"on_pipeline_complete hook failed: {e}")

    def _send_slack(self, message: str) -> None:
        """Send a message to Slack webhook.

        Args:
            message: Message text (supports Slack markdown).
        """
        if not self.config.webhook_url:
            logger.debug("No webhook URL configured, skipping Slack notification")
            return

        logger.debug(f"Sending Slack notification to {self.config.channel}")

        payload = {
            "channel": self.config.channel,
            "text": message,
            "mrkdwn": True,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            logger.debug("Slack notification sent successfully")

        except urllib.error.URLError as e:
            logger.warning(f"Failed to send Slack notification (network error): {e}")
        except Exception as e:
            # Don't fail pipeline on notification errors
            logger.warning(f"Failed to send Slack notification: {e}")


# Register the plugin
OutputHandlerRegistry.register(SlackNotificationHandler)
```

---

## 6. Best Practices

### 6.1 Choosing ConfigType

Use this decision tree to select the right configuration approach:

```
Need configuration?
        │
       No ──────────▶ Don't define ConfigType
        │              (accept None in __init__)
       Yes
        │
        ▼
Need validation?
(ranges, patterns,
 custom rules)
        │
       No ──────────▶ Use @dataclass
        │              (simple, built-in, no deps)
       Yes
        │
        ▼
Use Pydantic BaseModel
(rich validation, type coercion)
```

**Dataclass (Recommended for most cases)**:
```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    output_folder: str = "./output"  # Always provide defaults
    threshold: float = 0.5
    enabled: bool = True
```

**Pydantic (For complex validation)**:
```python
from pydantic import BaseModel, Field, field_validator

class MyConfig(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    url: str = ""

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    model_config = {"extra": "forbid"}  # Reject unknown keys
```

### 6.2 Required Attributes

| Attribute | Required | Type | Description |
|-----------|----------|------|-------------|
| `plugin_name` | ✅ Yes | `str` | Unique identifier (case-insensitive) |
| `name` | ❌ No | `str` | Human-readable name |
| `version` | ❌ No | `str` | Version string |
| `ConfigType` | ❌ No | `type` | Configuration class |

### 6.3 Exception Hierarchy and Error Handling

#### Exception Hierarchy

`meteor_core` provides a structured exception hierarchy for consistent error handling:

```
MeteorError (base)
├── MeteorLoadError (image loading failures)
│   └── MeteorUnsupportedFormatError (unsupported file formats)
├── MeteorOutputError (output operation failures)
│   ├── MeteorWriteError (file write failures)
│   └── MeteorProgressError (progress tracking errors)
├── MeteorValidationError (parameter/input validation)
└── MeteorConfigError (configuration errors)
```

**Import exceptions**:
```python
from meteor_core.exceptions import (
    MeteorError,
    MeteorLoadError,
    MeteorUnsupportedFormatError,
    MeteorOutputError,
    MeteorWriteError,
    MeteorProgressError,
    MeteorValidationError,
    MeteorConfigError,
)
```

**Exception attributes**:

Each exception includes rich context for debugging and issue reporting:

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable error description |
| `filepath` | `Optional[str]` | File path (if applicable) |
| `original_error` | `Optional[Exception]` | Original exception (for chained errors) |
| `context` | `Dict[str, Any]` | Additional context information |

**Creating exceptions with context**:
```python
from meteor_core.exceptions import MeteorLoadError

raise MeteorLoadError(
    "Failed to decode FITS file",
    filepath="/path/to/image.fits",
    original_error=original_exception,
    context={
        "loader": "fits",
        "bit_depth": 16,
        "compression": "RICE_1",
    },
)
```

**Output-specific exceptions**:

For output operations, use `MeteorWriteError` and `MeteorProgressError`:

```python
from meteor_core.exceptions import MeteorWriteError, MeteorProgressError

# File write error with destination path
error = MeteorWriteError(
    "Failed to copy candidate file",
    filepath="/source/image.CR2",           # Source path
    destination_path="/output/image.CR2",   # Destination path
    operation="copy",                       # Operation type
    original_error=os_error,
    context={"error_category": "copy_failed"},
)

# Progress tracking error
error = MeteorProgressError(
    "Failed to parse progress file",
    filepath="progress.json",
    operation="parse",                      # "load", "save", "parse", "serialize"
    original_error=json_error,
    context={"error_category": "parse_failed"},
)
```

| Exception | Use Case | Key Attributes |
|-----------|----------|----------------|
| `MeteorWriteError` | File copy, debug image save, directory creation | `destination_path`, `operation` |
| `MeteorProgressError` | Progress file read/write, JSON parse errors | `operation` (load/save/parse/serialize) |

#### Exception Policy by Plugin Type

Each plugin type has different expectations for when to raise exceptions vs. continue processing:

| Plugin Type | Method | Policy |
|-------------|--------|--------|
| **Input Loader** | `load()` | **Raise exceptions** - Pipeline cannot continue without image |
| **Input Loader** | `extract_metadata()` | **Return empty dict** - Metadata is optional |
| **Detector** | `detect()` | **Raise or return DetectionResult** - Pipeline marks failures as no detection |
| **Output Handler** | `save_candidate()` | **Depends on criticality** - See below |
| **Output Handler** | Lifecycle hooks | **Never raise** - Log errors, continue processing |

**Output Handler Exception Policy (Critical vs. Non-Critical)**:

Output handlers fall into two categories based on their role:

| Category | Examples | Policy |
|----------|----------|--------|
| **Primary (Critical)** | FileOutputHandler, S3Handler | **Raise exceptions** - Disk/storage errors are critical |
| **Secondary (Non-Critical)** | SlackHandler, WebhookHandler | **Return OutputResult(saved=False, ...)** - Notification failures are non-critical |

- **Primary handlers** persist the detection results (RAW files, debug images). If these fail, it usually indicates a systemic issue (disk full, permission denied, network storage unavailable) that will affect all subsequent writes. Raising an exception allows users to address the issue immediately rather than discovering hours later that no files were saved.

- **Secondary handlers** provide notifications or auxiliary outputs. Their failure should not stop the pipeline since the core detection work can still proceed.

The built-in `FileOutputHandler` follows the **primary handler** pattern and raises `MeteorWriteError` on write failures. The Slack example in this guide demonstrates the **secondary handler** pattern.

**Input Loader exceptions**:

```python
class MyLoader(DataclassInputLoader[MyConfig]):
    def load(self, filepath: str) -> InputContext:
        # Check file existence
        if not os.path.exists(filepath):
            raise MeteorLoadError(
                f"File not found: {filepath}",
                filepath=filepath,
            )

        # Check file format
        if not filepath.lower().endswith((".fits", ".fit")):
            raise MeteorUnsupportedFormatError(
                f"Unsupported format: {filepath}",
                filepath=filepath,
                context={"supported_formats": [".fits", ".fit"]},
            )

        try:
            # Load the image
            image = self._read_fits(filepath)
            return InputContext(
                image_data=image,
                filepath=filepath,
                metadata=self.extract_metadata(filepath),
                loader_info=self.get_info(),
            )
        except Exception as e:
            # Wrap low-level errors with context
            raise MeteorLoadError(
                f"Failed to load FITS file: {e}",
                filepath=filepath,
                original_error=e,
                context={"loader": self.plugin_name},
            )

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Metadata extraction should not raise exceptions."""
        try:
            return self._read_fits_header(filepath)
        except Exception:
            # Return empty dict, don't fail
            return {}
```

**Detector behavior** (raise or return):

```python
class MyDetector(DataclassDetector[MyConfig]):
    def detect(self, context: DetectionContext) -> DetectionResult:
        # Raise when configuration or inputs are invalid so the pipeline can
        # log the error for that file.
        if context.current_image.shape != context.previous_image.shape:
            raise ValueError("current_image and previous_image must match")

        # Or return a "no detection" result for recoverable cases.
        return DetectionResult(
            is_candidate=False,
            score=0.0,
            lines=[],
            aspect_ratio=0.0,
            debug_image=None,
            extras={"reason": "no contours"},
        )
```

**Output Handler error handling**:

For **primary handlers** (critical file/storage operations), raise exceptions:

```python
from meteor_core.exceptions import MeteorWriteError

class MyFileHandler(DataclassOutputHandler[MyConfig]):
    """Primary handler - raises exceptions on critical failures."""

    def save_candidate(self, source_path, filename, debug_image, roi_polygon) -> OutputResult:
        dest_path = os.path.join(self.config.output_folder, filename)
        try:
            shutil.copy2(source_path, dest_path)
            return OutputResult(
                saved=True,
                output_path=dest_path,
                debug_path=None,
            )
        except OSError as e:
            # Primary handlers raise to stop pipeline on critical errors
            raise MeteorWriteError(
                f"Failed to copy candidate file: {e}",
                filepath=source_path,
                destination_path=dest_path,
                operation="copy",
                original_error=e,
                context={"error_category": "copy_failed"},
            ) from e
```

For **secondary handlers** (notifications, webhooks), log and return `OutputResult(saved=False, ...)`:

```python
class MyNotificationHandler(DataclassOutputHandler[MyConfig]):
    """Secondary handler - logs errors and continues."""

    def save_candidate(self, source_path, filename, debug_image, roi_polygon) -> OutputResult:
        try:
            self._upload_to_cloud(source_path)
            return OutputResult(
                saved=True,
                output_path=source_path,
                debug_path=None,
            )
        except Exception as e:
            # Secondary handlers log but don't fail the pipeline
            logger.warning(f"Cloud upload failed (non-critical): {e}")
            return OutputResult(
                saved=False,
                output_path=source_path,
                debug_path=None,
            )

    def on_candidate_detected(self, filename, saved, score, aspect_ratio):
        try:
            self._send_notification(filename)
        except Exception as e:
            # NEVER raise in lifecycle hooks
            logger.warning(f"Notification failed: {e}")
```

### 6.4 Logging Guidelines

#### Setting Up Logging

Use Python's standard `logging` module with the `meteor_core` logger hierarchy:

```python
import logging

# Get a logger for your plugin
logger = logging.getLogger("meteor_core.inputs.my_loader")
# Or for detectors: logging.getLogger("meteor_core.detectors.my_detector")
# Or for outputs: logging.getLogger("meteor_core.outputs.my_handler")
```

#### Log Level Policy

| Level | When to Use | Example |
|-------|-------------|---------|
| `DEBUG` | Detailed trace info for troubleshooting | File paths, config values, intermediate results |
| `INFO` | Significant events during normal operation | Plugin loaded, processing started/completed |
| `WARNING` | Recoverable issues that don't stop processing | Missing optional metadata, slow operation, deprecated usage |
| `ERROR` | Failures that affect the current operation | Failed to save file, network timeout |
| `CRITICAL` | Severe errors requiring immediate attention | Rarely used in plugins |

**Log level examples**:

```python
import logging
from meteor_core.schema import InputContext

logger = logging.getLogger("meteor_core.inputs.fits_loader")


class FitsLoader(DataclassInputLoader[FitsConfig]):
    def __init__(self, config):
        super().__init__(config)
        logger.debug(f"FitsLoader initialized with config: {config}")

    def load(self, filepath: str) -> InputContext:
        logger.debug(f"Loading FITS file: {filepath}")

        # INFO: Significant events
        logger.info(f"Processing {os.path.basename(filepath)}")

        try:
            image = self._read_fits(filepath)
            logger.debug(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
            return InputContext(
                image_data=image,
                filepath=filepath,
                metadata=self.extract_metadata(filepath),
                loader_info=self.get_info(),
            )
        except Exception as e:
            # ERROR: Operation failed
            logger.error(f"Failed to load {filepath}: {e}")
            raise MeteorLoadError(str(e), filepath=filepath, original_error=e)

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        try:
            metadata = self._read_header(filepath)
            logger.debug(f"Extracted metadata: {list(metadata.keys())}")
            return metadata
        except Exception as e:
            # WARNING: Non-critical failure
            logger.warning(f"Could not extract metadata from {filepath}: {e}")
            return {}
```

#### Logging Best Practices

**Do**:
- Use appropriate log levels consistently
- Include relevant context (filenames, config values)
- Use `logger.exception()` to include stack traces for errors
- Keep log messages concise but informative

```python
# Good: Informative with context
logger.debug(f"Applying threshold {threshold} to image {filepath}")
logger.warning(f"Metadata missing 'exposure_time' in {filepath}, using default")
logger.error(f"Failed to save to {output_path}: {e}")
```

**Don't**:
- Log sensitive information (API keys, credentials)
- Use `print()` instead of logging
- Log excessively in tight loops (performance impact)
- Raise exceptions just to log them

```python
# Bad: Using print
print(f"Loading {filepath}")  # Use logger.info() instead

# Bad: Logging in tight loops
for pixel in image.flatten():  # Millions of iterations!
    logger.debug(f"Processing pixel: {pixel}")

# Bad: Raising just to log
try:
    process()
except Exception as e:
    logger.error(str(e))
    raise  # If you're re-raising anyway, use logger.exception()

# Good: Use logger.exception() for full traceback
try:
    process()
except Exception as e:
    logger.exception(f"Processing failed")  # Includes traceback
    raise
```

### Internationalization (i18n) Guidance

- **Localize UI/UX only**: user-facing interface text such as CLI prompts, progress summaries, and error headers should use localized messages.
- **Keep everything else in English**: logs, debug output, and developer-facing diagnostics remain in English to keep troubleshooting consistent.

When you need localized UI/UX strings, use the shared message catalog in
`meteor_core/locales/<locale>/messages.yaml` via `meteor_core.i18n.get_message`.
Avoid introducing plugin-specific translation files unless coordinated with the
core maintainers.

Example entries (matching the Slack output handler sample):

```yaml
ui:
  notification:
    meteor_detected: "🌠 *Meteor Detected!*\n• File: `{filename}`\n• Score: {score}\n• Aspect Ratio: {aspect_ratio}"
    detection_complete: "✅ *Detection Complete*\n• Processed: {processed} images\n• Detected: {detected} candidates\n• Time: {minutes} minutes\n• Rate: {rate} images/sec"
```

Add corresponding translations in other locale files (for example,
`meteor_core/locales/ja/messages.yaml`).

#### Using Diagnostic Reports

For errors that users might report as issues, use `format_for_issue()`:

```python
from meteor_core.exceptions import MeteorLoadError

try:
    image = load_image(filepath)
except MeteorLoadError as e:
    # Get diagnostic report for GitHub issue
    diagnostic_report = e.format_for_issue()
    logger.error(f"Load failed. Diagnostic info:\n{diagnostic_report}")
    raise
```

### 6.5 Performance Considerations

**Input Loaders**:
- Return single-channel arrays; use uint16 by default or float32 if you normalize
- Avoid unnecessary copies (`image.astype()` creates a copy)
- Consider memory-mapped files for very large images

**Detectors**:
- Use NumPy vectorized operations over Python loops
- Pre-allocate arrays when possible
- Consider using OpenCV's optimized functions

**Output Handlers**:
- Make lifecycle hooks non-blocking (use timeouts)
- Buffer notifications for batch sending if needed
- Use async I/O for network operations (advanced)

### 6.6 Thread Safety

The pipeline may process images in parallel. Ensure your plugins are thread-safe:

```python
class MyHandler(DataclassOutputHandler[MyConfig]):
    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.Lock()
        self._count = 0

    def on_candidate_detected(self, filename, saved, score, aspect_ratio):
        with self._lock:
            self._count += 1
```

---

## 7. Step-by-Step Tutorial

### 7.1 Step-by-Step: Creating a Plugin

#### Step 1: Choose Plugin Type and Base Class

```python
# For input loaders with dataclass config
from meteor_core.inputs import DataclassInputLoader, BaseMetadataExtractor

# For detectors with dataclass config
from meteor_core.detectors import DataclassDetector

# For output handlers with dataclass config
from meteor_core.outputs import DataclassOutputHandler
```

#### Step 2: Define Configuration (Optional)

```python
from dataclasses import dataclass

@dataclass
class MyPluginConfig:
    option1: str = "default"
    option2: int = 10
```

#### Step 3: Implement the Plugin Class

```python
class MyPlugin(DataclassInputLoader[MyPluginConfig]):
    plugin_name = "my_plugin"  # Required
    name = "My Plugin"         # Optional
    version = "1.0.0"          # Optional
    ConfigType = MyPluginConfig

    def load(self, filepath: str) -> InputContext:
        # Implementation
        pass
```

#### Step 4: Register the Plugin

**Option A: Runtime registration**
```python
from meteor_core.inputs import LoaderRegistry
LoaderRegistry.register(MyPlugin)
```

**Option B: Entry point (for packages)**
```toml
# pyproject.toml
[project.entry-points."detect_meteors.input"]
my_plugin = "my_package:MyPlugin"
```

**Option C: Plugin directory (for user plugins)**
```bash
# Save as ~/.detect_meteors/input_plugins/my_plugin.py
```

### 6.2 Testing Your Plugin

#### Unit Test Example

```python
import unittest
import numpy as np
from my_plugin import MyPlugin, MyPluginConfig


class TestMyPlugin(unittest.TestCase):
    def test_load_returns_correct_shape(self):
        config = MyPluginConfig(option1="test")
        plugin = MyPlugin(config)

        context = plugin.load("test_image.tiff")

        self.assertEqual(len(context.image_data.shape), 2)  # Grayscale
        self.assertEqual(context.image_data.dtype, np.uint16)

    def test_default_config(self):
        # Test with default configuration
        from meteor_core.inputs import LoaderRegistry
        LoaderRegistry.register(MyPlugin)

        loader = LoaderRegistry.create("my_plugin")  # Uses defaults
        self.assertIsNotNone(loader)


if __name__ == "__main__":
    unittest.main()
```

#### Integration Test

```python
def test_plugin_in_pipeline():
    from meteor_core.inputs import LoaderRegistry
    from my_plugin import MyPlugin

    LoaderRegistry.register(MyPlugin)

    # Verify registration
    assert "my_plugin" in LoaderRegistry.list_available()

    # Create instance
    loader = LoaderRegistry.create("my_plugin", {"option1": "test"})
    assert loader.config.option1 == "test"
```

### 6.3 Debugging Tips

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check plugin info**:
```python
from meteor_core.inputs import LoaderRegistry

LoaderRegistry.discover()
for name in LoaderRegistry.list_available():
    cls = LoaderRegistry.get(name)
    instance = cls(None)
    print(instance.get_info())
```

**Verify configuration coercion**:
```python
# Test that dict config works
handler = OutputHandlerRegistry.create("my_handler", {"option": "value"})

# Test that ConfigType instance works
config = MyConfig(option="value")
handler = OutputHandlerRegistry.create("my_handler", config)

# Test default config
handler = OutputHandlerRegistry.create("my_handler")  # Uses ConfigType()
```

---

## See Also

**Documentation**:
- [INSTALL_DEV.md](INSTALL_DEV.md) — Development environment setup
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [README.md](README.md) — User documentation

**Built-in plugin implementations** (reference code):
- [`meteor_core/inputs/raw.py`](meteor_core/inputs/raw.py) — RawImageLoader
- [`meteor_core/detectors/hough_default.py`](meteor_core/detectors/hough_default.py) — HoughDetector
- [`meteor_core/outputs/file_handler.py`](meteor_core/outputs/file_handler.py) — FileOutputHandler
