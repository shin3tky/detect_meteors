# Version 1.6 Release Notes

## Version 1.6.5 (2025-12-26) 📓

### 🔧 Pipeline Configuration Files and CLI Plugin Selection

Version 1.6.5 introduces configuration file support and CLI plugin selection, enabling users to configure the entire detection pipeline—including plugin choices—via YAML/JSON files or command-line arguments. This is a major step toward the v2.0 plugin architecture.

### Highlights

- **Pipeline configuration files**: Load all settings from YAML/JSON via `--config`
- **CLI plugin selection**: `--input-loader`, `--detector`, `--output-handler` for runtime plugin choice
- **CLI plugin configuration**: `--input-loader-config`, `--detector-config`, `--output-handler-config` for plugin-specific settings
- **DetectionContext normalization API**: Public `register_detection_context_converter()` and `normalize_detection_context()`
- **Unified pipeline execution**: CLI now routed through `MeteorDetectionPipeline`
- **Deprecation path**: Legacy parameter flags still work but are deprecated in favor of config files

### Why This Change?

This release bridges the gap between v1.x and v2.0 by externalizing pipeline configuration:

| Feature | Before (v1.6.4) | After (v1.6.5) |
|---------|-----------------|----------------|
| **Plugin selection** | Hardcoded defaults | `--config` or `--detector hough` |
| **Plugin configuration** | Not available | `--detector-config '{...}'` |
| **Detection params** | CLI flags only | Config file or CLI flags |
| **Pipeline execution** | Standalone functions | Routed through `MeteorDetectionPipeline` |
| **DetectionContext normalization** | Internal only | Public API |

Benefits:
1. **Reproducible runs**: Save and share complete pipeline configurations
2. **Plugin experimentation**: Easily switch between plugins without code changes
3. **CI/CD integration**: Configure detection pipelines via config files
4. **Gradual migration**: Legacy CLI flags still work alongside new config approach

### Configuration File Format

Create a YAML or JSON file with `PipelineConfig` fields:

```yaml
# pipeline.yaml
target_folder: ./rawfiles
output_folder: ./candidates
debug_folder: ./debug_masks

# Detection parameters
params:
  diff_threshold: 8
  min_area: 10
  min_aspect_ratio: 3.0
  min_line_score: 30.0

# Worker settings
num_workers: 4
batch_size: 50
enable_parallel: true

# Plugin selection and configuration
input_loader_name: raw
input_loader_config:
  binning: 2
  normalize: true

detector_name: hough
detector_config:
  use_probabilistic: true

output_handler_name: file
output_handler_config:
  overwrite: false
```

**Usage**:

```bash
# Load entire configuration from file
python detect_meteors_cli.py --config pipeline.yaml

# Override specific settings via CLI
python detect_meteors_cli.py --config pipeline.yaml --detector threshold
```

### CLI Plugin Selection

Select and configure plugins directly from the command line:

```bash
# Select plugins by name
python detect_meteors_cli.py \
    --input-loader raw \
    --detector hough \
    --output-handler file

# Provide plugin configs as JSON strings
python detect_meteors_cli.py \
    --detector hough \
    --detector-config '{"use_probabilistic": true}'

# Or as YAML strings
python detect_meteors_cli.py \
    --output-handler slack \
    --output-handler-config "webhook_url: https://hooks.slack.com/..."

# Or as file paths
python detect_meteors_cli.py \
    --detector-config my_detector_settings.yaml
```

**CLI plugin options**:

| Option | Description |
|--------|-------------|
| `--config FILE` | Load pipeline configuration from YAML/JSON file |
| `--input-loader NAME` | Select input loader plugin by name |
| `--input-loader-config VALUE` | JSON/YAML string or file path for loader config |
| `--detector NAME` | Select detector plugin by name |
| `--detector-config VALUE` | JSON/YAML string or file path for detector config |
| `--output-handler NAME` | Select output handler plugin by name |
| `--output-handler-config VALUE` | JSON/YAML string or file path for handler config |

### Configuration Precedence

When multiple sources provide configuration:

1. **CLI arguments** (highest priority)
2. **Configuration file** (`--config`)
3. **Built-in defaults** (lowest priority)

This allows loading a base configuration and overriding specific settings:

```bash
# Load base config, override detector
python detect_meteors_cli.py --config base.yaml --detector threshold

# Load base config, override detection threshold
python detect_meteors_cli.py --config base.yaml --diff-threshold 12
```

### Python API

Load and use configuration files programmatically:

```python
from meteor_core import MeteorDetectionPipeline, load_pipeline_config

# Load configuration from file
config = load_pipeline_config("pipeline.yaml")

# Create and run pipeline
pipeline = MeteorDetectionPipeline(config)
pipeline.run()
```

### DetectionContext Normalization API

The `DetectionContext` normalization API is now publicly available, consistent with `InputContext`, `DetectionResult`, and `OutputResult`:

```python
from meteor_core.schema import (
    DetectionContext,
    register_detection_context_converter,
    normalize_detection_context,
)

# Register a converter for older schema versions
def upgrade_v0_to_v1(context: DetectionContext) -> DetectionContext:
    # Migration logic
    return DetectionContext(
        current_image=context.current_image,
        previous_image=context.previous_image,
        roi_mask=context.roi_mask,
        runtime_params=context.runtime_params,
        metadata=context.metadata,
        schema_version=1,
    )

register_detection_context_converter(0, upgrade_v0_to_v1)

# Normalize a context (applies converters if needed)
normalized = normalize_detection_context(context)
```

**Available normalization functions**:

| Dataclass | normalize function | converter registration |
|-----------|-------------------|------------------------|
| `InputContext` | `normalize_input_context()` | `register_input_context_converter()` |
| `DetectionContext` | `normalize_detection_context()` | `register_detection_context_converter()` |
| `DetectionResult` | `normalize_detection_result()` | `register_detection_result_converter()` |
| `OutputResult` | `normalize_output_result()` | `register_output_result_converter()` |

### Migration Guide

**For CLI users**:

No immediate migration required. All existing CLI flags continue to work:

```bash
# These still work (but are deprecated)
python detect_meteors_cli.py --diff-threshold 8 --min-area 10

# Recommended: Use config files for complex setups
python detect_meteors_cli.py --config pipeline.yaml
```

**For plugin authors**:

1. **Document your ConfigType fields** for config file users:
   ```python
   @dataclass
   class MyDetectorConfig:
       """Configuration for MyDetector.
       
       YAML example:
           detector_name: my_detector
           detector_config:
             sensitivity: 0.8
             use_gpu: true
       """
       sensitivity: float = 0.5
       use_gpu: bool = False
   ```

2. **Provide sensible defaults** so users can omit optional settings in config files.

3. **Use meaningful field names** that work well as YAML/JSON keys.

**For Python API users**:

Use `load_pipeline_config()` for file-based configuration:

```python
# Before: Manual PipelineConfig construction
config = PipelineConfig(
    target_folder="./rawfiles",
    output_folder="./candidates",
    params={"diff_threshold": 8},
)

# After: Load from file
config = load_pipeline_config("pipeline.yaml")
```

### Unified Pipeline Execution

CLI execution is now routed through `MeteorDetectionPipeline`, ensuring consistent behavior between CLI and Python API:

```
┌─────────────────────────────────────────────────────────────────┐
│  CLI (detect_meteors_cli.py)                                    │
│                                                                 │
│  1. Parse CLI arguments                                         │
│  2. Load --config file (if provided)                            │
│  3. Apply CLI overrides                                         │
│  4. Build PipelineConfig                                        │
│  5. Create MeteorDetectionPipeline(config)                      │
│  6. pipeline.run()                                              │
└─────────────────────────────────────────────────────────────────┘
```

This ensures:
- Consistent plugin resolution between CLI and API
- Consistent normalization checkpoints
- Consistent lifecycle hook invocation

### Files Changed

| File | Changes |
|------|---------|
| `detect_meteors_cli.py` | Added `--config`, plugin selection args, routed through pipeline |
| `meteor_core/pipeline.py` | Added `load_pipeline_config()`, plugin config handling |
| `meteor_core/schema.py` | Added public `normalize_detection_context()`, `register_detection_context_converter()` |
| `meteor_core/completions/bash_completion.sh` | Added new options |
| `meteor_core/completions/zsh_completion.sh` | Added new options |
| `config_examples/pipeline.yaml` | New example configuration file |
| `PLUGIN_AUTHOR_GUIDE.md` | Added configuration file and CLI documentation |
| `CHANGELOG.md` | Added v1.6.5 entry |
| `README.md` | Updated configuration file section and "What's New" |
| `ROADMAP.md` | Added v1.6.5, marked config file support as implemented |

### Backward Compatibility

✅ **Fully backward compatible** with v1.6.4, v1.6.3, v1.6.2, v1.6.1, and v1.6.0:

- **CLI**: All existing flags work unchanged
- **Runtime**: No changes to detection behavior
- **API**: Existing code works unchanged
- **Plugins**: Existing plugins work unchanged

**Deprecation notice**: Legacy parameter flags (e.g., `--diff-threshold`, `--min-area`) will be deprecated in a future release. Migrate to `--config` or `params:` in config files.

---

## Version 1.6.4 (2025-12-25) 🎄

### 🔧 Output Handler Hooks and Frame Tracking

Version 1.6.4 adds the `on_detection_result` lifecycle hook for output handlers, propagates `DetectionResult` through the pipeline, and introduces frame index tracking for improved progress reporting and post-processing analysis.

### Highlights

- **Output handler `on_detection_result` hook**: New per-detection callback with serialized context payload
- **DetectionResult propagation**: `process_image_batch()` now returns `DetectionResult`, enabling access to `lines`, `extras`, and `metrics`
- **Frame indices tracking**: `frame_index` and `prev_frame_index` in detection context metadata and `progress.json`
- **Debug image optimization**: Debug images only generated for candidate detections, reducing memory usage
- **Performance improvements**: `_build_runtime_params()` moved outside processing loop
- **Batch progress recording**: Frame indices recorded in `progress.json` for post-processing analysis

### Why This Change?

This release enhances observability and post-processing capabilities:

| Feature | Before (v1.6.3) | After (v1.6.4) |
|---------|-----------------|----------------|
| **Per-detection hook** | Not available | `on_detection_result(context, result, filepath)` |
| **DetectionResult access** | Not propagated | Available in output handlers |
| **Frame tracking** | Filename only | `frame_index` and `prev_frame_index` |
| **Debug images** | Generated for all | Only for candidates |
| **Progress detail** | File count | Frame indices (e.g., "frames 42, 108, 215") |

Benefits:
1. **Real-time inspection**: Output handlers can inspect detection results before saving
2. **Rich diagnostics**: Access to `lines`, `extras`, and `metrics` from detectors
3. **Post-processing analysis**: Frame indices in `progress.json` enable correlation with external data
4. **Memory efficiency**: Debug images cleared for non-candidates
5. **Better progress UX**: Users see which specific frames detected meteors

### Output Handler Lifecycle (v1.6.4)

The pipeline now invokes hooks in this order per frame:

```
┌─────────────────────────────────────────────────────────────────┐
│  For each detection result:                                     │
│                                                                 │
│  1. on_detection_result(context, result, filepath)              │
│     └── Inspect result.lines, result.extras, result. metrics    │
│     └── context contains runtime_params, metadata               │
│                                                                 │
│  2. save_candidate() [only if result.is_candidate]              │
│     └── Save the candidate image                                │
│                                                                 │
│  3. on_candidate_detected(filename, saved, score, aspect_ratio) │
│     └── Send notifications, update counters                     │
└─────────────────────────────────────────────────────────────────┘
```

### Frame Indices in Detection Context

Detection context metadata now includes frame indices:

```python
context.metadata = {
    "current": {
        "frame_index": 42,      # 0-based index of current frame
        # ... other metadata
    },
    "previous": {
        "frame_index": 41,      # 0-based index of previous frame
        # ... other metadata
    },
}
```

### Progress File Changes (progress.json)

The `detected_details` entries now include frame indices:

```json
{
  "processed": 150,
  "detected": 3,
  "detected_details": [
    {
      "filename": "IMG_0042.CR2",
      "score": 85.5,
      "aspect_ratio": 3.2,
      "frame_index": 42,
      "prev_frame_index": 41
    },
    {
      "filename": "IMG_0108.CR2",
      "score": 92.1,
      "aspect_ratio": 4.1,
      "frame_index": 108,
      "prev_frame_index": 107
    }
  ]
}
```

This enables:
- Correlation with external timestamp logs
- Analysis of detection patterns across frame sequences
- Integration with video/timelapse metadata

### Migration Guide for Plugin Authors

**No immediate migration required.** All existing plugins continue to work.

**Recommended updates for Output Handler authors**:

1. **Implement `on_detection_result` hook** for per-detection processing:
   ```python
   def on_detection_result(
       self,
       context: Dict[str, Any],
       result: DetectionResult,
       filepath: str,
   ) -> None:
       """Called immediately after each detection result.
   
       Args:
           context: Serialized DetectionContext (no image data)
           result: The DetectionResult from the detector
           filepath: Path to the current image file
       """
       # Inspect detector outputs
       if result.is_candidate:
           logger.info(f"Detected {len(result.lines)} lines in {filepath}")
           logger.debug(f"Metrics: {result.metrics}")
   
       # Access runtime params from context
       runtime_params = context.get("runtime_params", {})
       logger.debug(f"Used params: {runtime_params}")
   ```

2. **Access frame indices** in the context:
   ```python
   def on_detection_result(self, context, result, filepath):
       metadata = context.get("metadata", {})
       current_frame = metadata.get("current", {}).get("frame_index")
       prev_frame = metadata.get("previous", {}).get("frame_index")
       logger.info(f"Processing frames {prev_frame} → {current_frame}")
   ```

### Debug Image Optimization

To reduce memory usage in large batch processing:

- `DetectionResult.debug_image` is now cleared before returning non-candidate results
- Debug images are only generated and retained for actual candidates
- This significantly reduces memory pressure when processing thousands of images

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/pipeline.py` | Added `on_detection_result` invocation, frame index tracking, debug image optimization |
| `meteor_core/schema.py` | Frame index fields in metadata |
| `detect_meteors_cli.py` | Updated result tuple unpacking, progress display with frame indices |
| `PLUGIN_AUTHOR_GUIDE.md` | Updated lifecycle hooks documentation |
| `CHANGELOG.md` | Added v1.6.4 entry |
| `README.md` | Updated "What's New" section |

### Backward Compatibility

✅ **Fully backward compatible** with v1.6.3, v1.6.2, v1.6.1, and v1.6.0:

- **CLI**: No changes to command-line interface
- **Runtime**: No changes to detection behavior
- **API**: Existing plugins work unchanged; new hooks are optional
- **progress.json**: New fields added; existing fields unchanged

---

## Version 1.6.3 (2025-12-24) 🎅

### 🔧 RuntimeParams Contract and Pipeline Normalization

Version 1.6.3 completes the schema versioning initiative by adding `RuntimeParams` as a versioned dataclass and implementing automatic normalization checkpoints throughout the pipeline.

### Highlights

- **RuntimeParams dataclass**: Formalized runtime parameter passing with `schema_version`, namespaced structure, and `to_dict()` serialization
- **DetectionContext.to_dict()**: New serialization method for logging and debugging (excludes image/mask payloads)
- **Pipeline normalization checkpoints**: Automatic normalization of `InputContext`, `DetectionResult`, and `OutputResult` at pipeline boundaries
- **Legacy boolean compatibility**: Output handlers returning `bool` are automatically wrapped in `OutputResult` with deprecation warning
- **Documentation updates**: Comprehensive Plugin Author Guide updates for all contract types

### Why This Change?

This release finalizes the contract standardization across all plugin interfaces:

| Contract | v1.6.1 | v1.6.2 | v1.6.3 |
|----------|--------|--------|--------|
| `DetectionContext` | ✅ schema_version | — | ✅ `to_dict()` |
| `DetectionResult` | ✅ schema_version, metrics, `to_dict()` | — | — |
| `InputContext` | — | ✅ schema_version, `to_dict()` | — |
| `OutputResult` | — | ✅ schema_version, metrics, `to_dict()` | — |
| `RuntimeParams` | — | — | ✅ NEW: schema_version, `to_dict()` |
| **Pipeline normalization** | — | — | ✅ All contracts |

Benefits:
1. **Complete serialization**: All contracts now support `to_dict()` for JSON-compatible logging
2. **Namespaced parameters**: `RuntimeParams` separates global params from detector-specific overrides
3. **Automatic validation**: Pipeline normalizes all plugin outputs, catching schema mismatches early
4. **Smooth migration**: Legacy `bool` returns from output handlers work with deprecation warnings

### Schema Changes (v1.6.3)

**RuntimeParams** (new in v1.6.3):
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

Serialized form via `to_dict()`:
```python
{
    "schema_version": 1,
    "global": {"diff_threshold": 8, "min_area": 10, ...},
    "detector": {"hough": {"hough_threshold": 50}, ...},
}
```

**DetectionContext.to_dict()** (new in v1.6.3):
```python
@dataclass
class DetectionContext:
    # ... existing fields ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging.

        Excludes current_image, previous_image, and roi_mask
        to avoid large binary data in logs.
        """
        ...
```

**Constants added**:
```python
RUNTIME_PARAMS_SCHEMA_VERSION = 1
```

### Pipeline Normalization

The pipeline now automatically normalizes plugin outputs at these checkpoints:

| Checkpoint | Function | Action |
|------------|----------|--------|
| After `loader.load()` | `normalize_input_context()` | Validates schema, applies converters |
| After `detector.detect()` | `normalize_detection_result()` | Validates schema, applies converters |
| After `handler.save_candidate()` | `normalize_output_result()` | Validates schema, wraps legacy `bool` |

**Legacy boolean compatibility**:

Output handlers returning `bool` instead of `OutputResult` are automatically wrapped:

```python
# Legacy handler (deprecated but still works)
def save_candidate(self, source_path, filename, ...) -> bool:
    # ... save logic ...
    return True  # or False

# Pipeline automatically converts to:
# OutputResult(saved=True, output_path=None, debug_path=None)
# with a deprecation warning logged
```

### Migration Guide for Plugin Authors

**No immediate migration required.** All existing plugins continue to work.

**Recommended updates**:

1. **For Detector authors**: Access runtime params via the namespaced structure
   ```python
   def detect(self, context: DetectionContext) -> DetectionResult:
       # Use helper methods from BaseDetector
       global_params, detector_params = self.split_runtime_params(
           context.runtime_params
       )
       params = {**global_params, **detector_params}
   ```

2. **For Output Handler authors**: Return `OutputResult` instead of `bool`
   ```python
   # Before (deprecated)
   def save_candidate(self, ...) -> bool:
       return True
   
   # After (recommended)
   def save_candidate(self, ...) -> OutputResult:
       return OutputResult(saved=True, output_path=dest_path, debug_path=None)
   ```

3. **For debugging/logging**: Use `to_dict()` methods
   ```python
   logger.debug(f"Detection context: {context.to_dict()}")
   logger.debug(f"Runtime params: {context.runtime_params.to_dict()}")
   ```

### BaseDetector Helper Methods

`BaseDetector` provides convenience methods for working with `RuntimeParams`:

| Method | Signature | Description |
|--------|-----------|-------------|
| `split_runtime_params` | `(runtime_params) -> (global_params, detector_params)` | Extract namespaced params |
| `build_runtime_params` | `(flat_params) -> RuntimeParams` | Convert flat dict to RuntimeParams |
| `detect_legacy` | `(current, previous, roi, params) -> DetectionResult` | Adapter for old signature |

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/schema.py` | Added `RuntimeParams`, `DetectionContext.to_dict()` |
| `meteor_core/pipeline.py` | Added normalization checkpoints, legacy bool wrapping |
| `PLUGIN_AUTHOR_GUIDE.md` | Comprehensive updates for all contracts |
| `CHANGELOG.md` | Added v1.6.3 entry |

### Backward Compatibility

✅ **Fully backward compatible** with v1.6.2, v1.6.1, and v1.6.0:

- **CLI**: No changes
- **Runtime**: No changes to detection behavior
- **API**: All existing plugins work unchanged
- **Output handlers**: Legacy `bool` returns work with deprecation warning
- **Detectors**: Legacy flat dict params still supported

---

## Version 1.6.2 (2025-12-23) 🇯🇵

### 🔧 Input/Output Context Contracts

Version 1.6.2 extends schema versioning to input loaders and output handlers, completing the contract standardization across all plugin types.

### Highlights

- **Input context contracts**: `InputContext` formalizes loader return values with `schema_version`, `loader_info`, and `to_dict()`
- **Output result contracts**: `OutputResult` formalizes handler return values with `schema_version`, `handler_info`, `metrics`, and `to_dict()`
- **Complete plugin contract coverage**: All three plugin types (loaders, detectors, handlers) now have versioned contracts
- **Documentation updates**: Plugin Author Guide updated with comprehensive `InputContext`/`OutputResult` documentation

### Why This Change?

This release completes the schema versioning initiative started in v1.6.1:

| Plugin Type | v1.6.1 | v1.6.2 |
|-------------|--------|--------|
| **Detectors** | ✅ `DetectionContext`, `DetectionResult` | — |
| **Input Loaders** | — | ✅ `InputContext` |
| **Output Handlers** | — | ✅ `OutputResult` |

Benefits:
1. **Consistent contracts**: All plugin types follow the same pattern (schema_version, info dict, to_dict())
2. **Future migration support**: Version field enables gradual migration without breaking existing plugins
3. **Improved diagnostics**: `metrics` field in `OutputResult` for standardized performance tracking
4. **Serialization support**: `to_dict()` methods for JSON-compatible logging and debugging

### Schema Changes (v1.6.2)

**InputContext** (new in v1.6.2):
```python
@dataclass
class InputContext:
    image_data: ImageLike              # Loaded image (numpy, torch, or PIL)
    filepath: str                      # Original file path
    metadata: Dict[str, Any] = {}      # Loader-extracted metadata (EXIF, etc.)
    loader_info: Dict[str, Any] = {}   # Loader identity (name, version)
    schema_version: int = 1            # INPUT_CONTEXT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for JSON/logging (excludes image_data)."""
        ...
```

**OutputResult** (new in v1.6.2):
```python
@dataclass
class OutputResult:
    saved: bool                        # Whether file was persisted
    output_path: Optional[str]         # Path to saved candidate
    debug_path: Optional[str]          # Path to saved debug image
    handler_info: Dict[str, Any] = {}  # Handler identity (name, version)
    metrics: Dict[str, Any] = {}       # Performance diagnostics
    schema_version: int = 1            # OUTPUT_RESULT_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for JSON/logging."""
        ...
```

**Constants added**:
```python
INPUT_CONTEXT_SCHEMA_VERSION = 1
OUTPUT_RESULT_SCHEMA_VERSION = 1
```

### Contract Pattern Summary

All plugin contracts now follow a consistent pattern:

| Contract | Plugin Type | Info Field | Metrics | Schema Version |
|----------|-------------|------------|---------|----------------|
| `DetectionContext` | Detector (input) | — | — | ✅ |
| `DetectionResult` | Detector (output) | — | ✅ | ✅ |
| `InputContext` | Loader (output) | `loader_info` | — | ✅ |
| `OutputResult` | Handler (output) | `handler_info` | ✅ | ✅ |

### Migration Guide for Plugin Authors

Plugin architecture remains **experimental**, and v1.6.2 introduces a breaking change for output handlers used by the pipeline.

**Required for Output Handler authors**:
1. Return `OutputResult` from `save_candidate`. The pipeline expects `.saved`, `.output_path`, and `.debug_path`.
2. Update any legacy `OutputWriter`-based handlers (or wrappers returning `bool`) to return `OutputResult` instead.

**For Input Loader authors** (optional enhancements):
1. Return `InputContext` instead of raw image array
2. Populate `loader_info` using `self.get_info()` from `BaseInputLoader`
3. Use `context.to_dict()` for logging

**Recommended for Output Handler authors**:
1. Populate `handler_info` using `self.get_info()` from `BaseOutputHandler`
2. Use `metrics` for performance tracking (duration_ms, bytes_written, etc.)
3. Use `result.to_dict()` for logging

### Example: Updated Loader Implementation

```python
from meteor_core.inputs import DataclassInputLoader
from meteor_core.schema import InputContext

class MyLoader(DataclassInputLoader[MyConfig]):
    def load(self, filepath: str) -> InputContext:
        image = self._load_image(filepath)
        return InputContext(
            image_data=image,
            filepath=filepath,
            metadata=self.extract_metadata(filepath),
            loader_info=self.get_info(),  # {"name": ..., "version": ...}
        )
```

### Example: Updated Handler Implementation

```python
from meteor_core.outputs import DataclassOutputHandler
from meteor_core.schema import OutputResult
import time

class MyHandler(DataclassOutputHandler[MyConfig]):
    def save_candidate(self, source_path, filename, ...) -> OutputResult:
        start = time.perf_counter()
        dest_path = self._save_file(source_path, filename)
        duration_ms = (time.perf_counter() - start) * 1000

        return OutputResult(
            saved=True,
            output_path=dest_path,
            debug_path=None,
            handler_info=self.get_info(),
            metrics={"duration_ms": duration_ms},
        )
```

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/schema.py` | Added `InputContext`, `OutputResult`, schema version constants |
| `PLUGIN_AUTHOR_GUIDE.md` | Updated loader/handler documentation with new contracts |
| `CHANGELOG.md` | Added v1.6.2 entry |
| `README.md` | Updated "What's New" section |
| `ROADMAP.md` | Added v1.6.2 milestone |

### Backward Compatibility

✅ **Fully backward compatible** with v1.6.1 and v1.6.0:

- **CLI**: No changes
- **Runtime**: No changes to detection behavior
- **API**: Existing loader/handler plugins work unchanged
- **Configuration**: No changes


---

## Version 1.6.1 (2025-12-22) 🌃

### 🔧 Plugin Schema Versioning and ML-Ready Architecture

Version 1.6.1 introduces schema versioning for detector plugin contracts and multi-framework image support, laying the groundwork for ML-based detectors in v2.x.

### Highlights

- **Schema versioning**: `DetectionContext` and `DetectionResult` now include `schema_version` field

- **Multi-framework images**: New `ImageLike` type supports numpy, PyTorch, and PIL
- **Enhanced diagnostics**: `DetectionResult.metrics` for standardized performance metrics
- **Serialization support**: `DetectionResult.to_dict()` for JSON-compatible output

### Why This Change?

This release prepares the plugin architecture for:
1. **Future schema migrations**: Version field enables gradual migration without breaking existing plugins
2. **ML detector support**: `ImageLike` type allows detectors to work with PyTorch tensors natively
3. **Standardized metrics**: Consistent diagnostics across different detector implementations

### Schema Changes (v1.6.1)

**DetectionContext** (v1.6.1):
```python
@dataclass
class DetectionContext:
    current_image: ImageLike      # NEW: Union[np.ndarray, torch.Tensor, PIL.Image]
    previous_image: ImageLike     # NEW: Union[np.ndarray, torch.Tensor, PIL.Image]
    roi_mask: Any
    runtime_params: Dict[str, Any]
    metadata: Dict[str, Any]
    schema_version: int = 1       # NEW: For migration support
```

**DetectionResult** (v1.6.1):
```python
@dataclass
class DetectionResult:
    is_candidate: bool
    score: float
    lines: List[Tuple[int, int, int, int]]
    aspect_ratio: float
    debug_image: Optional[Any]
    extras: Dict[str, Any]
    metrics: Dict[str, Any]       # NEW: Standard diagnostics
    schema_version: int = 1       # NEW: For migration support
    def to_dict(self) -> Dict[str, Any]:  # NEW: Serialization
        ...
```

### New Utility Functions

**`ensure_numpy()`** in `meteor_core.utils`:
```python
from meteor_core.utils import ensure_numpy

# Convert any ImageLike to numpy array
image = ensure_numpy(context.current_image)  # Works with numpy, torch, PIL
```

### Migration Guide for Plugin Authors

No migration required. Existing plugins work without modification.

**Optional enhancements**:
1. Use `ensure_numpy()` for type-safe image handling
2. Populate `metrics` dict with standard diagnostics
3. Use `result.to_dict()` for logging

### Files Changed

| File | Changes |
|------|---------|
| `meteor_core/schema.py` | Added `ImageLike`, schema versions, `to_dict()` |
| `meteor_core/utils.py` | Added `ensure_numpy()` function |
| `PLUGIN_AUTHOR_GUIDE.md` | Updated DetectionContext/DetectionResult documentation |
| `CHANGELOG.md` | Added v1.6.1 entry |
| `README.md` | Updated "What's New" section |

### Backward Compatibility

✅ **Fully backward compatible** with v1.6.0:

- **CLI**: No changes
- **Runtime**: No changes to detection behavior
- **API**: All existing detector plugins work unchanged
- **Configuration**: No changes

---

## Version 1.6.0 (2025-12-21) 💌

### ⚡ Development Toolchain Modernization

Version 1.6.0 modernizes the development toolchain by migrating from pip/black/flake8 to uv/Ruff. This change dramatically improves developer experience with faster dependency management and unified linting/formatting.

### Highlights

- **uv**: Rust-based Python package manager (10-100× faster than pip)
- **Ruff**: Rust-based linter and formatter replacing both black and flake8
- **Unified configuration**: All tool settings in `pyproject.toml`
- **Simplified setup**: Single command installs everything

### Why This Change?

| Aspect | Before (v1.5.x) | After (v1.6.0) |
|--------|-----------------|----------------|
| **Package manager** | pip | uv |
| **Formatter** | black | Ruff |
| **Linter** | flake8 | Ruff |
| **Config files** | `pyproject.toml` + `.flake8` | `pyproject.toml` only |
| **Install speed** | ~30-60 seconds | ~2-5 seconds |
| **Lint + format** | Two separate tools | Single unified tool |

### Performance Comparison

**Dependency installation:**
```bash
# Before (pip)
pip install -e ".[dev]"  # ~30-60 seconds

# After (uv)
uv sync --all-extras     # ~2-5 seconds
```

**Linting and formatting:**
```bash
# Before (black + flake8)
black .                  # ~2 seconds
flake8 .                 # ~1 second

# After (Ruff)
ruff check --fix .       # ~0.1 seconds
ruff format .            # ~0.1 seconds
```

### Migration Guide for Contributors

#### New Developer Setup

```bash
# 1. Install uv (if not already installed)
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Clone and setup
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
uv sync --all-extras

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Verify setup
uv run pre-commit run --all-files
```

#### For Existing Contributors

If you have an existing development environment:

```bash
# Update your local repository
git pull origin main

# Remove old virtual environment (optional but recommended)
rm -rf .venv

# Create new environment with uv
uv sync --all-extras

# Reinstall pre-commit hooks
uv run pre-commit install
```

### New Development Workflow

#### Running Commands

All Python commands should be prefixed with `uv run`:

```bash
# Run tests
uv run python run_tests.py

# Run the CLI
uv run python detect_meteors_cli.py --help

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .
```

#### Pre-commit Hooks

The pre-commit configuration now uses local Ruff hooks:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff check
        entry: .venv/bin/ruff check --fix
        language: system
        types: [python]
      - id: ruff-format
        name: ruff format
        entry: .venv/bin/ruff format
        language: system
        types: [python]
```

### Configuration Changes

#### pyproject.toml

The `[tool.ruff]` section replaces both black and flake8 configurations:

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = ["E", "W", "F", "C90"]
ignore = ["E203", "E501", "E226"]

[tool.ruff.lint.mccabe]
max-complexity = 40

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

### Ruff vs black/flake8 Compatibility

Ruff is designed to be a drop-in replacement for black and flake8. Key compatibility notes:

| Feature | black/flake8 | Ruff |
|---------|--------------|------|
| Line length | `max-line-length = 88` | `line-length = 88` |
| Quote style | black default (double) | `quote-style = "double"` |
| Complexity | `max-complexity = 40` | `[tool.ruff.lint.mccabe] max-complexity = 40` |
| Ignored rules | `.flake8` ignore list | `[tool.ruff.lint] ignore` |

### Dependencies Updated

**Development dependencies (optional):**

```toml
[project.optional-dependencies]
dev = [
    "ruff==0.14.10",      # Replaces black + flake8
    "pre-commit>=4.5.0",
    "coverage>=7.6.0",
]
```

**Removed from dev dependencies:**
- `black>=25.12.0`
- `flake8>=7.3.0`
- `flake8-pyproject>=1.2.4`

### FAQ

#### Q: Do I need to install uv?

Yes, uv is now the recommended way to manage dependencies for this project. It's a single binary with no dependencies.

#### Q: Can I still use pip?

While pip still works for basic installation (`pip install -e .`), the development workflow is optimized for uv. Pre-commit hooks expect Ruff to be available via the uv-managed virtual environment.

#### Q: Will Ruff format code differently than black?

Ruff's formatter is designed to be compatible with black. In most cases, the output is identical. Minor differences may occur in edge cases, but these are cosmetic and don't affect functionality.

#### Q: Is the linting stricter or more lenient?

The same rule set is applied. Ruff supports the same error codes as flake8 (E, W, F, C90), and the ignore list has been preserved.

### Files Changed

| File | Changes |
|------|---------|
| `pyproject.toml` | Added `[tool.ruff]` section, updated dev dependencies |
| `.pre-commit-config.yaml` | Replaced black/flake8 with local Ruff hooks |
| `INSTALL_DEV.md` | Rewritten for uv/Ruff workflow |
| `CHANGELOG.md` | Added v1.6.0 entry |
| `README.md` | Updated "What's New" section |

### Backward Compatibility

✅ **Fully backward compatible** with v1.5.x:

- **CLI**: No changes to command-line interface
- **Runtime**: No changes to detection behavior or output
- **API**: No changes to `meteor_core` module interfaces
- **Configuration**: No changes to detection parameters or sensor presets

This release only affects the development toolchain. Users who install the package without development dependencies will not notice any difference.

### Links

- [uv documentation](https://docs.astral.sh/uv/)
- [Ruff documentation](https://docs.astral.sh/ruff/)
- [INSTALL_DEV.md](INSTALL_DEV.md) - Updated developer setup guide

---

## Version Information

### v1.6.5 (Latest) 📓
- **Version**: 1.6.5
- **Release Date**: 2025-12-26
- **Major Changes**:
  - Pipeline configuration file support (YAML/JSON) via `--config`
  - CLI plugin selection (`--input-loader`, `--detector`, `--output-handler`)
  - CLI plugin configuration (`--input-loader-config`, `--detector-config`, `--output-handler-config`)
  - DetectionContext normalization API publicly available
  - CLI execution routed through `MeteorDetectionPipeline`
  - Legacy parameter flags deprecated in favor of config files

### v1.6.4 🎄
- **Version**: 1.6.4
- **Release Date**: 2025-12-25
- **Major Changes**:
  - Output handler `on_detection_result` lifecycle hook
  - DetectionResult propagation through pipeline
  - Frame indices (`frame_index`, `prev_frame_index`) in detection context and progress.json
  - Debug image optimization for memory efficiency
  - Batch progress recording with frame indices in progress.json

### v1.6.3
- **Version**: 1.6.3
- **Release Date**: 2025-12-24
- **Major Changes**:
  - RuntimeParams dataclass with schema versioning
  - DetectionContext.to_dict() for serialization
  - Pipeline normalization checkpoints for all contracts
  - Legacy boolean compatibility for output handlers
  - Comprehensive Plugin Author Guide updates

### v1.6.2
- **Version**: 1.6.2
- **Release Date**: 2025-12-23
- **Major Changes**:
  - InputContext for standardized loader return values
  - OutputResult for standardized handler return values
  - Schema versioning for all plugin contracts
  - Complete plugin contract documentation

### v1.6.1
- **Version**: 1.6.1
- **Release Date**: 2025-12-22
- **Major Changes**:
  - Schema versioning for DetectionContext/DetectionResult
  - ImageLike type for multi-framework support
  - DetectionResult.metrics for standardized diagnostics
  - ensure_numpy() utility for type-safe conversion

### v1.6.0
- **Version**: 1.6.0
- **Release Date**: 2025-12-21
- **Major Changes**:
  - pip → uv (package management)
  - black + flake8 → Ruff (formatting + linting)
  - Unified configuration in pyproject.toml
  - Simplified developer setup workflow


---

**Status**: Production Ready  
**Compatibility**: Fully backward compatible with v1.5.x  
**Recommendation**: Use configuration files (`--config`) for reproducible pipeline setups; plugin authors should document ConfigType fields for config file users and implement lifecycle hooks for enhanced observability

Happy meteor hunting! 🌠📓
