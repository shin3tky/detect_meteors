# Roadmap

## Version 1.x - Foundation and Auto-Parameter Optimization

-2025 4Q

### Core Detection System
- [x] **v1.0.1** - Initial release with manual parameter configuration
- [x] **v1.0.2** - Enhanced help descriptions for Hough transform parameters
- [x] **v1.0.3** - RAW file validation and automatic batch-size tuning

### Progress Management
- [x] **v1.1.0** - Resumable processing with JSON progress tracking
- [x] **v1.1.0** - Safe Ctrl-C interruption handling

### Auto-Parameter Estimation (Image-based)
- [x] **v1.2.0** - Initial `diff_threshold` auto-estimation (3-sigma rule)
- [x] **v1.2.1** - Improved percentile-based `diff_threshold` estimation
- [x] **v1.3.1** - Complete auto-parameter estimation for all three parameters
  - [x] Star size distribution analysis for `min_area`
  - [x] Image geometry-based `min_line_score` estimation
  - [x] Focal length support for lens-specific optimization

### Auto-Parameter Optimization (Physics-based)
- [x] **v1.4.1** - NPF Rule-based scientific optimization (MILESTONE)
  - [x] EXIF metadata extraction from RAW files
  - [x] NPF Rule integration for exposure time validation
  - [x] Star trail physics estimation (Earth's rotation)
  - [x] Meteor speed physics (3× faster than stars)
  - [x] Shooting condition quality scoring
  - [x] Sensor characterization (pixel pitch calculation)

### Safety Enhancement
- [x] **v1.4.2** - Output file protection and safety features
  - [x] Skip overwriting existing output files by default
  - [x] `--output-overwrite` flag for explicit overwrite permission
  - [x] Warning and exit when target and output directories are identical

### Sensor Type Presets
- [x] **v1.5.0** - Simplified NPF configuration with sensor presets
  - [x] Unified `SENSOR_PRESETS` configuration (MFT, APS-C, APS-C_CANON, APS-H, FF, 1INCH)
  - [x] New `--sensor-type` option for one-stop sensor configuration
  - [x] New `--list-sensor-types` option to display available presets
  - [x] Parameter override priority (individual args > preset > default)
  - [x] Updated shell completion scripts (bash/zsh)
  - [x] Backward compatibility with v1.4.x options

### Medium Format Support
- [x] **v1.5.1** - Medium format sensor support
  - [x] MF44X33 preset (Fujifilm GFX, Pentax 645Z, Hasselblad X2D)
  - [x] MF54X40 preset (Hasselblad H6D-100c)
  - [x] Sensor size ordering (smallest to largest)
  - [x] Updated shell completion scripts with medium format types

### Sensor Override Validation
- [x] **v1.5.2** - Automatic validation for sensor preset overrides
  - [x] Warning when `--sensor-width` deviates more than ±30% from preset
  - [x] Warning when `--pixel-pitch` deviates more than ±50% from preset
  - [x] Informational warnings only (processing continues normally)
  - [x] Enhanced `apply_sensor_preset()` with validation support
  - [x] Comprehensive test coverage (23 test cases)

### Fisheye Lens Correction
- [x] **v1.5.3** - Fisheye lens support with equisolid angle projection
  - [x] `--fisheye` flag for fisheye correction
  - [x] Position-dependent effective focal length calculation
  - [x] NPF calculation based on edge (worst case) focal length
  - [x] Star trail ratio adjustment for image edges
  - [x] Extensible projection model infrastructure
  - [x] Comprehensive test coverage (27 test cases)

### ROI Display and Documentation
- [x] **v1.5.4** - ROI display improvement and documentation
  - [x] Brightened ROI selection image for dark conditions
  - [x] NOTICE document for third-party attributions

### Code Architecture Refactoring
- [x] **v1.5.5** - Modular code structure for v2.x preparation
  - [x] Separated CLI interface from core logic
  - [x] Created `meteor_core/` package with modular components
  - [x] Introduced `detectors/` subpackage with base class
  - [x] Introduced `outputs/` subpackage for result handling
  - [x] Enhanced type safety with TypedDict
  - [x] Maintained full backward compatibility

### Input/Output Plugin Infrastructure
- [x] **v1.5.6** - Plugin-ready input/output foundation
  - [x] `InputLoader`/`MetadataExtractor` protocols with dataclass/Pydantic helper bases
  - [x] Built-in `RawImageLoader` plugin plus discovery via entry points and standardized plugin directories (`~/.detect_meteors/input_plugins`, `~/.detect_meteors/detector_plugins`, `~/.detect_meteors/output_plugins`; input path renamed from `~/.detect_meteors/plugins` in v1.5.11)
  - [x] `PipelineConfig` and `DetectionPipeline` protocol to centralize orchestration and loader resolution
  - [x] `OutputHandler` protocol to standardize candidate/debug persistence

### Progress Metadata & Quality Tooling
- [x] **v1.5.7** - Enhanced `progress.json` with CLI parameters, ROI, and processing metadata
- [x] **v1.5.8** - Added flake8 linting to complement Black formatting

### Packaging Modernization
- [x] **v1.5.9** - Migrated project configuration to PEP 621 `pyproject.toml`

### Plugin Architecture Hardening
- [x] **v1.5.10** - Migrated plugin interfaces from Protocol to ABC base classes
- [x] **v1.5.11** - Unified plugin registry behavior and standardized plugin directories

### Stability & Error Handling
- [x] **v1.5.12** - Custom exception hierarchy and diagnostic reporting
  - [x] Structured exception classes for inputs (`MeteorLoadError`, `MeteorUnsupportedFormatError`)
  - [x] Structured exception classes for outputs (`MeteorOutputError`, `MeteorWriteError`, `MeteorProgressError`)
  - [x] Structured exception classes for config (`MeteorValidationError`, `MeteorConfigError`)
  - [x] Diagnostic information with system details and dependency versions
  - [x] `--verbose` flag for detailed error info and DEBUG logging
  - [x] `--save-diagnostic` option for bug reporting
  - [x] Standard Python logging throughout all modules

### Internationalization
- [x] **v1.5.13** - Multi-language support for CLI messages
  - [x] `--locale` option for display language selection
  - [x] `DETECT_METEORS_LOCALE` environment variable for default locale
  - [x] ICU-style message templates with plural rule support
  - [x] YAML-based locale catalogs under `meteor_core/locales/`
  - [x] English (`en`) and Japanese (`ja`) translations
  - [x] UI/UX messages localized; system/debug output in English
  - [x] Progress file normalization helpers

### Development Tooling Modernization
- [x] **v1.6.0** - Migration to modern Python tooling
  - [x] Replace Black + flake8 with `ruff` (unified linter/formatter)
  - [x] Adopt `uv` for faster dependency management and virtual environment handling
  - [x] Update `pyproject.toml` for ruff configuration
  - [x] Update pre-commit hooks for ruff integration

### Plugin Contract Versioning
- [x] **v1.6.1** - Schema versioning and ML-ready architecture
  - [x] Schema versioning for `DetectionContext` and `DetectionResult`
  - [x] `ImageLike` type alias for multi-framework support (numpy, PyTorch, PIL)
  - [x] `ensure_numpy()` utility for type-safe image conversion
  - [x] `DetectionResult.metrics` field for standardized diagnostics
  - [x] `DetectionResult.to_dict()` method for serialization
  - [x] Updated Plugin Author Guide with detector schema documentation

### Input/Output Contract Standardization
- [x] **v1.6.2** - Input/output context contracts
  - [x] `InputContext` dataclass for standardized loader return values
  - [x] `OutputResult` dataclass for standardized handler return values
  - [x] Schema versioning (`INPUT_CONTEXT_SCHEMA_VERSION`, `OUTPUT_RESULT_SCHEMA_VERSION`)
  - [x] `loader_info`/`handler_info` fields for plugin identity tracking
  - [x] `metrics` field in `OutputResult` for performance diagnostics
  - [x] `to_dict()` methods for serialization support
  - [x] Updated Plugin Author Guide with loader/handler contract documentation

### Runtime Parameters and Pipeline Normalization
- [x] **v1.6.3** - RuntimeParams contract and pipeline normalization
  - [x] `RuntimeParams` dataclass for formalized runtime parameter passing
  - [x] Schema versioning (`RUNTIME_PARAMS_SCHEMA_VERSION`)
  - [x] Namespaced structure (`global_params`, `detector` per-plugin overrides)
  - [x] `RuntimeParams.to_dict()` method for serialization
  - [x] `DetectionContext.to_dict()` method for logging/debugging
  - [x] `BaseDetector` helpers: `split_runtime_params()`, `build_runtime_params()`, `detect_legacy()`
  - [x] Pipeline normalization checkpoints for `InputContext`, `DetectionResult`, `OutputResult`
  - [x] Legacy boolean compatibility for output handlers (with deprecation warning)
  - [x] Updated Plugin Author Guide with comprehensive contract documentation

### Output Handler Lifecycle Hooks
- [x] **v1.6.4** - Output handler hooks and DetectionResult propagation
  - [x] `on_detection_result` hook invoked per-detection with serialized context payload
  - [x] `DetectionResult` propagation through `process_image_batch()` result tuples
  - [x] Frame indices (`frame_index`, `prev_frame_index`) in detection context and progress tracking
  - [x] Debug image optimization (only generated for candidate detections)
  - [x] Performance improvement (`_build_runtime_params()` moved outside loop)
  - [x] Updated Plugin Author Guide with lifecycle hook documentation

### Pipeline Configuration and CLI Plugin Selection
- [x] **v1.6.5** - Configuration files and CLI plugin specification
  - [x] Pipeline configuration file support (YAML/JSON) via `--config`
  - [x] `load_pipeline_config()` utility for programmatic loading
  - [x] CLI plugin selection (`--input-loader`, `--detector`, `--output-handler`)
  - [x] CLI plugin configuration (`--input-loader-config`, `--detector-config`, `--output-handler-config`)
  - [x] Config values accept JSON strings, YAML strings, or file paths
  - [x] `DetectionContext` normalization API (`register_detection_context_converter()`, `normalize_detection_context()`)
  - [x] CLI execution routed through `MeteorDetectionPipeline` for unified processing
  - [x] Updated Plugin Author Guide with configuration file and CLI documentation

### Pipeline Hook System
- [x] **v1.6.6** - Pipeline hooks for extensible processing
  - [x] `on_file_found(filepath) -> bool` hook for file filtering before loading
  - [x] `on_image_loaded(context) -> InputContext` hook for image transformation/metadata enrichment
  - [x] `on_detection_complete(result, context) -> DetectionResult` hook for result adjustment
  - [x] `on_output_saved(result) -> None` hook for post-save notifications
  - [x] `HookRegistry` for centralized hook discovery and management
  - [x] Hook discovery via entry points (`detect_meteors.hook`) and plugin directory (`~/.detect_meteors/hook_plugins/`)
  - [x] `BaseHook`, `DataclassHook`, `PydanticHook` base classes for typed configurations
  - [x] CLI options (`--hooks`, `--hook-config`) for runtime hook configuration
  - [x] `PipelineConfig.hooks` and `hook_error_mode` for programmatic control
  - [x] Updated Plugin Author Guide with comprehensive hook documentation

### Static Type Checking
- [x] **v1.6.8** - ty type checker integration for plugin stability
  - [x] Astral's `ty` type checker configuration in `pyproject.toml`
  - [x] Pre-commit hook integration for automatic type checking
  - [x] Graduated rule enforcement (error vs warning levels)
  - [x] Plugin registry type safety improvements (detector, hook, input, output)
  - [x] Typed factory casts and validator callable pattern
  - [x] `MAX_NUM_WORKERS` constant with pipeline/CLI validation

### Sorted Detection Analysis
- [x] **v1.6.10 (2026-10-06) 🌠 Make Your Dream Come True Day** - Frame-ordered hooks and aircraft trail metadata (v1.6.9 skipped)
  - [x] `SortedDetection` records without image payloads
  - [x] `on_batch_results_sorted` for analysis within each result batch
  - [x] `on_all_detections_sorted` for analysis across the current run
  - [x] Sorted hooks executed in the main process, including parallel runs
  - [x] Built-in `aircraft_trail` hook with geometric tracking and likelihood/evidence metadata in `progress.json`
  - [x] Aircraft analysis preserves candidate decisions and scores

### Missed-Meteor Analysis and Parameter Tuning
- [x] **v1.6.11 (2026-11-07)** - Consolidate the 1.6 foundation and support missed-candidate analysis
  - [x] Separate Hough measurements, threshold decisions, and debug rendering while preserving normal detection behavior
  - [x] Read-only image-pair replay API and CLI with recorded parameters, ROI, provenance, and rejection reasons
  - [x] Bounded contour-derived tuning, optional finite measurement grids, and annotated target regions
  - [x] Sequence comparison with added/lost candidates, known-positive coverage, and explicit work limits
  - [x] Versioned JSON/PNG analysis bundles and explicitly selected reusable session configurations
  - [x] Persist ROI in pipeline configuration and support Bash/Zsh completion for the analysis workflow
  - [x] Align parallel candidate-save failures with sequential error propagation
  - [x] Add hook/error regression coverage, including real spawn workers (681 tests pass)
  - [x] Update runtime/development dependencies and document upstream security fixes

See the [release notes](RELEASE_NOTES_1.6.md#version-1611-2026-11-07) for usage
and the [design and validation record](parameter_analysis_1.6.11.md) for the
four reviewed implementation steps and extension points.

### Plugin and Hook Compatibility

**v1.6.12 (2026-11-11)**

- [x] Clarify the Hough detection and pipeline flow and improve function
  documentation for IDEs while preserving candidate criteria.
- [x] Use the plugin classes and schema converters selected by the main process
  in spawned workers. Restore loader and detector state per batch and report
  worker setup failures as configuration errors.
- [x] Read and write pipeline YAML/JSON with `schema_version: 1`, including
  exported `session.yaml`. Read unversioned files as v1 and reject unsupported
  explicit versions. Progress files and analysis reports have their own formats.
- [x] Report incompatible detector results with the detector name, frame path,
  and expected schema version. Leave the failed pair eligible for resume.
- [x] Validate image, detection-result, and sorted hook returns within the hook
  error policy. Verify `warn` and `raise` behavior and retry after a detector
  compatibility failure in sequential processing and with real spawn workers.

See the [release notes](RELEASE_NOTES_1.6.md#version-1612) for upgrade
and recovery guidance, and the [changelog](CHANGELOG.md#v1612) for the full list.

### 1.x Contract to Preserve

The [Plugin Author Guide](PLUGIN_AUTHOR_GUIDE.md) and its
[Japanese edition](PLUGIN_AUTHOR_GUIDE_ja.md) describe the public interfaces.
Use the following behavior as the baseline for running existing 1.x
configurations and plugins in 2.x.

| Area | 1.x behavior to preserve |
| --- | --- |
| Configuration and discovery | Existing unversioned and explicit v1 YAML/JSON resolve the same settings. Keep built-in defaults, explicit CLI override precedence, plugin names and lookup precedence, and the interpretation of plugin-specific settings. Preserve discovery through entry points and local plugin directories. |
| Plugin calls and data | Keep the documented loader, detector, output-handler, and hook signatures, defaults, and data contracts. Preserve the namespaced detector runtime parameters, registered converters for supported older payloads, and existing adapters for bare images and legacy boolean output results. The pipeline file version, plugin payload versions, and application/plugin releases are independent. |
| Hook order and process boundaries | Run hooks in configuration order. File filtering, output callbacks, and sorted hooks run in the main process. Per-pair image and detection hooks run in the main process for sequential execution or in workers for parallel execution, with instances shared within a batch. Keep candidate saving and progress recording ahead of sorted analysis. Sort each completed batch by frame index and sort the whole current invocation for final analysis, including after resume. |
| Worker requirements | Use the main process's plugin and converter selections in workers. Worker classes and converter functions must be importable, and loader and detector state must be serializable. |
| Errors and progress | A valid non-candidate counts as processed. An incompatible detector result stops the run and leaves the failed pair unrecorded. A parallel batch that fails before returning can leave other pairs unrecorded as well. Invalid hook returns follow `warn` (continue with the last accepted value, with in-place changes potentially retained) or `raise` (propagate `MeteorHookError`). Preserve progress already recorded and keep primary save failures fatal. |

### Pending (Deferred from v1.x)
The following features were originally planned for v1.x but have been deferred:

- [ ] Camera model database for automatic sensor detection from EXIF
- [ ] Per-image adaptive parameter adjustment
- [ ] Declination support with GPS coordinate extraction
- [ ] Advanced quality metrics (focus quality, atmospheric transparency)

## Version 2.x - Architecture and Extensibility

2026 1Q-

### Compatibility Before New Pipeline Features

Use the [1.x contract](#1x-contract-to-preserve) as the release baseline.
Multi-detector chains and other new stages are opt-in.

- [ ] Publish a compatibility matrix for the 1.x formats and plugin contracts
  supported by 2.x. Identify combinations that work directly, require an
  adapter, or produce a compatibility error.
- [ ] Extend the current regression coverage with fixtures captured from
  representative 1.x releases. Run the same configurations and plugins in 2.x
  sequentially and with spawn workers. Compare effective settings, candidate
  decisions, hook order and scope, and progress after failure and resume.
  Check saved progress, analysis reports, and replayable sessions with
  separate fixtures for their respective formats.
- [ ] Define how 2.x will read v1 pipeline files and preserve their defaults
  and CLI precedence. Add an in-memory migration when a new file schema is
  needed. Define progress and report migrations independently if those
  formats change.
- [ ] Preserve v1 plugin behavior when adding new stages. Prefer optional
  additions and introduce explicit adapters for incompatible changes. If a
  new plugin API needs a version declaration, keep undeclared 1.x plugins on
  the existing contract and require explicit opt-in to the new contract in
  both the main process and workers.

### Pipeline Modularity

- [ ] Swappable detector stacks (multi-detector chaining and fallback order)
- [ ] Pluggable pre/post processors (noise reduction, masking, ROI transforms)
- [ ] Pipeline presets and profiles (named configs with overrides)

### Plugin Ecosystem Expansion

- [ ] SDK templates and validation tooling for third-party plugins
- [ ] Plugin capability discovery (declared features/requirements)
- [ ] Distribution guidelines and example plugin gallery

### Integration & Interop
- [ ] Output adapters for popular annotation formats (COCO, YOLO, CSV/Parquet)
- [ ] Remote storage integration hooks (S3/GCS/Azure)
- [ ] Batch orchestration helpers (multiprocessing, queue-based workers)

## Version 3.x - Intelligence and Learning

2026 2Q-

### ML-based Detection
- [ ] Baseline ML detector integration (optional, non-default)
- [ ] Labeled dataset ingestion pipeline and annotation tooling
- [ ] Train/evaluate CLI workflow with reproducible configs
- [ ] Model registry and versioned model selection

### Intelligent Post-processing
- [ ] Advanced pattern recognition (meteor vs. noise discrimination)
- [ ] Adaptive learning from user feedback (false-positive suppression)
- [ ] Multi-object classification (meteors, aircraft, satellites); the existing aircraft hook supplies auxiliary heuristics only

### Performance & Deployment
- [ ] Accelerated inference options (ONNX, GPU backends)
- [ ] Streaming/near-real-time detection mode
- [ ] Edge-friendly lightweight model variants

---

**Latest release**: [v1.6.12](https://github.com/shin3tky/detect_meteors/releases/tag/1.6.12),
released 2026-11-11 — Plugin and hook compatibility and code readability
