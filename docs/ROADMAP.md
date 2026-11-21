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

### Release Build and Verification

**v1.6.13 (2026-11-16)**

- [x] Build the wheel, source distribution, and `SHA256SUMS` from the tagged
  commit. Check fresh installations and CLI startup before publication.
- [x] Verify that public downloads match the original build artifact and repeat
  the installation checks. Record the matching and mismatched workflow tests.

The v1.6.13 distribution supports Python 3.12 and 3.13. The current source
checkout also supports Python 3.14 and includes it in the test and release
workflows. See the [v1.6.13 release notes](RELEASE_NOTES_1.6.md#version-1613)
and the [installation guide](INSTALL.md#python-requirements).

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

### Continued Development on the 1.x Line

- [ ] Continue improving the built-in Hough detector and its replay and
  threshold-tuning workflow. Use the existing analysis tools to investigate
  reports such as bright, wide trails missed by the aspect-ratio threshold.
  [Issue #89](https://github.com/shin3tky/detect_meteors/issues/89) ·
  [Replay and tuning validation](parameter_analysis_1.6.11.md)
- [ ] Continue refining the built-in aircraft annotations on reviewed
  sequences. Evaluate track continuity and cases with both a meteor and an
  aircraft using the [aircraft sample](aircraft_sample_validation.md) and
  additional observations as they become available.

### Pending (Deferred from v1.x)
The following features were originally planned for v1.x but have been deferred:

- [ ] Camera model database for automatic sensor detection from EXIF
- [ ] Per-image adaptive parameter adjustment
- [ ] Declination support with GPS coordinate extraction
- [ ] Advanced quality metrics (focus quality, atmospheric transparency)

## Version 2.x - Architecture and Extensibility

Make detectors, hooks, and outputs independently useful and composable. A
plugin author should be able to introduce a different way to evaluate a frame
pair, add information at a documented stage, or deliver results to a new
destination using the public extension points. Start with the registries,
versioned contracts, ordered hooks, and YAML/JSON configuration shipped in
1.x. [Plugin Author Guide](PLUGIN_AUTHOR_GUIDE.md)

Keep Hough detection and file output as the default workflow. Bring validated
1.x improvements into 2.x with their tests and configuration behavior. Use
the [1.x contract](#1x-contract-to-preserve) as the compatibility baseline
and make new combinations opt-in.

### v2.0.1 - Try the Extension Points End to End

- [ ] Package an opt-in reference detector with a different decision rule.
  Start by comparing the [candidate combinations](#candidate-detection-and-classification-methods)
  that fit the current frame-pair interface. Run the reference detector and
  Hough on the same sequence using separate configurations. Record how each
  detector reaches its decision and what information it makes available to
  hooks and outputs. See the [paired comparison plan](detector_comparison_2.0.1_ja.md).
- [ ] Evaluate opt-in stellar registration before frame differencing as an
  early 2.x preprocessing candidate. Across 121 RAW pairs, the prototype
  added about 61 ms per pair and reduced stellar residuals. Start with
  translation, which accounted for most of the measured improvement. Compare
  adjacent subtraction and a temporal-median background with and without
  registration at matched false-positive rates. Record faint and short trail
  recovery and preprocessing time before selecting a configuration.
  [Stellar registration evaluation](star_registration_evaluation_ja.md)
- [ ] Package an opt-in local review output that writes a structured report
  through the existing output handler interface. Exercise it with both
  detectors using a common format for frame identity, candidate decision,
  and detector-specific evidence. Use this example to identify what becomes
  available before saving, after saving, and after sorted analysis.
- [ ] Exercise a per-pair annotation hook and the existing sorted hooks in
  the example workflows. Document which data each hook can use, when its
  changes are visible to consumers, and how execution differs between
  sequential runs, spawn workers, and resume.
- [ ] Publish the example configurations and compatibility matrix. Verify
  v1 configs, progress and resume, analysis bundles and exported sessions,
  and representative legacy plugins. Check the wheel and source
  distribution with the existing release workflows on Python 3.12–3.14.

### Candidate Detection and Classification Methods

These methods are candidates for experiments. A detector plugin can first
own processing that depends on both frames. Use image hooks for transforms
that operate on each loaded image independently, and sorted hooks for
analysis of the resulting evidence across frames.
[Plugin lifecycle](PLUGIN_AUTHOR_GUIDE.md#1-application-lifecycle)

| Stage | Method | Use and feasibility |
| --- | --- | --- |
| Preprocessing | Stellar registration | Match stars in the sky ROI and resample the reference exposure into the current frame's coordinates before subtraction. Use the unregistered pair if too few well-distributed matches support the transform. |
| Preprocessing | Difference of Gaussians (DoG) | Subtract two differently blurred images to emphasize a range of feature widths. Try it on the pair residual with the existing OpenCV dependency. Check how faint trails and star residuals respond to the chosen scales. |
| Preprocessing | White top-hat transform | Extract bright detail relative to a morphological opening. Try it as an alternative to DoG. Check how the structuring element affects wide or broken trails. |
| Background subtraction | Adjacent-frame subtraction and median filtering | Use the current frame difference as a control. Test a spatial median filter for isolated noise and check whether it also removes thin trails. Evaluate a temporal median of several reference frames as a separate background estimate. |
| Line segment detection | LSD (Line Segment Detector) | Produce segment endpoints from a grayscale residual. OpenCV LSD takes 8-bit input, so record the conversion from the binned RAW residual and check loss of faint detail and multiple edges from a single trail. |
| Line detection | Radon transform | Integrate brightness along lines at different angles. Try it on bounded regions, recover segment endpoints from the supporting pixels, and measure CPU time and memory before expanding the search. |
| Region analysis | Labeling and connected-component analysis | Label a binary foreground mask, then measure each region's shape and contrast using the residual. This fits the existing OpenCV-based frame-pair workflow. |
| Geometry estimation | RANSAC (Random Sample Consensus) | Fit a line robustly to selected pixels or fragmented regions. Check whether the fit improves the evidence for broken trails and record its random seed and work limits for repeatable comparisons. |
| Feature extraction | Brightness profile analysis | Measure brightness along and across a proposed trail. Test local contrast, width, and variation along its length as evidence for review or classification. |
| Classification | CNN (convolutional neural network) | Try an optional classifier over candidate crops after there are reviewed examples. Evaluate on separate shooting sequences and allow uncertain classifications. This can build on the 3.x training and evaluation workflow. |

[OpenCV filters](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html) ·
[OpenCV LSD](https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html) ·
[OpenCV connected components](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html) ·
[RANSAC line fitting](https://scikit-image.org/docs/stable/auto_examples/transform/plot_ransac.html) ·
[Radon and CNN for meteor trails](https://link.springer.com/article/10.1007/s11038-026-09600-7)

#### Combinations to Try

- **Region-based reference detector.** Start with the current pair residual,
  then use labeling and connected-component analysis to select regions.
  Score their shape and brightness profiles. Add RANSAC if fragmented
  regions make the geometry unreliable. This exercises region and score
  reporting with the existing frame-pair interface.
- **Line-segment detector.** Compare DoG and white top-hat as separate
  preprocessing options feeding the same LSD detector. Measure brightness
  profiles on the higher-bit-depth residual using the detected segments.
  Compare the resulting evidence with the region-based detector and Hough.
- **Faint-trail experiment.** Try the Radon transform on selected regions
  and compare whether it finds reviewed trails missed by the other methods.
  Check shorter trails, the cost of the angle search, and the effect of
  restricting the search to regions proposed by another method.
- **Temporal background and classification.** A temporal median needs a
  defined reference window, image alignment where needed, and consistent
  frame selection at batch boundaries and on resume. Evaluate it separately
  from sorted hooks, which receive detection records without image arrays.
  For aircraft and satellites, test whether track continuity and brightness
  profiles help label individual trails. Evaluate clouds as an image or
  region quality annotation. Keep these annotations advisory during trials.

Compare each experiment on the same RAW pairs and ROI, changing one stage at
a time. Record reviewed trail matches, missed trails, extra frames to review,
runtime and peak memory. Use the [aircraft sequence](aircraft_sample_validation.md)
as a starting point for mixed meteor and aircraft frames. Add independent
reviewed sequences with satellites, clouds, and faint or short meteors before
judging classification or detection quality across shooting conditions.

### Evolve the Detector and Hook Contracts

- [x] Add a pre-detection frame-pair hook and use it for opt-in stellar
  registration before the 1.6.13 Hough workflow.
  [Workflow and RAW validation](star_registration_workflow_ja.md)

- [ ] Define the result fields that hooks and outputs can share across
  detector types. Preserve the frame pair and candidate decision, identify
  the detector and the meaning of its score, and represent optional evidence
  such as line segments or regions with explicit image coordinates. Use the
  reference detector to test which Hough-specific requirements belong in
  optional capabilities.
- [ ] Let plugins describe the evidence they produce or need. Use the
  aircraft hook's dependence on line segments and the report's ability to
  show regions as the first examples. Decide which combinations should be
  validated at configuration time and how consumers handle missing evidence.
- [ ] Evolve hooks around the existing lifecycle. Keep image transforms and
  per-pair result changes before candidate saving. Use ordered hooks for
  batch and sequence analysis, and add an opt-in way to consume the enriched
  results after analysis. Specify callback order, process and state lifetime,
  and the scope of results after interruption or resume. Keep existing
  callbacks in their documented order.

### Compose Detectors and Outputs

- [ ] Compare detectors side by side before introducing a multi-detector
  pipeline. Prototype an explicit comparison mode first, with per-detector
  decisions and evidence. Consider a primary detector and an optional
  verifier once a workflow needs a combined decision. Define which result
  governs candidate saving and how errors affect progress before exposing
  that combination in configuration.
- [ ] Support adding a report or annotation exporter alongside the primary
  file output. Start with a local destination and define what gets emitted
  after sorted analysis, what happens when an output fails, and how resume
  avoids duplicate records. Evaluate annotation formats and remote outputs
  once a concrete consumer can validate the result.
- [ ] Turn the exercised examples into a plugin author workflow covering
  packaging, discovery, configuration, and worker compatibility. Add
  validation tooling and reusable configuration profiles as those examples
  establish which checks and settings authors need.

## Version 3.x - Intelligence and Learning

Scheduling follows validation of the detector and output extension points in 2.x.

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

**Published release baseline** —
[v1.6.13](https://github.com/shin3tky/detect_meteors/releases/tag/1.6.13),
released 2026-11-16 with release build and verification workflows.
