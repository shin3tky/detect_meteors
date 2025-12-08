# Changelog

## v1.5.7 - 2025-12-08
- **Progress metadata enrichment**: `progress.json` now records CLI parameters (`params`), ROI selection (`roi`), and finalized processing parameters (`processing_params`) for reference when reviewing or re-running detection sessions.
- **Pipeline consistency**: Both CLI and `PipelineConfig`-driven runs persist the same metadata fields, ensuring uniform progress files across entry points.

## v1.5.6 - 2025-12-07
- **Input loader pluginization**: Added `InputLoader`/`MetadataExtractor` protocols with dataclass/Pydantic helper bases, a built-in `RawImageLoader` plugin, and deterministic loader discovery via entry points (`detect_meteors.input`) and the local `~/.detect_meteors/plugins` directory
- **Pipeline configuration object**: New `PipelineConfig` dataclass centralizes all runtime settings and powers a `DetectionPipeline` protocol plus helpers to resolve input loaders and metadata extraction paths
- **Output extensibility**: Introduced `OutputHandler` protocol to formalize candidate/debug persistence ahead of v2.0 plugin architecture

## v1.5.5 - 2025-12-05
- **Code Architecture Refactoring**: Reorganized codebase into modular structure for v2.x plugin architecture preparation
  - `detect_meteors_cli.py`: CLI interface only (argument parsing, user interaction)
  - `meteor_core/`: Core logic modules
    - `schema.py`: Type definitions and data structures
    - `pipeline.py`: Processing pipeline orchestration
    - `image_io.py`: RAW image loading and EXIF extraction
    - `roi_selector.py`: ROI selection interface
    - `utils.py`: Utility functions
    - `detectors/`: Detection algorithm implementations
      - `base.py`: Abstract base detector class
      - `hough_default.py`: Default Hough transform detector
    - `outputs/`: Output handling
      - `writer.py`: Result file writer
- **Type Safety Improvements**: Enhanced type hints throughout codebase using TypedDict for structured data
- **Backward Compatibility**: CLI interface unchanged; all existing commands work without modification

## v1.5.4 - 2025-12-03
- **Improved ROI Selection Display**: Brightened the ROI selection image for better visibility in dark conditions
- **NOTICE Document**: Added NOTICE file for third-party license attributions and acknowledgments

## v1.5.3 - 2025-12-02
- **Fisheye Lens Correction**: Added `--fisheye` flag for equisolid angle projection compensation
  - Accounts for varying effective focal length across the fisheye image
  - Center: nominal focal length preserved
  - Edge: effective focal length reduced (cos(45°) ≈ 0.707× for 180° diagonal FOV)
  - Star trail length ratio increases toward edges (up to ~1.414× at corners)
- **NPF Calculation for Fisheye**: Uses edge (worst case) effective focal length
  - More conservative/lenient NPF recommended exposure time
  - Better compliance evaluation for wide-angle fisheye photography
- **New Fisheye Functions**:
  - `calculate_fisheye_effective_focal_length()`: Position-dependent focal length
  - `calculate_fisheye_edge_focal_length()`: Edge focal length for NPF
  - `calculate_fisheye_trail_length_ratio()`: Trail length variation across image
  - `get_fisheye_max_trail_ratio()`: Maximum trail ratio at image edge
  - `display_fisheye_info()`: Display fisheye correction parameters
- **Projection Model Infrastructure**: Prepared for future projection models (equidistant, stereographic)
  - Currently implements equisolid angle projection only
  - `FISHEYE_PROJECTION_MODELS` dictionary for extensibility
- **Updated Shell Completions**: Both bash and zsh completion scripts updated with `--fisheye` support
- **Comprehensive Test Coverage**: Added `test_fisheye_v1x.py` with 27 test cases

## v1.5.2 - 2025-12-01
- **Sensor Override Validation**: Added automatic validation when `--sensor-type` preset values are overridden with `--sensor-width` or `--pixel-pitch`.
  - Warns when `--sensor-width` deviates more than ±30% from preset value
  - Warns when `--pixel-pitch` deviates more than ±50% from preset value
  - Warnings are informational only - processing continues normally
  - Helps catch accidental misconfiguration while preserving flexibility
- **Enhanced `apply_sensor_preset()` Function**: Now returns sensor preset dictionary along with parameter values for validation
- **Comprehensive Test Coverage**: Added `test_sensor_validation_v1x.py` with 23 test cases covering validation scenarios
- **Updated Test Suite**: Modified `test_sensor_presets_v1x.py` to handle new 5-tuple return value from `apply_sensor_preset()`

## v1.5.1 - 2025-11-30
- **Medium Format Sensor Support**: Added support for medium format sensors larger than Full Frame (35mm).
  - `MF44X33`: Fujifilm GFX series, Pentax 645Z, Hasselblad X2D/X1D (43.8×32.9mm, crop factor 0.79)
  - `MF54X40`: Hasselblad H6D-100c (53.4×40mm, crop factor 0.64)
- **Sensor Size Ordering**: Reordered `--sensor-type` options by sensor size (smallest to largest): 1INCH → MFT → APSC → APSC_CANON → APSH → FF → MF44X33 → MF54X40
- **Updated Shell Completions**: Both bash and zsh completion scripts updated with medium format sensor types

## v1.5.0 - 2025-11-29 🦃
- **Sensor Type Presets**: Introduced unified `SENSOR_PRESETS` configuration that consolidates `DEFAULT_SENSOR_WIDTHS` and `CROP_FACTORS` into a single, comprehensive sensor database with typical pixel pitch values.
- **New `--sensor-type` Option**: Simplified NPF Rule configuration with a single parameter. Automatically sets `--focal-factor`, `--sensor-width`, and `--pixel-pitch` based on sensor type (MFT, APS-C, APS-C_CANON, APS-H, FF, 1INCH).
- **New `--list-sensor-types` Option**: Display all available sensor type presets with their configurations (focal factor, sensor width, pixel pitch) and exit.
- **Parameter Override Priority**: Individual CLI arguments (`--focal-factor`, `--sensor-width`, `--focal-length`, `--pixel-pitch`) take priority over `--sensor-type` preset values, allowing fine-tuned customization.
- **Helper Functions**: Added `get_sensor_preset()` and `apply_sensor_preset()` for programmatic access to sensor configurations.
- **Updated Shell Completions**: Both bash and zsh completion scripts updated with `--sensor-type` and `--list-sensor-types` support.
- **Backward Compatibility**: Fully compatible with v1.4.x. All existing command-line options work unchanged; `CROP_FACTORS` and `DEFAULT_SENSOR_WIDTHS` dictionaries preserved for legacy code.

## v1.4.2 - 2025-11-25
- **Output File Protection**: Changed behavior to skip overwriting existing files at the output destination instead of overwriting them.
- **New Command-Line Option**: Added `--output-overwrite` flag to allow overwriting existing output files when explicitly requested.
- **Safety Check**: Added warning and exit when `--target` and `--output` directories are identical to prevent accidental data loss.

## v1.4.1 - 2025-11-24
- **NPF Rule-based Scientific Optimization**: Implemented the NPF Rule for scientifically accurate exposure time validation and parameter optimization, marking a milestone in physics-based meteor detection.
- **EXIF Metadata Integration**: Comprehensive automatic extraction of shooting conditions (ISO, exposure time, aperture, focal length, resolution) from RAW files using multi-strategy approach (embedded thumbnail → PIL → rawpy).
- **Sensor Characterization**: Introduced pixel pitch calculation from sensor width and image resolution, with support for direct specification or sensor type lookup (MFT, APS-C, FF).
- **Star Trail Physics Estimation**: Implemented Earth's rotation-based calculation (15°/hour sidereal rate) to estimate star movement during exposure, accounting for field of view and declination.
- **Shooting Quality Assessment**: Comprehensive quality scoring (0.0-1.0) based on NPF compliance (60% weight), ISO sensitivity (25% weight), and focal length (15% weight), with levels (EXCELLENT/GOOD/FAIR/POOR).
- **Enhanced Parameter Optimization**:
  - `diff_threshold`: Adjusted for ISO sensitivity (×2 per doubling from 800) and NPF overshoot (×1.5 per 1× above 1.5×)
  - `min_area`: Based on star trail length estimation with focal length correction (wide 0.7×, telephoto 1.3×) and NPF adjustment
  - `min_line_score`: Based on meteor speed physics (3× faster than stars) with focal length and exposure time adjustments
- **Detailed Analysis Output**: Scientific reasoning for each parameter adjustment with NPF compliance display, pixel pitch calculation, star trail estimate, and quality score breakdown.
- **New Command-Line Options**:
  - `--sensor-width`: Physical sensor width in mm for accurate NPF calculation
  - `--pixel-pitch`: Direct pixel pitch specification in μm (highest accuracy)
  - `--show-npf`: Display NPF Rule analysis and exit
  - `--show-exif`: Display EXIF metadata only and exit
- **Real-world Validation**: Tested with OM Digital OM-1 (MFT, 24mm, ISO 1600, 5s) achieving 100% detection (9 candidates including 2 confirmed meteors) with quality score 1.00 (EXCELLENT).
- **Backward Compatibility**: Fully compatible with v1.3.1, falls back to image-based estimation if EXIF unavailable.

## v1.3.1 - 2025-11-23
- **Complete Auto-Parameter Estimation**: Extended `--auto-params` to automatically estimate all three critical detection parameters: `diff_threshold` (v1.2.1), `min_area` (NEW), and `min_line_score` (NEW).
- **Star size distribution analysis**: Introduced automatic `min_area` estimation by detecting and analyzing star sizes in sample images, using 98th percentile brightness threshold and robust 75th percentile × 2.0 formula.
- **Image geometry-based scoring**: Implemented automatic `min_line_score` estimation from image diagonal length (2.5% coefficient), with optional focal length adjustment for different lens types.
- **Focal length support**: Added `--focal-length` option to optimize meteor trail length expectations for wide-angle (14mm), standard (24mm), and telephoto (50mm+) lenses.
- **Progress tracking restored**: Re-integrated resumable processing from v1.1.0 with `progress.json` file, supporting safe Ctrl-C interruption and automatic parameter validation via hash.
- **Critical bug fixes from v1.3.0 (unreleased)**:
  - Fixed inverted focal length adjustment logic (was multiplying instead of dividing)
  - Reduced base coefficient from 4% to 2.5% based on real-world meteor data
  - Improved star detection threshold from 95th to 98th percentile with size filtering (2-100 pixels²)
- **Real-world validation**: Tested with OM Digital OM-1 (24mm, ISO 1600, 5s exposure) achieving 100% detection rate (2/2 meteors) with automatic parameters.

## v1.2.1 - 2025-11-22
- **Improved Auto-Parameter Estimation**: Revised `diff_threshold` auto-estimation algorithm from 3-sigma rule to percentile-based approach for better handling of peaked night sky brightness distributions.
- **Real-world validation**: Based on actual RAW image testing, reduced typical estimated thresholds from 25 to 15, significantly improving meteor detection sensitivity.
- **Enhanced statistics output**: Added 98th and 99th percentile reporting, plus detailed breakdown of three estimation methods (98th percentile, mean + 1.5σ, median × 3).
- **Optimized threshold range**: Adjusted clamp range from 4-25 to 3-18 based on real-world feedback for more sensitive meteor detection.

## v1.2.0 - 2025-11-22 (unreleased)
- **NEW: Auto-Parameter Estimation**: Added `--auto-params` flag to automatically estimate optimal `diff_threshold` from sample images using ROI statistics.
- Implemented 3-sigma rule-based estimation from first 5 sample images for initial auto-tuning capability.
- Added detailed statistical output (mean, std dev, median, percentiles) during auto-estimation.
- Preserves manual parameter specification priority when both `--auto-params` and explicit values are provided.

## v1.1.0 - 2025-11-22
- Added resumable processing with a JSON progress file that keeps processed/detected counts and resumes only when parameters match, bringing first-class support for both progress tracking and safe resumes.
- Added `--progress-file`, `--no-resume`, and `--remove-progress` flags to manage progress tracking or clear saved state before runs.
- Allow safe interruption with `Ctrl-C` without losing tracked progress so long runs can be paused and resumed later.

## v1.0.3 - 2025-11-21
- Added optional RAW file validation with progress reporting to skip corrupted inputs before processing.
- Added automatic batch-size tuning based on available memory (requires `psutil`) and a profiling flag for timing insights.
- Defaulted the input folder to `rawfiles` and expanded the CLI option reference in the README for easier configuration.

## v1.0.2 - 2025-11-20
- Added help descriptions with defaults for Hough transform parameters and minimum line score.

## v1.0.1 - 2025-11-20
- Initial release.
