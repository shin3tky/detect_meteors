# Command Line Options Reference

All command-line flags for `detect_meteors_cli.py`, with defaults and guidance:

Defaults below apply when no configuration override is supplied. For pipeline
settings, explicitly supplied CLI flags take precedence over the configuration
file, followed by built-in defaults. Relative paths use the current working
directory.

## Input/Output Options
- **`-t`/`--target`** (default: `rawfiles`): Source folder that contains RAW images to scan.
- **`-o`/`--output`** (default: `candidates`): Destination folder for RAW files flagged as meteor candidates.
- **`--debug-dir`** (default: `debug_masks`): Where to save generated mask and debug images (used with `--debug-image`).
- **`--debug-image`** (default: disabled): Save mask/debug images to `--debug-dir`.
- **`--no-debug-image`** (default): Do not save mask/debug images.

## Pipeline Configuration & Plugin Selection
- **`--config`**: Load pipeline settings from a YAML/JSON configuration file (see `config_examples/pipeline.yaml`). Configuration files can be partial; omitted settings are filled with defaults.
- **`--input-loader`**: Input loader plugin name (overrides `input_loader_name` in config).
- **`--input-loader-config`**: Input loader config as a JSON/YAML string or file path.
- **`--detector`**: Detector plugin name (overrides `detector_name` in config).
- **`--detector-config`**: Detector config as a JSON/YAML string or file path.
- **`--output-handler`**: Output handler plugin name (overrides `output_handler_name` in config).
- **`--output-handler-config`**: Output handler config as a JSON/YAML string or file path.
- **`--hooks`**: Comma-separated hook plugin names (execution order, overrides `hooks` in config). When omitted, hooks from the configuration file are retained. With no configured hooks, hooks are skipped. Use `--hooks ""` to override configured hooks with an empty list.
- **`--hook-config`**: Hook config as a JSON/YAML list or mapping (or file path) keyed by hook name.

Built-in plugin defaults are `raw`, `hough`, and `file`. The RAW loader supports
only `binning: 2` and defaults to `normalize: false`. The Hough detector accepts
`detector_config: {}` and reads thresholds from `params`. The file output
handler's overwrite field is `output_overwrite`. Set `hook_error_mode` to
`raise` (default) or `warn` in the configuration file; there is no CLI flag for it.
The optional `aircraft_trail` hook adds metadata after processing without
changing candidate decisions or removing copied files.
With an explicit `--output-handler file` (or `output_handler_name: file`), use
the handler config for output paths and overwrite settings; top-level pipeline
settings are only inherited when the default handler is selected implicitly.

> **Note**: The CLI now runs through `MeteorDetectionPipeline`. The legacy path
> remains for backward compatibility but is slated for deprecation.

## Detection Parameters
- **`--diff-threshold`** (default: `8`): Pixel-difference threshold used to binarize frame-to-frame differences. **TIP**: Use `--auto-params` to optimize automatically based on ISO and NPF compliance.
- **`--min-area`** (default: `10`): Smallest allowed contour area in pixels. **TIP**: Use `--auto-params` to optimize based on star trail length.
- **`--min-aspect-ratio`** (default: `3.0`): Minimum ratio of a contour's long side to its short side.

> **Legacy notice**: These parameter flags are mapped into `PipelineConfig.params` for backward compatibility.
> They will be deprecated in favor of YAML/JSON configuration files over time.

## Hough Transform Parameters
- **`--hough-threshold`** (default: `10`): Accumulator threshold for the probabilistic Hough transform.
- **`--hough-min-line-length`** (default: `15`): Minimum line length (in pixels) accepted by the Hough transform.
- **`--hough-max-line-gap`** (default: `5`): Maximum gap (in pixels) between segments on the same detected line.
- **`--min-line-score`** (default: `80.0`): Minimum summed line length score required to mark a meteor candidate. **TIP**: Use `--auto-params` to optimize based on expected meteor trail length.

## Region of Interest (ROI) Options
- **`--no-roi`**: Skip ROI selection and process the entire frame.
- **`--roi`**: Explicit polygon ROI as `"x1,y1;x2,y2;..."` (needs ≥3 vertices).

Interactive ROI selection is enabled by default. ROI coordinates, contour areas,
and line lengths use the binned image (half the RAW width and height with the
built-in loader).

## NPF Rule-based Auto-Parameter Optimization
- **`--auto-params`**: Automatically optimize all three critical detection parameters using NPF Rule and EXIF metadata. The algorithm:
  - Extracts EXIF data (ISO, exposure, aperture, focal length, resolution)
  - Calculates NPF recommended exposure and star trail length
  - Evaluates shooting condition quality (EXCELLENT/GOOD/FAIR/POOR)
  - Optimizes `diff_threshold` based on ISO sensitivity and NPF overshoot
  - Optimizes `min_area` based on star trail length
  - Optimizes `min_line_score` based on meteor speed (3× faster than stars)
  - Explicit `--diff-threshold`, `--min-area`, and `--min-line-score` flags take priority over auto-optimization; values supplied only in a configuration file may be recalculated

When the EXIF data required for NPF analysis is unavailable, automatic estimation
falls back to image-based sampling and geometry. The current auto-parameter
path uses the built-in RAW helpers even when another input plugin is selected.

## NPF Rule Options
- **`--sensor-type`**: Sensor type preset that automatically sets `--focal-factor`, `--sensor-width`, and `--pixel-pitch`. Valid types (ordered by sensor size):
  
  - `1INCH` - 1-inch sensor (13.2×8.8mm)
  - `MFT` - Micro Four Thirds (17.3×13mm)
  - `APS-C` (or `APSC`) - APS-C Sony/Nikon/Fuji (23.5×15.6mm)
  - `APS-C_CANON` - APS-C Canon (22.3×14.9mm)
  - `APS-H` - APS-H Canon (27.9×18.6mm)
  - `FF` (or `FULLFRAME`) - Full Frame 35mm (36×24mm)
  - `MF44X33` - Medium Format 44×33 (43.8×32.9mm) - Fujifilm GFX, Pentax 645Z, Hasselblad X2D
  - `MF54X40` - Medium Format 54×40 (53.4×40mm) - Hasselblad H6D-100c
  
  Individual options below override preset values when specified.
  
- **`--sensor-width`**: Physical sensor width in millimeters (e.g., `17.3` for MFT, `23.5` for APS-C, `36.0` for Full Frame, `43.8` for MF44×33, `53.4` for MF54×40). Used to calculate pixel pitch for NPF Rule. Overrides `--sensor-type` preset if specified.
- **`--pixel-pitch`**: Direct pixel pitch specification in micrometers (μm). If not specified, calculated from `--sensor-width` and image resolution, or uses default value (4.0μm). Overrides `--sensor-type` preset if specified.
- **`--focal-length`**: Focal length in 35mm equivalent (mm). If not specified, automatically extracted from EXIF metadata. Can be manually specified to override EXIF value.
- **`--focal-factor`**: Sensor type or crop factor (e.g., `MFT`, `APS-C`, `FF`, `MF44X33`, or numeric like `2.0`, `0.79`). Used to convert actual focal length to 35mm equivalent. Overrides `--sensor-type` preset if specified. Note: Medium format sensors have crop factors less than 1.0 (e.g., `0.79` for MF44×33, `0.64` for MF54×40).
- **`--list-sensor-types`**: Display available sensor type presets with their configurations and exit.
- **`--show-npf`**: Display detailed NPF Rule analysis and exit without processing. Shows pixel pitch, NPF recommended exposure, compliance level, star trail estimate, and impact assessment.
- **`--show-exif`**: Display EXIF metadata only and exit without processing. **Use this first** to verify focal length extraction before running `--auto-params`.

## Performance Options
- **`--workers`** (default: `max(1, CPU count - 1)`): Number of parallel worker processes; allowed range is `1` through the CPU count. One worker uses sequential processing.
- **`--batch-size`** (default: `10`): Number of adjacent-frame pairs in each parallel task. Sequential processing handles one pair at a time.
- **`--auto-batch-size` / `--no-auto-batch-size`**: Enable or disable auto-adjusted batch sizing to stay within ~60% of available RAM.
- **`--parallel` / `--no-parallel`**: Explicitly enable or disable parallel processing (defaults to enabled, unless overridden by config).

## Utility Options
- **`--profile`**: Print timing breakdowns after the run.
- **`--verbose`**: Show detailed diagnostic information on errors. Includes system info, dependency versions, and full error context for troubleshooting.
- **`--save-diagnostic FILE`**: Save diagnostic report to specified file on error. If FILE is omitted, generates a timestamped filename. The report is formatted as Markdown suitable for GitHub issue attachments.
- **`--validate-raw`**: Accepted but currently unused by both the CLI pipeline and the legacy detection function. Load failures are handled during processing. Python callers can invoke `meteor_core.pipeline.validate_raw_file()` explicitly for pre-validation.
- **`--progress-file`** (default: `progress.json`): Path to the JSON file that tracks processed frames.
- **`--locale`** (default: environment variable `DETECT_METEORS_LOCALE` or `en`): Locale code for CLI messages. Currently supports `en` (English) and `ja` (Japanese).
- **`--no-resume`**: Start fresh without loading existing progress; new progress replaces the previous contents. Existing candidate files still follow the overwrite setting.
- **`--remove-progress`**: Delete the progress file and exit immediately.
- **`--output-overwrite`**: Force overwrite existing files in output folder (default: skip existing files).
- **`--version`**: Display the application version and exit.
- **`-h` / `--help`**: Display command-line help and exit.

## Fisheye Correction Options
- **`--fisheye`**: Enable fisheye lens correction for equisolid angle projection lenses. Adjusts NPF calculations to use edge focal length (worst case) and accounts for longer star trails at image edges. Recommended for ultra-wide fisheye lenses (e.g., 8mm on Full Frame or MFT).
