# Command Line Options Reference

All command-line flags for `detect_meteors_cli.py`, with defaults and guidance:

Defaults below apply when no configuration override is supplied. For pipeline
settings, explicitly supplied CLI flags take precedence over the configuration
file, followed by built-in defaults. Relative paths use the current working
directory.

## Read-Only Image Analysis (v1.6.11)

- **`--analyze-image IMAGE`**: Analyze one image with the built-in Hough detector without copying candidates or updating progress.
- **`--reference-image IMAGE`**: Explicit preceding image. Without a reference or an explicitly selected sequence, only target-image information is shown.
- **`--analysis-sequence DIR`**: Select the preceding RAW using normal file collection and filename ordering within this explicit sequence. Mutually exclusive with `--reference-image`. No reference is available for the first frame.
- **`--analysis-progress JSON`**: Read original effective parameters and ROI from a progress file. This is separate from `--config` and the normal processing option `--progress-file`.

```bash
python detect_meteors_cli.py \
  --analyze-image rawfiles/frame002.ORF \
  --reference-image rawfiles/frame001.ORF \
  --analysis-progress previous_run/progress.json
```

In analysis mode, parameter precedence is defaults → `--config` values → recorded
`params` → recorded `processing_params` → explicit detection parameter flags.
The output shows each value's source and explicit changes from the baseline.
Both `--min-aspect-ratio 2.0` and `--min-aspect-ratio=2.0` are recognized as explicit
overrides; these values are examples, not recommendations. Recorded ROI is used
unless overridden with `--roi` or `--no-roi`; do not specify both.

The summary shows the selected pair, original progress path, loaded image size,
metadata, effective settings, rejection reasons, and measurement counts.
`--verbose` also lists contour measurements and Hough segment endpoints.
`--locale en` and `--locale ja` select the summary language.

Exit status is **0** for a measured pair, regardless of its candidate decision;
**1** for configuration, input, or processing errors; and **2** for invalid option
combinations or a missing preceding frame. A missing frame is reported as
`reference_missing`, with no candidate decision. The target's parent directory,
default `--target`, and config target folder are never used to guess a sequence.

Analysis rejects `--auto-params`, `--remove-progress`, `--show-exif`, `--show-npf`,
`--list-sensor-types`, `--debug-image`, `--progress-file`, and explicitly supplied
`--target`. Use the analysis-specific input options instead. Other pipeline
output and performance settings do not start a processing run or output handler.
The existing `--save-diagnostic` option can still explicitly request an error
diagnostic file. Without `--analysis-output`, analysis does not save reports or
images.

Only the built-in Hough detector is supported. Configured hooks are rejected;
use `--hooks ""` explicitly for a detector-only replay. Original progress alone
cannot establish the original loader configuration or hook effects. The
summary identifies those limits and any missing settings filled by defaults.
Bounded tuning and explicit report/configuration export are available through
the options below.
See the [analysis design](parameter_analysis_1.6.11.md).

## Bounded Parameter Tuning (v1.6.11)

- **`--tune-params`**: Generate bounded, evidence-derived proposals for `--analyze-image`. Nothing is applied or saved.
- **`--tuning-config JSON_OR_YAML`**: A mapping or file containing limits and an optional `measurement_grid`; see [`parameter_tuning.yaml`](../config_examples/parameter_tuning.yaml).
- **`--analysis-region X0,Y0,X1,Y1`**: Optional annotated target rectangle in loaded-image coordinates; end coordinates are exclusive. Without it, proposals recover a frame-level candidate, not a specific meteor.
- **`--comparison-sequence DIR`**: Explicit RAW sequence to evaluate before/after each returned proposal, using normal adjacent filename-ordered pairs.
- **`--known-meteor IMAGE`**: Known meteor frame for comparison, repeatable; requires `--comparison-sequence`. These are user-supplied positive labels, not `detected_files` inferred from progress.

All modifiers require `--tune-params`, which requires `--analyze-image`.
`--config`, `--analysis-progress`, and `--tuning-config` serve different purposes;
passing tuning limits as pipeline configuration is rejected.

```bash
python detect_meteors_cli.py \
  --analyze-image rawfiles/session/frame002.ORF \
  --reference-image rawfiles/session/frame001.ORF \
  --analysis-progress previous_run/progress.json \
  --tune-params \
  --tuning-config config_examples/parameter_tuning.yaml \
  --comparison-sequence rawfiles/session \
  --known-meteor rawfiles/session/frame002.ORF
```

| Configuration key | Default | Meaning |
| --- | --- | --- |
| `max_gate_relaxation` | `0.5` | Each area/ratio/score threshold stays at least this original value multiplied by `1 - max_gate_relaxation`; permitted fraction is 0 through 1 |
| `max_measurements` | `8` | Target measurements, including the original measurement |
| `max_evaluations` | `256` | Target candidate decisions, including the original decision |
| `max_proposals` | `5` | Maximum returned proposals; omitted feasible proposals are reported |
| `max_comparison_pairs` | `100` | Maximum complete comparison pairs |
| `max_comparison_measurements` | `600` | Maximum measurement runs across comparison pairs |
| `measurement_grid` | `{}` | Explicit finite integer lists for difference/Hough parameters; empty means gate-only tuning |

Gate values come from measured contour area/aspect ratio and the image-wide line
score. Strict area/ratio boundaries are preserved without rounding away a valid
proposal. The default relaxation fraction is a configurable search limit, not a
camera-specific recommendation. No proposal is generated without Hough line
evidence. An already passing target needs no adjustment; no feasible proposal
within the bounds is also a valid outcome.

Grid keys are `diff_threshold`, `hough_threshold`, `hough_min_line_length`, and
`hough_max_line_gap`. Original values are always included, with other values
ordered by distance from the original. A bounded Cartesian product is explored.
Each changed preprocessing/Hough setting receives fresh measurements from the
same loaded image/ROI snapshot. Area/ratio/score-only changes reuse evidence.

ROI is held fixed. An annotation fully outside ROI produces
`target_outside_roi`; a partially overlapping annotation uses evidence within
the original ROI. A contour must overlap actual foreground pixels in the
annotated region, but this association alone does not identify a meteor.

Proposals are initially ordered by the sum of
`abs(after - before) / max(abs(before), 1)` across changed parameters. Comparison
evaluates only the returned proposals; it does not claim a globally optimal
configuration. After comparison, fewer rejected known-positive frames take
precedence, then smaller change cost and fewer added candidates.

Comparison lists added/lost candidate frames and known meteor frames selected,
rejected, or unchecked. Limits apply to whole pairs shared by all proposals;
an unprocessed known frame is unchecked, never a negative. Resolution/dtype
changes and load failures are errors. Matching preprocessing/Hough settings
share measurements within a pair, with the original loader and ROI retained.
No comparison means candidate growth is unverified; no known-positive labels
means their retention is unverified. Added candidates are not a false-positive
rate, and selecting a known meteor frame does not prove that its meteor was
the detected feature.

The summary reports measurement/decision counts, planned grid size, proposal
truncation, comparison coverage, and stop reasons. Budget exhaustion or no
proposal does not turn a valid measured pair into a processing error; existing
analysis exit statuses still apply.

## Analysis Bundles and Session Configuration (v1.6.11)

- **`--analysis-output DIR`**: Explicitly create a new analysis directory with `report.json` and PNG evidence. Existing files, directories, and symlinks are rejected, including empty directories. Parent directories may be created.
- **`--analysis-proposal baseline|N`**: Also save `session.yaml` for the current effective baseline or a one-based index from the returned proposal list. Requires `--analysis-output`; numeric indices require `--tune-params`. With no selection, the bundle contains no executable session configuration.

```bash
python detect_meteors_cli.py \
  --analyze-image rawfiles/session/frame002.ORF \
  --reference-image rawfiles/session/frame001.ORF \
  --analysis-progress previous_run/progress.json \
  --tune-params \
  --analysis-output analysis/review-001 \
  --analysis-proposal 1

python detect_meteors_cli.py --config analysis/review-001/session.yaml
```

The second command starts normal detection; export alone does not apply any
settings, copy candidates, or update progress. Proposal indices refer to the
current run's returned order, including comparison ordering. Use `baseline` to
save the current settings explicitly. Review the proposal list before selecting
an index. Existing output is never replaced, even with `--output-overwrite`.

`report.json` has format `detect_meteors.analysis`, `schema_version: 1`, the
application and runtime versions, input fingerprints, exact parameters and their
sources, ROI, metadata, limitations, decisions, contours/lines, tuning bounds and
work counters, proposal changes, and comparison coverage. Primary images,
original progress, and configuration sources have SHA-256 hashes and file
identity/change stamps. Comparison inputs have stamps; their `sha256` is null
to avoid rereading the entire sequence. Changed or missing sources invalidate
export rather than being linked to stale evidence.

Measurement records reference lossless mask PNGs. Baseline/proposal records link
debug PNGs, including rejected-frame evidence. Identical measurement settings
share a mask. `assets` records each PNG/configuration's size and SHA-256; the
manifest does not hash or list itself. Non-JSON metadata is represented as text
with `serialization_notes`. Image arrays and live loader objects are not stored
in JSON. The manifest is written last; failed writes remove only artifacts
created by that invocation and empty newly created directories.

`session.yaml` is a full `PipelineConfig` loaded by existing `--config`. Exact
float thresholds, the actual registered loader and its serializable settings,
the built-in Hough detector, and effective ROI are retained. Configuration
coercion must reproduce loader values and types; unavailable/unserializable
loader settings fail explicitly. Sessions with active hooks or custom output
handlers cannot be exported as equivalent detector-only configurations.

The saved session uses one worker and sequential processing. Its absolute
candidate/debug/progress paths are under `DIR/run`, which is created only when
normal processing starts, with overwrite disabled in both pipeline and file
handler settings. Original run paths are not reused. The target folder is the
explicit analysis sequence, otherwise the configured target if supplied in the
original config, otherwise the analyzed image's parent. Plugin-specific relative
paths keep their existing meaning; the report records the working directory.
Review/edit the absolute paths when relocating a saved session.

Two optional configuration fields persist ROI: `roi_polygon` (null or at least
three integer points in loaded-image coordinates) and `enable_roi_selection`
(boolean; default true). Saved sessions set the latter false. Normal processing
uses configured ROI unless `--roi` or `--no-roi` explicitly overrides it. Analysis
uses recorded progress ROI ahead of configured ROI, and explicit ROI flags
ahead of both. Both null and the legacy progress marker `full_image` mean the
full image. Existing configurations retain their previous defaults.

A missing reference can still produce an honest report with no decision or
measurement PNGs and exit status 2. It cannot produce a reusable selected
session. Report/configuration validation errors exit with status 1 and do not
replace inputs or existing artifacts. The bundle is a record of exploratory
evidence, not automatic meteor classification or a claim of optimal settings.

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
