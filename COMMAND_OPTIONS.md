# Command Line Options Reference

All command-line flags for `detect_meteors_cli.py`, with defaults and guidance:

## Input/Output Options
- **`-t`/`--target`** (default: `rawfiles`): Source folder that contains RAW images to scan.
- **`-o`/`--output`** (default: `candidates`): Destination folder for RAW files flagged as meteor candidates.
- **`--debug-dir`** (default: `debug_masks`): Where to save generated mask and debug images.

## Detection Parameters
- **`--diff-threshold`** (default: `8`): Pixel-difference threshold used to binarize frame-to-frame differences. **TIP**: Use `--auto-params` to optimize automatically based on ISO and NPF compliance.
- **`--min-area`** (default: `10`): Smallest allowed contour area in pixels. **TIP**: Use `--auto-params` to optimize based on star trail length.
- **`--min-aspect-ratio`** (default: `3.0`): Minimum ratio of a contour's long side to its short side.

## Hough Transform Parameters
- **`--hough-threshold`** (default: `10`): Accumulator threshold for the probabilistic Hough transform.
- **`--hough-min-line-length`** (default: `15`): Minimum line length (in pixels) accepted by the Hough transform.
- **`--hough-max-line-gap`** (default: `5`): Maximum gap (in pixels) between segments on the same detected line.
- **`--min-line-score`** (default: `80.0`): Minimum summed line length score required to mark a meteor candidate. **TIP**: Use `--auto-params` to optimize based on expected meteor trail length.

## Region of Interest (ROI) Options
- **`--no-roi`**: Skip ROI selection and process the entire frame.
- **`--roi`**: Explicit polygon ROI as `"x1,y1;x2,y2;..."` (needs ≥3 vertices).

## NPF Rule-based Auto-Parameter Optimization
- **`--auto-params`**: Automatically optimize all three critical detection parameters using NPF Rule and EXIF metadata. The algorithm:
  - Extracts EXIF data (ISO, exposure, aperture, focal length, resolution)
  - Calculates NPF recommended exposure and star trail length
  - Evaluates shooting condition quality (EXCELLENT/GOOD/FAIR/POOR)
  - Optimizes `diff_threshold` based on ISO sensitivity and NPF overshoot
  - Optimizes `min_area` based on star trail length
  - Optimizes `min_line_score` based on meteor speed (3× faster than stars)
  - Manual parameter specifications always have priority over auto-optimization

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
- **`--workers`** (default: CPU count - 1): Number of parallel worker processes.
- **`--batch-size`** (default: `10`): How many RAW files each worker processes at a time.
- **`--auto-batch-size`**: Dynamically shrink batch size to stay within ~60% of available RAM.
- **`--no-parallel`**: Force single-threaded execution.

## Utility Options
- **`--profile`**: Print timing breakdowns after the run.
- **`--verbose`**: Show detailed diagnostic information on errors. Includes system info, dependency versions, and full error context for troubleshooting.
- **`--save-diagnostic FILE`**: Save diagnostic report to specified file on error. If FILE is omitted, generates a timestamped filename. The report is formatted as Markdown suitable for GitHub issue attachments.
- **`--validate-raw`**: Pre-validate RAW files to catch corruption before processing.
- **`--progress-file`** (default: `progress.json`): Path to the JSON file that tracks processed frames.
- **`--locale`** (default: environment variable `DETECT_METEORS_LOCALE` or `en`): Locale code for CLI messages. Currently supports `en` (English) and `ja` (Japanese).
- **`--no-resume`**: Ignore and remove any existing progress file before processing.
- **`--remove-progress`**: Delete the progress file and exit immediately.
- **`--output-overwrite`**: Force overwrite existing files in output folder (default: skip existing files).

## Fisheye Correction Options
- **`--fisheye`**: Enable fisheye lens correction for equisolid angle projection lenses. Adjusts NPF calculations to use edge focal length (worst case) and accounts for longer star trails at image edges. Recommended for ultra-wide fisheye lenses (e.g., 8mm on Full Frame or MFT).
