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

## NPF Rule-based Auto-Parameter Optimization (NEW in v1.4)
- **`--auto-params`**: Automatically optimize all three critical detection parameters using NPF Rule and EXIF metadata. The algorithm:
  - Extracts EXIF data (ISO, exposure, aperture, focal length, resolution)
  - Calculates NPF recommended exposure and star trail length
  - Evaluates shooting condition quality (EXCELLENT/GOOD/FAIR/POOR)
  - Optimizes `diff_threshold` based on ISO sensitivity and NPF overshoot
  - Optimizes `min_area` based on star trail length
  - Optimizes `min_line_score` based on meteor speed (3× faster than stars)
  - Manual parameter specifications always take priority over auto-optimization

## NPF Rule Options (NEW in v1.4)
- **`--sensor-width`**: Physical sensor width in millimeters (e.g., `17.3` for MFT, `23.5` for APS-C, `36.0` for Full Frame). Used to calculate pixel pitch for NPF Rule. Significantly improves optimization accuracy.
- **`--pixel-pitch`**: Direct pixel pitch specification in micrometers (μm). If not specified, calculated from `--sensor-width` and image resolution, or uses default value (4.0μm).
- **`--focal-length`**: Focal length in 35mm equivalent (mm). If not specified, automatically extracted from EXIF metadata. Can be manually specified to override EXIF value.
- **`--focal-factor`**: Sensor type or crop factor (e.g., `MFT`, `APS-C`, `FF`, or numeric like `2.0`). Used to convert actual focal length to 35mm equivalent.
- **`--show-npf`**: Display detailed NPF Rule analysis and exit without processing. Shows pixel pitch, NPF recommended exposure, compliance level, star trail estimate, and impact assessment.
- **`--show-exif`**: Display EXIF metadata only and exit without processing. **Use this first** to verify focal length extraction before running `--auto-params`.

## Performance Options
- **`--workers`** (default: CPU count - 1): Number of parallel worker processes.
- **`--batch-size`** (default: `10`): How many RAW files each worker processes at a time.
- **`--auto-batch-size`**: Dynamically shrink batch size to stay within ~60% of available RAM.
- **`--no-parallel`**: Force single-threaded execution.

## Utility Options
- **`--profile`**: Print timing breakdowns after the run.
- **`--validate-raw`**: Pre-validate RAW files to catch corruption before processing.
- **`--progress-file`** (default: `progress.json`): Path to the JSON file that tracks processed frames.
- **`--no-resume`**: Ignore and remove any existing progress file before processing.
- **`--remove-progress`**: Delete the progress file and exit immediately.
