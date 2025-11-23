# detect_meteors

Detect meteors in batches of RAW photos using configurable image processing pipelines.

**Note:** This software detects meteors by identifying changes between the current RAW photo and the previous one. The first RAW photo is not processed, so please check it manually.

## Overview
- CLI-first workflow for scanning folders of RAW photos and flagging potential meteors.
- Works with RAW photos supported by [`rawpy`](https://github.com/letmaik/rawpy) (tested with Olympus Raw Files (ORF) files).
- Provides region-of-interest (ROI) cropping and Hough transform tuning to focus on likely meteor streaks.
- **NEW in v1.3**: Complete automatic parameter estimation - all three critical parameters auto-tuned from your images.
- **NEW in v1.3**: Focal length support for optimal lens-specific detection.
- See the [CHANGELOG](CHANGELOG.md) for release history.

## Roadmap
- Upcoming plans are outlined in [ROADMAP](ROADMAP.md).

## Technical Overview
- See the project wiki for a deeper technical walkthrough: [Technical Processing Overview](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Overview)

## Requirements
- Python 3.12.12 (tested).
- macOS Tahoe 26.1 on an Intel MacBook Pro (tested); other Unix-like systems may work.
- Help wanted: verification on Apple Silicon Macs, Windows, and Linux would be greatly appreciated.
- Dependencies: `numpy`, `matplotlib`, `opencv-python`, `rawpy`, `psutil`.

## Installation
1) Set up Python (example uses `Homebrew`).
   ```bash
   brew install python@3.12
   ```
2) Create and activate a virtual environment.
   ```bash
   /usr/local/opt/python@3.12/bin/python3.12 -m venv venv
   source ./venv/bin/activate
   ```
3) Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start with Complete Auto-Parameter Estimation (NEW in v1.3)

The simplest way to get started - let the software automatically determine **all** optimal detection parameters:

```bash
python detect_meteors_cli.py --auto-params
```

For best results, add your lens focal length:

```bash
python detect_meteors_cli.py --auto-params --focal-length 24
```

This will:
1. Analyze sample images from your RAW files
2. Automatically estimate **all three critical parameters**:
   - `diff_threshold`: From frame-to-frame brightness differences
   - `min_area`: From star size distribution  
   - `min_line_score`: From image geometry and focal length
3. Display detailed statistical analysis
4. Process all images with optimized settings

Example output:
```
==================================================
Auto-estimating diff_threshold from 5 samples
==================================================
Loading samples... ✓ Loaded 5 images
Analyzing frame-to-frame differences in ROI... ✓

ROI Difference Statistics (from 11,531,648 pixels):
  Mean:         6.62
  Std Dev:      7.02
  Median:       5.00
  98th %ile:    20.00
✓ Selected threshold: 15 (minimum of all methods)

==================================================
Auto-estimating min_area from 3 samples
==================================================
Loading samples... ✓ Loaded 3 images
Detecting stars in ROI... ✓ Detected 266 stars

Star Size Statistics (from 266 stars):
  Median:       3.0 pixels²
  75th %ile:    5.0 pixels²
✓ Estimated min_area: 10

==================================================
Auto-estimating min_line_score from image geometry
==================================================
Image Geometry:
  Dimensions:   2620×1956 pixels
  Diagonal:     3270 pixels
  Focal length: 24.0mm
  Focal factor: 0.48×
✓ Estimated min_line_score: 40.0
```


### Traditional Manual Configuration

Show help:
```bash
python detect_meteors_cli.py --help
```

Quick start (defaults to `rawfiles` as input, `candidates` as output, and `debug_masks` for debug images):
```bash
python detect_meteors_cli.py
```

Specify input/output folders and a debug directory:
```bash
python detect_meteors_cli.py -t /path/to/raws -o meteors_out --debug-dir debug_out
```

Process the entire frame (disable ROI cropping):
```bash
python detect_meteors_cli.py --no-roi
```

Limit processing to a region of the starry sky using a polygon ROI (quote the value so the shell keeps the semicolons intact):
```bash
python detect_meteors_cli.py --roi "10,10;4000,10;4000,2000;10,2000"
```

When selecting the ROI in the GUI, left-click to place vertices. Press `ESC` to remove the last point, and move the cursor back near the first point until a circle appears; left-click there to close the polygon.

### Combining Auto-Parameters with Manual Overrides

You can use `--auto-params` while still manually specifying certain parameters:

```bash
# Auto-estimate all parameters, but manually set diff_threshold
python detect_meteors_cli.py --auto-params \
  --focal-length 24 \
  --diff-threshold 12

# Auto-estimate with pre-defined ROI
python detect_meteors_cli.py --auto-params \
  --roi "100,100;3900,100;3900,2900;100,2900" \
  --focal-length 24

# Mix auto and manual parameters
python detect_meteors_cli.py --auto-params \
  --focal-length 24 \
  --min-area 15 \
  --min-aspect-ratio 4.0
```

### Tuning Examples

Tune for short meteor streaks:
```bash
python detect_meteors_cli.py \
  --hough-threshold 10 \
  --hough-min-line-length 15 \
  --hough-max-line-gap 5 \
  --min-line-score 40
```

Tune for long meteor streaks:
```bash
python detect_meteors_cli.py \
  --hough-threshold 15 \
  --hough-min-line-length 40 \
  --min-line-score 120
```

Automatically shrink batch size when RAM is tight (uses up to 60% of free memory):
```bash
python detect_meteors_cli.py --auto-batch-size --workers 4 --batch-size 20
```

## Option Reference

All command-line flags for `detect_meteors_cli.py`, with defaults and guidance:

### Input/Output Options
- **`-t`/`--target`** (default: `rawfiles`): Source folder that contains RAW images to scan. All files supported by `rawpy` are considered.
- **`-o`/`--output`** (default: `candidates`): Destination folder for RAW files flagged as meteor candidates.
- **`--debug-dir`** (default: `debug_masks`): Where to save generated mask and debug images. Create the directory beforehand to keep outputs organized.

### Detection Parameters
- **`--diff-threshold`** (default: `8`): Pixel-difference threshold used to binarize frame-to-frame differences. Raise to suppress noise; lower to capture faint streaks. **TIP**: Use `--auto-params` to estimate this automatically.
- **`--min-area`** (default: `10`): Smallest allowed contour area in pixels. Increase to ignore tiny speckles or hot pixels; decrease to detect very small objects. **TIP**: Use `--auto-params` to estimate from star sizes (v1.3+).
- **`--min-aspect-ratio`** (default: `3.0`): Minimum ratio of a contour's long side to its short side. Meteors are elongated; higher values enforce skinnier shapes.

### Hough Transform Parameters
- **`--hough-threshold`** (default: `10`): Accumulator threshold for the probabilistic Hough transform. Higher values demand stronger line evidence and reduce false positives.
- **`--hough-min-line-length`** (default: `15`): Minimum line length (in pixels) accepted by the Hough transform. Tune together with `--hough-max-line-gap` to match expected streak lengths.
- **`--hough-max-line-gap`** (default: `5`): Maximum gap (in pixels) that can exist between segments on the same detected line. Lower gaps favor continuous streaks; higher gaps tolerate breaks from noise.
- **`--min-line-score`** (default: `80.0`): Minimum summed line length score required to mark a meteor candidate. Raise to capture only the clearest streaks; lower to catch faint or short lines. **TIP**: Use `--auto-params --focal-length MM` to optimize for your lens (v1.3+).

### Region of Interest (ROI) Options
- **`--no-roi`**: Skip ROI selection and process the entire frame. Useful for wide-field captures where meteors could appear anywhere.
- **`--roi`**: Explicit polygon ROI as `"x1,y1;x2,y2;..."` (needs ≥3 vertices). Overrides interactive ROI selection and can be scripted for repeatable crops.

### Auto-Parameter Estimation (NEW in v1.3)
- **`--auto-params`**: Automatically estimate all three critical detection parameters (`diff_threshold`, `min_area`, `min_line_score`) from sample images. The algorithm:
  - Analyzes first 5 images for frame differences (v1.2.1 percentile-based approach)
  - Detects and measures stars in first 3 images for size distribution (v1.3 NEW)
  - Calculates optimal line score from image geometry (v1.3 NEW)
  - Manual parameter specifications always take priority over auto-estimation
- **`--focal-length`** (NEW in v1.3): Focal length in millimeters (e.g., `14`, `24`, `50`). Used with `--auto-params` to optimize `min_line_score` for your specific lens. Wide-angle lenses (14mm) get lower thresholds, telephoto lenses (50mm+) get higher thresholds.

### Performance Options
- **`--workers`** (default: `psutil.cpu_count(logical=True)`): Number of parallel worker processes. Increase to speed up on multi-core machines; reduce if the system feels sluggish.
- **`--batch-size`** (default: `10`): How many RAW files each worker processes at a time. Larger batches reduce I/O overhead but consume more memory.
- **`--auto-batch-size`**: Dynamically shrink batch size to stay within ~60% of available RAM. Pair with `--workers` to balance speed and memory safety.
- **`--no-parallel`**: Force single-threaded execution. Handy for debugging or when parallelism conflicts with other workloads.

### Utility Options
- **`--profile`**: Print timing breakdowns (first load, processing time, totals) after the run.
- **`--validate-raw`**: Pre-validate RAW files to catch corruption before processing. Adds a quick sanity check step on large batches.
- **`--progress-file`** (default: `progress.json`): Path to the JSON file that tracks processed and detected frames so long runs can resume safely. Automatically created during processing.
- **`--no-resume`**: Ignore and remove any existing progress file before processing. Use when you want a clean run without picking up past state.
- **`--remove-progress`**: Delete the progress file and exit immediately. Handy for clearing progress without starting a new detection pass.

## What's New in v1.3

### v1.3.1 - Complete Auto-Parameter Estimation (Latest)
- **Complete automation**: `--auto-params` now estimates **all three** critical parameters automatically
- **Star size analysis**: NEW automatic `min_area` estimation from detected star sizes
- **Image geometry scoring**: NEW automatic `min_line_score` estimation from image diagonal
- **Focal length support**: NEW `--focal-length` option optimizes for wide-angle to telephoto lenses
- **Progress tracking**: Restored from v1.1.0 with `progress.json` support for resumable processing
- **Real-world validated**: 100% detection rate (2/2 meteors) on Olympus OM-1 test dataset
- See [RELEASE_NOTES_1.3.1.md](RELEASE_NOTES_1.3.1.md) for comprehensive details

### v1.2.1 - Improved Auto-Parameter Estimation
- **Percentile-based estimation**: Switched from 3-sigma rule to percentile-based approach for `diff_threshold`
- **Real-world validated**: Reduced typical thresholds from 25 to 15, significantly improving meteor detection
- **Enhanced output**: Added detailed statistical breakdown with multiple estimation methods

## Build a Single Binary with Nuitka
If you want to distribute `detect_meteors_cli` as a standalone executable, you can bundle it with [Nuitka](https://nuitka.net/):

```bash
pip install nuitka
python -m nuitka --onefile --standalone detect_meteors_cli.py
```

## Inputs and Outputs
- **Inputs:** A directory of RAW photos (all files supported by `rawpy` will be considered).
- **Outputs:**
  - Candidate images saved to the directory provided with `-o/--output`.
  - Optional debug masks written to the directory provided with `--debug-dir`.
  - `progress.json` file for tracking processed images (resumable processing).

## Tips for Best Results

### Using Auto-Parameter Estimation (v1.3+)
1. **Select a clean ROI**: The auto-estimation works best when the ROI contains only pure night sky without:
   - Artificial lights (streetlights, light pollution sources)
   - Ground objects (trees, buildings, horizon line)
   - Atmospheric features (clouds, fog, aurora)
2. **Provide focal length**: Use `--focal-length MM` for optimal `min_line_score` tuning
3. **Consistent shooting conditions**: Auto-estimation analyzes sample images, so ensure consistent ISO, exposure, and aperture throughout your session
4. **Manual override when needed**: You can still override any auto-estimated value for fine-tuning

### Expected Auto-Estimated Values

#### By ISO Setting
- **Low ISO (≤1600)**: diff_threshold ~3-5, min_area ~10-15
- **Medium ISO (~3200)**: diff_threshold ~6-10, min_area ~15-25  
- **High ISO (≥6400)**: diff_threshold ~10-18, min_area ~20-30

#### By Focal Length (with `--focal-length`)
- **Wide-angle (14mm)**: min_line_score ~40 (shorter apparent trails)
- **Standard (24mm)**: min_line_score ~40 (medium trails)
- **Telephoto (50mm+)**: min_line_score ~80-100 (longer apparent trails)

### Resumable Processing
- Long processing sessions can be interrupted with Ctrl-C
- Progress is automatically saved to `progress.json`
- Simply run the same command again to resume
- Use `--no-resume` for a fresh start
- Use `--remove-progress` to clear saved progress

### When to Use Manual Parameters
- Extreme ISOs (>12800) where auto-estimation may need adjustment
- Known problematic conditions (aurora, airglow, unusual atmospheric phenomena)
- Fine-tuning based on initial auto-params results
- Special requirements for specific research or publication needs

## Contributing
Issues and pull requests are welcome. Please open an issue to discuss substantial changes before submitting a PR.

## License
This project is licensed under the terms of the [Apache License 2.0](LICENSE).
