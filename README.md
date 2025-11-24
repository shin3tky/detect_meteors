# Detect Meteors CLI

Detect Meteors CLI for batch processing of RAW night-sky images.

**Note:** This software detects meteors by identifying changes between the current RAW image and the previous one. The first RAW image is not processed, so please check it manually.

## Overview
- CLI-first workflow for scanning folders of RAW images and flagging potential meteors.
- Works with RAW images supported by [`rawpy`](https://github.com/letmaik/rawpy) (tested with Olympus Raw Files (ORF) files).
- Provides Region of Interest (ROI) cropping and Hough transform tuning to focus on likely meteor streaks.
- **NEW in v1.4**: NPF Rule-based scientific parameter optimization with EXIF metadata integration.
- **NEW in v1.4**: Automatic sensor characterization and shooting condition quality assessment.
- Complete Auto-Parameter Estimation - all three critical parameters scientifically optimized from your images.
- Focal length support for optimal lens-specific detection.
- See the [CHANGELOG](CHANGELOG.md) for release history.

## Roadmap
- Upcoming plans are outlined in [ROADMAP](ROADMAP.md).

## Technical Overview
- See the project wiki for a deeper technical walkthrough: [Technical Processing Overview](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Overview)
- v1.3.1 Auto-Parameter Estimation Extensions: [v1.3.1 Additions](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Overview-v1.3.1-Additions)
- v1.4.1 NPF Rule-based Scientific Optimization: [v1.4.1 Additions](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Overview-v1.4.1-Additions)

## Requirements
- Python 3.12.12 (tested).
- macOS Tahoe 26.1 on an Intel MacBook Pro (tested); other Unix-like systems may work.
- Help wanted: verification on Apple Silicon Macs, Windows, and Linux would be greatly appreciated.
- Dependencies: `numpy`, `matplotlib`, `opencv-python`, `rawpy`, `psutil`, `pillow`.

## Installation

### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/shin3tky/detect_meteors.git
cd detect_meteors
```

### Step 2: Set up Python

Install Python 3.12 (example uses `Homebrew` on macOS):
```bash
brew install python@3.12
```

### Step 3: Install System Dependencies

Install OpenCV and LibRaw:
```bash
brew install opencv libraw
```

### Step 4: Create Virtual Environment

Create and activate a Python virtual environment:
```bash
# Create virtual environment
/usr/local/opt/python@3.12/bin/python3.12 -m venv venv

# Activate virtual environment
source ./venv/bin/activate
```

**Note**: On subsequent uses, you only need to activate the environment:
```bash
source ./venv/bin/activate
```

### Step 5: Install Python Dependencies

Install required Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

This will install:
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `opencv-python` - Image processing
- `rawpy` - RAW image reading
- `psutil` - System utilities
- `pillow` - Image handling and EXIF extraction

### Verification

Verify the installation:
```bash
python detect_meteors_cli.py --help
```

You should see the help message with all available options.

## Usage

### Quick Start with NPF Rule-based Auto-Parameters (NEW in v1.4)

**Step 1: Check EXIF Metadata (Recommended)**

Before processing, verify that your RAW files contain focal length information:

```bash
python detect_meteors_cli.py --show-exif
```

This displays EXIF metadata from your first RAW file:
```
Camera Settings (EXIF Metadata)
============================================================
  Camera:           Olympus OM-1
  Focal length:     24.0mm (35mm equiv.)  ← Check this!
  ISO:              1600
  Exposure:         5.0s
  Aperture:         f/2.8
  Resolution:       5240×3912 px
============================================================
```

**What to look for:**
- ✅ **If focal length is detected**: You can proceed directly to Step 2
- ❌ **If focal length is missing or incorrect**: You'll need to specify it manually using `--focal-length` or `--focal-factor`

**Step 2: Run Auto-Parameter Optimization**

The most scientific approach - let the software automatically optimize detection parameters using the PhotoPills NPF Rule and EXIF metadata:

**Option A: Focal length detected in EXIF (recommended)**
```bash
python detect_meteors_cli.py --auto-params --sensor-width 17.3
```

**Option B: Focal length missing - specify manually**
```bash
# Specify exact focal length (35mm equivalent)
python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-length 24

# Or specify crop factor (system will calculate from actual focal length)
python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-factor MFT
```

**Option C: Use sensor type shortcut**
```bash
python detect_meteors_cli.py --auto-params --focal-factor MFT
```

This will:
1. Extract EXIF metadata (ISO, exposure time, aperture, focal length) from your RAW files
2. Calculate NPF Rule recommended exposure time based on sensor characteristics
3. Evaluate shooting condition quality with scientific scoring
4. Automatically optimize **all three critical parameters** based on:
   - `diff_threshold`: ISO sensitivity and NPF compliance
   - `min_area`: Estimated star trail length from Earth's rotation
   - `min_line_score`: Expected meteor trail length (3× faster than stars)
5. Display detailed analysis with optimization reasoning
6. Process all images with scientifically-optimized settings

Example output:
```
============================================================
Auto-params: NPF Rule-based Optimization
============================================================

Camera Settings (EXIF Metadata)
============================================================
  Camera:           Olympus OM-1
  Focal length:     24.0mm (35mm equiv.)
  ISO:              1600
  Exposure:         5.0s
  Aperture:         f/2.8
  Resolution:       5240×3912 px
============================================================

NPF Rule Analysis
============================================================
  Pixel pitch:      3.30μm (sensor: 17.3mm)
  NPF recommended:  8.2s
  Actual exposure:  5.0s ✓ OK
  Star trail est.:  ~1.5 pixels
  Impact:           LOW
============================================================

Parameter Optimization (NPF Rule-based)
============================================================

Shooting Quality Score: 1.00 (EXCELLENT)

Parameter Adjustments:
  • diff_threshold: 8 → 7 (ISO/NPF-based)
  • min_area: 10 → 3 (star trail-based)
  • min_line_score: 80.0 → 30.0 (meteor trail-based)

============================================================
```

### Understanding NPF Rule

The **NPF Rule** (developed by PhotoPills) is a scientific method to calculate the maximum exposure time before stars show trailing due to Earth's rotation. Unlike traditional "500 Rule" or "600 Rule", it accounts for modern camera sensor pixel pitch:

```
NPF Exposure (seconds) = (35 × F-number + 30 × Pixel Pitch) / Focal Length
```

Where:
- **F-number**: Aperture (e.g., f/2.8 → 2.8)
- **Pixel Pitch**: Physical pixel size in micrometers (μm)
- **Focal Length**: 35mm equivalent in millimeters

The software uses this to:
- Assess whether your exposure settings are optimal
- Estimate star trail length during actual exposure
- Adjust detection parameters based on NPF compliance

### Sensor Width Specification

For best accuracy, specify your camera's sensor width:

```bash
# Micro Four Thirds (17.3mm)
python detect_meteors_cli.py --auto-params --sensor-width 17.3

# APS-C (23.5mm)
python detect_meteors_cli.py --auto-params --sensor-width 23.5

# Full Frame (36.0mm)
python detect_meteors_cli.py --auto-params --sensor-width 36.0
```

Or use sensor type shortcuts:
```bash
python detect_meteors_cli.py --auto-params --focal-factor MFT
python detect_meteors_cli.py --auto-params --focal-factor APS-C
python detect_meteors_cli.py --auto-params --focal-factor FF
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

Limit processing to a region of the starry sky using a polygon ROI:
```bash
python detect_meteors_cli.py --roi "10,10;4000,10;4000,2000;10,2000"
```

### Combining Auto-Parameters with Manual Overrides

You can use `--auto-params` while still manually specifying certain parameters:

```bash
# Auto-optimize with manual diff_threshold
python detect_meteors_cli.py --auto-params \
  --sensor-width 17.3 \
  --diff-threshold 12

# Auto-optimize with pre-defined ROI
python detect_meteors_cli.py --auto-params \
  --roi "100,100;3900,100;3900,2900;100,2900" \
  --sensor-width 17.3

# Mix auto and manual parameters
python detect_meteors_cli.py --auto-params \
  --sensor-width 17.3 \
  --min-area 15 \
  --min-aspect-ratio 4.0
```

### NPF Analysis Only

Display NPF analysis without processing images:

```bash
python detect_meteors_cli.py --show-npf --sensor-width 17.3
```

This shows:
- Calculated pixel pitch
- NPF recommended exposure time
- Your actual exposure time and compliance
- Estimated star trail length
- Impact assessment (LOW/MODERATE/HIGH)

### EXIF Metadata Check Only

Display EXIF metadata from your RAW files without processing:

```bash
python detect_meteors_cli.py --show-exif
```

**Use this to:**
- Verify focal length is correctly extracted
- Check ISO, exposure time, and aperture values
- Confirm camera and lens information
- Decide whether manual `--focal-length` or `--focal-factor` is needed

## Option Reference

All command-line flags for `detect_meteors_cli.py`, with defaults and guidance:

### Input/Output Options
- **`-t`/`--target`** (default: `rawfiles`): Source folder that contains RAW images to scan.
- **`-o`/`--output`** (default: `candidates`): Destination folder for RAW files flagged as meteor candidates.
- **`--debug-dir`** (default: `debug_masks`): Where to save generated mask and debug images.

### Detection Parameters
- **`--diff-threshold`** (default: `8`): Pixel-difference threshold used to binarize frame-to-frame differences. **TIP**: Use `--auto-params` to optimize automatically based on ISO and NPF compliance.
- **`--min-area`** (default: `10`): Smallest allowed contour area in pixels. **TIP**: Use `--auto-params` to optimize based on star trail length.
- **`--min-aspect-ratio`** (default: `3.0`): Minimum ratio of a contour's long side to its short side.

### Hough Transform Parameters
- **`--hough-threshold`** (default: `10`): Accumulator threshold for the probabilistic Hough transform.
- **`--hough-min-line-length`** (default: `15`): Minimum line length (in pixels) accepted by the Hough transform.
- **`--hough-max-line-gap`** (default: `5`): Maximum gap (in pixels) between segments on the same detected line.
- **`--min-line-score`** (default: `80.0`): Minimum summed line length score required to mark a meteor candidate. **TIP**: Use `--auto-params` to optimize based on expected meteor trail length.

### Region of Interest (ROI) Options
- **`--no-roi`**: Skip ROI selection and process the entire frame.
- **`--roi`**: Explicit polygon ROI as `"x1,y1;x2,y2;..."` (needs ≥3 vertices).

### NPF Rule-based Auto-Parameter Optimization (NEW in v1.4)
- **`--auto-params`**: Automatically optimize all three critical detection parameters using NPF Rule and EXIF metadata. The algorithm:
  - Extracts EXIF data (ISO, exposure, aperture, focal length, resolution)
  - Calculates NPF recommended exposure and star trail length
  - Evaluates shooting condition quality (EXCELLENT/GOOD/FAIR/POOR)
  - Optimizes `diff_threshold` based on ISO sensitivity and NPF overshoot
  - Optimizes `min_area` based on star trail length
  - Optimizes `min_line_score` based on meteor speed (3× faster than stars)
  - Manual parameter specifications always take priority over auto-optimization

### NPF Rule Options (NEW in v1.4)
- **`--sensor-width`**: Physical sensor width in millimeters (e.g., `17.3` for MFT, `23.5` for APS-C, `36.0` for Full Frame). Used to calculate pixel pitch for NPF Rule. Significantly improves optimization accuracy.
- **`--pixel-pitch`**: Direct pixel pitch specification in micrometers (μm). If not specified, calculated from `--sensor-width` and image resolution, or uses default value (4.0μm).
- **`--focal-length`**: Focal length in 35mm equivalent (mm). If not specified, automatically extracted from EXIF metadata. Can be manually specified to override EXIF value.
- **`--focal-factor`**: Sensor type or crop factor (e.g., `MFT`, `APS-C`, `FF`, or numeric like `2.0`). Used to convert actual focal length to 35mm equivalent.
- **`--show-npf`**: Display detailed NPF Rule analysis and exit without processing. Shows pixel pitch, NPF recommended exposure, compliance level, star trail estimate, and impact assessment.
- **`--show-exif`**: Display EXIF metadata only and exit without processing. **Use this first** to verify focal length extraction before running `--auto-params`.

### Performance Options
- **`--workers`** (default: CPU count - 1): Number of parallel worker processes.
- **`--batch-size`** (default: `10`): How many RAW files each worker processes at a time.
- **`--auto-batch-size`**: Dynamically shrink batch size to stay within ~60% of available RAM.
- **`--no-parallel`**: Force single-threaded execution.

### Utility Options
- **`--profile`**: Print timing breakdowns after the run.
- **`--validate-raw`**: Pre-validate RAW files to catch corruption before processing.
- **`--progress-file`** (default: `progress.json`): Path to the JSON file that tracks processed frames.
- **`--no-resume`**: Ignore and remove any existing progress file before processing.
- **`--remove-progress`**: Delete the progress file and exit immediately.

## What's New in v1.4

### v1.4.1 - NPF Rule-based Scientific Optimization (Latest)
- **NPF Rule integration**: Implemented PhotoPills NPF Rule for scientifically accurate exposure time validation
- **EXIF metadata extraction**: Automatic extraction of focal length, ISO, exposure time, aperture, and resolution from RAW files
- **Sensor characterization**: Pixel pitch calculation from sensor width and image resolution
- **Star trail estimation**: Physics-based calculation of star movement during exposure (Earth's rotation: 15°/hour)
- **Shooting quality score**: Comprehensive assessment (0.0-1.0) based on NPF compliance, ISO sensitivity, and focal length
- **Intelligent parameter optimization**: 
  - `diff_threshold`: Adjusted for ISO sensitivity and NPF overshoot
  - `min_area`: Based on estimated star trail length
  - `min_line_score`: Based on expected meteor trail length (meteors move ~3× faster than stars)
- **Detailed analysis output**: Shows NPF compliance, pixel pitch, star trail estimate, quality score, and optimization reasoning
- **Real-world validated**: 100% detection rate (9 candidates including 2 confirmed meteors) on Olympus OM-1 test dataset
- See [RELEASE_NOTES_1.4.1.md](RELEASE_NOTES_1.4.1.md) for comprehensive details

### v1.3.1 - Complete Auto-Parameter Estimation
- **Complete automation**: `--auto-params` estimates **all three** critical parameters automatically
- **Star size analysis**: Automatic `min_area` estimation from detected star sizes
- **Image geometry scoring**: Automatic `min_line_score` estimation from image diagonal
- **Focal length support**: `--focal-length` option optimizes for wide-angle to telephoto lenses
- **Progress tracking**: Restored from v1.1.0 with `progress.json` support

### v1.2.1 - Improved Auto-Parameter Estimation
- **Percentile-based estimation**: Switched from 3-sigma rule to percentile-based approach for `diff_threshold`
- **Real-world validated**: Reduced typical thresholds from 25 to 15, significantly improving meteor detection

## Build a Single Binary with Nuitka
If you want to distribute `detect_meteors_cli` as a standalone executable, you can bundle it with [Nuitka](https://nuitka.net/):

```bash
pip install nuitka
python -m nuitka --onefile --standalone detect_meteors_cli.py
```

## Inputs and Outputs
- **Inputs:** A directory of RAW images (all files supported by `rawpy` will be considered).
- **Outputs:**
  - Candidate images saved to the directory provided with `-o/--output`.
  - Optional debug masks written to the directory provided with `--debug-dir`.
  - `progress.json` file for tracking processed images (resumable processing).

## Tips for Best Results

### Recommended Workflow (v1.4+)

**Step 1: Check EXIF metadata**
```bash
python detect_meteors_cli.py --show-exif
```
- Verify focal length is correctly extracted
- Note your ISO, exposure, and aperture settings
- If focal length is missing, prepare to use `--focal-length` or `--focal-factor`

**Step 2: Verify NPF compliance (optional)**
```bash
python detect_meteors_cli.py --show-npf --sensor-width 17.3
```
- Check if your exposure time is within NPF recommendation
- Understand your shooting quality score
- See estimated star trail length

**Step 3: Run auto-parameter optimization**
```bash
# If focal length was detected in EXIF
python detect_meteors_cli.py --auto-params --sensor-width 17.3

# If focal length was missing in EXIF
python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-length 24
```

**Step 4: Review results and adjust if needed**
- Check detected candidates
- If too many false positives: increase thresholds manually
- If missing meteors: decrease thresholds manually

### Using NPF Rule-based Auto-Parameters (v1.4+)

1. **Provide sensor information**: Use `--sensor-width MM` or `--focal-factor TYPE` for accurate pixel pitch calculation
   ```bash
   # Micro Four Thirds
   python detect_meteors_cli.py --auto-params --sensor-width 17.3
   
   # Or use sensor type
   python detect_meteors_cli.py --auto-params --focal-factor MFT
   ```

2. **Check EXIF before processing**: Use `--show-exif` to verify focal length extraction
   ```bash
   python detect_meteors_cli.py --show-exif
   ```
   - If focal length is missing: add `--focal-length MM` or `--focal-factor TYPE`

3. **Check NPF compliance first**: Use `--show-npf` to understand your shooting conditions
   ```bash
   python detect_meteors_cli.py --show-npf --sensor-width 17.3
   ```

4. **Optimal shooting conditions**:
   - **Exposure time**: Keep within NPF recommended limit for best results
   - **ISO**: Lower ISO reduces noise but requires longer exposure
   - **Aperture**: Wider aperture (lower f-number) allows shorter exposures

5. **Select a clean ROI**: Ensure the ROI contains only pure night sky without:
   - Artificial lights (streetlights, light pollution sources)
   - Ground objects (trees, buildings, horizon line)
   - Atmospheric features (clouds, fog, aurora)

### Expected Auto-Optimized Values

#### By Sensor Type
- **Micro Four Thirds (MFT)**: pixel_pitch ~3.3μm, NPF ~6-10s
- **APS-C**: pixel_pitch ~3.9μm, NPF ~8-12s
- **Full Frame**: pixel_pitch ~5.9μm, NPF ~12-16s

#### By Shooting Conditions
- **Low ISO (≤1600), NPF OK**: diff_threshold ~5-7, min_area ~3-5, min_line_score ~30-40
- **Medium ISO (~3200), NPF exceeded**: diff_threshold ~8-12, min_area ~4-8, min_line_score ~30-50
- **High ISO (≥6400), NPF exceeded**: diff_threshold ~12-18, min_area ~6-12, min_line_score ~40-60

#### By NPF Compliance
- **Under NPF (OK)**: Optimal star trail control, lower thresholds
- **Slight overshoot (1.0-1.5×)**: Acceptable, minor adjustments
- **Moderate overshoot (1.5-2.5×)**: Noticeable star trails, increased thresholds
- **Critical overshoot (>2.5×)**: Significant star trails, may affect detection quality

### Shooting Quality Assessment

The software provides a quality score (0.0-1.0) based on:

1. **NPF Compliance** (most important, 60% weight):
   - OK (≤1.0×): Perfect score
   - WARNING (≤1.5×): 0.8
   - MODERATE (≤2.5×): 0.5
   - CRITICAL (>2.5×): 0.3

2. **ISO Sensitivity** (25% weight):
   - ≤1600: 1.0 (clean images)
   - ≤3200: 0.9 (acceptable noise)
   - ≤6400: 0.7 (noticeable noise)
   - >6400: 0.5 (high noise)

3. **Focal Length** (15% weight, wide angle advantageous):
   - ≤24mm: 1.0 (excellent meteor coverage)
   - ≤35mm: 0.95
   - ≤50mm: 0.85
   - >50mm: 0.7 (narrow field limits meteor visibility)

**Overall Quality Levels**:
- **EXCELLENT** (≥0.8): Ideal conditions for meteor detection
- **GOOD** (≥0.6): Acceptable conditions, good detection probability
- **FAIR** (≥0.4): Suboptimal conditions, detection may be limited
- **POOR** (<0.4): Challenging conditions, consider adjusting shooting parameters

### Resumable Processing
- Long processing sessions can be interrupted with Ctrl-C
- Progress is automatically saved to `progress.json`
- Simply run the same command again to resume
- Use `--no-resume` for a fresh start
- Use `--remove-progress` to clear saved progress

### When to Use Manual Parameters
- Extreme ISOs (>12800) where auto-optimization may need adjustment
- Known problematic conditions (aurora, airglow, unusual atmospheric phenomena)
- Fine-tuning based on initial auto-params results
- Special requirements for specific research or publication needs

## Common Sensor Widths Reference

| Camera System | Sensor Width (mm) | Crop Factor | Typical Pixel Pitch (μm) |
|--------------|-------------------|-------------|-------------------------|
| Micro Four Thirds | 17.3 | 2.0 | 3.3-3.7 |
| APS-C (Sony, Nikon, Fuji) | 23.5 | 1.5 | 3.9-4.3 |
| APS-C (Canon) | 22.3 | 1.6 | 4.1-4.5 |
| APS-H (Canon) | 27.9 | 1.3 | 5.0-6.4 |
| Full Frame | 36.0 | 1.0 | 4.3-8.4 |
| 1-inch | 13.2 | 2.7 | 2.4-2.9 |

## Contributing
Issues and pull requests are welcome. Please open an issue to discuss substantial changes before submitting a PR.

## License
This project is licensed under the terms of the [Apache License 2.0](LICENSE).
