# Detect Meteors CLI

[![Python application test](https://github.com/shin3tky/detect_meteors/actions/workflows/python-test.yml/badge.svg)](https://github.com/shin3tky/detect_meteors/actions/workflows/python-test.yml)

Automatically detect meteors in RAW astrophotography images using frame-to-frame difference analysis.

**Note:** Detection compares consecutive RAW images. The first image is skipped—please check it manually.

## Motivation

During meteor shower events, manually reviewing thousands of RAW images to find meteors is tedious and time-consuming. This tool automates the initial detection process, allowing astrophotographers to quickly identify candidate images for further review.

I developed this tool hoping it would be useful for fellow astrophotography enthusiasts who face the same challenge.

![workflow](workflow.png)

📅 **Planning your meteor photography?** Check out the [Meteor Showers Calendar](https://github.com/shin3tky/detect_meteors/wiki/Meteor-Showers-Calendar) for upcoming meteor shower dates and viewing tips.

## Overview
- **Fully automated**: NPF Rule-based optimization analyzes EXIF metadata (ISO, exposure, focal length) and scientifically tunes all detection parameters
- **Scientifically validated**: 100% detection rate on real-world test dataset (OM Digital OM-1, 1000+ RAW images)
- **RAW format support**: Works with any format supported by [`rawpy`](https://github.com/letmaik/rawpy) (tested: Olympus/OM Digital ORF)
- **Intelligent processing**: ROI cropping, Hough transform line detection, and resumable batch processing
- See [CHANGELOG](CHANGELOG.md) for release history

## Roadmap
- Upcoming plans are outlined in [ROADMAP](ROADMAP.md).

## Technical Overview
- See the project wiki for a technical summary: [Technical Processing Summary](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Summary)
- See the project wiki for a deeper technical walkthrough: [Technical Processing Overview](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Overview)
- Auto-Parameter Estimation Extensions: [v1.3 Additions](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Overview-v1.3-Additions)
- NPF Rule-based Scientific Optimization: [v1.4 Additions](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Overview-v1.4-Additions)
- Sensor presets and fisheye correction: [v1.5 Additions](https://github.com/shin3tky/detect_meteors/wiki/Technical-Processing-Overview-v1.5-Additions)

## Requirements
- Python 3.12.12 (tested).
- macOS Tahoe 26.1 on an Intel MacBook Pro 16-inch, 2019 (tested); other Unix-like systems may work.
- Windows 11 Pro 25H2 on HP ZBook 14u G5 (tested).
- Help wanted: verification on Apple Silicon Macs, Windows, and Linux would be greatly appreciated.
- Dependencies: `numpy`, `matplotlib`, `opencv-python`, `rawpy`, `psutil`, `pillow`.

## Installation

For detailed installation instructions for macOS and Windows, please refer to [INSTALL.md](INSTALL.md).

## What's New in v1.5

### v1.5.5 - Internal Structure Improvement
- **Code architecture refactoring**: Separated CLI interface from core logic modules
- **New `meteor_core/` package**: Modular components for better maintainability
- **Type safety improvements**: Enhanced type hints with TypedDict
- **v2.x preparation**: Foundation for plugin architecture

### v1.5.0–1.5.4 Summary
- **v1.5.4**: Improved ROI selection display, added NOTICE document
- **v1.5.3**: Fisheye lens correction (`--fisheye` flag)
- **v1.5.2**: Sensor override validation with automatic warnings
- **v1.5.1**: Medium format sensor support (MF44X33, MF54X40)
- **v1.5.0**: Sensor type presets (`--sensor-type` option)

👉 See [RELEASE_NOTES_1.5.md](RELEASE_NOTES_1.5.md) for complete details and usage examples.

### Earlier Versions

- **v1.4.2**: Output file protection - skip overwriting existing files by default ([details](RELEASE_NOTES_1.4.md))
- **v1.4.1**: NPF Rule-based scientific optimization with EXIF metadata extraction ([details](RELEASE_NOTES_1.4.md))
- **v1.3.1**: Complete auto-parameter estimation with star size and image geometry analysis ([details](RELEASE_NOTES_1.3.md))
- **v1.2.1**: Percentile-based threshold estimation ([details](RELEASE_NOTES_1.2.md))

## Usage

### Quick Start

**Step 1: Check EXIF Metadata (Recommended)**

Before processing, verify that your RAW files contain focal length information:

```bash
python detect_meteors_cli.py --show-exif
```

This displays EXIF metadata from your first RAW file:
```
Camera Settings (EXIF Metadata)
============================================================
  Camera:           OM Digital OM-1
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

The most scientific approach - let the software automatically optimize detection parameters using the NPF Rule and EXIF metadata:

**Option A: Use sensor type preset (recommended, v1.5+)**
```bash
# Micro Four Thirds
python detect_meteors_cli.py --auto-params --sensor-type MFT

# APS-C (Sony/Nikon/Fuji)
python detect_meteors_cli.py --auto-params --sensor-type APS-C

# APS-C (Canon)
python detect_meteors_cli.py --auto-params --sensor-type APS-C_CANON

# Full Frame
python detect_meteors_cli.py --auto-params --sensor-type FF

# Medium Format (Fujifilm GFX, Pentax 645Z, Hasselblad X2D)
python detect_meteors_cli.py --auto-params --sensor-type MF44X33

# Fisheye lens (v1.5.3+) - add --fisheye flag
python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 16 --fisheye
```

**Option B: If focal length is missing in EXIF - specify manually**
```bash
# Specify exact focal length (35mm equivalent)
python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 24

# Or override specific values from preset
python detect_meteors_cli.py --auto-params --sensor-type MFT --pixel-pitch 3.3
```

**Option C: Legacy manual specification (still supported)**
```bash
python detect_meteors_cli.py --auto-params --sensor-width 17.3 --focal-factor MFT
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
  Camera:           OM Digital OM-1
  Focal length:     24.0mm (35mm equiv.)
  ISO:              1600
  Exposure:         5.0s
  Aperture:         f/2.8
  Resolution:       5240×3912 px
============================================================

NPF Rule Analysis
============================================================
  Pixel pitch:      3.30μm (sensor: 17.3mm)
  NPF recommended:  8.3s
  Actual exposure:  5.0s ✓ OK (0.60× NPF)
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

For more details on the NPF Rule and focal length handling, see [NPF_RULE.md](NPF_RULE.md).

For complete command line options reference, see [COMMAND_OPTIONS.md](COMMAND_OPTIONS.md).

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
python detect_meteors_cli.py --show-npf --sensor-type MFT
```
- Check if your exposure time is within NPF recommendation
- Understand your shooting quality score
- See estimated star trail length

**Step 3: Run auto-parameter optimization**
```bash
# Use sensor type preset (recommended, v1.5+)
python detect_meteors_cli.py --auto-params --sensor-type MFT

# If focal length was missing in EXIF, add --focal-length
python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 24

# Or override specific preset values if needed
python detect_meteors_cli.py --auto-params --sensor-type MFT --pixel-pitch 3.3
```

**Step 4: Review results and adjust if needed**
- Check detected candidates
- If too many false positives: increase thresholds manually
- If missing meteors: decrease thresholds manually

### Using NPF Rule-based Auto-Parameters (v1.4+, simplified in v1.5)

1. **Use sensor type preset (v1.5+, recommended)**: Use `--sensor-type TYPE` for automatic configuration
   ```bash
   # Micro Four Thirds
   python detect_meteors_cli.py --auto-params --sensor-type MFT
   
   # Sony/Nikon/Fuji APS-C camera
   python detect_meteors_cli.py --auto-params --sensor-type APS-C
   
   # Full Frame camera
   python detect_meteors_cli.py --auto-params --sensor-type FF
   
   # Medium Format (Fujifilm GFX, Pentax 645Z)
   python detect_meteors_cli.py --auto-params --sensor-type MF44X33
   
   # List all available sensor types
   python detect_meteors_cli.py --list-sensor-types
   ```

2. **Override preset values when needed**: Individual parameters take priority over presets
   ```bash
   # Use MFT preset but override pixel pitch for specific camera
   python detect_meteors_cli.py --auto-params --sensor-type MFT --pixel-pitch 3.3
   ```

3. **Check EXIF before processing**: Use `--show-exif` to verify focal length extraction
   ```bash
   python detect_meteors_cli.py --show-exif
   ```
   - If focal length is missing: add `--focal-length MM`

4. **Check NPF compliance first**: Use `--show-npf` to understand your shooting conditions
   ```bash
   python detect_meteors_cli.py --show-npf --sensor-type MFT
   ```

5. **Optimal shooting conditions**:
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
- **Medium Format 44×33**: pixel_pitch ~3.76μm, NPF ~10-14s

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

## Contributing
Issues and pull requests are welcome. Please open an issue to discuss substantial changes before submitting a PR.

## License
This project is licensed under the terms of the [Apache License 2.0](LICENSE).
