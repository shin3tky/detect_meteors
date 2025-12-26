# Detect Meteors CLI

![social_preview](social_preview.jpg)

[![tests](https://github.com/shin3tky/detect_meteors/actions/workflows/python-test.yml/badge.svg)](https://github.com/shin3tky/detect_meteors/actions/workflows/python-test.yml)

Automatically detect meteors in RAW astrophotography images using frame-to-frame difference analysis.

**Note:** Detection compares consecutive RAW images. The first image is skipped—please check it manually.

## Motivation

During meteor shower events, manually reviewing thousands of RAW images to find meteors is tedious and time-consuming. This tool automates the initial detection process, allowing astrophotographers to quickly identify candidate images for further review.

![workflow](workflow.png)

📅 **Planning your meteor photography?** Check out the [Meteor Showers Calendar](https://github.com/shin3tky/detect_meteors/wiki/Meteor-Showers-Calendar) for upcoming meteor shower dates and viewing tips.

## Features

- **Fully automated**: NPF Rule-based optimization analyzes EXIF metadata and scientifically tunes detection parameters
- **Scientifically validated**: 100% detection rate on real-world test dataset (OM Digital OM-1, 1000+ RAW images)
- **RAW format support**: Works with any format supported by [`rawpy`](https://github.com/letmaik/rawpy)
- **Intelligent processing**: ROI cropping, Hough transform line detection, resumable batch processing
- **High performance**: ~0.18 sec/image with multi-core parallel processing

## Requirements

- Python 3.12+
- macOS, Windows, or Linux
- Dependencies: `numpy`, `opencv-python`, `rawpy`, `psutil`, `pillow`, `pydantic`, `pyyaml`

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Quick Start

### Step 1: Check EXIF Metadata

```bash
python detect_meteors_cli.py --show-exif
```

Verify focal length is detected. If missing, you'll need to specify it with `--focal-length`.

### Step 2: Run Detection

```bash
# Micro Four Thirds camera
python detect_meteors_cli.py --auto-params --sensor-type MFT

# APS-C camera (Sony/Nikon/Fuji)
python detect_meteors_cli.py --auto-params --sensor-type APS-C

# Full Frame camera
python detect_meteors_cli.py --auto-params --sensor-type FF

# With fisheye lens
python detect_meteors_cli.py --auto-params --sensor-type MFT --focal-length 16 --fisheye
```

### Step 3: Review Candidates

Check the `candidates/` folder for detected meteor images.

## Available Sensor Types

| Sensor Type | Description |
|-------------|-------------|
| `1INCH` | 1-inch sensor |
| `MFT` | Micro Four Thirds |
| `APS-C` | APS-C (Sony/Nikon/Fuji) |
| `APS-C_CANON` | APS-C (Canon) |
| `APS-H` | APS-H |
| `FF` | Full Frame 35mm |
| `MF44X33` | Medium Format 44×33mm |
| `MF54X40` | Medium Format 54×40mm |

List all presets: `python detect_meteors_cli.py --list-sensor-types`

## Inputs and Outputs

- **Input**: Directory of RAW images (default: `rawfiles/`)
- **Output**: 
  - Candidate images in `candidates/` (or custom `-o` path)
  - Optional debug masks with `--debug-image` and `--debug-dir`
  - `progress.json` for resumable processing

## Configuration Files (YAML/JSON)

The CLI can load pipeline settings from a configuration file. The file must be a
JSON or YAML object whose keys align with `PipelineConfig`.

**Top-level keys**

- `target_folder`, `output_folder`, `debug_folder` (required paths)
- `params` (detection parameters)
- `num_workers`, `batch_size`, `auto_batch_size`, `enable_parallel`
- `progress_file`, `output_overwrite`
- `input_loader_name`, `input_loader_config`
- `detector_name`, `detector_config`
- `output_handler_name`, `output_handler_config`

**Example (YAML)**

```yaml
target_folder: ./rawfiles
output_folder: ./candidates
debug_folder: ./debug_masks
params:
  diff_threshold: 8
  min_area: 10
  min_aspect_ratio: 3.0
input_loader_name: raw
input_loader_config:
  binning: 1
  normalize: true
detector_name: hough
output_handler_name: file
```

**Example file**: [`config_examples/pipeline.yaml`](config_examples/pipeline.yaml)

**Usage (Python)**

```python
from meteor_core import MeteorDetectionPipeline, load_pipeline_config

config = load_pipeline_config("config_examples/pipeline.yaml")
pipeline = MeteorDetectionPipeline(config)
pipeline.run()
```

## Resumable Processing

- Interrupt with Ctrl-C anytime
- Resume by running the same command again
- Use `--no-resume` for a fresh start

## Documentation

| Document | Description |
|----------|-------------|
| [COMMAND_OPTIONS.md](COMMAND_OPTIONS.md) | Complete CLI options reference |
| [NPF_RULE.md](NPF_RULE.md) | NPF Rule and focal length handling |
| [INSTALL.md](INSTALL.md) | Installation guide |
| [INSTALL_DEV.md](INSTALL_DEV.md) | Developer setup |
| [PLUGIN_AUTHOR_GUIDE.md](PLUGIN_AUTHOR_GUIDE.md) | Plugin development |
| [Wiki](https://github.com/shin3tky/detect_meteors/wiki) | Technical details |

## What's New in v1.6.4 🎄

- **Frame indices in detection context**: `frame_index` and `prev_frame_index` tracked throughout the pipeline
  - `progress.json` `detected_details` now includes frame indices for post-processing analysis
  - Progress messages show which frames detected meteors
- **Output handler `on_detection_result` hook**: New callback for per-detection processing
  - Receives serialized context payload with runtime params, file info, timestamps
  - Documented lifecycle: `on_detection_result` → `on_candidate_image` → `on_debug_image`
- **DetectionResult propagation**: Access detector `lines`, `extras`, and `metrics` in output handlers
- **Performance improvements**: Optimized debug image generation and runtime param building

For detailed migration information, see [RELEASE_NOTES_1.6.md](RELEASE_NOTES_1.6.md).

### Previous Releases

| Version | Highlights | Details |
|---------|------------|---------|
| v1.6.x | Schema versioning, ML-ready architecture, uv/Ruff toolchain | [RELEASE_NOTES_1.6.md](RELEASE_NOTES_1.6.md) |
| v1.5.x | Plugin architecture, sensor presets, fisheye support | [RELEASE_NOTES_1.5.md](RELEASE_NOTES_1.5.md) |
| v1.4.x | NPF Rule optimization, EXIF extraction | [RELEASE_NOTES_1.4.md](RELEASE_NOTES_1.4.md) |
| v1.3.x | Auto-parameter estimation | [RELEASE_NOTES_1.3.md](RELEASE_NOTES_1.3.md) |
| v1.2.x | Threshold estimation improvements | [RELEASE_NOTES_1.2.md](RELEASE_NOTES_1.2.md) |

See [CHANGELOG.md](CHANGELOG.md) for complete release history.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for upcoming features.

## Authors

Detect Meteors CLI was created by Shinichi Morita (shin3tky).

The NPF Rule implementation is based on the formula developed by Frédéric Michaud of the Société Astronomique du Havre (SAH). See [NOTICE](NOTICE) for full attribution.

## Contributing

Issues and pull requests are welcome. Please open an issue to discuss substantial changes before submitting a PR.

For development setup, see [INSTALL_DEV.md](INSTALL_DEV.md).

## License

This project is licensed under the [Apache License 2.0](LICENSE).
