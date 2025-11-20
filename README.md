# detect_meteors

Detect meteors in batches of RAW photos using configurable image processing pipelines.

## Overview
- CLI-first workflow for scanning folders of RAW photos and flagging potential meteors.
- Works with RAW photos supported by [`rawpy`](https://github.com/letmaik/rawpy) (tested with ORF files).
- Provides region-of-interest (ROI) cropping and Hough transform tuning to focus on likely meteor streaks.
- See the [CHANGELOG](CHANGELOG.md) for release history.

## Requirements
- Python 3.12.12 (tested).
- macOS Tahoe on an Intel MacBook Pro (tested); other Unix-like systems may work.
- Help wanted: verification on Apple Silicon Macs, Windows, and Linux would be greatly appreciated.
- Dependencies: `numpy`, `matplotlib`, `opencv-python`, `rawpy`, `psutil`.

## Installation
1) Set up Python (example uses `brew`).
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
   pip install numpy matplotlib opencv-python rawpy psutil
   ```

## Usage
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

### Option reference

All command-line flags for `detect_meteors_cli.py`, with defaults and guidance:

- **`-t`/`--target`** (default: `rawfiles`): Source folder that contains RAW images to scan. All files supported by `rawpy` are considered.
- **`-o`/`--output`** (default: `candidates`): Destination folder for RAW files flagged as meteor candidates.
- **`--debug-dir`** (default: `debug_masks`): Where to save generated mask and debug images. Create the directory beforehand to keep outputs organized.
- **`--diff-threshold`** (default: `8`): Pixel-difference threshold used to binarize frame-to-frame differences. Raise to suppress noise; lower to capture faint streaks.
- **`--min-area`** (default: `10`): Smallest allowed contour area in pixels. Increase to ignore tiny speckles or hot pixels; decrease to detect very small objects.
- **`--min-aspect-ratio`** (default: `3.0`): Minimum ratio of a contour’s long side to its short side. Meteors are elongated; higher values enforce skinnier shapes.
- **`--hough-threshold`** (default: `10`): Accumulator threshold for the probabilistic Hough transform. Higher values demand stronger line evidence and reduce false positives.
- **`--hough-min-line-length`** (default: `15`): Minimum line length (in pixels) accepted by the Hough transform. Tune together with `--hough-max-line-gap` to match expected streak lengths.
- **`--hough-max-line-gap`** (default: `5`): Maximum gap (in pixels) that can exist between segments on the same detected line. Lower gaps favor continuous streaks; higher gaps tolerate breaks from noise.
- **`--min-line-score`** (default: `80.0`): Minimum summed line length score required to mark a meteor candidate. Raise to capture only the clearest streaks; lower to catch faint or short lines.
- **`--no-roi`**: Skip ROI selection and process the entire frame. Useful for wide-field captures where meteors could appear anywhere.
- **`--roi`**: Explicit polygon ROI as `"x1,y1;x2,y2;..."` (needs ≥3 vertices). Overrides interactive ROI selection and can be scripted for repeatable crops.
- **`--workers`** (default: `psutil.cpu_count(logical=True)`): Number of parallel worker processes. Increase to speed up on multi-core machines; reduce if the system feels sluggish.
- **`--batch-size`** (default: `10`): How many RAW files each worker processes at a time. Larger batches reduce I/O overhead but consume more memory.
- **`--auto-batch-size`**: Dynamically shrink batch size to stay within ~60% of available RAM. Pair with `--workers` to balance speed and memory safety.
- **`--no-parallel`**: Force single-threaded execution. Handy for debugging or when parallelism conflicts with other workloads.
- **`--profile`**: Print timing breakdowns (first load, processing time, totals) after the run.
- **`--validate-raw`**: Pre-validate RAW files to catch corruption before processing. Adds a quick sanity check step on large batches.
- **`--progress-file`** (default: `progress.json`): Path to the JSON file that tracks processed and detected frames so long runs can resume safely.
- **`--no-resume`**: Ignore and remove any existing progress file before processing. Use when you want a clean run without picking up past state.
- **`--remove-progress`**: Delete the progress file and exit immediately. Handy for clearing progress without starting a new detection pass.

## Build a single binary with Nuitka
If you want to distribute `detect_meteors_cli` as a standalone executable, you can bundle it with [Nuitka](https://nuitka.net/):

```bash
pip install nuitka
python -m nuitka --onefile --standalone detect_meteors_cli.py
```

## Inputs and outputs
- **Inputs:** A directory of RAW photos (all files supported by `rawpy` will be considered).
- **Outputs:**
  - Candidate images saved to the directory provided with `-o/--output`.
  - Optional debug masks written to the directory provided with `--debug-dir`.

## Contributing
Issues and pull requests are welcome. Please open an issue to discuss substantial changes before submitting a PR.

## License
This project is licensed under the terms of the [Apache License 2.0](LICENSE).
