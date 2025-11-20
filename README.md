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

Quick start (defaults to `examples` as input, `candidates` as output, and `debug_masks` for debug images):
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
