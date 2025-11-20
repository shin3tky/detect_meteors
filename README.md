# detect_meteors

Detect meteors in batches of RAW photos using configurable image processing pipelines.

## Overview
- CLI-first workflow for scanning folders of RAW photos and flagging potential meteors.
- Works with RAW photos supported by [`rawpy`](https://github.com/letmaik/rawpy) (tested with ORF files).
- Provides region-of-interest (ROI) cropping and Hough transform tuning to focus on likely meteor streaks.

## Requirements
- Python 3.13.7 (tested).
- macOS Tahoe (tested); other Unix-like systems may work.
- Dependencies: `numpy`, `matplotlib`, `opencv-python`, `rawpy`.

## Installation
1) Set up Python (example uses `pyenv`).
   ```bash
   pyenv local 3.13.7
   ```
2) Create and activate a virtual environment.
   ```bash
   python3 -m venv venv
   source ./venv/bin/activate
   ```
3) Install dependencies.
   ```bash
   pip install numpy matplotlib opencv-python rawpy
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

Limit processing to a region of the starry sky:
```bash
python detect_meteors_cli.py --roi 10,10,4000,2000
```

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

### Build a single binary with Nuitka
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
