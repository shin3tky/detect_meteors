# detect_meteors
Detecting meteors from a set of RAW images

## What does this solve?
- This helps with the task of searching for meteors in thousands of RAW photos every time a meteor shower occurs.

## System Requirements
- Tested on macOS Tahoe.
- Tested on Python 3.13.7.
- Compatible with RAW photos supported by rawpy (Tested on ORF).

## Environment setup

```
pyenv local 3.13.7

python3 -m venv venv
source ./venv/bin/activate
pip install numpy matplotlib opencv-python rawpy
```

## How to use

Show help
```
python detect_meteors_cli.py --help
```

Use by default (-t examples -o candidates --debug-dir debug_masks)
```
python detect_meteors_cli.py
```

Specify the input and output folders
```
python detect_meteors_cli.py -t /path/to/raws -o meteors_out --debug-dir debug_out
```

Apply to the entire image
```
python detect_meteors_cli.py --no-roi
```

Specify the region of the starry sky
```
python detect_meteors_cli.py --roi 10,10,4000,2000
```

Candidate short-line-length meteors
```
python detect_meteors_cli.py \
  --hough-threshold 10 \
  --hough-min-line-length 15 \
  --hough-max-line-gap 5 \
  --min-line-score 40
```

Candidate long-line-length meteors
```
python detect_meteors_cli.py \
  --hough-threshold 15 \
  --hough-min-line-length 40 \
  --min-line-score 120
```

