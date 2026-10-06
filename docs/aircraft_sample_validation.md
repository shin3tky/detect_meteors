# Aircraft Trail Sample Validation: v1.6.10

Validated locally on 2026-10-06 with Python 3.12.14 on macOS. This is a
reproducible usage example for auxiliary metadata, not a classification accuracy
benchmark. The RAW files are tracked in the repository for reproducibility and
excluded from the Python wheel and source distribution. See the
[sample README and checksum instructions](../rawfiles/2024GEMINI_AIRCRAFT/README.md).

## Ground Truth and Processing

- Input: `rawfiles/2024GEMINI_AIRCRAFT`, 12 files from `_C140335.ORF` through
  `_C140346.ORF`, ordered by filename.
- The owner reports aircraft in all 12 images and confirms meteors only in
  `_C140338.ORF` and `_C140344.ORF`.
- EXIF: Olympus E-M1MarkII, actual focal length 12 mm, ISO 3200, 6 s, f/2.8.
- The built-in RAW loader produces 2620×1956 binned pixels; no ROI was applied.
- `_C140335.ORF` is the baseline and is not scored; 11 adjacent pairs are analyzed.

NPF-based auto-estimation with `--sensor-type MFT` yielded `diff_threshold: 9`,
`min_area: 3`, and `min_line_score: 30.0`. Other detection settings were the
built-in defaults. These exact values are saved in the sample configuration.

## Commands

Run from a repository checkout containing the tracked RAW files. For an
extracted source distribution, copy the sample directory from a repository
checkout first. All commands below use the project root as the working directory.

### Default Aircraft Matching

```bash
uv run python detect_meteors_cli.py \
  --target rawfiles/2024GEMINI_AIRCRAFT \
  --output candidates/aircraft_v1.6.10/default \
  --debug-dir debug_masks/aircraft_v1.6.10/default \
  --progress-file candidates/aircraft_v1.6.10/default/progress.json \
  --auto-params --sensor-type MFT --hooks aircraft_trail \
  --no-roi --no-resume --workers 2 --batch-size 4 --debug-image --profile
```

### Sample Matching Tolerances

```bash
uv run python detect_meteors_cli.py \
  --config config_examples/aircraft_trail_sample.yaml \
  --no-roi --no-resume --debug-image --profile
```

Compared with defaults, this configuration changes endpoint matching distances
from 6 to 150 binned pixels and angle tolerance from 3 to 10 degrees. Other
aircraft settings are unchanged. These are sample-specific exploratory values,
not new product defaults or validated settings for other cameras/sequences.

### Control Without the Hook

```bash
uv run python detect_meteors_cli.py \
  --config config_examples/aircraft_trail_sample.yaml --hooks "" \
  --output candidates/aircraft_v1.6.10/no_hook \
  --debug-dir debug_masks/aircraft_v1.6.10/no_hook \
  --progress-file candidates/aircraft_v1.6.10/no_hook/progress.json \
  --no-roi --no-resume
```

## Results

All three runs processed 11 pairs and retained 11 candidates, including both
owner-confirmed meteor images. Detection scores, line counts, aspect ratios, and
candidate sets agreed across the runs. Aircraft metadata was absent with the
hook disabled and present for all candidates in the two hook-enabled runs.

| Frame | Owner-confirmed meteor | Default likelihood | Sample likelihood | Sample track | Track observations |
|-------|------------------------|--------------------|-------------------|--------------|--------------------|
| `_C140335.ORF` | No | Baseline | Baseline | — | — |
| `_C140336.ORF` | No | 0.1667 | 0.1667 | `air-0001` | 1 |
| `_C140337.ORF` | No | 0.1667 | 0.6447 | `air-0001` | 2 |
| `_C140338.ORF` | Yes | 0.1667 | 0.1667 | `air-0002` | 1 |
| `_C140339.ORF` | No | 0.1667 | 0.7621 | `air-0002` | 2 |
| `_C140340.ORF` | No | 0.1667 | 0.1667 | `air-0003` | 1 |
| `_C140341.ORF` | No | 0.1667 | 0.6494 | `air-0003` | 2 |
| `_C140342.ORF` | No | 0.1667 | 0.8051 | `air-0003` | 3 |
| `_C140343.ORF` | No | 0.1667 | 0.7847 | `air-0003` | 4 |
| `_C140344.ORF` | Yes | 0.1667 | 0.8002 | `air-0003` | 5 |
| `_C140345.ORF` | No | 0.1667 | 0.8189 | `air-0003` | 6 |
| `_C140346.ORF` | No | 0.1667 | 0.8344 | `air-0003` | 7 |

Likelihoods are rounded to four decimals. Track IDs reset on each analysis pass.

## Interpretation

The default matching tolerances did not link any observations in this sample.
Each record therefore has a single-observation continuity score of 0.1667. A low
score does not mean that aircraft are absent.

The wider sample tolerances link several records, but `_C140344.ORF`, which
contains a meteor as well as aircraft, receives likelihood 0.8002. Discarding
images at a threshold of 0.7 would discard that meteor too. The configuration's
`likelihood_threshold` field does not implement such filtering.

The hook uses only the longest detected line per frame pair. The longest line
in `_C140338.ORF`'s difference has a different direction from nearby aircraft
lines, and the next pair can retain the disappearing meteor because the detector
uses absolute differences. These observations explain why a linked track alone
does not establish an aircraft-only image or reliably separate simultaneous
objects.

Use the metadata to support manual inspection of candidate RAWs and debug masks.
See the [aircraft usage guide](aircraft_light_trails_hook_design.md) for commands,
configuration fields, and resume/interruption behavior.
