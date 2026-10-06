# Aircraft Light Trails Hook: Current Implementation

This document describes the implementation in v1.6.10. It replaces the original
plan to perform tracking in `on_detection_complete`.

## Purpose

The optional `aircraft_trail` pipeline hook annotates detections with aircraft
light trail likelihood and geometric evidence. It preserves `is_candidate`,
`score`, candidate counts, and copied RAW files. The likelihood is a heuristic
score in [0, 1], not a calibrated probability or an automatic classification.

Implementation: `meteor_core/hooks/aircraft_trail.py`.

## Enable the Hook

```bash
uv run python detect_meteors_cli.py --hooks aircraft_trail --no-roi
```

Or add the following to a pipeline YAML configuration:

```yaml
hooks:
  - name: aircraft_trail
    config:
      min_track_frames: 3
hook_error_mode: warn
```

Hooks are disabled by default. The hook is included in built-in discovery and
registered under the `detect_meteors.hook` entry point. Custom hooks can also be
discovered from `~/.detect_meteors/hook_plugins/`.

### Configure from the Command Line

Use `--hook-config` to set matching tolerances without a pipeline file:

```bash
uv run python detect_meteors_cli.py \
  --target rawfiles/2024GEMINI_AIRCRAFT \
  --output candidates/aircraft_v1.6.10/cli \
  --debug-dir debug_masks/aircraft_v1.6.10/cli \
  --progress-file candidates/aircraft_v1.6.10/cli/progress.json \
  --auto-params --sensor-type MFT --no-roi --no-resume --debug-image \
  --hooks aircraft_trail \
  --hook-config '{"aircraft_trail":{"max_start_distance_px":150,"max_end_distance_px":150,"max_angle_diff_deg":10}}'
```

The distances above were tried on the local sample and are not universal
defaults. Start with the built-in defaults for a new sequence, inspect the
evidence, and adjust distances for the displacement in binned image coordinates.
Wider tolerances can join unrelated lines. `min_track_frames` changes continuity
weighting; `likelihood_threshold` currently has no effect.

### Run the Local Sample

The Git-tracked `rawfiles/2024GEMINI_AIRCRAFT` sequence contains 12 RAW files, all with
aircraft trails. The owner identifies meteors only in `_C140338.ORF` and
`_C140344.ORF`. Clone the repository to obtain the RAW files and verify the
checksums in the [sample README](../rawfiles/2024GEMINI_AIRCRAFT/README.md).
The RAW images are excluded from the Python wheel and source distribution.

From the repository or extracted source distribution directory, run:

```bash
uv run python detect_meteors_cli.py \
  --config config_examples/aircraft_trail_sample.yaml \
  --no-roi --no-resume --debug-image --profile
```

This sample configuration uses the detection parameters obtained by NPF-based
auto-estimation and wider endpoint/angle matching tolerances. It keeps separate
output, debug, and progress paths so it does not reuse the normal `progress.json`.
On a single-core machine, add `--workers 1` to override the sample's two workers.

### Read the Results

After the run completes, inspect the `aircraft` blocks in the progress file:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("candidates/aircraft_v1.6.10/tuned/progress.json")
data = json.loads(path.read_text(encoding="utf-8"))
for item in sorted(data["detected_details"], key=lambda item: item["frame_index"]):
    aircraft = item.get("aircraft", {})
    evidence = aircraft.get("evidence", {})
    print(item["filename"], aircraft.get("likelihood"),
          aircraft.get("track_id"), evidence.get("track_frames"))
PY
```

Review these values alongside the original RAW and debug masks. A higher
likelihood describes the selected line's continuity, not whether a meteor is
absent from the image. In this sample, `_C140344.ORF` has both a meteor and a high
aircraft likelihood. Filtering files by a likelihood threshold would lose that
meteor. The absolute frame difference can also retain a meteor's disappearance
in the following pair, so a track need not represent a persistent aircraft.

See [the sample validation results](aircraft_sample_validation.md) for both the
built-in defaults and the sample configuration.

## Processing and Persistence

```text
Detector -> DetectionResult -> on_detection_complete (aircraft hook: unchanged)
  -> Output handler and ProgressManager.record_result
  -> SortedDetection records, sorted within each result batch
  -> on_batch_results_sorted (aircraft hook: unchanged)
  -> Collect records from the current invocation
  -> OutputHandler.on_pipeline_complete
  -> Sort collected records by frame_index
  -> AircraftTrailHook.on_all_detections_sorted
       attach SortedDetection.extras["aircraft"]
  -> ProgressManager.update_extras_from_sorted_detections
       update existing candidate entries in progress.json
```

Both sorted hooks run in the main process. Parallel worker completion order
therefore does not affect final tracking order. `SortedDetection` excludes image
arrays and retains frame indices, line segments, candidate flags, scores, and
extras. Successful non-candidate records participate in tracking too.

The built-in progress manager persists only the `aircraft` namespace from extras,
under candidate entries in `detected_details`. Non-candidates have no detail
entry, and other extras require custom persistence. Candidate files have already
been saved before aircraft analysis runs.

## Metadata

The hook attaches a dictionary of this shape to `SortedDetection.extras`:

```json
{
  "aircraft": {
    "likelihood": 0.85,
    "track_id": "air-0001",
    "evidence": {
      "track_frames": 3,
      "angle_diff_deg": 1.2,
      "start_distance_px": 4.5,
      "end_distance_px": 3.2,
      "speed_consistency": 0.85
    }
  }
}
```

The numbers above illustrate the payload shape, not a computed score. In
`progress.json`, the `aircraft` dictionary is added directly to the candidate's
existing detail entry alongside `filename`, `score`, `lines` (line count), `ratio`,
`frame_index`, and `prev_frame_index`.

Records without lines receive likelihood `0.0`, track ID `null`, track frame
count `0`, and `null` geometric/speed evidence. Tracking errors for individual
records are logged and receive the same default metadata.

## Tracking and Scoring

1. Select the longest line segment in each frame and derive consistently ordered
   endpoints, midpoint, and normalized angle.
2. Expire tracks whose last observation is more than `track_ttl_frames` earlier.
3. Match against endpoint distance, angle, and available speed-variance limits.
   Choose the qualifying track with the smallest sum of endpoint distances and
   angle difference, or create a new track.
4. Update track state and calculate likelihood using continuity (50%), angle
   consistency (20%), endpoint consistency (20%), and speed consistency (10%).

`min_track_frames` determines when the continuity contribution reaches full
weight. It is not a minimum observation count for emitting metadata. Tracks can
bridge frame gaps within the TTL; strict consecutive-frame continuity is not
required. Earlier records retain the score/evidence calculated at their own
observation; later observations do not retroactively revise them.

## Configuration

`AircraftTrailConfig` is a dataclass with these defaults:

| Field | Default | Current use |
|-------|---------|-------------|
| `min_track_frames` | `3` | Observation count for full continuity weight |
| `max_start_distance_px` | `6.0` | Start endpoint matching limit |
| `max_end_distance_px` | `6.0` | End endpoint matching limit |
| `max_angle_diff_deg` | `3.0` | Angle matching limit |
| `max_speed_variance` | `0.3` | Normalized speed-variance limit |
| `likelihood_threshold` | `0.7` | Currently unused; does not filter or change results |
| `track_ttl_frames` | `5` | Maximum inactive frame gap before expiry |

Pixel distances use detector image coordinates, which are the binned RAW image
coordinates with the built-in loader.

## Run Boundaries and Limitations

- Track state and track IDs reset at the start of each final analysis pass.
- On resume, only newly processed records are analyzed. Previous line data and
  tracks are not reconstructed from `progress.json`. Use a fresh complete run
  (`--no-resume`) when analysis needs the entire sequence.
- Ctrl-C saves normal processing progress and returns before final aircraft
  analysis, so newly processed candidates may lack aircraft metadata.
- Failed frame pairs without valid frame indices are omitted from sorted records.
- Only one line (the longest) is used per frame; simultaneous objects are not
  independently tracked within a frame.
- Aircraft analysis catches and logs its own errors. `hook_error_mode` controls
  exceptions escaping hooks generally; it does not force internally caught
  aircraft analysis errors to propagate.

## Existing Tests

`tests/test_aircraft_trail_hook_v1x.py` contains three tests covering linked
metadata across frames, default metadata for missing lines, and persistence via
`ProgressManager.record_result`. See `PLUGIN_AUTHOR_GUIDE.md` for the sorted hook
contracts and broader pipeline lifecycle.
