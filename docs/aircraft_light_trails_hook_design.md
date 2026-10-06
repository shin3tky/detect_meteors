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
