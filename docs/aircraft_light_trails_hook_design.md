# Aircraft Light Trails Hook Design

## Purpose
Implement an **aircraft light trail likelihood** feature as a **pipeline hook** that:
- Does **not** change meteor detection outcomes.
- Adds **auxiliary metadata** to `progress.json` (via `detected_details`).
- Is enabled/disabled via `PipelineConfig.hooks`.

This document describes the **design** and **implementation plan** before code changes.

## Scope
- **Primary hook:** `on_detection_complete` (pipeline hook).
- **Data carrier:** `DetectionResult.extras` for per-frame metadata.
- **Persistence:** `ProgressManager.record_result` writes `detected_details` to `progress.json`.
- **No changes** to candidate decision logic (`is_candidate`, `score`, etc.).

## Background constraints (from existing code)
- `DetectionResult` includes `extras` for auxiliary data; this is the safest extension point.
- `progress.json` is written via `ProgressManager.record_result` with `detected_details` entries.
- Pipeline hooks are resolved via `PipelineConfig.hooks` and executed during detection.
- Hooks must be discoverable for multiprocessing (entry points or `~/.detect_meteors/hook_plugins`).

## Proposed Data Flow
```
Detector -> DetectionResult
  -> Hook: on_detection_complete
        - analyze current/previous frame line data
        - compute aircraft likelihood signals
        - attach to DetectionResult.extras
  -> Pipeline continues unchanged
  -> ProgressManager.record_result(...)
        - read extras (if present)
        - persist to detected_details in progress.json
```

## Extension Points
### 1) DetectionResult.extras
Add a structured payload under a namespaced key (example):
```
extras["aircraft"] = {
  "likelihood": 0.0..1.0,
  "track_id": "air-<hash>",
  "evidence": {
    "track_frames": 3,
    "angle_diff_deg": 1.2,
    "start_distance_px": 4.5,
    "end_distance_px": 3.2,
    "speed_consistency": 0.85
  }
}
```

### 2) progress.json detected_details
`ProgressManager.record_result` currently persists:
```
{
  "filename": ...,
  "score": ...,
  "lines": ...,
  "ratio": ...,
  "frame_index": ...,
  "prev_frame_index": ...
}
```
Extend with:
```
"aircraft": {
  "likelihood": 0.0..1.0,
  "track_id": "air-<hash>",
  "evidence": { ... }
}
```
This is purely additive and should not affect existing consumers.

## Hook Selection Rationale
### Why `on_detection_complete`
- It receives `DetectionResult` **and** `DetectionContext`.
- It runs after detector output is normalized and before output handlers.
- It allows us to **attach metadata** without modifying detection outcomes.

### Why not `on_output_saved`
- That hook does not mutate results and is intended for notifications.

## Aircraft Heuristic (Design)
### Core criteria
1. **Continuity:** a track persists across **>= 3 consecutive frames**.
2. **Geometric stability:** small deltas in **start/end positions** and **angle**.
3. (Optional) **Speed stability:** consistent per-frame movement distance.

### Inputs required
- Line segments from `DetectionResult.lines`.
- Frame index (from `DetectionContext.metadata`).
- Per-frame line endpoints (start/end) and angle.

### Tracking concept
Maintain a lightweight **track buffer** keyed by recent frames to link candidate lines across frames:
- Track state: last endpoints, last angle, last frame index, derived speed.
- When a new frame arrives, match to existing tracks using thresholds.
- Update track evidence and compute likelihood.

## ConfigType Design (Hook)
Use a dataclass config for ease of use and default construction.

Recommended config fields:
- `min_track_frames: int = 3`
- `max_start_distance_px: float = 6.0`
- `max_end_distance_px: float = 6.0`
- `max_angle_diff_deg: float = 3.0`
- `max_speed_variance: float = 0.3` (normalized)
- `likelihood_threshold: float = 0.7`
- `track_ttl_frames: int = 5` (cleanup for stale tracks)

## Multi-processing considerations
- Hook must be discoverable in worker processes:
  - `detect_meteors.hook` entry point or
  - `~/.detect_meteors/hook_plugins`.
- Avoid global mutable state unless guarded for concurrency.
- If using in-memory track state, ensure it is **per-process** and does not attempt shared memory.

## Persistence Strategy
### Option A: Direct write in record_result
- Modify `ProgressManager.record_result` to read `DetectionResult.extras` when available.
- Map `extras["aircraft"]` into `detected_details`.

### Option B: Separate hook-based writer
- A dedicated hook could write to progress file, but this adds coordination complexity.
- **Recommendation:** Option A (centralized, consistent with existing writes).

## Failure/Edge Cases
- No line data: store `aircraft.likelihood = 0.0` or omit the field.
- Missing frame indices: skip track linkage for that entry.
- Parallel processing: ensure track data is per worker, or restrict logic to within frame pairs.

## Implementation Checklist (Engineering Steps)
- [ ] Define **hook class** `AircraftTrailHook` (DataclassHook).
- [ ] Implement `ConfigType` dataclass with defaults.
- [ ] Add tracking logic in `on_detection_complete`:
  - [ ] Extract line segments.
  - [ ] Compute line angle and endpoints.
  - [ ] Match/update tracks.
  - [ ] Compute likelihood + evidence.
  - [ ] Attach to `DetectionResult.extras` under `"aircraft"`.
- [ ] Extend `ProgressManager.record_result`:
  - [ ] Accept optional `DetectionResult` or `extras` input.
  - [ ] Write `aircraft` block into `detected_details` when present.
  - [ ] Keep backward compatibility with existing progress files.
- [ ] Ensure hook discovery setup:
  - [ ] Add entry point or document plugin placement.
  - [ ] Update configuration examples with `PipelineConfig.hooks`.
- [ ] Add tests:
  - [ ] Hook attaches extras to result.
  - [ ] Progress JSON includes `aircraft` field.
  - [ ] No regression in detection counts.

## Acceptance Criteria
- `progress.json` has `aircraft` metadata for frames that meet criteria.
- `is_candidate` and `score` are unchanged by the hook.
- Hook can be enabled/disabled via `PipelineConfig.hooks`.
- Works in parallel processing (no shared-state errors).
