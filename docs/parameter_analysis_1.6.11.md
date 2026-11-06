# v1.6.11: Missed-Meteor Analysis and Parameter Tuning

[日本語版](parameter_analysis_1.6.11_ja.md)

Status: the explicit-pair replay API and read-only analysis CLI are reviewed.
Bounded tuning and sequence comparison are implemented and reviewed.
Analysis bundles and selected session configuration export are implemented and
reviewed.
All four implementation steps are included in v1.6.11. See the
[release notes](RELEASE_NOTES_1.6.md#version-1611-2026-11-07) for the released workflow.

## Goal and User Workflow

Starting from an image containing a visually confirmed meteor that was not
selected as a candidate, explain why it was missed and allow the shooting
sequence to be reprocessed with suitable parameters. Implement this as a 1.x
extension that completes the existing frame-difference and Hough approach.

1. The user specifies the visually confirmed image and the original processing conditions.
2. Reproduce the decision using the same input image, preceding frame, ROI, and preprocessing as normal detection.
3. Show the difference mask, line segments, contours, and whether each threshold was passed.
4. Compare several tuning proposals that limit the amount of change, including results before and after adjustment.
5. Save the selected proposal to a separate configuration file and reprocess the entire shooting sequence.
6. Review newly added candidates and known meteors to decide which settings to adopt.

Distinguish a frame becoming a candidate from detecting the meteor within that
frame. When multiple trails are present, check the correspondence between the
detected shapes and the meteor region identified by the user.

## Current Implementation

| Processing | Location | Current role and limitations |
| --- | --- | --- |
| Shooting conditions | `schema.py`, `image_io.py`, `utils.py` | Sensor presets, EXIF, pixel pitch, focal length, and fisheye models |
| Physics-based estimation | NPF-related functions in `utils.py` | Estimate initial values from exposure, star motion, and other factors; do not use ground-truth labels for individual meteors |
| Image-based initial estimation | `estimate_*` in `pipeline.py` | Use difference distributions, star areas, image dimensions, and other image properties |
| Estimation orchestration and display | CLI `_run_auto_params` | Select NPF or existing image-based estimation and handle user-specified values |
| Feature measurement | `HoughDetector.analyze` | Absolute difference, thresholding, ROI, 3×3 morphological opening, Hough transform, and external contour measurements |
| Candidate decision | `HoughAnalysis.evaluate` | Apply area, aspect-ratio, and score conditions to the measurements |
| Normal detection output | `HoughDetector.detect` | Use the shared measurement and decision logic to return the existing `DetectionResult` |
| Auxiliary aircraft information | `hooks/aircraft_trail.py` | Track line segments over time; do not establish meteor ground truth or automatically exclude candidates |
| Configuration and progress | `config_io.py`, `outputs/progress.py` | Record execution conditions and candidate details; progress does not retain non-candidate measurements or Hough line endpoints |

Treat initial estimation, tuning based on visually confirmed images, and normal
detection as separate responsibilities. Changing the initial estimation alone
does not reproduce a previous run.

## Foundation Implemented in This Change

Introduce the following types in `meteor_core/detectors/hough_analysis.py`:

- `ContourMeasurement`: contour area and rotated rectangle before candidate filtering.
- `CandidateThresholds`: area, aspect-ratio, and score thresholds that can be changed without repeating measurements.
- `CandidateDecision`: passing contours, the maximum aspect ratio among area-eligible contours, and rejection reasons.
- `HoughAnalysis`: mask, image-wide score, line segments, contour measurements, and original candidate thresholds.

`HoughDetector.detect()` combines `analyze()` → `evaluate()` → debug rendering
when needed. Analysis uses the same measurements as normal detection rather than
a separate analysis-only algorithm. Also consolidate the Hough line-scoring
calculation previously duplicated between the method and compatibility function.

Preserve the current candidate conditions:

```text
For at least one individual contour:
    area > min_area
    aspect_ratio > min_aspect_ratio
And for the image as a whole:
    line_score >= min_line_score
```

The aspect ratio is `max(width, height) / max(min(width, height), 1)`.
Exclude contours whose width or height is zero. As before, compute the returned
maximum aspect ratio only from contours that pass the area condition. Do not
combine the area of one contour with the aspect ratio of another to pass the rule.

Rejection reason codes are `no_contours`, `min_area`, `invalid_geometry`,
`min_aspect_ratio`, and `min_line_score`. These identify conditions that blocked
the decision; they do not classify an image as having no meteor. In particular,
`no_contours` alone cannot establish whether a meteor was outside the ROI or
removed by preprocessing. For contours, report the first stage that could not
be passed, and also report an insufficient image-wide score when applicable.

`evaluate()` does not modify the original thresholds, images, or measurements.
`render_debug()` can be called explicitly for non-candidates, while normal
detection continues to render candidates only. Preserve normal-result `extras`,
existing metrics keys, the progress format, plugin contracts, and defaults.
This API is specific to the built-in Hough detector; other detectors are not
required to implement it.

Example of using the foundation API with an existing `DetectionContext`:

```python
from dataclasses import replace

from meteor_core.detectors.hough_default import HoughDetector

analysis = HoughDetector().analyze(context)
original = analysis.evaluate()
trial = analysis.evaluate(replace(analysis.thresholds, min_aspect_ratio=2.0))
debug_image = analysis.render_debug(original)
```

The `2.0` in this example is a trial value supplied by the caller, not a
recommendation or a new default. The example does not select a tuning proposal
or save files.

## Implementation Step 1: Explicit-Pair Replay API

Add a read-only API in `meteor_core/analysis.py`. This step covers input
resolution and replay, independently of the CLI, within the roadmap's original-
condition reproduction milestone. Complete testing and review before connecting
the CLI.

- `load_reproduction_settings()`: read parameters and ROI from original progress, retaining each value's source, explicit changes, and gaps in the recorded conditions.
- `reproduce_hough_pair()`: load the explicitly selected target and reference with the existing loader and use the shared built-in Hough API for measurements and decisions.
- `HoughReproduction`: retain the input pair, effective settings, loader information, image dimensions, metadata for both images, measurements, and decision in memory.

Apply parameters in this order: defaults → caller-supplied `params` → recorded
`params` → recorded `processing_params` → `parameter_overrides`. Supplied
`params` also fill missing values; they do not silently replace recorded values.
Accept progress with missing values, but identify fallbacks in
`parameter_sources` and `limitations`. Reject a pipeline configuration passed as
progress, invalid thresholds, and invalid ROI data. Do not silently ignore
parameters outside the built-in Hough detector's seven parameters.

Use the recorded ROI unless explicitly overridden with `roi_polygon` or
`use_full_image=True`; reject specifying both. If no ROI is recorded or supplied,
use the full image and record that assumption. An ROI that does not overlap the
image at all is an input error. ROI coordinates refer to the image returned by
the loader.

```python
from meteor_core.analysis import load_reproduction_settings, reproduce_hough_pair

settings = load_reproduction_settings(progress_path="previous_run/progress.json")
replay = reproduce_hough_pair(
    "raw/frame002.ORF",
    "raw/frame001.ORF",
    settings=settings,
)
print(replay.status)
if replay.decision is not None:
    print(replay.decision.is_candidate, replay.decision.rejection_reasons)
```

Omitting the reference returns `status="reference_missing"` with target-image
information only. Measurements and decision are `None`; do not substitute a
zero image or another nearby file. Explicitly reject identical files (including
hardlinks), loading failures, dimension or dtype mismatches, and non-grayscale
images.

Specify either a loader instance or a loader name/configuration; reject
combining these approaches. Reuse normal processing's loader resolution and
input normalization, and do not fall back for an unknown loader. In the shared
unknown-loader error path, also replace the nonexistent `list_loaders()` call
with `list_available()`.

This API replays the detector itself. Original progress alone does not reproduce
loader settings or hooks, and `limitations` records that restriction. It does
not run hooks or auto-params, infer references, copy candidates, update progress,
save reports, or automatically generate tuning proposals.

Validation for this step:

- Added 14 API tests; all 595 tests, including the existing suite, passed.
- Across 11 adjacent RAW pairs and both full-image and polygon ROIs, all 22 comparisons matched normal pipeline decisions, scores, line segments, aspect ratios, and candidate debug-image pixels. Original progress was unchanged.
- Loaded recorded settings from the user's progress for `P8136507.ORF` and verified `reference_missing` without its preceding image. This does not count as reproducing the original difference-based decision.
- Ruff lint/format checks and ty type checking passed.

## Implementation Step 2: Analysis CLI

Connect the reviewed API to a separate read-only CLI path:

```bash
python detect_meteors_cli.py \
  --analyze-image rawfiles/frame002.ORF \
  --reference-image rawfiles/frame001.ORF \
  --analysis-progress previous_run/progress.json
```

Alternatively, replace `--reference-image` with `--analysis-sequence DIR` to
select the preceding RAW in an explicitly selected original sequence, using
normal file collection and filename ordering. Do not infer a sequence from the
target's parent, the default target folder, or a configuration's target folder.
The reference-selection methods are mutually exclusive. The first frame has
no preceding frame. Reject a target outside the selected sequence.

Use `--config` for loader/detector settings and fallback parameters, and
`--analysis-progress` for original recorded parameters and ROI. Reject swapping
these input formats. Recorded values override config parameters; explicit
detection parameter flags override recorded values, with before/after changes
shown separately. Preserve explicit-override tracking for both `--key value`
and `--key=value`, including argparse's unambiguous abbreviations. `--roi` and
`--no-roi` explicitly override ROI and cannot be combined.

Display the pair and reference-selection method, original progress path, image
dimensions, target metadata, loader identity, effective parameter sources, ROI,
explicit changes, limitations, decision, rejection reasons, and measurement
counts. `--verbose` lists contours and line endpoints. Provide English and
Japanese summaries and Bash/Zsh completion for all four new options.

Return exit status 0 for a measured pair, including rejected frames, 1 for
configuration/input/processing errors, and 2 for invalid option combinations or
`reference_missing`. Missing references produce no decision. Check incompatible
modes before any progress deletion or parameter estimation can run. Reject
`--auto-params`, `--remove-progress`, `--show-exif`, `--show-npf`,
`--list-sensor-types`, `--debug-image`, `--progress-file`, and explicit `--target`
in analysis mode. Other output/performance settings do not invoke the normal
pipeline. Explicitly requested error diagnostics through `--save-diagnostic`
retain their existing behavior.

Reject other detectors and configured hooks. Users may explicitly select
`--hooks ""` for a detector-only replay. Do not create output directories, copy
candidates, update progress, generate tuning proposals, or save analysis reports
in this step. See the [option reference](COMMAND_OPTIONS.md) for details.

Validation for this step:

- Added 19 CLI tests; all 614 tests, including the existing suite, passed.
- On the real `_C140338.ORF`/`_C140337.ORF` pair, explicit reference selection and filename-order sequence selection returned exit status 0 with identical decisions and measurement summaries.
- With the user's RAW and recorded progress, the Japanese CLI returned `reference_missing` and exit status 2. All three subprocess runs left an empty working directory unchanged, and the original progress bytes were preserved.
- Ruff lint/format checks, ty, and Bash/Zsh completion syntax checks passed. English/Japanese analysis message keys match.

## Input and Reproducibility Design

An analysis job takes the target image, preceding image, effective parameters,
ROI, loader configuration, and detector configuration. It may also take a region
containing the visually confirmed meteor. When locating the preceding image by
filename order, use the same file collection and ordering as normal detection,
and record the selected pair in the report. An explicitly specified reference
image takes precedence. Do not guess the preceding image when the shooting
sequence cannot be identified.

When the original `progress.json` is available, use its recorded effective
parameters and ROI as the reproduction baseline. Record explicit overrides
separately from the original values and display changes to the reproduction
conditions. Allow older progress files to omit fields. Treat normal
configuration files and progress files as distinct input formats.

For a single image without its preceding frame, EXIF, a preview, and background
information can still be provided. However, mark reproduction of difference
detection as having an insufficient reference image. Do not substitute an
invented background or a zero image and claim that tuning succeeded. Explicitly
handle identical target and reference images, dimension mismatches, loading
failures, and mismatches between the target region and ROI.

When reproducing the original processing, do not implicitly reselect the ROI
or rerun `--auto-params`. Distinguish results of runs with hooks from the
detector's own results. Do not silently replace an unsupported loader or detector
with the built-in Hough implementation.

## Implementation Step 3: Bounded Tuning and Sequence Comparison

Add `meteor_core/tuning.py` with `TuningOptions`, `TuningProposal`, `TuningRun`,
and `SequenceComparison`. `tune_reproduction()` generates proposals;
`compare_tuning_proposals()` compares the returned proposals with the current
effective baseline on an explicit list of `(current, previous)` pairs.

Retain the target pair's loaded image/ROI snapshot and original loader in
`HoughReproduction`, excluding these internal resources from its representation.
`remeasure()` changes runtime parameters while preserving that snapshot;
`reproduce_pair()` uses the same loader and settings for another pair. The result
also records the loaded dtype. Images are retained for the lifetime of a replay;
comparison processes one pair at a time rather than retaining the whole sequence.

Default search limits are a 0.5 maximum fractional relaxation per candidate
gate, eight target measurements, 256 target decisions, five returned proposals,
100 comparison pairs, and 600 comparison measurements. Original target evidence
counts toward the measurement and decision budgets. These configurable limits
are computational/exploration bounds, not physical camera calibrations.

For each eligible contour with Hough evidence, derive the least gate changes
needed to pass the existing rule. An integer area threshold becomes at most
`ceil(area) - 1`, aspect ratio at most the adjacent representable float below the
measurement, and score at most the measured global score. Preserve already
passing gate values. Discard changes outside the allowed relaxation range,
deduplicate complete parameter sets, and evaluate candidates in normalized
change-cost order. Report omitted proposals when the output limit is reached.

Optional `measurement_grid` lists are limited to the four difference/Hough
parameters. Include original values, order other values by distance, and visit
the Cartesian product only up to the work limits. Recompute masks, lines, and
contours for changed measurement settings; reuse evidence only for gate changes.
Do not claim to find the global optimum of an unfinished or bounded search.

An optional rectangle `(x0, y0, x1, y1)` uses loaded-image coordinates with
exclusive end coordinates. Match contour foreground pixels in that region,
using clipped contour bounds to avoid full-image allocation per contour. An
unrelated accepted contour cannot satisfy this target. ROI stays fixed; reject
invalid rectangles, report `target_outside_roi` when there is no ROI overlap,
and evaluate only the original ROI when overlap is partial. Region overlap is
an association with a shape, not proof of meteor identity. Without annotation,
scope remains frame-level. An already passing target, missing reference, or no
feasible proposal are explicit outcomes.

Comparison evaluates the same complete pairs for every returned proposal.
Reuse measurements only within a pair and only for identical difference/Hough
settings; release each pair before proceeding. Count actual measurement runs,
and stop before a pair that cannot fit the remaining budget. Record total and
evaluated pairs, completion, and stop reason. Reject duplicate targets,
resolution/dtype mismatches, and input errors rather than counting failures as
negative detections.

List added/lost baseline candidate frames. Explicit known-positive labels are
classified as candidate, rejected, or unchecked for each proposal; do not infer
ground truth from progress candidates. A label outside the evaluated pairs is
unchecked. Rank fewer rejected known frames first, then smaller change cost,
then fewer additional candidates. Comparison is limited to the returned
proposals. Missing comparison or known labels remains unverified, not a clean
accuracy result. These are frame-level results even for labeled meteor images.

CLI options are `--tune-params`, `--tuning-config`, `--analysis-region`,
`--comparison-sequence`, and repeatable `--known-meteor`. All modifiers require
tuning mode; known labels also require comparison. Configuration formats are
kept separate, with invalid fields and input-role swaps rejected. English and
Japanese output and shell completion are provided. See
[`parameter_tuning.yaml`](../config_examples/parameter_tuning.yaml) for defaults
and the [option reference](COMMAND_OPTIONS.md#bounded-parameter-tuning-v1611)
for usage. No proposal is applied and no report/configuration is saved.

Validation for this step:

- Added 16 core tests and eight CLI tests; all 638 tests passed. Coverage includes exact threshold boundaries, target-region disambiguation, work limits, snapshot reuse, known-frame coverage, and input failures.
- With the real `_C140338.ORF`/`_C140337.ORF` pair and an intentionally strict ratio threshold of 20, derived a threshold of `13.29749164836932` from evidence and compared all 11 pairs in the sample sequence. This is a controlled frame-recovery check, not validation of the unavailable user-reported pair.
- A two-pair comparison cap reported 2/11 pairs and left a later known meteor unchecked. The user's RAW without its reference returned no proposals. All three CLI subprocesses left an empty working directory unchanged and preserved the original user progress.
- An explicit RAW difference-threshold grid stopped at two of three planned measurements; both returned proposals matched fresh file loads and exact candidate decisions.
- Ruff lint/format checks, ty, shell completion syntax, bilingual messages/examples, sample defaults, and local documentation links passed verification.

## Tuning Proposal Design

Tune parameters per shooting session. Sky brightness, clouds, ground lighting,
star motion, effective resolution, saturation, and noise vary alongside the
camera and lens, so equipment names alone cannot determine universally optimal
thresholds. A shooting location may be recorded as optional session metadata,
but use statistics from the actual images to assess background effects as well.

Separate tuning into stages:

1. Keep measurements fixed and re-evaluate area, aspect-ratio, and score thresholds. Respect the exact comparison boundaries so that rounding does not cause a proposed adjustment to fail on rerun.
2. When needed, search bounded ranges of difference thresholds and Hough settings, repeating measurements for each change. Also discard previous measurements when the ROI, preprocessing, or resolution changes.
3. Check evidence corresponding to the target meteor region, and compare proposals by the amount of parameter change, additional candidates, and retention of known meteors. Without a specified region, report recovery at the frame level rather than claiming to have identified the meteor itself.

Provide configurable search ranges and evaluation limits, and record the number
of measurement runs and the reason the search ended. Valid outcomes include no
proposal, failure to reproduce, or a need for more data when evidence is
insufficient. A single positive image cannot establish the false-positive rate.
When comparison images are provided, process them before and after adjustment
under the same effective conditions and list new candidates and lost known
candidates.

Do not interpret the sum of line lengths as a meteor probability or the strength
of a single trail. Add any future strong-score recovery rule only after
validating aspects such as the association between lines and contours. This
foundation change does not introduce a hard-coded `2.0` or score tailored to the
reported data, automatic aircraft exclusion, or propagation of aircraft
classification based solely on temporal adjacency.

## Implementation Step 4: Analysis Bundles and Session Configuration

`meteor_core/analysis_reports.py` provides `save_analysis_bundle()`. Export is
explicit: `--analysis-output DIR` creates a new bundle, and optional
`--analysis-proposal baseline|N` saves the effective baseline or a one-based
returned proposal as `session.yaml`. No selection means report/evidence only.
Numeric indices require tuning; every selection requires an output directory.

```bash
python detect_meteors_cli.py \
  --analyze-image rawfiles/session/frame002.ORF \
  --reference-image rawfiles/session/frame001.ORF \
  --analysis-progress previous_run/progress.json \
  --tune-params \
  --analysis-output analysis/review-001 \
  --analysis-proposal 1

python detect_meteors_cli.py --config analysis/review-001/session.yaml
```

The second command is a separate normal run. Export does not apply settings,
copy candidates, or update original progress. Indices refer to the current
returned proposal order after comparison. Reports remain valid exploratory
evidence even if no proposal is feasible or a reference is missing; a selected
reusable configuration requires a measured pair.

The JSON manifest uses format `detect_meteors.analysis`, schema version 1.
It records application/Python/NumPy/OpenCV versions, detector and loader identity,
the effective loader configuration, image dimensions/dtype, metadata, original
progress path, exact effective/baseline parameters, value sources, ROI, changes,
and limitations. `pipeline_config` reflects effective analysis parameters/ROI;
fingerprinted source files retain the original configuration identities.
Unsupported metadata values are converted to text with serialization notes.
Live image contexts and loader objects are not serialized.

`measurements` maps identifiers to difference/Hough settings, scores, full line
coordinates, contour measurements, mask pixel counts, and lossless mask PNGs.
Baseline/proposal decisions reference those identifiers and debug PNGs, including
rejected-frame evidence. Gate-only proposals share masks; changed measurement
settings use fresh evidence from the original snapshot. Tuning bounds, normalized
grids, work counters, termination/truncation, region associations, and comparison
coverage/results are retained. Explicit selection and `session.yaml` are linked
from the manifest. Asset entries contain file sizes and SHA-256 hashes; the
manifest does not hash or list itself.

Images, progress, and supplied configuration sources receive read-time file
identity/change stamps. Export verifies those stamps and hashes primary sources
with SHA-256. Comparison records retain source stamps without hashing the whole
sequence again. Source changes during load, repeated comparison, hashing, or
export are errors. The original target snapshot must agree with comparison
sources. These checks detect changes; they do not lock source files.

Create the destination exclusively and never replace existing paths, including
empty directories or symlinks. Write the manifest last. On failure, remove only
files created by that invocation, checking their identities, and remove only
empty new directories. This leaves original RAWs, progress, settings, and
unowned files untouched. No force-overwrite mode is provided for bundles.

Selected configurations preserve the built-in Hough rule and exact thresholds,
the actual registered loader and its value/type-preserving configuration, and
effective ROI. Reject unavailable loader state, coercion that changes values or
types, active hooks, unsupported detector settings, and custom output handlers.
The latter cannot have their output paths redirected generically. File handler
outputs and progress use new absolute paths under `DIR/run`, with overwrite
disabled; normal processing creates these paths later. Use one worker and
sequential processing for portable replay. Preserve other supported settings.

Add optional `PipelineConfig.roi_polygon` and `enable_roi_selection` fields.
Existing configurations default to interactive ROI selection; saved sessions
disable it. The normal CLI and pipeline honor configured ROI, with explicit
`--roi`/`--no-roi` overrides. In analysis, recorded progress ROI precedes the
configuration fallback, and explicit flags precede both. Accept the existing
progress marker `full_image` as full-image ROI. The API's explicit empty polygon
argument selects full image rather than falling back to configured ROI.

Session targets use the explicit analysis sequence, otherwise an explicitly
configured target, otherwise the analyzed image's parent. Top-level run paths
are absolute; plugin-specific relative paths retain their existing semantics.
The report records the working directory. See the
[option reference](COMMAND_OPTIONS.md#analysis-bundles-and-session-configuration-v1611)
for output contents, selection, and replay examples.

Validation for this step:

- Added 20 export/replay tests; all 658 tests passed. Covered PNG/JSON fidelity, hashes, exact float thresholds, loader/ROI round trips, normal pipeline/CLI replay without GUI, legacy progress, changed sources or parameters, existing-path refusal, and failed-write cleanup including preservation of unowned files.
- Exported a selected proposal from the real sample and processed all 11 pairs through normal CLI using the saved configuration. The target's score, aspect ratio, and line count matched the selected report exactly.
- Original config/progress/candidate/debug markers were unchanged; repeated export was refused. The user's RAW without a reference saved a truthful incomplete report with exit status 2.
- Ruff lint/format checks, ty, shell completion syntax, and bilingual documentation were verified. The implementation is reviewed; v1.6.11 release preparation is documented in the release notes.

## Acceptance Criteria for 1.6.11

- Normal detection retains the existing candidate set, scores, line segments, aspect ratios, and debug rendering.
- Preserve strict `>` for area and aspect ratio, `>=` for score, and the handling of ROI and zero dimensions.
- Reproduce a visually confirmed missed detection under the original conditions, showing blocking conditions and evidence before and after adjustment.
- Do not reuse previous evidence after a settings change that requires new measurements.
- Report missing information and distinguish becoming a candidate from successful meteor detection.
- Rerunning with a saved configuration produces results consistent with the proposal's evaluation.
- Continue to support existing progress, configurations, and detector plugins without adding side effects to normal processing.

## Foundation Validation Results

- Added six regression tests; all 581 tests, including the existing suite, passed.
- Python code passed Ruff lint and formatting checks, and ty type checking.
- Compared the refactored detector with the previous implementation on 11 adjacent pairs from `2024GEMINI_AIRCRAFT`. With difference threshold 9, minimum area 3, and a full-image ROI fixed, tested four combinations of aspect-ratio thresholds 2.0/3.0 and score thresholds 30.0/200.0. All 44 comparisons matched in candidate decisions, scores, line segments, maximum aspect ratios, extras, non-timing metrics, and debug-image pixels.

The user-reported `P8136507.ORF` is not yet counted as an acceptance case with a
reproduced original difference-based decision: its preceding frame,
`P8136506.ORF`, has not been provided.
