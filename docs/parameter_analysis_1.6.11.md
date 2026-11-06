# v1.6.11: Missed-Meteor Analysis and Parameter Tuning

[日本語版](parameter_analysis_1.6.11_ja.md)

Status: design and foundation refactoring. The CLI for selecting an image, tuning
recommendations, and saving adjusted settings are not implemented yet.
This foundation change does not update the release number; the runtime version
remains 1.6.10.

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

## Next Stage: Inputs and Reproducibility

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

## Next Stage: Generating Tuning Proposals

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

## Next Stage: Saving Reports and Configurations

Finalize CLI option names during implementation. Provide capabilities to analyze
a target image, specify a reference image, read original progress, choose an
analysis output location, and save a tuning proposal as configuration. Use an
execution path separate from normal detection; running analysis alone must not
copy candidate files or update the original progress.

Use a separate, versioned analysis report format containing at least:

- The input pair, application and analysis-format versions, image dimensions, effective settings, and ROI.
- EXIF and user-supplied information, the sources of original values, estimates, and overrides, and whether reproduction succeeded.
- The original decision, rejection reasons, line and contour evidence, and references to masks.
- Changes in each tuning proposal, re-evaluation results, search ranges, and unverified aspects.
- Candidate additions and losses when a comparison sequence is available, and correspondence with visually confirmed regions.

Save proposals as complete configurations that the existing `--config` option
can load. Keep the original configuration and analysis report separate, and
retain loader and detector settings. For items without a storage field in the
current `PipelineConfig`, such as a session ROI, add a save-and-reload mechanism
consistent with existing CLI arguments during implementation. Also verify that
output destinations cannot overwrite input RAW files, original progress, or
the original configuration.

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
