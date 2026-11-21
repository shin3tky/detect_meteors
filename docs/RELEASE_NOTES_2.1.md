# Version 2.1 Release Notes

[日本語版](RELEASE_NOTES_2.1_ja.md)

<a id="version-211"></a>

## Version 2.1.1 (2026-11-21) 📺 World Television Day

2.1.1 adds a way to preprocess a frame pair immediately before detection while
using the existing detector and output handler. Conventional absolute-difference
Hough and file output remain the defaults. Stellar registration and FRT retry
are enabled through configuration.

### Highlights

- **Frame-pair preprocessing hook.** The pipeline calls
  `on_before_detection(context)` in configuration order after loading both
  frames and before invoking the detector. The built-in `star_registration`
  hook illustrates this by rotating and translating the previous image using
  stars in the ROI. [Stellar registration workflow (Japanese)](star_registration_workflow_ja.md)
- **Positive-difference Hough.** Set `detector_config.difference_mode: positive`
  to pass only pixels that brighten from the previous frame to Hough. This
  excludes negative residuals left when a meteor disappears.
  [Difference-mode comparison (Japanese)](hough_registration_thresholds_ja.md#正の差分だけを渡した場合)
- **Experimental FRT retry.** Set `detector_config.frt_retry: {}` to recheck
  pairs rejected by unaligned positive-difference Hough. The retry can accept
  faint trails with brief gaps and adds processing time and memory use per
  worker. [Settings, costs, and limitations (Japanese)](frt_retry_workflow_ja.md)

### Switching configurations and upgrading

Existing YAML and JSON configurations remain usable. The pipeline configuration
format uses `schema_version: 1`. The conventional workflow can resume from its
existing progress file with matching settings. Changing frame-pair hooks, the
difference mode, or FRT retry settings triggers reprocessing when the progress
hash changes. Use separate output and progress paths for comparisons.

With the aircraft sample in place, run from the source root.
[Getting the sample RAWs](INSTALL.md#sample-raw-images)

```bash
uv run detect-meteors --config config_examples/hough_legacy_sample.yaml --debug-image
```

Change the file passed to `--config` to compare workflows with the same
inputs, ROI, and Hough thresholds. Each configuration has separate candidate,
debug, and progress paths.

| Workflow | Configuration |
| --- | --- |
| Conventional absolute-difference Hough | [hough_legacy_sample.yaml](../config_examples/hough_legacy_sample.yaml) |
| Stellar registration before Hough | [hough_registered_sample.yaml](../config_examples/hough_registered_sample.yaml) |
| Unaligned positive-difference Hough | [hough_positive_sample.yaml](../config_examples/hough_positive_sample.yaml) |
| Positive-difference Hough with experimental FRT retry | [hough_frt_retry_sample.yaml](../config_examples/hough_frt_retry_sample.yaml) |

Select `hough_legacy_sample.yaml` to return to conventional detection. Use
`hooks: []` or `--hooks ""` to clear configured hooks. To disable FRT retry
separately, omit `frt_retry` or set it to `null`.
[Switching configurations in the installation guide](INSTALL.md#switch-detection-configurations) · [Switching configurations on the Wiki](https://github.com/shin3tky/detect_meteors/wiki/Home#switch-detection-configurations)

### Validation

- **Conventional workflow compatibility.** The same 121 RAW pairs and fixed
  conventional settings produced identical candidates, lines, scores, and debug
  images on the feature branch and the preceding local `main`. The feature
  branch also resumed from the original progress files. The four sample
  configurations were exercised on 11 aircraft sample pairs.
- **121-pair leave-one-sequence-out evaluation.** Unaligned positive-difference
  Hough matched 3/6 meteors with 0/115 false positives. Adding FRT retry matched
  6/6 meteors with 0/115 false positives. Each held-out sequence was excluded
  from threshold selection. The retry procedure was developed using diagnostics
  from this collection. [Per-sequence results and reproduction (Japanese)](frt_retry_workflow_ja.md#同じ121ペアの系列除外評価)
- **Separate 2025LEO evaluation.** On 95 consecutive pairs with fixed FRT
  thresholds, false positives increased from 2 to 20. The added detections
  showed moving objects across successive exposures. Visual review confirmed
  no meteors in this sequence, so recall could not be measured.
  [Separate-sequence results and measured cost (Japanese)](frt_retry_workflow_ja.md#2025leoでの結果と処理負荷)

### Plugin development

`on_before_detection()` accepts and returns a `DetectionContext`. It runs
in the main process for a sequential run and in workers for parallel runs.
The returned context reaches the detector and `on_detection_complete()`.
The author guide covers image coordinates, the ROI, `hook_error_mode`,
and how to add or remove hooks and plugins.
[Frame-pair hook contract](PLUGIN_AUTHOR_GUIDE.md#frame-pair-hook) · [Configuration and discovery](PLUGIN_AUTHOR_GUIDE.md#enable-disable-plugins)

### Runtime and distribution

The published 2.1.1 wheel and source distribution can both be installed with uv
on Python 3.12, 3.13, and 3.14. Verify the downloads against `SHA256SUMS` before
installing. [GitHub Release 2.1.1](https://github.com/shin3tky/detect_meteors/releases/tag/2.1.1) · [Installation guide](INSTALL.md) · [Verify release downloads](INSTALL.md#verify-release-downloads) · [2.1.1 changelog](CHANGELOG.md#v211)

Runtime dependency updates since 1.6.13 include Pydantic 2.14.4 and PyYAML 6.0.4.
Development updates include Ruff 0.17.6, ty 0.0.107, and coverage 7.17.4.
The source distribution includes the guides and sample configurations. Get
the sample RAWs, visual labels, and checksums from `detect_meteors_assets`.
Use a Git checkout to rerun the experiment scripts.
[Release readiness checklist](INSTALL_DEV.md#release-211-checklist) · [Post-publication checks (step 5)](INSTALL_DEV.md#post-publication-documentation)
