# Changelog

## 1.2.1 - 2025.11.22
- **Improved auto-parameter estimation**: Revised `diff_threshold` auto-estimation algorithm from 3-sigma rule to percentile-based approach for better handling of peaked night sky brightness distributions.
- **Real-world validation**: Based on actual RAW image testing, reduced typical estimated thresholds from 25 to 15, significantly improving meteor detection sensitivity.
- **Enhanced statistics output**: Added 98th and 99th percentile reporting, plus detailed breakdown of three estimation methods (98th percentile, mean + 1.5σ, median × 3).
- **Optimized threshold range**: Adjusted clamp range from 4-25 to 3-18 based on real-world feedback for more sensitive meteor detection.

## 1.2.0 - 2025.11.22 (Not release)
- **NEW: Auto-parameter estimation**: Added `--auto-params` flag to automatically estimate optimal `diff_threshold` from sample images using ROI statistics.
- Implemented 3-sigma rule-based estimation from first 5 sample images for initial auto-tuning capability.
- Added detailed statistical output (mean, std dev, median, percentiles) during auto-estimation.
- Preserves manual parameter specification priority when both `--auto-params` and explicit values are provided.

## 1.1.0 - 2025.11.22
- Added resumable processing with a JSON progress file that keeps processed/detected counts and resumes only when parameters match, bringing first-class support for both progress tracking and safe resumes.
- Added `--progress-file`, `--no-resume`, and `--remove-progress` flags to manage progress tracking or clear saved state before runs.
- Allow safe interruption with `Ctrl-C` without losing tracked progress so long runs can be paused and resumed later.

## 1.0.3 - 2025.11.21
- Added optional RAW file validation with progress reporting to skip corrupted inputs before processing.
- Added automatic batch-size tuning based on available memory (requires `psutil`) and a profiling flag for timing insights.
- Defaulted the input folder to `rawfiles` and expanded the CLI option reference in the README for easier configuration.

## 1.0.2 - 2025.11.20
- Added help descriptions with defaults for Hough transform parameters and minimum line score.

## 1.0.1 - 2025.11.20
- Initial release.
