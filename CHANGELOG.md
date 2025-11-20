# Changelog

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
