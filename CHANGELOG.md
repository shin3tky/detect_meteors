# Changelog

## 1.0.3 - 2025.11.21
- Added optional RAW file validation with progress reporting to skip corrupted inputs before processing.
- Added automatic batch-size tuning based on available memory (requires `psutil`) and a profiling flag for timing insights.
- Defaulted the input folder to `rawfiles` and expanded the CLI option reference in the README for easier configuration.

## 1.0.2 - 2025.11.20
- Added help descriptions with defaults for Hough transform parameters and minimum line score.

## 1.0.1 - 2025.11.20
- Initial release.
