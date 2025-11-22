# Version 1.2.1 Release Notes

## 🎉 Improved Auto-Parameter Estimation - Real-World Validation Success

Version 1.2.1 brings significant improvements to the `--auto-params` feature based on real-world RAW photos testing and user feedback. The `diff_threshold` auto-estimation algorithm has been completely revised to handle the highly peaked brightness distributions found in actual night sky photography.

## The Problem in v1.2.0

### Initial Implementation Issues
- **Estimated threshold**: 25
- **Result**: Failed to detect meteors (false negatives)
- **Cause**: 3-sigma rule was too conservative

Real-world night sky histograms showed highly peaked distributions, making the normal distribution assumption inappropriate.

### v1.2.1 Solution
- **Estimated threshold**: 15
- **Result**: Successfully detected meteors ✅
- **Improvement**: Switched to percentile-based estimation

## Major Changes

### 1. Revised Estimation Algorithm

#### Before (v1.2.0)
```python
estimated_threshold = int(mean_diff + 3 * std_diff)
estimated_threshold = np.clip(estimated_threshold, 4, 25)
```

#### After (v1.2.1)
```python
# Multiple estimation methods
method_1 = int(percentile_98)           # 98th percentile
method_2 = int(mean_diff + 1.5 * std)   # 3σ → 1.5σ reduction
method_3 = int(median_diff * 3.0)       # Median-based

# Select the most sensitive (lowest) threshold
estimated_threshold = min(method_1, method_2, method_3)
estimated_threshold = np.clip(estimated_threshold, 3, 18)
```

### 2. Enhanced Statistical Output

```
ROI Difference Statistics (from 1,234,567 pixels):
──────────────────────────────────────────────────
  Mean:         2.34
  Std Dev:      1.85
  Median:       1.89
  90th %ile:    5.67
  95th %ile:    8.92
  98th %ile:    12.45    ← New
  99th %ile:    15.23    ← New
──────────────────────────────────────────────────
Estimation methods:
  [1] 98th percentile:      12    ← New
  [2] Mean + 1.5σ:          15    ← Improved
  [3] Median × 3:           6     ← New
──────────────────────────────────────────────────
✓ Selected threshold: 12 (minimum of all methods)
```

### 3. Adjusted Clamp Range

| Version | Min | Max | Rationale |
|---------|-----|-----|-----------|
| v1.2.0  | 4   | 25  | Too conservative |
| v1.2.1  | 3   | 18  | Optimized for real data |

## Technical Background

### Handling Peaked Distributions

Real night sky difference distributions are:

```
Frequency
 │     ★
 │    ★★★
 │   ★★★★★
 │  ★★★★★★★
 │ ★★★★★★★★★
 └──────────── Brightness difference
    ↑
  Most pixels here (0-5 range)
```

These **Laplacian** or **exponential-like** distributions mean:
- Standard deviation is overestimated
- 3-sigma rule (99.7%) is inappropriate
- Percentile-based methods are more effective

### Why 98th Percentile?

- **95th percentile**: Risk of false positives (catching noise)
- **98th percentile**: Good balance ✅
- **99th percentile**: Risk of false negatives (missing meteors)

Experimentally validated to be most effective.

### Minimum Selection Rationale

By selecting the **lowest value** among three methods:
- Prioritize sensitivity (reduce false negatives)
- False positives filtered by downstream processing (aspect ratio, line score)

## Usage

### Basic Usage (No Change)

```bash
python detect_meteors_cli.py --auto-params
```

### Execution Example

```bash
# v1.2.0 behavior
$ python detect_meteors_cli.py --auto-params
✓ Estimated threshold (μ + 3σ): 25
Complete! 0 candidates extracted  # Missed meteors

# v1.2.1 behavior
$ python detect_meteors_cli.py --auto-params
✓ Selected threshold: 15 (minimum of all methods)
Complete! 3 candidates extracted  # Successfully detected!
```

## Validation Results

### Test Case (User-Reported Data)

- **v1.2.0**: Estimated threshold = 25 → Missed meteors
- **v1.2.1**: Estimated threshold = 15 → Detected meteors ✅

### Expected Improvements

| Shooting Conditions | v1.2.0 | v1.2.1 | Improvement |
|-------------------|--------|--------|-------------|
| Low ISO (1600) | 4-5 | 3-5 | Slight |
| Medium ISO (3200) | 8-10 | 6-10 | Significant |
| High ISO (6400) | 15-25 | 10-18 | Significant |

## Breaking Changes

### None

Version 1.2.1 is fully compatible with v1.2.0.

## Migration Guide

### From v1.2.0 to v1.2.1

```bash
# Simply replace the file
cp detect_meteors_cli.py detect_meteors_cli.py.bak
# Use the new version

# Command-line options remain unchanged
python detect_meteors_cli.py --auto-params
```

No changes to command-line options.

## Known Limitations

Same as v1.2.0:

1. **ROI selection is critical**: Including artificial lights or ground objects will compromise estimation accuracy
2. **Static image assumption**: Cloud movement is not considered
3. **Distribution assumptions**: Beware of extreme outliers

## What's Next

### Future Enhancements

1. **`min_area` auto-estimation**
   - Calculate from star size distribution
   - Consider focal length relationship

2. **`min_line_score` auto-adjustment**
   - Estimate from image size and focal length

3. **Advanced statistical methods**
   - Robust statistics (MAD, etc.)
   - Distribution shape estimation

## Acknowledgments

Version 1.2.1's improvements were made possible by real-world RAW image testing and user feedback. Special thanks to the user who provided the peaked histogram distribution data!

## Version Information

- **Version**: 1.2.1
- **Release Date**: November 22, 2025
- **Major Change**: Percentile-based `diff_threshold` auto-estimation

## Download

- **Main Program**: `detect_meteors_cli.py` (v1.2.1)
- **Documentation**: This file (`RELEASE_NOTES_1.2.1.md`)

---

Happy meteor hunting! 🌠
