"""Check faint-meteor recovery with unaligned positive differences and Hough."""

import argparse
from dataclasses import replace
import hashlib
from itertools import pairwise, product
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from benchmark_hough_registration import GRID
from benchmark_star_registration import MARGIN, SKY_ROIS
from benchmark_v2 import inputs
from meteor_core.config_io import load_pipeline_config
from meteor_core.detectors.hough_default import HoughDetector, HoughDetectorConfig
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.roi_selector import create_roi_mask_from_polygon
from meteor_core.schema import DetectionContext
from review_metrics import read_truth
from summarize import trail_coverage

CONFIG = Path("config_examples/hough_positive_sample.yaml")
FOCUS = ("_C150319.ORF", "_C150326.ORF", "_C140521.ORF")
GATES = {
    "diff_threshold": (3, 4, *GRID["diff_threshold"]),
    "min_area": (0, *GRID["min_area"]),
    "min_aspect_ratio": (1.5, *GRID["min_aspect_ratio"]),
    "min_line_score": (10, *GRID["min_line_score"]),
}
HOUGH_PARAMS = ("threshold", "min_line_length", "max_line_gap")
HOUGH = tuple(product((10, 6), (15, 10), (5, 12)))
KERNELS = {"square": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
SHAPE = (len(HOUGH), *(len(values) for values in GATES.values()))


def analysis_mask(shape, sequence):
    roi = create_roi_mask_from_polygon(SKY_ROIS[sequence], shape)
    return cv2.erode(roi, np.ones((2 * MARGIN + 1, 2 * MARGIN + 1), np.uint8))


def measure(detector, context, settings, axis):
    """Reuse the threshold mask and contours across Hough settings."""
    ratios = []
    scores = np.empty(SHAPE[:2])
    coverage = np.zeros(SHAPE[:2])
    areas = np.asarray(GATES["min_area"])
    for cutoff_index, cutoff in enumerate(GATES["diff_threshold"]):
        analysis = detector.analyze(
            replace(context, runtime_params={**settings, "diff_threshold": cutoff})
        )
        contours = np.asarray(
            [
                (contour.area, contour.aspect_ratio or 0.0)
                for contour in analysis.contours
            ]
        ).reshape(-1, 2)
        ratios.append(
            np.max(
                np.where(contours[:, :1] > areas, contours[:, 1:], 0.0),
                axis=0,
                initial=0.0,
            )
        )
        scores[0, cutoff_index] = analysis.score
        if axis is not None:
            coverage[0, cutoff_index] = trail_coverage(
                axis, [(line[:2], line[2:]) for line in analysis.lines]
            )
        for profile, values in enumerate(HOUGH[1:], start=1):
            score, lines = detector.compute_line_score(
                analysis.mask, dict(zip(HOUGH_PARAMS, values, strict=True))
            )
            scores[profile, cutoff_index] = score
            if axis is not None:
                coverage[profile, cutoff_index] = trail_coverage(
                    axis, [(line[:2], line[2:]) for line in lines]
                )

    shape_passed = (
        np.asarray(ratios)[:, :, None, None]
        > np.asarray(GATES["min_aspect_ratio"])[None, None, :, None]
    )
    score_passed = scores[:, :, None, None, None] >= np.asarray(GATES["min_line_score"])
    return shape_passed & score_passed, coverage


def collect(sequence, raw_root, settings):
    source, frames, labels, input_hash = inputs(sequence, raw_root)
    readme = source / "README.md"
    checksum_file = source / ("SHA256SUMS" if sequence == "bundled" else "SHA256SUM")
    inventory = {
        line.split()[1].lstrip("*") for line in checksum_file.read_text().splitlines()
    }
    truth = read_truth(readme, frames[1:], inventory)
    axes = {axis["current"]: axis for axis in labels}
    if set(axes) != {name for name, meteor in truth.items() if meteor}:
        raise ValueError(f"Expected an annotated axis for each meteor in {sequence}")
    detectors = {
        kernel: HoughDetector(HoughDetectorConfig("positive")) for kernel in KERNELS
    }
    for kernel, detector in detectors.items():
        detector._kernel = cv2.getStructuringElement(KERNELS[kernel], (3, 3))

    previous = load_and_bin_raw_fast(str(source / frames[0]))
    roi = analysis_mask(previous.shape, sequence)
    pairs = {kernel: {} for kernel in KERNELS}
    for _, name in pairwise(frames):
        current = load_and_bin_raw_fast(str(source / name))
        context = DetectionContext(current, previous, roi, {}, {})
        for kernel, detector in detectors.items():
            pairs[kernel][name] = measure(detector, context, settings, axes.get(name))
        previous = current
    return {
        "input_sha256": input_hash,
        "truth_sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "sky_mask_pixels": cv2.countNonZero(roi),
        "truth": truth,
        "pairs": pairs,
    }


def baseline_index(settings):
    expected_hough = tuple(settings[f"hough_{name}"] for name in HOUGH_PARAMS)
    if expected_hough != HOUGH[0]:
        raise ValueError("The sample Hough settings have changed")
    return (0, *(values.index(settings[key]) for key, values in GATES.items()))


def totals(sequences, kernel, baseline, *, retry_rejected=False):
    """Count positive exposures and retain the baseline's matched meteors."""
    eligible = np.ones(SHAPE, dtype=bool)
    false_positives = np.zeros(SHAPE, dtype=int)
    matched = np.zeros(SHAPE, dtype=int)
    for sequence in sequences:
        for name, meteor in sequence["truth"].items():
            candidate, coverage = sequence["pairs"][kernel][name]
            original, original_coverage = sequence["pairs"]["square"][name]
            if retry_rejected and original[baseline]:
                if meteor:
                    matched += original_coverage[baseline[:2]] >= 0.5
                else:
                    false_positives += 1
                continue
            if not meteor:
                false_positives += candidate
                continue
            appearance = candidate & (coverage >= 0.5)[:, :, None, None, None]
            matched += appearance
            if original[baseline]:
                eligible &= candidate
                if original_coverage[baseline[:2]] >= 0.5:
                    eligible &= appearance
    return eligible, false_positives, matched


def select_thresholds(counts_by_kernel, baseline):
    """Among zero-FP training choices, maximize geometrically matched meteors."""
    best = None
    for kernel, (eligible, fp, matched) in counts_by_kernel.items():
        zero_fp = eligible & (fp == 0)
        if not zero_fp.any():
            continue
        most_matches = matched[zero_fp].max()
        for index in zip(*np.nonzero(zero_fp & (matched == most_matches)), strict=True):
            distance = sum(
                a != b for a, b in zip(HOUGH[index[0]], HOUGH[baseline[0]], strict=True)
            ) + sum(abs(a - b) for a, b in zip(index[1:], baseline[1:], strict=True))
            rank = (-int(most_matches), distance, list(KERNELS).index(kernel), index)
            if best is None or rank < best[0]:
                best = (rank, kernel, index)
    if best is None:
        raise ValueError("No zero-FP training setting preserves the control meteors")
    return best[1:]


def evaluate(data, kernel, index, *, keep_baseline=None):
    settings = dict(
        zip((f"hough_{name}" for name in HOUGH_PARAMS), HOUGH[index[0]], strict=True)
    )
    settings.update(
        {
            key: values[i]
            for (key, values), i in zip(GATES.items(), index[1:], strict=True)
        }
    )
    sequences = {}
    for name, sequence in data.items():
        frames = {}
        for frame, truth in sequence["truth"].items():
            candidate, coverage = sequence["pairs"][kernel][frame]
            decision, axis_coverage = bool(candidate[index]), float(coverage[index[:2]])
            if keep_baseline is not None:
                original, original_coverage = sequence["pairs"]["square"][frame]
                if original[keep_baseline]:
                    decision = True
                    axis_coverage = float(original_coverage[keep_baseline[:2]])
            row = {"candidate": decision}
            if truth:
                row["appearance_coverage"] = axis_coverage
                row["appearance_match"] = (
                    row["candidate"] and row["appearance_coverage"] >= 0.5
                )
            frames[frame] = row
        sequences[name] = frames
    return {"opening": kernel, "settings": settings, "sequences": sequences}


def benchmark(data, settings):
    baseline = baseline_index(settings)
    folds = {}
    for held, evaluation in data.items():
        train = [rows for name, rows in data.items() if name != held]
        counts = {kernel: totals(train, kernel, baseline) for kernel in KERNELS}
        retry_counts = {
            kernel: totals(train, kernel, baseline, retry_rejected=True)
            for kernel in KERNELS
        }
        folds[held] = {
            "square": evaluate(
                {held: evaluation},
                *select_thresholds({"square": counts["square"]}, baseline),
            ),
            "square_or_cross": evaluate(
                {held: evaluation}, *select_thresholds(counts, baseline)
            ),
            "retry_rejected": evaluate(
                {held: evaluation},
                *select_thresholds(retry_counts, baseline),
                keep_baseline=baseline,
            ),
        }

    all_counts = {kernel: totals(data.values(), kernel, baseline) for kernel in KERNELS}
    all_retry_counts = {
        kernel: totals(data.values(), kernel, baseline, retry_rejected=True)
        for kernel in KERNELS
    }
    cost = {}
    for focus in FOCUS:
        sequence = next(rows for rows in data.values() if focus in rows["truth"])
        cost[focus] = {}
        for kernel, (eligible, fp, _) in all_counts.items():
            candidates, coverage = sequence["pairs"][kernel][focus]
            allowed = eligible & candidates & (coverage >= 0.5)[:, :, None, None, None]
            cost[focus][kernel] = int(fp[allowed].min()) if allowed.any() else None

    return {
        "baseline": evaluate(data, "square", baseline),
        "leave_one_sequence_out": folds,
        "zero_fp_with_all_labels": evaluate(
            data, *select_thresholds(all_counts, baseline)
        ),
        "zero_fp_retry_rejected_with_all_labels": evaluate(
            data, *select_thresholds(all_retry_counts, baseline), keep_baseline=baseline
        ),
        "minimum_fp_with_all_labels_for_each_meteor": cost,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    settings = load_pipeline_config(CONFIG).params.to_dict()
    sequences = {}
    for sequence in SKY_ROIS:
        sequences[sequence] = collect(sequence, args.raw_root, settings)
        print("Measured", sequence, flush=True)
    report = {
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "opening": list(KERNELS),
        "hough_profiles": [dict(zip(HOUGH_PARAMS, row, strict=True)) for row in HOUGH],
        "grid": GATES,
        "sequences": {
            name: {key: value for key, value in rows.items() if key != "pairs"}
            for name, rows in sequences.items()
        },
        "evaluation": benchmark(sequences, settings),
    }
    (args.output / "recovery.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
