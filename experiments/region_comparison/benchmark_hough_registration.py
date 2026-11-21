"""Sweep Hough thresholds on the 121 reviewed adjacent RAW pairs."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform

import cv2
import numpy as np

from benchmark_star_registration import MARGIN, SKY_ROIS
from benchmark_v2 import inputs
from meteor_core.config_io import load_pipeline_config
from meteor_core.detectors.hough_default import HoughDetector, HoughDetectorConfig
from meteor_core.hooks.star_registration import StarRegistrationHook
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.pipeline import _build_runtime_params
from meteor_core.roi_selector import create_roi_mask_from_polygon
from meteor_core.schema import DetectionContext
from review_metrics import event_matches, frame_metrics, read_truth
from summarize import trail_coverage

CONFIG = Path("config_examples/star_registration_sample.yaml")
POLICIES = ("unaligned", "aligned")
POSITIVE_POLICIES = ("positive_unaligned", "positive_aligned")
GRID = {
    "diff_threshold": (*range(5, 21), 22, 24, 26, 28, 30, 35, 40),
    "min_area": (1, 3, 6, 10, 20, 40, 80),
    "min_aspect_ratio": (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0),
    "min_line_score": (
        15,
        30,
        45,
        60,
        90,
        120,
        180,
        240,
        360,
        480,
        720,
        1000,
        1500,
        2000,
    ),
}
SHAPE = tuple(map(len, GRID.values()))


def measure(detector, context, settings, references):
    """Run Hough once per pixel cutoff, then apply the frame gates in the grid."""
    line_scores = []
    ratios = []
    coverage = {label["current"]: [] for label in references}
    areas = np.array(GRID["min_area"])
    for diff_threshold in GRID["diff_threshold"]:
        runtime = _build_runtime_params(
            {**settings, "diff_threshold": diff_threshold}, detector
        )
        analysis = detector.analyze(replace(context, runtime_params=runtime))
        line_scores.append(analysis.score)
        max_ratio = np.zeros(len(areas))
        for contour in analysis.contours:
            ratio = contour.aspect_ratio
            if ratio is not None:
                np.maximum(max_ratio, ratio, out=max_ratio, where=contour.area > areas)
        ratios.append(max_ratio)
        if references:
            lines = [(line[:2], line[2:]) for line in analysis.lines]
            for label in references:
                coverage[label["current"]].append(trail_coverage(label, lines))

    # Area and ratio must pass on the same contour. Line score is frame-wide.
    shape_passed = (
        np.asarray(ratios)[:, :, None, None]
        > np.asarray(GRID["min_aspect_ratio"])[None, None, :, None]
    )
    line_passed = np.asarray(line_scores)[:, None, None, None] >= np.asarray(
        GRID["min_line_score"]
    )
    return shape_passed & line_passed, coverage


def collect(sequence, raw_root, settings, *, positive_only=False):
    source, frames, labels, digest = inputs(sequence, raw_root)
    readme = source / "README.md"
    checksums = source / ("SHA256SUMS" if sequence == "bundled" else "SHA256SUM")
    inventory = {
        line.split()[1].lstrip("*") for line in checksums.read_text().splitlines()
    }
    truth = read_truth(readme, frames[1:], inventory)
    if any(not truth[label["current"]] for label in labels):
        raise ValueError(f"An axis disagrees with exposure truth for {sequence}")
    axes = {label["current"]: label for label in labels}
    detector = HoughDetector()
    positive_detector = HoughDetector(HoughDetectorConfig(difference_mode="positive"))
    hook = StarRegistrationHook()
    previous = load_and_bin_raw_fast(str(source / frames[0]))
    roi = create_roi_mask_from_polygon(SKY_ROIS[sequence], previous.shape)
    margin = cv2.erode(roi, np.ones((2 * MARGIN + 1, 2 * MARGIN + 1), np.uint8))
    # Keep the absolute, unaligned control as the threshold-selection reference.
    policies = ("unaligned", *POSITIVE_POLICIES) if positive_only else POLICIES
    pairs = {policy: {} for policy in policies}
    registration = {}
    for index, (prior, name) in enumerate(zip(frames, frames[1:]), start=1):
        current = load_and_bin_raw_fast(str(source / name))
        context = DetectionContext(current, previous, roi, {}, {"frame_index": index})
        aligned = hook.on_before_detection(context)
        common_roi = cv2.bitwise_and(margin, aligned.roi_mask)
        registration[name] = {
            **aligned.metadata["star_registration"],
            "extra_pixels_excluded": cv2.countNonZero(margin)
            - cv2.countNonZero(common_roi),
        }
        references = [axes[frame] for frame in (name, prior) if frame in axes]
        contexts = (context, aligned)
        if positive_only:
            contexts = (context, *contexts)
        for policy, pair in zip(policies, contexts, strict=True):
            pairs[policy][name] = measure(
                positive_detector if policy in POSITIVE_POLICIES else detector,
                replace(pair, roi_mask=common_roi),
                settings,
                references,
            )
        previous = current
    return {
        "input_sha256": digest,
        "truth_sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "frames": frames,
        "labels": labels,
        "truth": truth,
        "sky_mask_pixels": cv2.countNonZero(margin),
        "sky_polygon": SKY_ROIS[sequence],
        "registration": registration,
        "pairs": pairs,
    }


def select_thresholds(sequences, policy, baseline, *, diff_only=False):
    """Minimize training false positives while retaining original meteor evidence."""
    available = np.ones(SHAPE, dtype=bool)
    if diff_only:
        available.fill(False)
        available[(slice(None), *baseline[1:])] = True
    counts = {
        key: np.zeros(SHAPE, dtype=int) for key in ("fp", "tp", "appearance", "events")
    }
    for sequence in sequences:
        pairs = sequence["pairs"][policy]
        original = sequence["pairs"]["unaligned"]
        for name, truth in sequence["truth"].items():
            candidates, _ = pairs[name]
            counts["tp" if truth else "fp"] += candidates
            if truth and original[name][0][baseline]:
                available &= candidates
        for label in sequence["labels"]:
            current = label["current"]
            index = sequence["frames"].index(current)
            follow = sequence["frames"][index + 1 : index + 2]
            matched = []
            originally_matched = []
            for name in [current, *follow]:
                candidates, coverage = pairs[name]
                enough = np.asarray(coverage[current]) >= 0.5
                matched.append(candidates & enough[:, None, None, None])
                reference_candidates, reference_coverage = original[name]
                originally_matched.append(
                    reference_candidates[baseline]
                    and reference_coverage[current][baseline[0]] >= 0.5
                )
            if originally_matched[0]:
                available &= matched[0]
            events = np.logical_or.reduce(matched)
            if any(originally_matched):
                available &= events
            counts["appearance"] += matched[0]
            counts["events"] += events

    def rank(index):
        return (
            counts["fp"][index],
            -counts["appearance"][index],
            -counts["tp"][index],
            -counts["events"][index],
            sum(abs(i - j) for i, j in zip(index, baseline, strict=True)),
            index,
        )

    return min(zip(*np.nonzero(available), strict=True), key=rank, default=None)


def evaluate(sequences, policy, index):
    if index is None:
        return None
    out = {
        "settings": {
            key: values[i] for (key, values), i in zip(GRID.items(), index, strict=True)
        },
        "sequences": {},
    }
    for name, sequence in sequences.items():
        pairs = {
            frame: {
                "candidate": bool(decisions[index]),
                "coverage": {
                    meteor: values[index[0]] for meteor, values in coverage.items()
                },
            }
            for frame, (decisions, coverage) in sequence["pairs"][policy].items()
        }
        metrics = frame_metrics(pairs, sequence["truth"])
        out["sequences"][name] = {
            "frames": metrics,
            "events": event_matches(pairs, sequence["frames"], sequence["labels"]),
        }
    return out


def benchmark(sequences, settings, policies=POLICIES):
    baseline = tuple(values.index(settings[key]) for key, values in GRID.items())
    return {
        policy: {
            "baseline": evaluate(sequences, policy, baseline),
            "diff_only": evaluate(
                sequences,
                policy,
                select_thresholds(sequences.values(), policy, baseline, diff_only=True),
            ),
            "all_thresholds": evaluate(
                sequences,
                policy,
                select_thresholds(sequences.values(), policy, baseline),
            ),
            "leave_one_sequence_out": {
                held_out: evaluate(
                    {held_out: data},
                    policy,
                    select_thresholds(
                        (
                            train
                            for name, train in sequences.items()
                            if name != held_out
                        ),
                        policy,
                        baseline,
                    ),
                )
                for held_out, data in sequences.items()
            },
        }
        for policy in policies
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Compare positive differences before and after registration.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    settings = load_pipeline_config(CONFIG).params.to_dict()
    sequences = {}
    for sequence in SKY_ROIS:
        sequences[sequence] = collect(
            sequence, args.raw_root, settings, positive_only=args.positive_only
        )
        print("Measured", sequence, flush=True)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "grid": GRID,
        "baseline_settings": settings,
        "sequences": {
            name: {key: value for key, value in sequence.items() if key != "pairs"}
            for name, sequence in sequences.items()
        },
        "evaluation": benchmark(
            sequences, settings, POSITIVE_POLICIES if args.positive_only else POLICIES
        ),
    }
    if args.positive_only:
        report["selection_reference"] = (
            "unaligned absolute difference at baseline settings"
        )
    (args.output / "thresholds.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
