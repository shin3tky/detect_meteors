"""Test the existing FRT on pairs rejected by unaligned positive-difference Hough."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
from itertools import pairwise
import json
from math import inf, nextafter
from pathlib import Path
import platform
from time import perf_counter

import cv2
import numpy as np

from benchmark_hough_recovery import CONFIG, FOCUS, analysis_mask
from benchmark_registered_median import _scores
from benchmark_star_registration import SKY_ROIS
from benchmark_v2 import _record, inputs
from frt_detector import FrtConfig, FrtDetector
from meteor_core.config_io import load_pipeline_config
from meteor_core.detectors.hough_default import HoughDetector, HoughDetectorConfig
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.schema import DetectionContext
from review_metrics import event_matches, frame_metrics, read_truth
from summarize import trail_coverage

FRT_CONFIG = FrtConfig(max_report_segments=100_000)


def collect(sequence, raw_root, settings):
    cv2.setNumThreads(1)
    source, frames, labels, digest = inputs(sequence, raw_root)
    checksum_file = source / ("SHA256SUMS" if sequence == "bundled" else "SHA256SUM")
    inventory = {
        line.split()[1].lstrip("*") for line in checksum_file.read_text().splitlines()
    }
    readme = source / "README.md"
    truth = read_truth(readme, frames[1:], inventory)
    axes = {axis["current"]: axis for axis in labels}
    if set(axes) != {name for name, meteor in truth.items() if meteor}:
        raise ValueError(f"Expected an annotated axis for each meteor in {sequence}")

    hough = HoughDetector(HoughDetectorConfig(difference_mode="positive"))
    frt = FrtDetector(FRT_CONFIG)
    previous = load_and_bin_raw_fast(str(source / frames[0]))
    roi = analysis_mask(previous.shape, sequence)
    pairs = {}
    for index, (prior, name) in enumerate(pairwise(frames), start=1):
        current = load_and_bin_raw_fast(str(source / name))
        context = DetectionContext(
            current, previous, roi, settings, {"frame_index": index}
        )
        references = [axes[frame] for frame in (name, prior) if frame in axes]
        analysis = hough.analyze(context)
        decision = analysis.evaluate()
        row = {"hough": _record(decision.is_candidate, analysis.lines, references)}
        if not decision.is_candidate:
            started = perf_counter()
            result = frt.detect(context)
            row["frt"] = {
                **_scores(result, references),
                "detector_seconds": perf_counter() - started,
            }
        pairs[name] = row
        previous = current
        if index % 15 == 0 or index == len(frames) - 1:
            print(f"{sequence}: {index}/{len(frames) - 1} pairs", flush=True)

    return {
        "input_sha256": digest,
        "truth_sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "sky_mask_pixels": cv2.countNonZero(roi),
        "frames": frames,
        "labels": labels,
        "truth": truth,
        "pairs": pairs,
    }


def negative_scores(sequences):
    """Return only scores that can produce an additional false positive."""
    return [
        (name, frame, row["frt"]["score"])
        for name, sequence in sequences.items()
        for frame, row in sequence["pairs"].items()
        if not sequence["truth"][frame] and "frt" in row
    ]


def train_zero_fp(sequences):
    """Use the lowest inclusive cutoff that excludes all training negatives."""
    negatives = negative_scores(sequences)
    max_score = max((score for _, _, score in negatives), default=0.0)
    return {
        "min_snr": max(FRT_CONFIG.min_snr, nextafter(max_score, inf)),
        "negative_count": len(negatives),
        "max_negative_score": max_score,
        "max_negative_frames": [
            {"sequence": name, "frame": frame}
            for name, frame, score in negatives
            if score == max_score
        ],
    }


def axis_coverage(label, scored_lines, threshold):
    return trail_coverage(
        label,
        [(line[:2], line[2:]) for score, line in scored_lines if score >= threshold],
    )


def evaluate(sequence, min_snr):
    """Keep Hough detections; apply the same FRT cutoff to decisions and geometry."""
    axes = {axis["current"]: axis for axis in sequence["labels"]}
    pairs = {}
    added = []
    for name, row in sequence["pairs"].items():
        frt = row.get("frt")
        if min_snr is None or frt is None:
            pairs[name] = row["hough"]
            continue
        candidate = frt["score"] >= min_snr
        pairs[name] = {
            "candidate": candidate,
            "coverage": {
                frame: axis_coverage(axes[frame], lines, min_snr)
                for frame, lines in frt["relevant_lines"].items()
            },
        }
        if candidate:
            added.append(name)
    return {
        "frames": frame_metrics(pairs, sequence["truth"]),
        "events": event_matches(pairs, sequence["frames"], sequence["labels"]),
        "added_by_frt": added,
    }


def evaluate_folds(sequences):
    folds = {}
    for held_out, sequence in sequences.items():
        train = {name: data for name, data in sequences.items() if name != held_out}
        calibration = train_zero_fp(train)
        cutoff = calibration["min_snr"]
        folds[held_out] = {
            "training_sequences": list(train),
            "calibration": calibration,
            "training": {name: evaluate(data, cutoff) for name, data in train.items()},
            "held_out": evaluate(sequence, cutoff),
        }
    return folds


def recovery_cost(sequences):
    """Measure the cost of preserving each faint axis using all labels."""
    negatives = negative_scores(sequences)
    costs = {}
    for frame in FOCUS:
        sequence = next(data for data in sequences.values() if frame in data["pairs"])
        label = next(axis for axis in sequence["labels"] if axis["current"] == frame)
        lines = sequence["pairs"][frame]["frt"]["relevant_lines"][frame]
        highest = next(
            (
                score
                for score in sorted({score for score, _ in lines}, reverse=True)
                if axis_coverage(label, lines, score) >= 0.5
            ),
            None,
        )
        costs[frame] = {
            "highest_recovering_min_snr": highest,
            "minimum_false_positives": (
                sum(score >= highest for _, _, score in negatives)
                if highest is not None
                else None
            ),
        }
    return costs


def benchmark(sequences):
    calibration = train_zero_fp(sequences)
    return {
        "hough_only": {name: evaluate(data, None) for name, data in sequences.items()},
        "frt_at_prototype_threshold": {
            name: evaluate(data, FRT_CONFIG.min_snr) for name, data in sequences.items()
        },
        "leave_one_sequence_out": evaluate_folds(sequences),
        "zero_fp_with_all_labels": {
            "calibration": calibration,
            "results": {
                name: evaluate(data, calibration["min_snr"])
                for name, data in sequences.items()
            },
        },
        "cost_with_all_labels": recovery_cost(sequences),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    parser.add_argument("--workers", type=int, choices=range(1, 4), default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = load_pipeline_config(CONFIG).params.to_dict()
    if args.workers == 1:
        sequences = {name: collect(name, args.raw_root, settings) for name in SKY_ROIS}
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                name: pool.submit(collect, name, args.raw_root, settings)
                for name in SKY_ROIS
            }
            sequences = {name: job.result() for name, job in jobs.items()}
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "hough_settings": settings,
        "frt_config": asdict(FRT_CONFIG),
        "sequences": sequences,
        "evaluation": benchmark(sequences),
    }
    (args.output / "hough_frt_retry.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
