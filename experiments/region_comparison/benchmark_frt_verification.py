"""Calibrate local contrast and continuity on FRT proposals from rejected pairs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise, product
import json
from math import inf, nextafter
from pathlib import Path

import cv2
import numpy as np

from benchmark_hough_frt_retry import FRT_CONFIG, analysis_mask, axis_coverage
from benchmark_v2 import inputs
from meteor_core.detectors.frt_support import (
    MIN_DENSITY,
    WIDTH,
    noise_tiles,
    profiles,
    smooth_profiles,
    support_runs,
)
from meteor_core.image_io import load_and_bin_raw_fast
from review_metrics import event_matches, frame_metrics
from summarize import trail_coverage

CONTRASTS = tuple(
    (unit, value)
    for unit, values in (
        ("dn", (2, 3, 4, 5, 6, 8, 9, 12)),
        ("noise", (0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5)),
        ("both_sides_noise", (0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5)),
    )
    for value in values
)
GAPS = (0, 3)
LENGTHS = (16, 24, 32, 45, 60, 80)
COMBINATIONS = tuple(
    product(range(len(CONTRASTS)), range(len(GAPS)), range(len(LENGTHS)))
)
POOLED = tuple(c for c in COMBINATIONS if CONTRASTS[c[0]][0] != "both_sides_noise")
BILATERAL = tuple(c for c in COMBINATIONS if CONTRASTS[c[0]][0] == "both_sides_noise")
VERIFIERS = {
    "support_only": (POOLED, False),
    "support_and_snr": (POOLED, True),
    "both_sides_only": (BILATERAL, False),
    "both_sides_and_snr": (BILATERAL, True),
}


def key(combination):
    contrast, gap, length = combination
    unit, value = CONTRASTS[contrast]
    return {
        "contrast_unit": unit,
        "min_local_contrast": value,
        "max_gap_px_approx": GAPS[gap],
        "min_support_px": LENGTHS[length],
    }


def measure(proposals, residual, roi, refs, tiles):
    """Sweep contrast and continuity, keeping supported geometry near references."""
    by_length = np.zeros((len(CONTRASTS), len(GAPS), len(LENGTHS)))
    relevant = {
        label["current"]: np.array(
            [
                index
                for index, line in enumerate(proposals)
                if trail_coverage(label, [(line[:2], line[2:4])]) > 0
            ],
            dtype=np.intp,
        )
        for label in refs
    }
    matching = {name: {} for name in relevant}
    for ids, (axis, spacing, valid, center, contrast, bilateral, sigma) in profiles(
        proposals, residual, roi, tiles
    ):
        eligible = valid & (center > 0)
        values = {
            "dn": smooth_profiles(contrast, valid),
            "noise": smooth_profiles(contrast / sigma, valid),
            "both_sides_noise": smooth_profiles(bilateral / sigma, valid),
        }
        for contrast_idx, (unit, threshold) in enumerate(CONTRASTS):
            support = eligible & (values[unit] >= threshold)
            for gap_idx, gap in enumerate(GAPS):
                row, begin, end, distance = support_runs(support, valid, spacing, gap)
                level = np.searchsorted(LENGTHS, distance, side="right") - 1
                keep = level >= 0
                row, begin, end, distance, level = (
                    values[keep] for values in (row, begin, end, distance, level)
                )
                snr = proposals[ids[row], 4]
                np.maximum.at(by_length[contrast_idx, gap_idx], level, snr)
                for name, indices in relevant.items():
                    selected = np.flatnonzero(np.isin(ids[row], indices))
                    if not selected.size:
                        continue
                    geometry = matching[name].setdefault((contrast_idx, gap_idx), [])
                    geometry.extend(
                        (
                            float(distance[i]),
                            float(snr[i]),
                            [*axis[row[i], begin[i]], *axis[row[i], end[i] - 1]],
                        )
                        for i in selected
                    )
    scores = np.maximum.accumulate(by_length[..., ::-1], axis=-1)[..., ::-1]
    return {"scores": scores, "matching": matching}


def collect(sequence, source_data, proposal_root, raw_root):
    cv2.setNumThreads(1)
    source, frames, labels, digest = inputs(sequence, raw_root)
    if (frames, labels, digest) != (
        source_data["frames"],
        source_data["labels"],
        source_data["input_sha256"],
    ):
        raise ValueError(f"Input and FRT report differ for {sequence}")
    axes = {axis["current"]: axis for axis in labels}
    previous = load_and_bin_raw_fast(str(source / frames[0]))
    roi = analysis_mask(previous.shape, sequence) != 0
    if int(roi.sum()) != source_data["sky_mask_pixels"]:
        raise ValueError(f"ROI differs for {sequence}")
    rows = {}
    for index, (prior, name) in enumerate(pairwise(frames), start=1):
        current = load_and_bin_raw_fast(str(source / name))
        row = source_data["pairs"][name]
        if "frt" in row:
            proposals = np.load(
                proposal_root / sequence / f"{name}.npy", allow_pickle=False
            )
            if proposals[:, 4].max(initial=0.0) != row["frt"]["score"]:
                raise ValueError(f"FRT scores differ for {name}")
            residual = cv2.subtract(current, previous, dtype=cv2.CV_32F)
            references = [axes[frame] for frame in (name, prior) if frame in axes]
            rows[name] = measure(
                proposals, residual, roi, references, noise_tiles(residual)
            )
        previous = current
        if index % 15 == 0 or index == len(frames) - 1:
            print(f"{sequence}: verified {index}/{len(frames) - 1} pairs", flush=True)
    return rows


def score_verified(data, verified, combination, min_snr):
    contrast, gap, length = combination
    axes = {axis["current"]: axis for axis in data["labels"]}
    measurements = {}
    for name, pair in data["pairs"].items():
        if name not in verified:
            measurements[name] = pair["hough"]
            continue
        row = verified[name]
        measurements[name] = {
            "candidate": bool(row["scores"][combination] >= min_snr),
            "coverage": {
                frame: float(
                    axis_coverage(
                        axes[frame],
                        [
                            (score, line)
                            for distance, score, line in choices.get(
                                (contrast, gap), ()
                            )
                            if distance >= LENGTHS[length]
                        ],
                        min_snr,
                    )
                )
                for frame, choices in row["matching"].items()
            },
        }
    return {
        "frames": frame_metrics(measurements, data["truth"]),
        "events": event_matches(measurements, data["frames"], data["labels"]),
    }


def select(train, verified, combinations, *, calibrate_snr):
    """Give zero training FPs priority, then maximize new matching appearances."""
    negatives = [
        row["scores"]
        for sequence, data in train.items()
        for frame, row in verified[sequence].items()
        if not data["truth"][frame]
    ]
    maximum = np.maximum.reduce(negatives)
    best = None
    for i, combination in enumerate(combinations):
        max_negative = float(maximum[combination])
        if not calibrate_snr and max_negative >= FRT_CONFIG.min_snr:
            continue
        cutoff = max(FRT_CONFIG.min_snr, nextafter(max_negative, inf))
        evaluations = {
            name: score_verified(data, verified[name], combination, cutoff)
            for name, data in train.items()
        }
        matches = sum(
            event["appearance"]["match"]
            for result in evaluations.values()
            for event in result["events"].values()
        )
        # Favor the least restrictive SNR cutoff and strict continuity at equal recall.
        rank = (-matches, cutoff, GAPS[combination[1]], -LENGTHS[combination[2]], i)
        if best is None or rank < best[0]:
            best = (rank, combination, cutoff, evaluations)
    if best is None:
        return None
    _, combination, cutoff, evaluations = best
    return combination, cutoff, evaluations


def test_selection(train, held_out, verified, combinations, *, calibrate_snr):
    selected = select(train, verified, combinations, calibrate_snr=calibrate_snr)
    if selected is None:
        return None
    combination, cutoff, results = selected
    return {
        "settings": {**key(combination), "min_snr": cutoff},
        "training": results,
        "evaluation": {
            name: score_verified(data, verified[name], combination, cutoff)
            for name, data in held_out.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frt_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    parser.add_argument("--workers", type=int, choices=range(1, 4), default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = json.loads((args.frt_run / "hough_frt_retry.json").read_text())
    sequences = source["sequences"]
    proposal_root = args.frt_run / "proposals"
    if args.workers == 1:
        verified = {
            name: collect(name, data, proposal_root, args.raw_root)
            for name, data in sequences.items()
        }
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                name: pool.submit(collect, name, data, proposal_root, args.raw_root)
                for name, data in sequences.items()
            }
            verified = {name: job.result() for name, job in jobs.items()}

    report = {
        "grid": {
            "contrasts": CONTRASTS,
            "max_gap_samples": GAPS,
            "min_support_px": LENGTHS,
            "min_supported_density": MIN_DENSITY,
            "smoothing_samples": WIDTH,
        },
        "source_checks": {
            name: {
                key: data[key]
                for key in ("input_sha256", "truth_sha256", "sky_mask_pixels")
            }
            for name, data in sequences.items()
        },
        "previous_hough_only": source["evaluation"]["hough_only"],
        "previous_frt_snr_only": source["evaluation"]["leave_one_sequence_out"],
        "fixed": {
            unit: {
                name: score_verified(
                    data,
                    verified[name],
                    (CONTRASTS.index((unit, cutoff)), 0, LENGTHS.index(45)),
                    FRT_CONFIG.min_snr,
                )
                for name, data in sequences.items()
            }
            for unit, cutoff in (("dn", 9), ("noise", 2))
        },
        "leave_one_sequence_out": {
            name: {
                method: test_selection(
                    {other: data for other, data in sequences.items() if other != name},
                    {name: data},
                    verified,
                    combinations,
                    calibrate_snr=calibrate,
                )
                for method, (combinations, calibrate) in VERIFIERS.items()
            }
            for name, data in sequences.items()
        },
        "zero_fp_with_all_labels": {
            method: test_selection(
                sequences, sequences, verified, combinations, calibrate_snr=calibrate
            )
            for method, (combinations, calibrate) in VERIFIERS.items()
        },
    }
    (args.output / "frt_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
