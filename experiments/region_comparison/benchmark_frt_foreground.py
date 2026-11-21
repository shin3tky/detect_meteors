"""Test a preceding-exposure background texture gate for FRT support."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise, product
import json
from math import inf, nextafter, sqrt
from pathlib import Path

import cv2
import numpy as np

from benchmark_frt_verification import CONTRASTS, GAPS, LENGTHS, profiles
from benchmark_hough_frt_retry import FRT_CONFIG, analysis_mask, axis_coverage
from benchmark_v2 import inputs
from frt_support import smooth_profiles, support_runs
from meteor_core.image_io import load_and_bin_raw_fast
from review_metrics import event_matches, frame_metrics
from segment_verification import noise_tiles
from summarize import trail_coverage

LEVELS = tuple(value for unit, value in CONTRASTS if unit == "both_sides_noise")
TEXTURE_CAPS = (0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
TEXTURE_QUANTILE = 0.9
TEXTURE_SIGMA_PX = 4.0
TEXTURE_WINDOW_PX = 13
SHAPE = (len(LEVELS), len(GAPS), len(LENGTHS), len(TEXTURE_CAPS))
CALIBRATIONS = {
    "fixed_snr": {"calibrate_snr": False},
    "calibrated_snr": {"calibrate_snr": True},
    "floor_ten": {"calibrate_snr": True, "snr_floor": 10.0},
    "separation_margin": {"calibrate_snr": True, "use_margin": True},
    "training_coverage": {
        "calibrate_snr": True,
        "snr_floor": 10.0,
        "prefer_coverage": True,
    },
    "validate_gap_relaxation": {
        "calibrate_snr": True,
        "snr_floor": 10.0,
        "verify_gaps": True,
    },
}


def texture_map(previous):
    """Average the amplitude of spatial structure in the preceding RAW."""
    raw = previous.astype(np.float32)
    background = cv2.GaussianBlur(raw, (0, 0), TEXTURE_SIGMA_PX)
    return cv2.blur(np.abs(raw - background), (TEXTURE_WINDOW_PX, TEXTURE_WINDOW_PX))


def interval_texture(profile, row, start, stop):
    """Use a percentile of background texture over each supported interval."""
    counts = stop - start
    output = np.empty(len(row))
    for count in np.unique(counts):
        selected = np.flatnonzero(counts == count)
        values = profile[row[selected, None], start[selected, None] + np.arange(count)]
        output[selected] = np.quantile(values, TEXTURE_QUANTILE, axis=1)
    return output


def interval_batches(proposals, residual, roi, tiles, texture):
    """Yield bilateral support regions with their preceding-RAW texture."""
    for ids, (axis, spacing, valid, center, _, bilateral, sigma) in profiles(
        proposals, residual, roi, tiles
    ):
        x, y = np.rint(axis).astype(np.intp).transpose(2, 0, 1)
        raw_texture = texture[y, x] / sigma
        brightness = smooth_profiles(bilateral / sigma, valid)
        for i, cutoff in enumerate(LEVELS):
            supported = valid & (center > 0) & (brightness >= cutoff)
            for j, gap in enumerate(GAPS):
                row, first, stop, length = support_runs(supported, valid, spacing, gap)
                keep = length >= LENGTHS[0]
                row, first, stop, length = (
                    value[keep] for value in (row, first, stop, length)
                )
                if not row.size:
                    continue
                roughness = interval_texture(raw_texture, row, first, stop)
                lines = np.column_stack((axis[row, first], axis[row, stop - 1]))
                yield (i, j), ids[row], length, roughness, lines


def measure(proposals, residual, roi, tiles, texture, references):
    scores = np.zeros(SHAPE)
    relevant = {
        label["current"]: np.array(
            [
                i
                for i, line in enumerate(proposals)
                if trail_coverage(label, [(line[:2], line[2:4])]) > 0
            ],
            dtype=np.intp,
        )
        for label in references
    }
    matching = {frame: {} for frame in relevant}
    for (i, j), source, length, roughness, lines in interval_batches(
        proposals, residual, roi, tiles, texture
    ):
        dig_length = np.searchsorted(LENGTHS, length, side="right") - 1
        dig_roughness = np.searchsorted(TEXTURE_CAPS, roughness, side="left")
        usable = dig_roughness < len(TEXTURE_CAPS)
        snr = proposals[source, 4]
        np.maximum.at(
            scores[i, j], (dig_length[usable], dig_roughness[usable]), snr[usable]
        )
        for frame, indices in relevant.items():
            hits = np.flatnonzero(usable & np.isin(source, indices))
            if hits.size:
                matching[frame].setdefault((i, j), []).extend(
                    (int(dig_length[k]), int(dig_roughness[k]), float(snr[k]), lines[k])
                    for k in hits
                )
    scores = np.maximum.accumulate(scores, axis=-1)
    scores = np.maximum.accumulate(scores[..., ::-1, :], axis=-2)[..., ::-1, :]
    return scores, matching


def collect(sequence, data, proposals, raw_root):
    cv2.setNumThreads(1)
    source, frames, labels, digest = inputs(sequence, raw_root)
    if (frames, labels, digest) != (
        data["frames"],
        data["labels"],
        data["input_sha256"],
    ):
        raise ValueError(f"Input and FRT report differ for {sequence}")
    axes = {label["current"]: label for label in labels}
    previous = load_and_bin_raw_fast(str(source / frames[0]))
    roi = analysis_mask(previous.shape, sequence) != 0
    if int(roi.sum()) != data["sky_mask_pixels"]:
        raise ValueError(f"ROI differs for {sequence}")
    output = {}
    for index, (prior, name) in enumerate(pairwise(frames), start=1):
        current = load_and_bin_raw_fast(str(source / name))
        row = data["pairs"][name]
        if "frt" in row:
            lines = np.load(proposals / sequence / f"{name}.npy", allow_pickle=False)
            if lines[:, 4].max(initial=0.0) != row["frt"]["score"]:
                raise ValueError(f"FRT scores differ for {name}")
            residual = cv2.subtract(current, previous, dtype=cv2.CV_32F)
            references = [axes[frame] for frame in (name, prior) if frame in axes]
            output[name] = measure(
                lines,
                residual,
                roi,
                noise_tiles(residual),
                texture_map(previous),
                references,
            )
        previous = current
        if index % 15 == 0 or index == len(frames) - 1:
            print(f"{sequence}: {index}/{len(frames) - 1} pairs", flush=True)
    return output


def scored_intervals(matching, trial):
    i, j, length, cap = trial
    return [
        (snr, line)
        for n, t, snr, line in matching.get((i, j), ())
        if n >= length and t <= cap
    ]


def evaluate(data, verified, trial, min_snr):
    labels = {label["current"]: label for label in data["labels"]}
    rows = {}
    for name, entry in data["pairs"].items():
        if name not in verified:
            rows[name] = entry["hough"]
            continue
        scores, matching = verified[name]
        rows[name] = {
            "candidate": bool(scores[trial] >= min_snr),
            "coverage": {
                frame: float(
                    axis_coverage(
                        labels[frame],
                        scored_intervals(values, trial),
                        min_snr,
                    )
                )
                for frame, values in matching.items()
            },
        }
    return {
        "frames": frame_metrics(rows, data["truth"]),
        "events": event_matches(rows, data["frames"], data["labels"]),
    }


def highest_axis_cutoff(label, lines, low):
    """Highest inclusive SNR cutoff preserving a 50% match to a training axis."""
    return next(
        (
            score
            for score in sorted(
                {score for score, _ in lines if score >= low}, reverse=True
            )
            if axis_coverage(label, lines, score) >= 0.5
        ),
        None,
    )


def improves_training_coverage(original, relaxed):
    """Require zero FPs, preserved matches and a wider training-axis match."""
    if any(result["frames"]["fp"] for result in relaxed.values()):
        return False
    improved = False
    for name, result in original.items():
        for axis, event in result["events"].items():
            before = event["appearance"]
            after = relaxed[name]["events"][axis]["appearance"]
            if before["match"] and after["coverage"] < before["coverage"]:
                return False
            improved |= after["match"] and after["coverage"] > before["coverage"]
    return improved


def train_and_test(
    train,
    test,
    verified,
    *,
    calibrate_snr,
    snr_floor=FRT_CONFIG.min_snr,
    use_margin=False,
    prefer_coverage=False,
    verify_gaps=False,
):
    negative = np.maximum.reduce(
        [
            scores
            for name, data in train.items()
            for frame, (scores, _) in verified[name].items()
            if not data["truth"][frame]
        ]
    )
    misses = [
        (label, verified[name][label["current"]][1][label["current"]])
        for name, data in train.items()
        for label in data["labels"]
        if label["current"] in verified[name]
    ]
    best = None
    for trial in product(*(range(n) for n in SHAPE)):
        peak = float(negative[trial])
        if not calibrate_snr and peak >= snr_floor:
            continue
        cutoff = max(snr_floor, nextafter(peak, inf))
        margin = 1.0
        if use_margin:
            recovered_limits = [
                limit
                for label, matching in misses
                if (
                    limit := highest_axis_cutoff(
                        label, scored_intervals(matching, trial), cutoff
                    )
                )
                is not None
            ]
            if recovered_limits:
                weakest = min(recovered_limits)
                margin = weakest / cutoff
                # Center the threshold in log-score space while preserving training matches.
                cutoff = max(cutoff, min(weakest, sqrt(weakest * cutoff)))
        training = {
            name: evaluate(data, verified[name], trial, cutoff)
            for name, data in train.items()
        }
        matches = [
            item["appearance"]["coverage"]
            for rows in training.values()
            for item in rows["events"].values()
            if item["appearance"]["match"]
        ]
        i, j, length, cap = trial
        rank = (
            -len(matches),
            -margin,
            cutoff,
            -sum(matches) if prefer_coverage else 0.0,
            GAPS[j],
            -LENGTHS[length],
            TEXTURE_CAPS[cap],
            -LEVELS[i],
        )
        if best is None or rank < best[0]:
            best = (rank, trial, cutoff, training)
    if best is None:
        return None
    _, (i, j, length, cap), cutoff, training = best
    if verify_gaps and GAPS[j] == 0:
        relaxed_index = GAPS.index(3)
        relaxed = {
            name: evaluate(
                data, verified[name], (i, relaxed_index, length, cap), cutoff
            )
            for name, data in train.items()
        }
        if improves_training_coverage(training, relaxed):
            j = relaxed_index
            training = relaxed
    return {
        "settings": {
            "bilateral_noise_min": LEVELS[i],
            "max_gap_samples": GAPS[j],
            "min_support_px": LENGTHS[length],
            "maximum_prior_texture_noise": TEXTURE_CAPS[cap],
            "frt_min_snr": cutoff,
        },
        "training": training,
        "evaluation": {
            name: evaluate(data, verified[name], (i, j, length, cap), cutoff)
            for name, data in test.items()
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
    seq = json.loads((args.frt_run / "hough_frt_retry.json").read_text())["sequences"]
    proposals = args.frt_run / "proposals"
    if args.workers == 1:
        measured = {
            name: collect(name, data, proposals, args.raw_root)
            for name, data in seq.items()
        }
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {
                name: pool.submit(collect, name, data, proposals, args.raw_root)
                for name, data in seq.items()
            }
            measured = {name: job.result() for name, job in jobs.items()}
    report = {
        "grid": {
            "bilateral_noise_min": LEVELS,
            "max_gap_samples": GAPS,
            "min_support_px": LENGTHS,
            "maximum_prior_texture_noise": TEXTURE_CAPS,
            "texture_quantile": TEXTURE_QUANTILE,
            "texture_sigma_px": TEXTURE_SIGMA_PX,
            "texture_window_px": TEXTURE_WINDOW_PX,
        },
        "source_checks": {
            name: {
                key: data[key]
                for key in ("input_sha256", "truth_sha256", "sky_mask_pixels")
            }
            for name, data in seq.items()
        },
        "leave_one_sequence_out": {
            name: {
                method: train_and_test(
                    {other: rows for other, rows in seq.items() if other != name},
                    {name: data},
                    measured,
                    **options,
                )
                for method, options in CALIBRATIONS.items()
            }
            for name, data in seq.items()
        },
        "all_labels": {
            method: train_and_test(seq, seq, measured, **options)
            for method, options in CALIBRATIONS.items()
        },
    }
    (args.output / "background_verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
