"""Measure three RAW backgrounds using a fixed FRT and shared false-positive budgets."""

import argparse
import hashlib
import json
import platform
from collections import deque
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from benchmark_star_registration import MARGIN, SKY_ROIS, _contrast
from benchmark_v2 import detector_for, inputs, temporal_median
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.pipeline import _build_runtime_params
from meteor_core.roi_selector import create_roi_mask_from_polygon
from meteor_core.schema import DetectionContext
from registered_median import HISTORY_DEPTH, registered_median
from review_frt import _crop, _gray, _overlay
from review_metrics import event_matches, frame_metrics, read_truth
from meteor_core.star_registration import star_centroids
from summarize import trail_coverage

POLICIES = ("adjacent", "median", "registered_median")
RATES = (0.05, 0.10, 0.20, 0.50)
CASE_FRAMES = {"_C140344.ORF", "_C140521.ORF"}
CALIBRATION = range(200, 241)
SEQUENCES = ("bundled", "gemini2021", "late2024")


def _load_calibration(raw_root):
    source = raw_root / "2024GEMINI"
    frames = [f"_C140{i:03d}.ORF" for i in CALIBRATION]
    manifest = dict(
        (name.lstrip("*"), sha)
        for sha, name in map(str.split, (source / "SHA256SUM").read_text().splitlines())
    )
    digest = hashlib.sha256()
    for name in frames:
        with (source / name).open("rb") as raw:
            sha = hashlib.file_digest(raw, "sha256").hexdigest()
        if sha != manifest[name]:
            raise ValueError(f"Unexpected checksum for {name}")
        digest.update(f"{sha}  {name}\n".encode())
    return source, frames, [], digest.hexdigest()


def _figures(out, current, backgrounds, axes, name):
    if not axes:
        return
    label = axes[0]
    trail = label["current"]
    if trail not in CASE_FRAMES:
        return
    crop, origin, side = _crop(label, current.shape)
    appear = name == trail
    status = "appearance" if appear else "following"
    canvas = Image.new("RGB", (950, 674), "white")
    draw = ImageDraw.Draw(canvas)
    text = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    draw.text(
        (20, 13),
        f"{name} | {status} pair | background and signed residual",
        font=text,
        fill="#19304b",
    )
    controls = (
        "Previous exposure",
        "Median of previous three",
        "Translation-aligned median",
    )
    low, high = np.percentile(current[crop], [5, 99.8])
    for i, key in enumerate(POLICIES):
        x = i * 310 + 20
        background = backgrounds[key][crop]
        draw.text((x, 47), controls[i], font=small, fill="#19304b")
        canvas.paste(_gray(background, low, high, (285, 285)), (x, 70))
        residual = cv2.subtract(current[crop], background, dtype=cv2.CV_32F)
        panel = _gray(residual, -18, 26, (285, 285))
        _overlay(panel, label, origin, 285 / side)
        canvas.paste(panel, (x, 365))
    draw.text(
        (20, 652),
        "Identical scales by row. Signed residual: -18 to 26 DN. Yellow: reference corridor.",
        font=small,
        fill="#19304b",
    )
    canvas.save(
        out
        / f"registered-median-{trail.removesuffix('.ORF').lstrip('_').lower()}-{status}.png"
    )


def _scores(result, labels):
    extras = result.extras["frt"]
    # accepted includes the original 8-point floor and common non-score gates.
    segments = [s for s in extras["segments"] if s["accepted"]]
    if len(segments) != extras["accepted_segments"]:
        raise ValueError("Candidate geometry was truncated before calibration")
    relevant = {
        label["current"]: [
            (s["response"], s["line_xyxy"])
            for s in segments
            if trail_coverage(label, [(s["line_xyxy"][:2], s["line_xyxy"][2:])]) > 0
        ]
        for label in labels
    }
    return {
        "score": max((s["response"] for s in segments), default=0.0),
        "relevant_lines": relevant,
    }


def collect(sequence, raw_root, output):
    calibration = sequence == "calibration"
    source, frames, labels, digest = (
        _load_calibration(raw_root) if calibration else inputs(sequence, raw_root)
    )
    checksums = source / ("SHA256SUMS" if sequence == "bundled" else "SHA256SUM")
    inventory = {
        path.lstrip("*")
        for _, path in map(str.split, checksums.read_text().splitlines())
    }
    readme = source / "README.md"
    truth = read_truth(readme, frames, inventory)
    hough_params = yaml.safe_load(
        Path("config_examples/aircraft_trail_sample.yaml").read_text()
    )["params"]
    detector = detector_for("frt", False)
    runtime = _build_runtime_params(hough_params, detector)
    polygon = SKY_ROIS["late2024" if calibration else sequence]
    previous = load_and_bin_raw_fast(str(source / frames[0]))
    roi = cv2.erode(
        create_roi_mask_from_polygon(polygon, previous.shape),
        np.ones((MARGIN * 2 + 1, MARGIN * 2 + 1), np.uint8),
    )
    mask = roi != 0
    history = deque([previous], maxlen=HISTORY_DEPTH)
    start = perf_counter()
    catalogues = deque([star_centroids(previous, mask)], maxlen=HISTORY_DEPTH)
    first_frame_seconds = perf_counter() - start
    axes = {label["current"]: label for label in labels}
    records = {}
    for i, name in enumerate(frames[1:], 1):
        current = load_and_bin_raw_fast(str(source / name))
        start = perf_counter()
        stars = star_centroids(current, mask)
        catalog_seconds = perf_counter() - start
        warmup = len(history) < HISTORY_DEPTH
        # Calibration uses fully populated, clean four-exposure windows. The
        # benchmark uses every pair and reports warm-up separately.
        eligible = not calibration or (
            not warmup and not any(truth[f] for f in frames[i - HISTORY_DEPTH : i + 1])
        )
        if eligible:
            start = perf_counter()
            median = temporal_median(history) if not warmup else history[-1]
            median_seconds = perf_counter() - start
            start = perf_counter()
            registered, registration = (
                registered_median(history, catalogues, stars, max_shift=MARGIN - 2)
                if not warmup
                else (history[-1], None)
            )
            registration_seconds = perf_counter() - start
            backgrounds = dict(
                zip(POLICIES, (history[-1], median, registered), strict=True)
            )
            preprocessing = dict(
                zip(
                    POLICIES,
                    (0.0, median_seconds, catalog_seconds + registration_seconds),
                    strict=True,
                )
            )
            active = [
                axes[f] for f in frames[max(0, i - HISTORY_DEPTH) : i + 1] if f in axes
            ]
            row = {
                "warmup": warmup,
                "registration": registration,
                "policies": {},
            }
            order = POLICIES[i % len(POLICIES) :] + POLICIES[: i % len(POLICIES)]
            for policy in order:
                background = backgrounds[policy]
                context = DetectionContext(
                    current, background, roi, runtime, {"frame_index": i}
                )
                start = perf_counter()
                result = detector.detect(context)
                detect_seconds = perf_counter() - start
                residual = cv2.subtract(current, background, dtype=cv2.CV_32F)
                row["policies"][policy] = {
                    **_scores(result, active),
                    "detector_seconds": detect_seconds,
                    "background_seconds": preprocessing[policy],
                    "abs_residual_p99_dn": float(
                        np.percentile(np.abs(residual[::4, ::4][mask[::4, ::4]]), 99)
                    ),
                    "appearance_contrast_dn": _contrast(axes[name], residual, mask)
                    if name in axes
                    else None,
                }
            _figures(
                output,
                current,
                backgrounds,
                [axes[f] for f in (name, frames[i - 1]) if f in axes],
                name,
            )
            records[name] = row
        history.append(current)
        catalogues.append(stars)
        if i % 15 == 0:
            print(sequence, i, "/", len(frames) - 1, flush=True)
    return {
        "frames": frames,
        "annotations": labels,
        "input_sha256": digest,
        "labels_sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
        "truth": truth,
        "sky_polygon": polygon,
        "first_catalogue_seconds": first_frame_seconds,
        "records": records,
    }


def cutoff(scores, rate):
    """Strict > cutoff with conservative treatment of tied scores at the boundary."""
    ordered = sorted(scores, reverse=True)
    if not ordered or not 0 <= rate < 1:
        raise ValueError("A budget requires negative scores and a rate below one")
    return max(0.0, ordered[int(rate * len(ordered))])


def score_sequence(sequence, policy, threshold):
    labels = {label["current"]: label for label in sequence["annotations"]}
    records = {}
    for current, item in sequence["records"].items():
        row = item["policies"][policy]
        covered = {
            name: trail_coverage(
                labels[name],
                [(xy[:2], xy[2:]) for score, xy in lines if score > threshold],
            )
            for name, lines in row["relevant_lines"].items()
        }
        records[current] = {"candidate": row["score"] > threshold, "coverage": covered}
    mature = {
        name: value
        for name, value in records.items()
        if not sequence["records"][name]["warmup"]
    }
    follow = event_matches(records, sequence["frames"], sequence["annotations"])
    # Check all three following pairs for any displaced or repeated trail.
    persistence = {}
    for name, label in labels.items():
        i = sequence["frames"].index(name)
        nearby = sequence["frames"][i + 1 : i + 4]
        persistence[name] = {
            frame: records[frame]["coverage"][label["current"]]
            if records[frame]["candidate"]
            else 0.0
            for frame in nearby
            if frame in records
        }
    return {
        "threshold_exclusive": threshold,
        "all_pairs": frame_metrics(records, sequence["truth"]),
        "mature_pairs": frame_metrics(mature, sequence["truth"]),
        "events": follow,
        "following_three_coverage": persistence,
    }


def evaluate(summary):
    cal = summary["sequences"]["calibration"]
    testing = {k: v for k, v in summary["sequences"].items() if k != "calibration"}
    evaluation = {
        "calibrated": {},
        "matched_on_test": {},
        "matched_on_mature_test": {},
        "native": {},
    }
    for policy in POLICIES:
        negative_scores = [
            r["policies"][policy]["score"] for r in cal["records"].values()
        ]
        calibrated = evaluation["calibrated"][policy] = {}
        for rate in RATES:
            threshold = cutoff(negative_scores, rate)
            calibrated[str(rate)] = {
                "training_negative_count": len(negative_scores),
                "training_fp": sum(score > threshold for score in negative_scores),
                "test": {
                    name: score_sequence(data, policy, threshold)
                    for name, data in testing.items()
                },
            }
        for group, mature_only in (
            ("matched_on_test", False),
            ("matched_on_mature_test", True),
        ):
            test_negatives = {
                name: [
                    r["policies"][policy]["score"]
                    for frame, r in data["records"].items()
                    if not data["truth"][frame] and not (mature_only and r["warmup"])
                ]
                for name, data in testing.items()
            }
            evaluation[group][policy] = {
                str(rate): {
                    name: score_sequence(
                        data, policy, cutoff(test_negatives[name], rate)
                    )
                    for name, data in testing.items()
                }
                for rate in RATES
            }
        evaluation["native"][policy] = {
            name: score_sequence(data, policy, 0.0) for name, data in testing.items()
        }
    return evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "threads": cv2.getNumThreads(),
        "numpy": np.__version__,
        "sequences": {},
    }
    for sequence in ("calibration", *SEQUENCES):
        result["sequences"][sequence] = collect(sequence, args.raw_root, args.output)
        (args.output / f"{sequence}.json").write_text(
            json.dumps(result["sequences"][sequence], indent=2) + "\n"
        )
        print("Completed", sequence, flush=True)
    result["evaluation"] = evaluate(result)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
