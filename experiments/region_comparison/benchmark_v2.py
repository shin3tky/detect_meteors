"""Compare historic detector configurations on the README-labelled RAW sequences."""

import argparse
import hashlib
import json
import platform
from collections import deque
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
import yaml

from benchmark_ridge_beamlet import EXPERIMENT, METHODS, SEQUENCES
from meteor_core.detectors import DetectorRegistry
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.pipeline import _build_runtime_params
from meteor_core.schema import DetectionContext
from segment_verification import (
    THRESHOLDS,
    noise_scaled_pair,
    noise_tiles,
    verify_segments,
)
from summarize import trail_coverage

SEQUENCES = {
    **SEQUENCES,
    "full2024": ("2024GEMINI", "140", 200, 527, "2024_gemini_late_labels.json"),
}
METHODS = {**METHODS, "tophat_lsd": ("lsd", {"preprocess": "tophat"})}


def inputs(name, raw_root):
    directory, prefix, first, last, annotation = SEQUENCES[name]
    source = raw_root / directory
    frames = [f"_C{prefix}{i:03d}.ORF" for i in range(first, last + 1)]
    sums = source / ("SHA256SUMS" if name == "bundled" else "SHA256SUM")
    expected = {
        frame.lstrip("*"): sha
        for sha, frame in map(str.split, sums.read_text().splitlines())
    }
    digest = hashlib.sha256()
    for name in frames:
        with (source / name).open("rb") as frame:
            sha = hashlib.file_digest(frame, "sha256").hexdigest()
        if sha != expected[name]:
            raise ValueError(f"Checksum mismatch for {source / name}")
        digest.update(f"{sha}  {name}\n".encode())
    axes = json.loads((EXPERIMENT / annotation).read_text())
    if directory == "2024GEMINI" and first < 335:
        axes += json.loads((EXPERIMENT / "meteor_labels.json").read_text())
    return source, frames, axes, digest.hexdigest()


def _lines(result, plugin, *, proposals=False):
    if plugin == "hough":
        return result.lines
    if plugin == "region":
        return [
            (*a, *b)
            for row in result.extras["region"]["regions"]
            if proposals or row["accepted"]
            for a, b in row["axis_support"]
        ]
    return [
        row["line_xyxy"]
        for row in result.extras[plugin]["segments"]
        if proposals or row["accepted"]
    ]


def _record(candidate, lines, reference):
    return {
        "candidate": bool(candidate),
        "coverage": {
            item["current"]: trail_coverage(
                item, [(line[:2], line[2:]) for line in lines]
            )
            for item in reference
        },
    }


def detector_for(method, common):
    plugin, settings = METHODS[method]
    if common and plugin in ("lsd", "frt", "region"):
        limit = "max_report_regions" if plugin == "region" else "max_report_segments"
        settings = {**settings, limit: 100_000}
    return DetectorRegistry.create(plugin, settings)


def _summarize_times(pairs, repeats):
    times = [
        sum(records[repeat]["detector_seconds"] for records in pairs.values())
        for repeat in range(repeats)
    ]
    return {
        "times": times,
        "detector_seconds": median(times),
        "verification_seconds": sum(
            records[0]["verification_seconds"] for records in pairs.values()
        ),
        "pairs": {frame: records[0] for frame, records in pairs.items()},
    }


def _bench(
    sequence,
    methods,
    repeats,
    raw_root,
    *,
    median_depth=0,
    common=False,
    noise_scaled=False,
):
    source, frames, labels, input_hash = inputs(sequence, raw_root)
    params = yaml.safe_load(
        Path("config_examples/aircraft_trail_sample.yaml").read_text()
    )["params"]
    detectors = {method: detector_for(method, common) for method in methods}
    runtime = {
        k: _build_runtime_params(params, detector) for k, detector in detectors.items()
    }
    axes = {item["current"]: item for item in labels}
    results = {method: {} for method in methods}
    verifications = {
        policy: {method: {} for method in methods} for policy in THRESHOLDS
    }
    background_total = 0.0
    sigma_total = 0.0
    previous = load_and_bin_raw_fast(str(source / frames[0]))
    roi = np.full_like(previous, 255, dtype=np.uint8)
    history = deque([previous], maxlen=max(1, median_depth))
    for index, name in enumerate(frames[1:], 1):
        current = load_and_bin_raw_fast(str(source / name))
        prior = frames[index - 1]
        reference = [axes[key] for key in (name, prior) if key in axes]
        if len(history) >= median_depth:
            if median_depth:
                started = perf_counter()
                reference_image = temporal_median(history)
                preprocessing = perf_counter() - started
            else:
                reference_image, preprocessing = previous, 0.0
            background_total += preprocessing
            analysis_current, analysis_prior = current, reference_image
            if common or noise_scaled:
                started = perf_counter()
                signed = cv2.subtract(current, reference_image, dtype=cv2.CV_32F)
                if common:
                    tiles = noise_tiles(signed)
                if noise_scaled:
                    analysis_current, analysis_prior = noise_scaled_pair(signed)
                sigma_total += perf_counter() - started
            for repeat in range(repeats):
                shift = (repeat + index) % len(methods)
                for method in methods[shift:] + methods[:shift]:
                    context = DetectionContext(
                        analysis_current,
                        analysis_prior,
                        roi,
                        runtime[method],
                        {"frame_index": index},
                    )
                    started = perf_counter()
                    result = detectors[method].detect(context)
                    elapsed = perf_counter() - started
                    plugin = METHODS[method][0]
                    if plugin == "region" and result.extras["region"][
                        "accepted_regions"
                    ] > len(result.extras["region"]["regions"]):
                        raise ValueError(f"Incomplete accepted region geometry: {name}")
                    row = _record(
                        result.is_candidate,
                        _lines(result, plugin) if reference else (),
                        reference,
                    )
                    if common:
                        extras = result.extras.get(plugin, {})
                        if extras.get("regions_omitted", 0) or extras.get(
                            "segments_omitted", 0
                        ):
                            raise ValueError(
                                f"Incomplete {plugin} proposals for {name}"
                            )
                        proposals = _lines(result, plugin, proposals=True)
                        started = perf_counter()
                        shared = verify_segments(proposals, signed, tiles)
                        verification = perf_counter() - started
                        row["proposals"] = len(proposals)
                        row["proposal_coverage"] = _record(
                            bool(proposals), proposals, reference
                        )["coverage"]
                        for policy, kept in shared.items():
                            verified = _record(bool(kept), kept, reference)
                            old = verifications[policy][method].setdefault(name, [])
                            if old and any(
                                verified[key] != old[0][key] for key in verified
                            ):
                                raise ValueError(
                                    f"Shared decisions changed: {policy}, {method}, {name}"
                                )
                            old.append(
                                {
                                    **verified,
                                    "detector_seconds": elapsed,
                                    "verification_seconds": verification,
                                }
                            )
                    observations = results[method].setdefault(name, [])
                    if observations and any(
                        row[key] != observations[0][key] for key in row
                    ):
                        raise ValueError(
                            f"Decision changed across repeats: {method}, {name}"
                        )
                    observations.append(
                        {
                            **row,
                            "detector_seconds": elapsed,
                            "verification_seconds": 0.0,
                        }
                    )
        history.append(current)
        previous = current
        if index % 25 == 0:
            print(sequence, index, "/", len(frames) - 1, flush=True)
    return {
        "sequence": sequence,
        "input_sha256": input_hash,
        "frames": frames,
        "annotations": labels,
        "median_depth": median_depth,
        "noise_scaled_input": noise_scaled,
        "background_seconds": background_total,
        "sigma_preparation_seconds": sigma_total,
        "results": {
            method: _summarize_times(pairs, repeats)
            for method, pairs in results.items()
        }
        | {
            f"{method}/{policy}": _summarize_times(pairs, repeats)
            for policy, by_method in verifications.items()
            for method, pairs in by_method.items()
            if common
        },
    }


def temporal_median(history):
    """Three-frame causal median with no look-ahead or current-frame leakage."""
    a, b, c = history
    lower, upper = np.minimum(a, b), np.maximum(a, b)
    return np.maximum(lower, np.minimum(upper, c))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    parser.add_argument("--sequences", nargs="+", choices=SEQUENCES, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    alternate = parser.add_mutually_exclusive_group()
    alternate.add_argument("--temporal-median", action="store_true")
    alternate.add_argument("--common-verifier", action="store_true")
    alternate.add_argument("--noise-scaled", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or len(args.methods) != len(set(args.methods)):
        parser.error("Use a positive repeat count and unique methods")
    args.output.mkdir(parents=True)
    (args.output / "environment.json").write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "opencv_threads": cv2.getNumThreads(),
                "methods": {
                    method: asdict(detector_for(method, args.common_verifier).config)
                    for method in args.methods
                },
                "runtime_params": yaml.safe_load(
                    Path("config_examples/aircraft_trail_sample.yaml").read_text()
                )["params"],
                "common_verifier": args.common_verifier,
                "noise_scaled_input": args.noise_scaled,
            },
            indent=2,
        )
        + "\n"
    )
    for name in args.sequences:
        result = _bench(
            name,
            args.methods,
            args.repeats,
            args.raw_root,
            median_depth=3 if args.temporal_median else 0,
            common=args.common_verifier,
            noise_scaled=args.noise_scaled,
        )
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print("Completed", name, flush=True)


if __name__ == "__main__":
    main()
