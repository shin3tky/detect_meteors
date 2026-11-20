"""Run a RAW-pair detector comparison from the repository root with an installed wheel."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
import yaml

from summarize import (
    line_trail_evidence,
    load_pairs,
    region_trail_evidence,
    trail_coverage,
)

SEQUENCES = {
    "bundled": ("2024GEMINI_AIRCRAFT", "140", 335, 346, "meteor_labels.json"),
    "gemini2021": ("2021GEMINI", "150", 295, 330, "2021_gemini_labels.json"),
    "late2024": ("2024GEMINI", "140", 452, 527, "2024_gemini_late_labels.json"),
}
METHODS = {
    "hough": ("hough", {}),
    "region": (
        "region",
        {
            "enable_ransac": True,
            "ransac_max_regions": 64,
            "ransac_min_contrast": 9,
            "ransac_min_contiguous_support_px": 45,
            "ransac_priority": "area_contrast_aspect",
            "reject_negative_axis_fraction": 0.75,
            "min_positive_axis_fraction": 0.75,
        },
    ),
    "dog_lsd": ("lsd", {"preprocess": "dog"}),
    "ridge": ("ridge", {}),
    "beamlet": ("beamlet", {}),
    "radon": ("radon", {}),
    "frt": ("frt", {}),
}
DEFAULT_METHODS = ["hough", "region", "dog_lsd", "ridge", "beamlet"]
EXPERIMENT = Path(__file__).resolve().parent


def _inputs(output, settings):
    directory, prefix, first, last, label_file = settings
    source = Path("rawfiles", directory).resolve()
    frames = [f"_C{prefix}{i:03d}.ORF" for i in range(first, last + 1)]
    digest = hashlib.sha256()
    data = output / "raw"
    data.mkdir()
    for frame in frames:
        path = source / frame
        with path.open("rb") as raw:
            sha = hashlib.file_digest(raw, "sha256").hexdigest()
        digest.update(f"{sha}  {frame}\n".encode())
        (data / frame).symlink_to(path)
    labels = json.loads((EXPERIMENT / label_file).read_text())
    return data, frames, labels, digest.hexdigest()


def _trail(label, row, detector):
    if detector == "hough":
        coverage = trail_coverage(label, [(x[:2], x[2:]) for x in row["lines"]])
        evidence = {
            "candidate": row["is_candidate"],
            "reported_coverage": coverage,
            "accepted_coverage": coverage,
            "omitted_count": 0,
        }
    elif detector == "region":
        evidence = region_trail_evidence(label, row)
    else:
        evidence = line_trail_evidence(label, row, detector)
        evidence["overlapping_proposals"] = [
            {**segment, "coverage": coverage}
            for segment in row["extras"][detector]["segments"]
            if (
                coverage := trail_coverage(
                    label, [(segment["line_xyxy"][:2], segment["line_xyxy"][2:])]
                )
            )
            > 0
        ]
    return {
        **evidence,
        "match": evidence["candidate"] and evidence["accepted_coverage"] >= 0.5,
    }


def _threshold_sweep(rows, labels, detector):
    if detector not in ("ridge", "beamlet"):
        return {}
    result = {}
    for level in (3, 6, 9, 15, 20):
        # Raising contrast can only remove accepted proposals. All accepted
        # proposals are in the report, regardless of the diagnostic limit.
        selected = {}
        for name, row in rows.items():
            selected[name] = [
                (s["line_xyxy"][:2], s["line_xyxy"][2:])
                for s in row["extras"][detector]["segments"]
                if s["accepted"] and s["contrast"] >= level
            ]
        result[str(level)] = {
            "candidates": sum(bool(lines) for lines in selected.values()),
            "trails": {
                label["current"]: trail_coverage(label, selected[label["current"]])
                for label in labels
            },
        }
    return result


def _frt_sweep(rows, labels):
    """Raise the score gate without moving proposals or changing other gates."""
    result = {}
    for threshold in (8, 10, 12, 16, 20, 30):
        selected = {}
        for name, row in rows.items():
            selected[name] = [
                (segment["line_xyxy"][:2], segment["line_xyxy"][2:])
                for segment in row["extras"]["frt"]["segments"]
                if segment["accepted"] and segment["response"] >= threshold
            ]
        appear = {
            label["current"]: trail_coverage(label, selected[label["current"]])
            for label in labels
        }
        frame_names = list(rows)
        following = {}
        for label in labels:
            index = frame_names.index(label["current"]) + 1
            if index < len(frame_names):
                following[label["current"]] = trail_coverage(
                    label, selected[frame_names[index]]
                )
        result[str(threshold)] = {
            "candidates": sum(bool(lines) for lines in selected.values()),
            "appearance_coverage": appear,
            "following_coverage": following,
        }
    return result


def _run(output, method, data, frames, labels, base):
    output.mkdir()
    detector, detector_config = METHODS[method]
    paths = {
        "output_folder": str(output / "candidates"),
        "debug_folder": str(output / "debug"),
        "report_path": str(output / "pairs.jsonl"),
    }
    config = {
        **base,
        "target_folder": str(data),
        "detector_name": detector,
        "detector_config": detector_config,
        "output_handler_name": "pair_report",
        "output_handler_config": paths,
        "output_folder": paths["output_folder"],
        "debug_folder": paths["debug_folder"],
        "progress_file": str(output / "progress.json"),
        "hooks": [],
        "enable_parallel": False,
        "num_workers": 1,
        "enable_roi_selection": False,
    }
    config_path = output / "pipeline.json"
    config_path.write_text(json.dumps(config, indent=2))
    command = [
        str(Path(sys.executable).with_name("detect-meteors")),
        "--config",
        str(config_path),
        "--no-resume",
        "--no-roi",
        "--no-debug-image",
        "--profile",
    ]
    started = perf_counter()
    with (output / "cli.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = perf_counter() - started
    rows = load_pairs(Path(paths["report_path"]), frames)
    candidates = [name for name, row in rows.items() if row["is_candidate"]]
    saved = sorted(path.name for path in Path(paths["output_folder"]).glob("*.ORF"))
    if saved != candidates:
        raise ValueError(f"Candidate files differ from the per-pair report in {output}")
    trails = {}
    events = {}
    for label in labels:
        row = rows[label["current"]]
        if frames[row["prev_frame_index"]] != label["previous"]:
            raise ValueError(f"Unexpected previous frame for {label['current']}")
        trails[label["current"]] = _trail(label, row, detector)
        index = row["frame_index"] + 1
        following = frames[index] if index < len(frames) else None
        next_pair = _trail(label, rows[following], detector) if following else None
        events[label["current"]] = {
            "following_frame": following,
            "following": next_pair,
            "event_match": trails[label["current"]]["match"]
            or (next_pair is not None and next_pair["match"]),
        }
    records = [
        {
            "current": name,
            "candidate": row["is_candidate"],
            "detector_ms": row["metrics"]["duration_ms"],
            **{
                key: value
                for key, value in row["extras"].get(detector, {}).items()
                if key
                in (
                    "proposals",
                    "block_pruned",
                    "proposals_omitted",
                    "segments_omitted",
                    "accepted_segments",
                    "projection_cells",
                )
            },
        }
        for name, row in rows.items()
    ]
    result = {
        "candidate_frames": candidates,
        "trails": trails,
        "events": events,
        "detector_seconds": sum(r["detector_ms"] for r in records) / 1000,
        "wall_seconds": elapsed,
        "pairs": records,
        "contrast_sweep_dn": _threshold_sweep(rows, labels, detector),
    }
    if detector == "frt":
        result["snr_sweep"] = _frt_sweep(rows, labels)
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        output.parent.name,
        output.name,
        f"{len(candidates)}/{len(rows)} candidates",
        f"{result['detector_seconds']:.3f} s detector",
        f"{elapsed:.3f} s CLI",
        {name: round(t["accepted_coverage"], 3) for name, t in trails.items()},
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="new directory for reports and RAW copies"
    )
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, default=DEFAULT_METHODS
    )
    parser.add_argument(
        "--sequences", nargs="+", choices=SEQUENCES, default=list(SEQUENCES)
    )
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    output = args.output.resolve()
    output.mkdir()
    base = yaml.safe_load(
        Path("config_examples/aircraft_trail_sample.yaml").read_text()
    )
    summary = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "hough_parameters": base["params"],
        "methods": {method: METHODS[method] for method in args.methods},
        "sequences": {},
    }
    for seq_index, name in enumerate(args.sequences):
        sequence = output / name
        sequence.mkdir()
        data, frames, labels, checksum = _inputs(sequence, SEQUENCES[name])
        results = {method: [] for method in args.methods}
        for repeat in range(args.repeats):
            shift = (repeat + seq_index) % len(args.methods)
            order = args.methods[shift:] + args.methods[:shift]
            for method in order:
                run = _run(
                    sequence / f"{method}-{repeat + 1}",
                    method,
                    data,
                    frames,
                    labels,
                    base,
                )
                previous_runs = results[method]
                if previous_runs and any(
                    run[key] != previous_runs[0][key]
                    for key in (
                        "candidate_frames",
                        "trails",
                        "events",
                        "contrast_sweep_dn",
                        "snr_sweep",
                    )
                    if key in run
                ):
                    raise ValueError(
                        f"{name} {method} decisions changed across repeats"
                    )
                previous_runs.append(run)
        summary["sequences"][name] = {
            "input_sha256": checksum,
            "pair_count": len(frames) - 1,
            "labels": labels,
            "results": {
                method: {
                    **runs[0],
                    "detector_seconds": median(r["detector_seconds"] for r in runs),
                    "wall_seconds": median(r["wall_seconds"] for r in runs),
                    "times": [[r["detector_seconds"], r["wall_seconds"]] for r in runs],
                }
                for method, runs in results.items()
            },
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
