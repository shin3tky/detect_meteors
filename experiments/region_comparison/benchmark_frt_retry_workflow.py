"""Replay independently calibrated FRT settings through the RAW workflow."""

import argparse
import hashlib
from itertools import pairwise
import json
from math import isclose
from pathlib import Path
from unittest.mock import patch

import cv2

from benchmark_hough_recovery import CONFIG, analysis_mask
from benchmark_star_registration import SKY_ROIS
from benchmark_v2 import _record, inputs
from meteor_core.config_io import load_pipeline_config
from meteor_core.pipeline import MeteorDetectionPipeline
from review_metrics import event_matches, frame_metrics, read_truth

MODE = "validate_gap_relaxation"
REFERENCE = (
    Path(__file__).parent / "results" / "frt_c140521" / "background_verification.json"
)


def run_sequence(sequence, raw_root, work, source_checks, fold, workers):
    source, frames, labels, digest = inputs(sequence, raw_root)
    readme = source / "README.md"
    truth_digest = hashlib.sha256(readme.read_bytes()).hexdigest()
    if (digest, truth_digest) != (
        source_checks["input_sha256"],
        source_checks["truth_sha256"],
    ):
        raise ValueError(f"RAW or label checksums changed for {sequence}")
    checksums = source / ("SHA256SUMS" if sequence == "bundled" else "SHA256SUM")
    inventory = {
        line.split()[1].lstrip("*") for line in checksums.read_text().splitlines()
    }
    truth = read_truth(readme, frames[1:], inventory)
    if set(fold["training"]) != set(SKY_ROIS) - {sequence}:
        raise ValueError(f"FRT settings for {sequence} include the evaluation sequence")

    input_dir = work / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        link = input_dir / frame
        if not link.is_symlink():
            link.symlink_to(source.resolve() / frame)
        if link.resolve() != source.resolve() / frame:
            raise ValueError(f"Unexpected benchmark input {link}")

    config = load_pipeline_config(CONFIG)
    config.target_folder = str(input_dir)
    config.output_folder = str(work / "candidates")
    config.debug_folder = str(work / "debug")
    config.progress_file = str(work / "progress.json")
    config.detector_config = {
        "difference_mode": "positive",
        "frt_retry": fold["settings"],
    }
    config.hooks = []
    config.roi_polygon = None
    config.num_workers = workers
    config.enable_parallel = workers > 1
    pipeline = MeteorDetectionPipeline(config)

    def select_roi(image, **_kwargs):
        # The 2021 ROI was rasterized and then eroded. Pass the exact mask used
        # in calibration to the pipeline through the ROI selector.
        mask = analysis_mask(image.shape, sequence)
        if cv2.countNonZero(mask) != source_checks["sky_mask_pixels"]:
            raise ValueError(f"ROI changed for {sequence}")
        return {"polygon": SKY_ROIS[sequence], "mask": mask}

    with patch("meteor_core.pipeline.select_roi", select_roi):
        count = pipeline.run(enable_roi_selection=True, resume=False)

    detections = pipeline._all_sorted_detections
    if [
        (row.frame_index, row.prev_frame_index, row.filename) for row in detections
    ] != [(index, index - 1, frame) for index, frame in enumerate(frames[1:], 1)]:
        raise ValueError(f"Workflow did not process the expected pairs for {sequence}")
    axes = {label["current"]: label for label in labels}
    pairs = {}
    candidates = {}
    retry_count = 0
    for (prior, frame), detection in zip(pairwise(frames), detections, strict=True):
        retry = detection.extras.get("frt_retry")
        retry_count += retry is not None
        lines = (
            [row["line_xyxy"] for row in retry["intervals"]]
            if retry is not None
            else detection.lines
        )
        pairs[frame] = _record(
            detection.is_candidate,
            lines,
            [axes[name] for name in (frame, prior) if name in axes],
        )
        if detection.is_candidate:
            candidates[frame] = {
                "stage": "frt_retry" if retry is not None else "hough",
                "score": detection.score,
                "line_count": len(lines),
            }

    results = {
        "frames": frame_metrics(pairs, truth),
        "events": event_matches(pairs, frames, labels),
    }
    reference = fold["evaluation"][sequence]
    if results["frames"] != reference["frames"] or count != len(candidates):
        raise ValueError(f"Workflow decisions differ from the reference for {sequence}")
    for name, event in results["events"].items():
        for stage in ("appearance", "following"):
            match, expected = event[stage], reference["events"][name][stage]
            if (match is None) != (expected is None) or (
                match is not None
                and (
                    match["match"] != expected["match"]
                    or not isclose(
                        match["coverage"], expected["coverage"], abs_tol=1e-12
                    )
                )
            ):
                raise ValueError(f"Workflow axis differs for {name} ({stage})")
    return {
        "training_sequences": list(fold["training"]),
        "settings": fold["settings"],
        **source_checks,
        "pairs": len(pairs),
        "frt_retries": retry_count,
        "candidates": candidates,
        **results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    parser.add_argument("--calibration", type=Path, default=REFERENCE)
    parser.add_argument("--workers", type=int, choices=range(1, 4), default=2)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    calibration = json.loads(args.calibration.read_text())
    results = {
        sequence: run_sequence(
            sequence,
            args.raw_root,
            args.output / sequence,
            calibration["source_checks"][sequence],
            calibration["leave_one_sequence_out"][sequence][MODE],
            args.workers,
        )
        for sequence in SKY_ROIS
    }
    (args.output / "frt_retry_workflow.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
