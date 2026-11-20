"""Inspect annotated RAW pairs and render figures from the ridge/beamlet reports."""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from beamlet_detector import BeamletConfig, _proposals
from benchmark_ridge_beamlet import SEQUENCES
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.schema import DetectionContext
from ridge_detector import RidgeConfig, RidgeDetector
from summarize import trail_coverage
from trail_measurement import difference_of_gaussians, measure_segment

IMAGE_SIZE = 248
FOCUS = ("_C140344.ORF", "_C150319.ORF", "_C150326.ORF", "_C140521.ORF")


def _pair(sequence, label):
    raw = Path("rawfiles", SEQUENCES[sequence][0])
    return [
        load_and_bin_raw_fast(str(raw / label[key])) for key in ("previous", "current")
    ]


def _matching_proposals(label, response, config):
    """Inspect all beams after tile ranking and before the measurement budget."""
    matched = []
    per_block = Counter()
    beams = sorted(_proposals(response, config), key=lambda beam: beam[1], reverse=True)
    for line, score, size in beams:
        x, y = (line[:2] + line[2:]) // 512
        block = (size, int(x), int(y))
        per_block[block] += 1
        coverage = trail_coverage(label, [(line[:2], line[2:])])
        if coverage > 0:
            matched.append(
                {
                    "line_xyxy": line.tolist(),
                    "response": score,
                    "tile_size_px": size,
                    "block_rank": per_block[block],
                    "coverage": coverage,
                }
            )
    return matched


def _coverage(label, segments):
    return trail_coverage(
        label, [(s["line_xyxy"][:2], s["line_xyxy"][2:]) for s in segments]
    )


def inspect_pair(sequence, label):
    previous, current = _pair(sequence, label)
    roi = np.ones_like(previous, dtype=bool)
    ridge = RidgeDetector(RidgeConfig(max_report_segments=100_000)).detect(
        DetectionContext(current, previous, roi, {}, {})
    )
    ridge_report = ridge.extras["ridge"]
    if ridge_report["segments_omitted"]:
        raise ValueError("Ridge diagnostic requires a higher report limit")
    ridge_measurements = [
        {**segment, "coverage": fraction}
        for segment in ridge_report["segments"]
        if (fraction := _coverage(label, [segment])) > 0
    ]

    signed = cv2.subtract(current, previous, dtype=cv2.CV_32F)
    response = difference_of_gaussians(signed)
    config = BeamletConfig()
    beam_candidates = _matching_proposals(
        label, response, replace(config, min_response=0.01)
    )
    nominal = [
        beam for beam in beam_candidates if beam["response"] >= config.min_response
    ]
    for beam in nominal:
        measurement = measure_segment(
            beam["line_xyxy"],
            signed,
            roi,
            min_length_px=config.min_length_px,
            min_contrast=config.min_contrast,
            background_offset_px=6,
        )
        beam.update(measurement)
    return {
        "ridge": {
            "candidate": ridge.is_candidate,
            "measured_coverage": _coverage(label, ridge_measurements),
            "accepted_coverage": _coverage(
                label,
                [segment for segment in ridge_measurements if segment["accepted"]],
            ),
            "overlapping_proposals": ridge_measurements,
        },
        "beamlet": {
            "before_budget_coverage": _coverage(label, nominal),
            "before_budget_passing_coverage": _coverage(
                label, [beam for beam in nominal if beam["accepted"]]
            ),
            "before_budget_overlaps": nominal,
            "before_score_and_budget_coverage": _coverage(label, beam_candidates),
            "largest_overlap_before_score_and_budget": max(
                beam_candidates, key=lambda item: item["coverage"], default=None
            ),
        },
    }


def _crop(sequence, label):
    previous, current = _pair(sequence, label)
    start, end = np.array(label["axis"], dtype=np.float64)
    center = (start + end) / 2
    side = max(80, int(np.max(np.abs(end - start))) + 56)
    upper = np.array((current.shape[1], current.shape[0])) - side
    left, top = np.clip(np.rint(center - side / 2).astype(int), 0, upper)
    slices = np.s_[top : top + side, left : left + side]
    signed = cv2.subtract(current[slices], previous[slices], dtype=cv2.CV_32F)
    # Use one display scale for all panels and frames. The pixel values in
    # the experiment and geometry are computed before this display mapping.
    image = Image.fromarray(np.uint8(np.clip((signed + 6) / 36, 0, 1) * 255))
    return (
        image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST).convert("RGB"),
        np.array((left, top)),
        IMAGE_SIZE / side,
    )


def _overlay(image, origin, scale, label, segments, passing_color):
    draw = ImageDraw.Draw(image)

    def xy(point):
        return tuple((np.asarray(point) - origin) * scale)

    a, b = np.array(label["axis"], dtype=np.float64)
    dx, dy = b - a
    normal = np.array((-dy, dx)) * label["half_width_px"] / np.hypot(dx, dy)
    for side in (normal, -normal):
        draw.line((xy(a + side), xy(b + side)), fill="#ffdc60", width=2)
    # Show the three longest overlaps of each status so the RAW stays legible.
    for accepted, color in ((False, "#ff84bf"), (True, passing_color)):
        ordered = sorted(
            (s for s in segments if s["accepted"] == accepted),
            key=lambda item: item["coverage"],
            reverse=True,
        )
        for segment in ordered[:3]:
            line = segment["line_xyxy"]
            draw.line((xy(line[:2]), xy(line[2:])), fill=color, width=2)


def render(summary, diagnostics, path):
    rows = [
        (name, label, series["results"])
        for name, series in summary["sequences"].items()
        for label in series["labels"]
        if label["current"] in FOCUS
    ]
    canvas = Image.new("RGB", (1082, 66 + 302 * len(rows)), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    heading = ImageFont.load_default(size=20)
    text = ImageFont.load_default(size=16)
    ink = "#152538"
    columns = (
        "Signed RAW difference",
        "Hessian ridge",
        "Beamlet",
        "Beamlet before budget",
    )
    for col, name in enumerate(columns):
        draw.text((16 + col * 266, 13), name, fill=ink, font=heading)
    draw.text(
        (16, 40),
        "Yellow: annotated corridor  /  Blue: accepted  /  Pink: rejected  /  Green: passes contrast before budget",
        font=text,
        fill=ink,
    )
    for row, (sequence, label, methods) in enumerate(rows):
        y = 66 + row * 302
        base, origin, scale = _crop(sequence, label)
        frame = label["current"]
        for col, method in enumerate((None, "ridge", "beamlet")):
            image = base.copy()
            evidence = methods[method]["trails"][frame] if method else None
            segments = evidence["overlapping_proposals"] if evidence else []
            _overlay(image, origin, scale, label, segments, "#70ddff")
            x = 16 + col * 266
            canvas.paste(image, (x, y))
            caption = frame
            if evidence:
                caption = (
                    f"Before {evidence['reported_coverage']:.1%} / "
                    f"After {evidence['accepted_coverage']:.1%}"
                )
            draw.text((x, y + 254), caption, fill=ink, font=text)
        beamlet = diagnostics[frame]["beamlet"]
        passing = [
            beam for beam in beamlet["before_budget_overlaps"] if beam["accepted"]
        ]
        image = base.copy()
        _overlay(image, origin, scale, label, passing, "#79e6a1")
        x = 16 + 3 * 266
        canvas.paste(image, (x, y))
        caption = f"Passing coverage {beamlet['before_budget_passing_coverage']:.1%}"
        draw.text((x, y + 254), caption, fill=ink, font=text)
    canvas.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {}
    for sequence, series in summary["sequences"].items():
        for label in series["labels"]:
            frame = label["current"]
            diagnostic = inspect_pair(sequence, label)
            expected = series["results"]["ridge"]["trails"][frame]
            observed = diagnostic["ridge"]
            if (
                observed["candidate"] != expected["candidate"]
                or observed["accepted_coverage"] != expected["accepted_coverage"]
            ):
                raise ValueError(
                    f"Expanded ridge report changed the result for {frame}"
                )
            measured = series["results"]["beamlet"]["trails"][frame][
                "reported_coverage"
            ]
            if diagnostic["beamlet"]["before_budget_coverage"] + 1e-9 < measured:
                raise ValueError(
                    f"Unpruned beamlets do not cover measured beams for {frame}"
                )
            evidence[frame] = diagnostic
    (args.output / "diagnostics.json").write_text(json.dumps(evidence, indent=2) + "\n")
    render(summary, evidence, args.output / "ridge-beamlet-raw.png")


if __name__ == "__main__":
    main()
