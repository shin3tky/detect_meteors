"""Render the difference, contours, and lines for three missed meteors."""

import argparse
from dataclasses import asdict
import json
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_hough_recovery import CONFIG, FOCUS, analysis_mask
from benchmark_star_registration import _contrast
from benchmark_v2 import inputs
from meteor_core.config_io import load_pipeline_config
from meteor_core.detectors.hough_analysis import ContourMeasurement
from meteor_core.detectors.hough_default import HoughDetector, HoughDetectorConfig
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.schema import DetectionContext
from review_frt import _crop, _gray, _overlay
from summarize import trail_coverage


def corridor_mask(label, origin, shape):
    a, b = np.asarray(label["axis"], dtype=float)
    dx, dy = b - a
    normal = np.array((-dy, dx)) * label["half_width_px"] / np.hypot(dx, dy)
    polygon = np.rint(
        np.array([a + normal, b + normal, b - normal, a - normal]) - origin
    )
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon.astype(np.int32), 1)
    return mask.astype(bool)


def contours_touching(mask, corridor):
    """Measure components in the crop whose filled contour touches the corridor."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    touched = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        section = corridor[y : y + h, x : x + w]
        if not section.any():
            continue
        region = np.zeros(section.shape, dtype=np.uint8)
        cv2.drawContours(region, [contour - (x, y)], 0, 1, cv2.FILLED)
        if np.any(region[section]):
            touched.append(contour)

    largest = max(touched, key=cv2.contourArea, default=None)
    geometry = None
    if largest is not None:
        measurement = ContourMeasurement.from_contour(largest)
        geometry = {**asdict(measurement), "aspect_ratio": measurement.aspect_ratio}
    return {
        "corridor_pixels": int(np.count_nonzero(mask[corridor])),
        "touching_contours": len(touched),
        "largest_touching_contour": geometry,
    }, largest


def diagnose(label, raw_root, source, sequence, settings):
    previous, current = (
        load_and_bin_raw_fast(str(raw_root / source / label[key]))
        for key in ("previous", "current")
    )
    roi = analysis_mask(current.shape, sequence)
    context = DetectionContext(current, previous, roi, settings, {})
    signed = cv2.subtract(current, previous, dtype=cv2.CV_32F)
    crop, origin, side = _crop(label, current.shape)
    corridor = corridor_mask(label, origin, (side, side))
    a, b = np.asarray(label["axis"])
    length = float(np.linalg.norm(b - a))
    x, y = np.rint(np.linspace(a, b, ceil(length) + 1)).astype(int).T
    if not roi[crop][corridor].all():
        raise ValueError(
            f"Part of the meteor corridor leaves the ROI: {label['current']}"
        )
    before = (
        np.uint8((signed[crop] > settings["diff_threshold"]) & (roi[crop] != 0)) * 255
    )
    before_row, before_contour = contours_touching(before, corridor)
    stages = {
        "length_px": length,
        "axis_positive_fraction": float(np.mean(signed[y, x] > 0)),
        "axis_above_threshold_fraction": float(
            np.mean(signed[y, x] > settings["diff_threshold"])
        ),
        "center_minus_sides_dn": _contrast(label, signed, roi != 0),
        "crop_origin_xy": tuple(map(int, origin)),
        "crop_size_px": side,
        "before_opening": before_row,
    }
    panels = [(signed[crop], None, ()), (before, before_contour, ())]
    detector = HoughDetector(HoughDetectorConfig("positive"))
    for name, kernel in (("square", cv2.MORPH_RECT), ("cross", cv2.MORPH_CROSS)):
        detector._kernel = cv2.getStructuringElement(kernel, (3, 3))
        analysis = detector.analyze(context)
        decision = analysis.evaluate()
        row, largest = contours_touching(analysis.mask[crop], corridor)
        matching = [
            line
            for line in analysis.lines
            if trail_coverage(label, [(line[:2], line[2:])]) > 0
        ]
        stages[name] = {
            **row,
            "line_score": analysis.score,
            "lines": analysis.lines,
            "lines_touching_axis_at_matching_angle": matching,
            "appearance_coverage": trail_coverage(
                label, [(line[:2], line[2:]) for line in analysis.lines]
            ),
            "max_eligible_aspect_ratio": decision.max_aspect_ratio,
            "candidate": decision.is_candidate,
            "rejection_reasons": decision.rejection_reasons,
        }
        panels.append((analysis.mask[crop], largest, analysis.lines))
    return stages, panels


def render(path, rows):
    size = 252
    canvas = Image.new("RGB", (1094, 1000), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    title, heading, note = (ImageFont.load_default(size=pt) for pt in (23, 17, 15))
    ink = "#172639"
    draw.text(
        (18, 14), "Faint meteors in unaligned frame differences", font=title, fill=ink
    )
    draw.text(
        (18, 46),
        "Gold: annotated corridor    Cyan: Hough lines    Pink: largest contour touching the corridor",
        font=note,
        fill=ink,
    )
    headings = (
        "Signed difference (±24 DN)",
        "Threshold > 9 DN",
        "Square opening + Hough",
        "Cross opening + Hough",
    )
    for col, text in enumerate(headings):
        draw.text((18 + 270 * col, 76), text, font=heading, fill=ink)
    for index, (label, stages, panels) in enumerate(rows):
        origin = stages["crop_origin_xy"]
        side = stages["crop_size_px"]
        scale = size / side
        y = 124 + 286 * index
        draw.text((18, y - 22), label["current"], font=heading, fill=ink)
        for col, (pixels, contour, lines) in enumerate(panels):
            low, high = (-24, 24) if col == 0 else (0, 255)
            panel = _gray(pixels, low, high, (size, size))
            _overlay(panel, label, origin, scale)
            annotations = ImageDraw.Draw(panel)
            if contour is not None:
                points = [tuple(point * scale) for point in contour.reshape(-1, 2)]
                annotations.line([*points, points[0]], fill="#ff7eae", width=1)
            for x1, y1, x2, y2 in lines:
                x0, y0 = origin
                visible, p1, p2 = cv2.clipLine(
                    (0, 0, side, side), (x1 - x0, y1 - y0), (x2 - x0, y2 - y0)
                )
                if visible:
                    annotations.line(
                        [tuple(item * scale for item in point) for point in (p1, p2)],
                        fill="#23c6ff",
                        width=2,
                    )
            canvas.paste(panel, (18 + 270 * col, y))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    settings = load_pipeline_config(CONFIG).params.to_dict()
    rows = []
    report = {"settings": settings, "input_hashes": {}, "pairs": {}}
    for sequence in ("gemini2021", "late2024"):
        source, _, labels, digest = inputs(sequence, args.raw_root)
        report["input_hashes"][sequence] = digest
        for label in labels:
            if label["current"] in FOCUS:
                stages, panels = diagnose(
                    label, args.raw_root, source.name, sequence, settings
                )
                rows.append((label, stages, panels))
                report["pairs"][label["current"]] = {"annotation": label, **stages}
    (args.output / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    render(args.output / "positive-hough-missed-meteors.png", rows)


if __name__ == "__main__":
    main()
