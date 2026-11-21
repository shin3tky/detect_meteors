"""Inspect the additional candidate in the stellar registration workflow."""

import argparse
from dataclasses import asdict, replace
import json
from math import ceil, hypot
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_hough_registration import CONFIG, POLICIES
from benchmark_star_registration import _contrast
from benchmark_v2 import inputs
from meteor_core.config_io import load_pipeline_config
from meteor_core.detectors.hough_default import HoughDetector
from meteor_core.hooks.star_registration import StarRegistrationHook
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.pipeline import _build_runtime_params
from meteor_core.roi_selector import create_roi_mask_from_polygon
from meteor_core.schema import DetectionContext
from review_frt import _crop, _gray, _overlay
from summarize import trail_coverage

FOCUS = "_C140345.ORF"


def score_sign(detector, context, difference):
    evidence = detector.analyze(
        replace(
            context, current_image=difference, previous_image=np.zeros_like(difference)
        )
    )
    decision = evidence.evaluate()
    return {
        "candidate": decision.is_candidate,
        "score": evidence.score,
        "num_lines": len(evidence.lines),
        "max_aspect_ratio": decision.max_aspect_ratio,
    }


def diagnose(detector, context, label):
    evidence = detector.analyze(context)
    decision = evidence.evaluate()
    signed = cv2.subtract(
        context.current_image, context.previous_image, dtype=cv2.CV_32F
    )
    lines = []
    for x1, y1, x2, y2 in evidence.lines:
        length = hypot(x2 - x1, y2 - y1)
        x, y = np.rint(np.linspace((x1, y1), (x2, y2), ceil(length) + 1)).astype(int).T
        lines.append(
            {
                "xyxy": (x1, y1, x2, y2),
                "length_px": length,
                "negative_fraction_on_line": float(np.mean(signed[y, x] < 0)),
                "previous_meteor_coverage": trail_coverage(
                    label, [((x1, y1), (x2, y2))]
                ),
            }
        )
    strongest_shape = max(
        (c for c in evidence.contours if c.area > evidence.thresholds.min_area),
        key=lambda c: c.aspect_ratio or 0.0,
    )
    row = {
        "candidate": decision.is_candidate,
        "rejection_reasons": decision.rejection_reasons,
        "mask_pixels": cv2.countNonZero(evidence.mask),
        "contour_count": len(evidence.contours),
        "max_aspect_ratio": decision.max_aspect_ratio,
        "highest_ratio_contour": asdict(strongest_shape),
        "accepted_contours": [
            asdict(evidence.contours[i]) for i in decision.accepted_contours
        ],
        "line_score": evidence.score,
        "lines": lines,
        "previous_meteor_coverage": trail_coverage(
            label, [(xyxy[:2], xyxy[2:]) for xyxy in evidence.lines]
        ),
        "previous_meteor_center_minus_sides_dn": _contrast(
            label, signed, context.roi_mask != 0
        ),
        "positive_only": score_sign(
            detector,
            context,
            cv2.subtract(context.current_image, context.previous_image),
        ),
        "negative_only": score_sign(
            detector,
            context,
            cv2.subtract(context.previous_image, context.current_image),
        ),
    }
    return row, signed, evidence


def render(path, context, label, diagnosis):
    crop, origin, side = _crop(label, context.current_image.shape)
    size = 300
    canvas = Image.new("RGB", (984, 792), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    title, heading, note = (ImageFont.load_default(size=pt) for pt in (23, 18, 15))
    ink = "#172639"
    draw.text((18, 16), "_C140344.ORF to _C140345.ORF", font=title, fill=ink)
    draw.text(
        (18, 47),
        "Gold: prior meteor corridor   Cyan: Hough lines   Red: accepted contour box",
        font=note,
        fill=ink,
    )
    draw.text(
        (18, 67),
        "Signed differences use one scale: black -48 DN, gray 0, white +48 DN.",
        font=note,
        fill=ink,
    )
    frames = (context.previous_image[crop], context.current_image[crop])
    low, high = np.percentile(np.concatenate(frames), (5, 99))
    scale = size / side
    for index, (policy, raw, row_title) in enumerate(
        zip(POLICIES, frames, ("Unaligned", "Aligned"), strict=True)
    ):
        row, signed, evidence = diagnosis[policy]
        y = 117 + index * 337
        raw_panel = _gray(raw, low, high, (size, size))
        diff_panel = _gray(signed[crop], -48, 48, (size, size))
        mask_panel = _gray(evidence.mask[crop], 0, 255, (size, size))
        for panel in (raw_panel, diff_panel, mask_panel):
            _overlay(panel, label, origin, scale)
        overlay = ImageDraw.Draw(mask_panel)
        for x1, y1, x2, y2 in evidence.lines:
            if (
                max(x1, x2) < origin[0]
                or min(x1, x2) >= origin[0] + side
                or max(y1, y2) < origin[1]
                or min(y1, y2) >= origin[1] + side
            ):
                continue
            points = (np.array(((x1, y1), (x2, y2))) - origin) * scale
            overlay.line([tuple(point) for point in points], fill="#23c6ff", width=2)
        for contour in row["accepted_contours"]:
            points = (cv2.boxPoints(contour["rect"]) - origin) * scale
            overlay.polygon(
                [tuple(point) for point in points], outline="#ff5151", width=2
            )
        frame_title = (
            "Previous RAW (_C140344)" if index == 0 else "Current RAW (_C140345)"
        )
        status = "candidate" if row["candidate"] else "rejected"
        titles = (
            frame_title,
            f"{row_title} | signed difference",
            f"Mask and lines | {status}",
        )
        for x, panel, label_text in zip(
            (18, 342, 666), (raw_panel, diff_panel, mask_panel), titles, strict=True
        ):
            draw.text((x, y - 23), label_text, font=heading, fill=ink)
            canvas.paste(panel, (x, y))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    source, frames, labels, digest = inputs("bundled", args.raw_root)
    prior = frames[frames.index(FOCUS) - 1]
    label = next(item for item in labels if item["current"] == prior)
    previous, current = (
        load_and_bin_raw_fast(str(source / frame)) for frame in (prior, FOCUS)
    )
    config = load_pipeline_config(CONFIG)
    detector = HoughDetector()
    context = DetectionContext(
        current,
        previous,
        create_roi_mask_from_polygon(config.roi_polygon, current.shape),
        _build_runtime_params(config.params.to_dict(), detector),
        {},
    )
    aligned = StarRegistrationHook().on_before_detection(context)
    results = {
        key: diagnose(detector, pair, label)
        for key, pair in zip(POLICIES, (context, aligned), strict=True)
    }
    report = {
        "input_sha256": digest,
        "current": FOCUS,
        "previous": prior,
        "previous_meteor": label,
        "roi_polygon": config.roi_polygon,
        "settings": config.params.to_dict(),
        "registration": aligned.metadata["star_registration"],
        "diagnosis": {key: row for key, (row, _, _) in results.items()},
    }
    (args.output / "c140345.json").write_text(json.dumps(report, indent=2) + "\n")
    render(args.output / "c140345-hough-registration.png", context, label, results)


if __name__ == "__main__":
    main()
