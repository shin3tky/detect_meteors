"""Render the RAW review and detector evidence for the annotated 2024 trail."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from meteor_core.detectors import DetectorRegistry
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.schema import DetectionContext
from summarize import load_pairs, trail_coverage

LABELS = Path(__file__).with_name("2024_gemini_late_labels.json")
CONTEXT_CROP = (slice(690, 910), slice(750, 1010))
DETAIL_BOX = (750, 687, 1010, 882)
ANNOTATION = "#ffdb5c"
DETECTOR = "#67dcff"


def render_review(images, frame_numbers, path):
    """Reproduce the seven-frame contact sheet used to review the RAW."""
    patches = [image[CONTEXT_CROP].astype(np.float32) for image in images]
    low, high = np.percentile(patches, [7, 98.2])
    raw_panels, difference_panels = [], []
    for index, patch in enumerate(patches):
        neighbors = patches[max(0, index - 1) : index] + patches[index + 1 : index + 2]
        residual = patch - np.maximum.reduce(neighbors)
        raw = np.clip((patch - low) / (high - low), 0, 1)
        enhanced = np.clip((cv2.GaussianBlur(residual, (0, 0), 0.65) + 10) / 39, 0, 1)
        for data, panels in ((raw, raw_panels), (enhanced, difference_panels)):
            panel = cv2.cvtColor((data * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            cv2.putText(
                panel,
                str(frame_numbers[index]),
                (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (50, 205, 255),
                1,
                cv2.LINE_AA,
            )
            panels.append(panel)
    cv2.imwrite(
        str(path), np.vstack((np.hstack(raw_panels), np.hstack(difference_panels)))
    )


def diagnostic_refit(previous, current, label, reference):
    """Recover the rejected refit from the fixed detector with an expanded report."""
    plugin = DetectorRegistry.get("region")
    before = reference["extras"]["region"]
    ransac = before["ransac"]
    signed = before["signed_difference"]
    config = plugin.ConfigType(
        enable_ransac=True,
        ransac_priority=ransac["priority"],
        ransac_seed=ransac["seed"],
        ransac_max_regions=64,
        ransac_min_contrast=ransac["min_contrast"],
        ransac_min_contiguous_support_px=ransac["min_contiguous_support_px"],
        reject_negative_axis_fraction=signed["negative_fraction_limit"],
        min_positive_axis_fraction=signed["positive_fraction_min"],
        max_report_regions=10000,
    )
    result = plugin(config).detect(
        DetectionContext(current, previous, np.ones_like(current, dtype=bool), {}, {})
    )
    after = result.extras["region"]
    if (
        result.is_candidate != reference["is_candidate"]
        or after["regions_omitted"]
        or after["accepted_regions"] != before["accepted_regions"]
    ):
        raise ValueError("Expanded region report did not reproduce the pair decision")
    areas = [area for area in after["regions"] if area.get("ransac", {}).get("refit")]
    area = max(
        areas,
        key=lambda item: trail_coverage(label, item["ransac"]["refit"]["axis_support"]),
    )
    refit = area["ransac"]["refit"]
    if refit["accepted"] or not trail_coverage(label, refit["axis_support"]):
        raise ValueError("Expected a rejected refit following the annotated trail")
    return refit


def render_evidence(previous, current, following, label, hough, refit, path):
    width, height = 520, 390
    detail_left, detail_top, detail_right, detail_bottom = DETAIL_BOX
    zoom = (slice(detail_top, detail_bottom), slice(detail_left, detail_right))

    raw = cv2.resize(current, (width, height), interpolation=cv2.INTER_AREA)
    low, high = np.percentile(raw, [1, 98])
    overview = Image.fromarray(
        (
            np.clip((raw.astype(np.float32) - low) / (high - low), 0, 1) ** 0.65 * 255
        ).astype(np.uint8)
    ).convert("RGB")
    hough_panel = overview.copy()
    for panel in (overview, hough_panel):
        draw = ImageDraw.Draw(panel)
        if panel is hough_panel:
            for xa, ya, xb, yb in hough["lines"]:
                draw.line(
                    (
                        xa * width / current.shape[1],
                        ya * height / current.shape[0],
                        xb * width / current.shape[1],
                        yb * height / current.shape[0],
                    ),
                    fill=DETECTOR,
                )
        draw.rectangle(
            (
                detail_left * width / current.shape[1],
                detail_top * height / current.shape[0],
                detail_right * width / current.shape[1],
                detail_bottom * height / current.shape[0],
            ),
            outline=ANNOTATION,
            width=2,
        )

    current_crop = current[zoom].astype(np.float32)
    temporal = current_crop - np.maximum(previous[zoom], following[zoom])
    adjacent = current_crop - previous[zoom]
    detail_panels = []
    for residual, limit in ((temporal, 39), (adjacent, 44)):
        values = np.clip((cv2.GaussianBlur(residual, (0, 0), 0.65) + 10) / limit, 0, 1)
        panel = (
            Image.fromarray((values * 255).astype(np.uint8))
            .resize((width, height), Image.Resampling.NEAREST)
            .convert("RGB")
        )
        detail_panels.append(panel)

    start, end = np.array(label["axis"])
    dx, dy = end - start
    normal = np.array((-dy, dx)) * (label["half_width_px"] / np.hypot(dx, dy))
    offset = np.array((detail_left, detail_top))
    for panel in detail_panels:
        draw = ImageDraw.Draw(panel)
        for side in (normal, -normal):
            draw.line(
                [tuple(2 * (point + side - offset)) for point in (start, end)],
                fill=ANNOTATION,
                width=2,
            )
    for span in refit["axis_support"]:
        ImageDraw.Draw(detail_panels[1]).line(
            [tuple(2 * (point - offset)) for point in span], fill=DETECTOR, width=3
        )

    sheet = Image.new("RGB", (1088, 938), "#fff")
    draw = ImageDraw.Draw(sheet)
    title = ImageFont.load_default(size=22)
    caption = ImageFont.load_default(size=17)
    draw.text((16, 12), label["current"], fill="#17202b", font=title)
    panels = (
        (overview, "A   Current RAW", "Yellow box: detail shown below"),
        (
            hough_panel,
            "B   Hough segments",
            f"{len(hough['lines'])} returned segments (blue)",
        ),
        (
            detail_panels[0],
            "C   Current - max(previous, next)",
            "Yellow lines: annotated trail corridor",
        ),
        (
            detail_panels[1],
            "D   RANSAC refit on current - previous",
            f"Blue: rejected ({refit['contrast']:g} DN, "
            f"{refit['max_contiguous_support_px']:.1f} px continuous)",
        ),
    )
    for index, (panel, heading, legend) in enumerate(panels):
        x, y = 16 + (index % 2) * 536, 44 + (index // 2) * 446
        draw.text((x, y), heading, fill="#17202b", font=title)
        sheet.paste(panel, (x, y + 28))
        draw.text((x, y + 422), legend, fill="#17202b", font=caption)
    sheet.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("hough_report", type=Path)
    parser.add_argument("region_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    (label,) = json.loads(LABELS.read_text(encoding="utf-8"))
    current_name = label["current"]
    frames = sorted(path.name for path in args.raw_dir.glob("*.ORF"))
    index = frames.index(current_name)
    if index < 3 or index + 3 >= len(frames) or frames[index - 1] != label["previous"]:
        raise ValueError(
            "The review needs the annotated frame and three RAWs on each side"
        )
    hough = load_pairs(args.hough_report, frames)[current_name]
    region = load_pairs(args.region_report, frames)[current_name]
    names = frames[index - 3 : index + 4]
    images = [load_and_bin_raw_fast(str(args.raw_dir / name)) for name in names]
    previous, current, following = images[2:5]
    refit = diagnostic_refit(previous, current, label, region)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_review(
        images,
        [int(Path(name).stem[-3:]) for name in names],
        args.output_dir / "c140521-raw-review.png",
    )
    render_evidence(
        previous,
        current,
        following,
        label,
        hough,
        refit,
        args.output_dir / "c140521-detector-evidence.png",
    )


if __name__ == "__main__":
    main()
