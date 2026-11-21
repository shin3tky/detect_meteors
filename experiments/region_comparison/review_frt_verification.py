"""Compare high-scoring false FRT lines with lines covering the missed meteors."""

import argparse
import json
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_hough_frt_retry import FOCUS, analysis_mask
from benchmark_v2 import inputs
from meteor_core.detectors.frt_support import (
    noise_tiles,
    sample_profiles,
    smooth_profiles,
    support_runs,
)
from meteor_core.image_io import load_and_bin_raw_fast
from review_frt import _gray
from summarize import trail_coverage

SUPPORT_PROBES = (("dn", 9, 0), ("noise", 1, 0), ("noise", 1, 3), ("noise", 0.6, 3))


def describe(proposal, residual, roi, tiles, coverage):
    length = float(np.linalg.norm(proposal[2:4] - proposal[:2]))
    _, spacing, valid, center, contrast, bilateral, sigma = sample_profiles(
        proposal[None], ceil(length), residual, roi, tiles
    )
    eligible = valid & (center > 0)
    median_dn = float(np.median(contrast[valid])) if valid.any() else None
    median_noise = float(np.median((contrast / sigma)[valid])) if valid.any() else None
    values = {
        "dn": smooth_profiles(contrast, valid),
        "noise": smooth_profiles(contrast / sigma, valid),
    }
    continuity = {
        f"{unit}_{cutoff}_gap_{gap}": float(
            support_runs(eligible & (values[unit] >= cutoff), valid, spacing, gap)[
                3
            ].max(initial=0.0)
        )
        for unit, cutoff, gap in SUPPORT_PROBES
    }
    return {
        "line_xyxy": proposal[:4].tolist(),
        "frt_snr": float(proposal[4]),
        "frt_positive_fraction": float(proposal[5]),
        "length_px": length,
        "valid_band_fraction": float(valid.mean()),
        "median_contrast_dn": median_dn,
        "median_contrast_noise": median_noise,
        "median_both_sides_contrast_dn": float(np.median(bilateral[valid]))
        if valid.any()
        else None,
        "longest_support_px": continuity,
        "meteor_axis_coverage": coverage,
    }


def oriented_strip(residual, line):
    """View the same signed RAW pixels along and across a proposed chord."""
    start, end = np.asarray(line).reshape(2, 2)
    delta = end - start
    unit = delta / np.linalg.norm(delta)
    normal = np.array((-unit[1], unit[0]))
    points = (
        np.linspace(start - 5 * unit, end + 5 * unit, 300)[None, :, :]
        + np.arange(-14, 15)[:, None, None] * normal[None, None, :]
    ).astype(np.float32)
    return cv2.remap(
        residual, points[..., 0].copy(), points[..., 1].copy(), cv2.INTER_LINEAR
    )


def inspect_sequence(sequence, data, raw_root, proposal_root, negatives, cutoff):
    source, frames, labels, digest = inputs(sequence, raw_root)
    if (frames, labels, digest) != (
        data["frames"],
        data["labels"],
        data["input_sha256"],
    ):
        raise ValueError("Review inputs do not match the FRT run")
    labels = {item["current"]: item for item in labels}
    output, panels = {}, {}
    for index, frame in enumerate(frames[1:], start=1):
        if frame not in negatives and frame not in FOCUS:
            continue
        previous, current = [
            load_and_bin_raw_fast(str(source / name))
            for name in (frames[index - 1], frame)
        ]
        residual = cv2.subtract(current, previous, dtype=cv2.CV_32F)
        roi = analysis_mask(residual.shape, sequence) != 0
        tiles = noise_tiles(residual)
        proposals = np.load(
            proposal_root / sequence / f"{frame}.npy", allow_pickle=False
        )
        if frame in negatives:
            group = "false_positive"
            lines = [(row, None) for row in proposals if row[4] >= cutoff]
        else:
            group = "meteor"
            lines = [
                (row, coverage)
                for row in proposals
                if (coverage := trail_coverage(labels[frame], [(row[:2], row[2:4])]))
                >= 0.5
            ]
        if not lines:
            raise ValueError(f"No representative FRT lines for {frame}")
        records = [
            describe(row, residual, roi, tiles, coverage) for row, coverage in lines
        ]
        output[frame] = {"group": group, "line_count": len(records), "lines": records}
        panels[frame] = oriented_strip(residual, records[0]["line_xyxy"])
    return output, panels


def render(path, rows, panels):
    width, cell_h = 315, 115
    columns = 3
    count = ceil(len(rows) / columns)
    canvas = Image.new(
        "RGB", (columns * (width + 12) + 14, 75 + count * cell_h), "#f7f9fc"
    )
    draw = ImageDraw.Draw(canvas)
    title, heading, small = (ImageFont.load_default(size=pt) for pt in (24, 15, 13))
    ink = "#172639"
    draw.text((14, 13), "FRT chords in signed RAW differences", font=title, fill=ink)
    draw.text(
        (14, 44),
        "Strongest chord per frame   |   White: +32 DN   Gray: 0   Black: -32 DN   |   Gold: center band",
        font=small,
        fill=ink,
    )
    for rank, (name, group) in enumerate(rows.items()):
        i, j = divmod(rank, columns)
        x, y = 14 + j * (width + 12), 75 + i * cell_h
        line = group["lines"][0]
        label = "Meteor" if group["group"] == "meteor" else "FP"
        draw.text(
            (x, y),
            f"{name}  {label}  |  FRT {line['frt_snr']:.1f}",
            font=heading,
            fill=ink,
        )
        strip = _gray(panels[name], -32, 32, (width, 58))
        marks = ImageDraw.Draw(strip)
        for level in (25, 33):
            marks.line((0, level, width - 1, level), fill="#ffdb62", width=1)
        canvas.paste(strip, (x, y + 22))
        support = line["longest_support_px"]["noise_1_gap_3"]
        contrast = line["median_contrast_dn"]
        median = "—" if contrast is None else f"{contrast:.1f} DN"
        draw.text(
            (x, y + 85),
            f"Median {median}    |    1-noise support {support:.0f} px",
            font=small,
            fill=ink,
        )
    canvas.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frt_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    report = json.loads((args.frt_run / "hough_frt_retry.json").read_text())
    fold = report["evaluation"]["leave_one_sequence_out"]["gemini2021"]
    negatives = set(fold["held_out"]["frames"]["fp"])
    cutoff = fold["calibration"]["min_snr"]
    rows, panels = {}, {}
    for sequence in ("gemini2021", "late2024"):
        data, views = inspect_sequence(
            sequence,
            report["sequences"][sequence],
            args.raw_root,
            args.frt_run / "proposals",
            negatives,
            cutoff,
        )
        rows.update(data)
        panels.update(views)
    # Display the three comparison meteors first, then all twenty false positives.
    ordered = {name: rows[name] for name in (*FOCUS, *sorted(negatives))}
    (args.output / "line_comparison.json").write_text(
        json.dumps({"high_score_cutoff": cutoff, "frames": ordered}, indent=2) + "\n"
    )
    render(args.output / "frt-false-positive-lines.png", ordered, panels)


if __name__ == "__main__":
    main()
