"""Review the survived FRT intervals and their RAW surroundings."""

import argparse
import json
from math import ceil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_frt_foreground import (
    LEVELS,
    analysis_mask,
    interval_batches,
    texture_map,
)
from benchmark_frt_verification import GAPS
from benchmark_v2 import inputs
from frt_support import sample_profiles, smooth_profiles, support_runs
from meteor_core.image_io import load_and_bin_raw_fast
from review_frt import _gray
from segment_verification import TILE, noise_tiles
from summarize import trail_coverage


def difference(left, right):
    return cv2.subtract(left, right, dtype=cv2.CV_32F)


def longest_next_support(lines, following, roi, cutoff, gap):
    """Check whether the same clipped interval is bright against the next RAW."""
    tiles = noise_tiles(following)
    results = []
    for line in lines:
        count = ceil(np.linalg.norm(line[2:4] - line[:2]))
        _, spacing, valid, center, _, bilateral, sigma = sample_profiles(
            line[None], count, following, roi, tiles
        )
        supported = (
            valid & (center > 0) & (smooth_profiles(bilateral / sigma, valid) >= cutoff)
        )
        distance = support_runs(supported, valid, spacing, gap)[3]
        results.append(float(distance.max(initial=0)))
    return results


def inspect(proposals, previous, current, following, roi, settings, label):
    selected = proposals[proposals[:, 4] >= settings["min_snr"]]
    residual = difference(current, previous)
    tiles = noise_tiles(residual)
    texture = texture_map(previous)
    co = LEVELS.index(settings["min_local_contrast"])
    gi = GAPS.index(settings["max_gap_px_approx"])
    rows = []
    for (i, j), source, distance, roughness, lines in interval_batches(
        selected, residual, roi, tiles, texture
    ):
        if (i, j) != (co, gi):
            continue
        for index in np.flatnonzero(distance >= settings["min_support_px"]):
            line = lines[index]
            count = ceil(distance[index])
            _, _, valid, center, _, bilateral, _ = sample_profiles(
                line[None], count, residual, roi, tiles
            )
            positive = np.maximum(center[valid], 0)
            brightest = ceil(positive.size / 10)
            share = np.sort(positive)[-brightest:].sum() / positive.sum()
            coverage = (
                float(trail_coverage(label, [(line[:2], line[2:4])]))
                if label is not None
                else None
            )
            rows.append(
                {
                    "interval_xyxy": line.tolist(),
                    "frt_snr": float(selected[source[index], 4]),
                    "length_px": float(distance[index]),
                    "prior_texture_p90_in_noise": float(roughness[index]),
                    "median_bilateral_dn": float(np.median(bilateral[valid])),
                    "peak_center_dn": float(center[valid].max()),
                    "brightest_tenth_fraction": float(share),
                    "appearance_coverage": coverage,
                }
            )
    rows.sort(key=lambda row: row["frt_snr"], reverse=True)
    if not rows:
        raise ValueError("No passing intervals to review")
    after = difference(current, following)
    for row, longest in zip(
        rows,
        longest_next_support(
            np.asarray([row["interval_xyxy"] for row in rows]),
            after,
            roi,
            settings["min_local_contrast"],
            settings["max_gap_px_approx"],
        ),
        strict=True,
    ):
        row["next_supported_length_px"] = longest

    representative = (
        max(rows, key=lambda row: row["appearance_coverage"] or 0.0)
        if label
        else max(rows, key=lambda row: row["length_px"])
    )
    center = np.asarray(representative["interval_xyxy"]).reshape(2, 2).mean(axis=0)
    x0, y0 = np.rint(center).astype(int) - 72
    crop = np.s_[y0 : y0 + 145, x0 : x0 + 145]
    yy, xx = np.ogrid[y0 : y0 + 145, x0 : x0 + 145]
    view = (
        (int(x0), int(y0)),
        [
            previous[crop],
            current[crop],
            following[crop],
            residual[crop],
            after[crop],
            texture[crop] / tiles[yy // TILE, xx // TILE],
        ],
    )
    return rows, view


def render(path, records, views):
    canvas = Image.new("RGB", (996, 876), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    title, heading, note = (ImageFont.load_default(size=pt) for pt in (23, 15, 13))
    ink = "#172639"
    draw.text(
        (12, 11),
        "FRT intervals that passed bilateral contrast and continuity",
        font=title,
        fill=ink,
    )
    draw.text(
        (12, 38),
        "Gold: passed intervals   |   Signed differences: -22 to +22 DN   |   Prior texture: 0 to 3 local noise units",
        font=note,
        fill=ink,
    )
    for col, caption in enumerate(
        (
            "Previous RAW",
            "Current RAW",
            "Next RAW",
            "Current - previous",
            "Current - next",
            "Prior texture",
        )
    ):
        draw.text((12 + col * 163, 61), caption, font=heading, fill=ink)
    for idx, (frame, rows) in enumerate(records.items()):
        origin, panels = views[frame]
        y = 98 + idx * 153
        draw.text((12, y - 18), frame, font=heading, fill=ink)
        raw_range = np.percentile(np.stack(panels[:3]), (5, 99.8))
        for col, values in enumerate(panels):
            lo, hi = raw_range if col < 3 else (0, 3) if col == 5 else (-22, 22)
            panel = _gray(values, lo, hi, (154, 131))
            if col >= 3:
                overlay = ImageDraw.Draw(panel)
                for row in rows:
                    points = np.asarray(row["interval_xyxy"]).reshape(2, 2) - origin
                    overlay.line(
                        [tuple(p * (154 / 145, 131 / 145)) for p in points],
                        fill="#ffcb42",
                        width=1,
                    )
            canvas.paste(panel, (12 + col * 163, y))
    canvas.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frt_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    sequence = "gemini2021"
    saved = json.loads((args.frt_run / "hough_frt_retry.json").read_text())
    report = json.loads(
        Path(
            "experiments/region_comparison/results/frt_verification/frt_verification.json"
        ).read_text()
    )
    trial = report["leave_one_sequence_out"][sequence]["both_sides_and_snr"]
    baseline = report["previous_hough_only"][sequence]["frames"]["tp"]
    actual = trial["evaluation"][sequence]["frames"]
    selected = [*actual["fp"], *(name for name in actual["tp"] if name not in baseline)]
    source, frames, axes, digest = inputs(sequence, args.raw_root)
    if digest != saved["sequences"][sequence]["input_sha256"]:
        raise ValueError("Mismatched RAW sequence")
    labels = {axis["current"]: axis for axis in axes}
    records, views = {}, {}
    for name in selected:
        idx = frames.index(name)
        previous, current, following = (
            load_and_bin_raw_fast(str(source / frame))
            for frame in frames[idx - 1 : idx + 2]
        )
        path = args.frt_run / "proposals" / sequence / f"{name}.npy"
        records[name], views[name] = inspect(
            np.load(path, allow_pickle=False),
            previous,
            current,
            following,
            analysis_mask(current.shape, sequence) != 0,
            trial["settings"],
            labels.get(name),
        )
    (args.output / "surviving_intervals.json").write_text(
        json.dumps(
            {
                "input_sha256": digest,
                "settings": trial["settings"],
                "intervals": records,
            },
            indent=2,
        )
        + "\n"
    )
    render(args.output / "frt-foreground-intervals.png", records, views)


if __name__ == "__main__":
    main()
