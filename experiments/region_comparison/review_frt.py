"""Render matched RAW pairs and inspect the full-field FRT of the late meteor."""

import argparse
import json
from math import cos, radians
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_ridge_beamlet import SEQUENCES
from frt_detector import FrtConfig
from meteor_core.detectors.frt_projection import (
    _frt_stages,
    _projection_input,
    _segment,
)
from meteor_core.image_io import load_and_bin_raw_fast
from summarize import load_pairs, trail_coverage

FOCUS = {"_C140344.ORF", "_C150326.ORF", "_C140521.ORF"}
SIZE = 248
INK = "#172639"
REFERENCE = "#ffdb62"
ACCEPTED = "#72ddff"
REJECTED = "#ff8bbe"


def _signed_pair(raw, label):
    previous, current = [
        load_and_bin_raw_fast(str(raw / label[key])) for key in ("previous", "current")
    ]
    return cv2.subtract(current, previous, dtype=cv2.CV_32F)


def _window_fraction(label, windows):
    a, b = np.array(label["axis"], dtype=np.float64)
    points = np.linspace(a, b, 1001)
    x, y = points.T
    inside = np.zeros(len(points), dtype=bool)
    for x0, y0, x1, y1 in windows:
        inside |= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    return float(inside.mean())


def _gray(values, low, high, size):
    normalized = np.clip((values - low) / (high - low), 0, 1)
    return (
        Image.fromarray(np.uint8(255 * normalized))
        .resize(size, Image.Resampling.NEAREST)
        .convert("RGB")
    )


def _crop(label, shape):
    endpoints = np.array(label["axis"], dtype=np.float64)
    center = endpoints.mean(axis=0)
    side = max(85, int(np.ptp(endpoints, axis=0).max()) + 64)
    left, top = np.clip(
        np.rint(center - side / 2).astype(int), 0, (shape[1] - side, shape[0] - side)
    )
    return np.s_[top : top + side, left : left + side], (left, top), side


def _overlay(image, label, origin, scale, segments=(), windows=()):
    def xy(point):
        return tuple((np.asarray(point) - origin) * scale)

    draw = ImageDraw.Draw(image)
    a, b = np.array(label["axis"], dtype=np.float64)
    dx, dy = b - a
    normal = np.array((-dy, dx)) * label["half_width_px"] / np.hypot(dx, dy)
    for side in (normal, -normal):
        draw.line((xy(a + side), xy(b + side)), fill=REFERENCE, width=2)
    for x0, y0, x1, y1 in windows:
        draw.rectangle((xy((x0, y0)), xy((x1 - 1, y1 - 1))), outline="#6795f8")
    for accepted, color in ((False, REJECTED), (True, ACCEPTED)):
        ranked = sorted(
            (item for item in segments if item["accepted"] == accepted),
            key=lambda item: (item["coverage"], item.get("response", 0)),
            reverse=True,
        )
        for item in ranked[:2]:
            line = item["line_xyxy"]
            draw.line((xy(line[:2]), xy(line[2:])), fill=color, width=2)


def _closest_peak(signed, label, config):
    """Find the highest-score scan cell with >=50% geometric overlap, before peak limits."""
    roi = np.ones_like(signed, dtype=bool)
    best = None
    projection = None
    axis = np.array(label["axis"], dtype=np.float64)
    for transposed in (False, True):
        oriented, mask = (signed.T, roi.T) if transposed else (signed, roi)
        coords = axis[:, ::-1] if transposed else axis
        a, b = coords[np.argsort(coords[:, 1])]
        dx, dy = b - a
        if dy == 0:
            continue
        channels = _projection_input(oriented, mask, config)
        for span, hop, sums in _frt_stages(channels, config.max_span):
            shifts = np.arange(1 - span, span)
            dy_line = span - 1
            angles = (shifts * dx + dy_line * dy) >= cos(radians(15)) * np.hypot(
                shifts, dy_line
            ) * np.hypot(dx, dy)
            for index in range(sums.shape[1]):
                start = index * hop
                if start > b[1] or start + dy_line < a[1]:
                    continue
                # Every matching chord crosses the annotated strip. Keep a
                # conservative intercept interval before exact trail_coverage.
                y = np.clip(start, a[1], b[1])
                cross = a[0] + (y - a[1]) * dx / dy
                center = cross - (y - start) * shifts / dy_line
                reach = (
                    abs(shifts / dy_line - dx / dy) * span
                    + label["half_width_px"] * np.hypot(dx, dy) / dy
                    + 3
                )
                for shift_index in np.flatnonzero(angles):
                    delta = shifts[shift_index]
                    low = max(
                        0, int(np.floor(center[shift_index] - reach[shift_index]))
                    )
                    high = min(
                        channels.shape[1],
                        int(np.ceil(center[shift_index] + reach[shift_index])) + 1,
                    )
                    values = sums[shift_index, index, low:high]
                    if low >= high:
                        continue
                    signal, variance, count = values.T
                    score = signal / np.sqrt(np.maximum(variance, 1e-10))
                    floor = best["response"] if best else float("-inf")
                    for j in np.flatnonzero((score > floor) & (count >= span * 0.75)):
                        if best and score[j] <= best["response"]:
                            continue
                        intercept = low + int(j)
                        peak = (
                            int(delta),
                            intercept,
                            float(score[j]),
                            float(signal[j]),
                            float(np.sqrt(variance[j])),
                        )
                        evidence = _segment(
                            channels,
                            oriented,
                            transposed,
                            span,
                            start,
                            peak,
                            config.min_snr,
                        )
                        line = evidence["line_xyxy"]
                        c = trail_coverage(label, [(line[:2], line[2:])])
                        if c < 0.5:
                            continue
                        best = {
                            **evidence,
                            "coverage": c,
                            "band_start": start,
                            "intercept": intercept,
                            "delta": int(delta),
                        }
                        region = sums[
                            max(0, shift_index - 22) : shift_index + 23,
                            index,
                            max(0, intercept - 65) : intercept + 66,
                        ]
                        projection = (
                            region[..., 0] / np.sqrt(np.maximum(region[..., 1], 1e-10)),
                            intercept - max(0, intercept - 65),
                            shift_index - max(0, shift_index - 22),
                        )
    if best is None or projection is None:
        raise ValueError("No FRT scan cell overlaps half of the late reference trail")
    return best, projection


def _new_sheet(columns, rows, headers, legend):
    sheet = Image.new("RGB", (18 + 276 * columns, 66 + 303 * rows), "white")
    draw = ImageDraw.Draw(sheet)
    title = ImageFont.load_default(size=20)
    caption = ImageFont.load_default(size=15)
    for col, header in enumerate(headers):
        draw.text((16 + col * 276, 12), header, font=title, fill=INK)
    draw.text((16, 40), legend, font=caption, fill=INK)
    return sheet, draw, caption


def _put(sheet, draw, font, image, col, row, caption):
    x, y = 16 + 276 * col, 64 + 303 * row
    sheet.paste(image, (x, y))
    draw.text((x, y + SIZE + 5), caption, font=font, fill=INK)


def run_comparison_review(root, output):
    summary = json.loads((root / "summary.json").read_text())
    results = {}
    panels = []
    for sequence, series in summary["sequences"].items():
        raw = Path("rawfiles", SEQUENCES[sequence][0])
        frames = sorted(p.name for p in (root / sequence / "raw").glob("*.ORF"))
        radon_rows = load_pairs(root / sequence / "radon-1" / "pairs.jsonl", frames)
        for label in series["labels"]:
            frame = label["current"]
            windows = radon_rows[frame]["extras"]["radon"]["windows_xyxy"]
            results[frame] = {
                "reference_axis_in_radon_windows": _window_fraction(label, windows),
                "radon_windows_xyxy": windows,
            }
            if frame in FOCUS:
                panels.append((raw, label, windows, series["results"]))
    sheet, draw, font = _new_sheet(
        3,
        len(panels),
        ("Signed RAW difference", "Windowed Radon", "Full-field FRT"),
        "Yellow: reference corridor  /  Cyan: accepted  /  Pink: rejected  /  Blue outline: Radon window",
    )
    for index, (raw, label, windows, methods) in enumerate(panels):
        frame = label["current"]
        signed = _signed_pair(raw, label)
        crop, origin, side = _crop(label, signed.shape)
        base = _gray(signed[crop], -6, 30, (SIZE, SIZE))
        for col, method in enumerate((None, "radon", "frt")):
            panel = base.copy()
            evidence = methods[method]["trails"][frame] if method else None
            segments = evidence["overlapping_proposals"] if evidence else ()
            _overlay(
                panel,
                label,
                origin,
                SIZE / side,
                segments,
                windows if method == "radon" else (),
            )
            caption = (
                frame
                if evidence is None
                else (f"Coverage {evidence['accepted_coverage']:.1%}")
            )
            _put(sheet, draw, font, panel, col, index, caption)
        if frame == "_C140521.ORF":
            best, projection = _closest_peak(signed, label, FrtConfig())
            results[frame]["best_frt_scan_cell_with_half_overlap"] = best
            _render_late(
                signed,
                label,
                windows,
                methods["frt"]["trails"][frame],
                best,
                projection,
                output / "frt-c140521-projection.png",
            )
    sheet.save(output / "frt-comparison-raw.png", optimize=True)
    (output / "diagnostics.json").write_text(json.dumps(results, indent=2) + "\n")


def _render_late(signed, label, windows, evidence, best, projection, output):
    sheet, draw, font = _new_sheet(
        2,
        2,
        ("Radon search windows", "Local FRT input"),
        "_C140521.ORF  /  Yellow: reference  /  Cyan: accepted  /  Pink: rejected",
    )
    h, w = signed.shape
    overview = _gray(
        cv2.resize(signed, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
        -6,
        30,
        (SIZE, SIZE),
    )
    overview_draw = ImageDraw.Draw(overview)
    sx, sy = SIZE / w, SIZE / h
    for x0, y0, x1, y1 in windows:
        overview_draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline="#6795f8")
    crop, origin, side = _crop(label, signed.shape)
    x0, y0 = origin
    overview_draw.rectangle(
        (x0 * sx, y0 * sy, (x0 + side) * sx, (y0 + side) * sy),
        outline=REFERENCE,
        width=2,
    )
    _put(sheet, draw, font, overview, 0, 0, "Full signed difference, scaled to square")

    transposed = best["transposed"]
    oriented = signed.T if transposed else signed
    channels = _projection_input(
        oriented, np.ones_like(oriented, dtype=bool), FrtConfig()
    )
    z = channels[..., 0] / np.sqrt(channels[..., 1])
    z = z.T if transposed else z
    filtered = _gray(z[crop], -2, 5, (SIZE, SIZE))
    _overlay(filtered, label, origin, SIZE / side)
    _put(sheet, draw, font, filtered, 1, 0, "Centered transverse DoG / local sigma")

    scores, x, y = projection
    plane = _gray(scores, 0, 18, (SIZE, SIZE))
    plane_draw = ImageDraw.Draw(plane)
    x = (x + 0.5) * SIZE / scores.shape[1]
    y = (y + 0.5) * SIZE / scores.shape[0]
    plane_draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=REFERENCE, width=2)
    draw.text((16, 350), "FRT projection / sigma", fill=INK, font=font)
    _put(
        sheet,
        draw,
        font,
        plane,
        0,
        1,
        f"Intercept x shift, span {best['span_px']} px, peak {best['response']:.2f}",
    )

    panel = _gray(signed[crop], -6, 30, (SIZE, SIZE))
    _overlay(panel, label, origin, SIZE / side, evidence["overlapping_proposals"])
    draw.text((292, 350), "Reported FRT segments", fill=INK, font=font)
    _put(
        sheet,
        draw,
        font,
        panel,
        1,
        1,
        f"Accepted coverage {evidence['accepted_coverage']:.1%}",
    )
    sheet.save(output, optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_comparison_review(args.benchmark, args.output)
