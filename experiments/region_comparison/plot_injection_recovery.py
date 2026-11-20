"""Render recovery curves from the measured injection report."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def recovery(records):
    """Keep raw matches, sham matches, and paired matches attributable to injection."""
    table = {}
    for record in records:
        length = record["length_px"]
        strength = record["target_snr"]
        strength = "5 DN" if strength is None else str(strength)
        for method, result in record["injected"].items():
            cell = (
                table.setdefault(method, {})
                .setdefault(length, {})
                .setdefault(strength, {"count": 0, "raw": 0, "sham": 0, "new": 0})
            )
            sham = record["control"][method]
            found = result["candidate"] and result["coverage"] >= 0.5
            chance = sham["candidate"] and sham["coverage"] >= 0.5
            cell["count"] += 1
            cell["raw"] += found
            cell["sham"] += chance
            cell["new"] += found and not chance
    return table


def plot(table, path):
    width, height = 1440, 830
    canvas = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=33)
    label = ImageFont.load_default(size=21)
    small = ImageFont.load_default(size=17)
    ink, grid = "#1f2937", "#d8e1e9"
    palette = {25: "#007d9a", 50: "#ad436e", 100: "#a35b00"}
    names = {
        "dog_lsd": "DoG + LSD",
        "tophat_lsd": "Top-hat + LSD",
        "hough": "Hough",
        "region": "Region + RANSAC",
        "radon": "Windowed Radon",
        "frt": "Full-field FRT",
        "beamlet": "Beamlet",
        "ridge": "Hessian ridge",
    }
    draw.text(
        (72, 30),
        "Meteor-trail injection recovery on real RAW backgrounds",
        font=title,
        fill=ink,
    )
    draw.text(
        (72, 83),
        "New geometric matches relative to the unmodified control · nine RAW crops per point",
        font=label,
        fill=ink,
    )
    for index, (length, color) in enumerate(palette.items()):
        x = 72 + index * 125
        draw.line((x, 135, x + 30, 135), fill=color, width=4)
        draw.text((x + 39, 124), f"{length} px", font=small, fill=ink)
    for index, (method, lengths) in enumerate(table.items()):
        top = 193 + (index // 4) * 295
        left = 72 + (index % 4) * 346
        bottom, right = top + 202, left + 290
        draw.text((left, top - 31), names[method], font=label, fill=ink)
        for rate in (0, 0.5, 1):
            y = bottom - round(rate * (bottom - top))
            draw.line((left, y, right, y), fill=grid, width=2)
            if index % 4 == 0:
                draw.text(
                    (left - 5, y), f"{rate:.0%}", anchor="rm", font=small, fill=ink
                )
        for i, target in enumerate(("2", "4", "8", "16", "32")):
            x = left + round(i * (right - left) / 4)
            draw.text((x, bottom + 7), target, anchor="mt", font=small, fill=ink)
        for length, color in palette.items():
            points = [
                (
                    left + round(i * (right - left) / 4),
                    bottom - round((bottom - top) * cell["new"] / cell["count"]),
                )
                for i, cell in enumerate(
                    lengths[length][target] for target in ("2", "4", "8", "16", "32")
                )
            ]
            draw.line(points, fill=color, width=3)
            for x, y in points:
                r = {25: 8, 50: 5, 100: 2}[length]
                draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
    draw.text(
        (720, height - 42),
        "Target integrated S/N before quantization (diagonal-noise model)",
        anchor="mm",
        font=label,
        fill=ink,
    )
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(recovery(json.loads(args.data.read_text())["records"]), args.output)


if __name__ == "__main__":
    main()
