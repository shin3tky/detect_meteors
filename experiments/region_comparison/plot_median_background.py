"""Show appearance and disappearance residuals from two measured background policies."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_v2 import temporal_median
from meteor_core.image_io import load_and_bin_raw_fast
from review_frt import _crop, _overlay

FOCUS = "_C140344.ORF"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    source = args.raw_root / "2024GEMINI_AIRCRAFT"
    sequence = sorted(source.glob("*.ORF"))
    index = next(i for i, path in enumerate(sequence) if path.name == FOCUS)
    label_file = Path(__file__).with_name("meteor_labels.json")
    annotation = next(
        item for item in json.loads(label_file.read_text()) if item["current"] == FOCUS
    )
    images = [
        load_and_bin_raw_fast(str(path)) for path in sequence[index - 3 : index + 2]
    ]
    crop, origin, side = _crop(annotation, images[0].shape)
    previous3, previous2, previous1, current, following = [
        image[crop] for image in images
    ]
    cases = [
        ("Appearance", current, previous1, (previous3, previous2, previous1)),
        ("Next pair", following, current, (previous2, previous1, current)),
    ]
    colorbar = np.array(
        ((19, 78, 171), (245, 247, 251), (177, 30, 39)), dtype=np.float32
    )
    canvas = Image.new("RGB", (820, 748), "white")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=17)
    for i, (name, frame, preceding, history) in enumerate(cases):
        for j, background in enumerate((preceding, temporal_median(history))):
            residual = cv2.subtract(frame, background, dtype=cv2.CV_32F)
            rgb = np.stack(
                [np.interp(residual, [-16, 0, 16], channel) for channel in colorbar.T],
                axis=-1,
            ).astype(np.uint8)
            panel = Image.fromarray(rgb).resize((318, 318), Image.Resampling.NEAREST)
            _overlay(panel, annotation, origin, 318 / side)
            x, y = 26 + 398 * j, 43 + 350 * i
            title_prefix = ("Preceding exposure", "Causal temporal median")[j]
            draw.text(
                (x, y - 28), f"{name} / {title_prefix}", font=title, fill="#15253a"
            )
            canvas.paste(panel, (x, y))
    draw.text(
        (26, 721),
        "Signed residual: blue -16 DN · white 0 DN · red +16 DN. Yellow: reference band.",
        font=small,
        fill="#15253a",
    )
    canvas.save(args.output)


if __name__ == "__main__":
    main()
