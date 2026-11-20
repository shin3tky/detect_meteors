"""Seeded injection–recovery on cropped, real adjacent-RAW backgrounds."""

import argparse
import json
from math import cos, sin
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

from benchmark_v2 import METHODS, _lines, detector_for, inputs
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.pipeline import _build_runtime_params
from meteor_core.schema import DetectionContext
from review_metrics import read_truth
from segment_verification import TILE, noise_tiles
from summarize import trail_coverage

SEQUENCES = ("bundled", "gemini2021", "late2024")
METHOD_ORDER = [
    "hough",
    "region",
    "dog_lsd",
    "tophat_lsd",
    "radon",
    "frt",
    "ridge",
    "beamlet",
]
LENGTHS = (25, 50, 100)
TARGETS = (2, 4, 8, 16, 32)
PSF = 1.2
SIDE = 768
SEED = 20261121


def template(shape, axis, width=PSF):
    """Unit-peak finite, uniform-brightness line convolved with a Gaussian PSF."""
    axis = np.asarray(axis, dtype=float)
    padding = int(np.ceil(5 * width))
    x0, y0 = np.floor(axis.min(axis=0)).astype(int) - padding
    x1, y1 = np.ceil(axis.max(axis=0)).astype(int) + padding + 1
    if not (0 <= x0 < x1 <= shape[1] and 0 <= y0 < y1 <= shape[0]):
        raise ValueError("Profile support is outside the synthetic image")
    mask = np.zeros(((y1 - y0) * 4, (x1 - x0) * 4), dtype=np.float32)
    a, b = np.rint((axis - (x0, y0)) * 4).astype(int)
    cv2.line(mask, a.tolist(), b.tolist(), 1.0, 1, cv2.LINE_8)
    mask = cv2.GaussianBlur(mask, (0, 0), 4 * width)
    mask = cv2.resize(mask, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    mask /= mask.max()
    return np.s_[y0:y1, x0:x1], mask


def inject(current, tiles, axis, *, target=None, peak_dn=None):
    """Insert one trail in the original analysis exposure, then round once to DN."""
    region, profile = template(current.shape, axis)
    ys, xs = region
    sigma = tiles[
        np.arange(ys.start, ys.stop)[:, None] // TILE,
        np.arange(xs.start, xs.stop) // TILE,
    ]
    if peak_dn is None:
        peak_dn = float(target / np.linalg.norm(profile / sigma))
    injected = current.copy()
    injected[region] = np.clip(
        np.rint(current[region] + peak_dn * profile), 0, 65535
    ).astype(np.uint16)
    delta = injected[region].astype(np.float32) - current[region]
    return injected, peak_dn, float(np.linalg.norm(delta / sigma))


def _context_metrics(detectors, params, prior, current, roi):
    results = {}
    for method, det in detectors.items():
        output = det.detect(DetectionContext(current, prior, roi, params[method], {}))
        lines = _lines(output, METHODS[method][0])
        results[method] = (
            output.is_candidate,
            [(line[:2], line[2:]) for line in lines],
        )
    return results


def _evidence(measured, axis):
    label = {"axis": axis.tolist(), "half_width_px": 5}
    return {
        method: {"candidate": bool(candidate), "coverage": trail_coverage(label, lines)}
        for method, (candidate, lines) in measured.items()
    }


def _example(path, residual, injected, axis):
    """Show the actual analysis residual, not a noiseless template."""
    out = Image.new("RGB", (960, 350), "white")
    drawing = ImageDraw.Draw(out)
    center = np.rint(axis.mean(axis=0)).astype(int)
    x, y = center - 78
    for i, (name, data) in enumerate(
        (
            ("Original signed RAW difference", residual),
            ("Same difference with injected trail", injected),
        )
    ):
        area = data[y : y + 156, x : x + 156]
        view = (
            Image.fromarray(np.uint8(np.clip((area + 8) / 28, 0, 1) * 255))
            .resize((304, 304), Image.Resampling.NEAREST)
            .convert("RGB")
        )
        origin = (26 + i * 460, 33)
        out.paste(view, origin)
        drawing.text((origin[0], 10), name, fill="black")
        a, b = ((axis - (x, y)) * 304 / 156 + origin).tolist()
        for px, py in (a, b):
            drawing.ellipse(
                (px - 5, py - 5, px + 5, py + 5), outline="#ffd36b", width=1
            )
    out.crop((0, 0, 818, 350)).save(path)


def _trials(rng):
    for length in LENGTHS:
        for target in TARGETS:
            yield length, target, None, rng.uniform(0, np.pi)
    yield 100, None, 5.0, -0.835


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    settings = yaml.safe_load(
        Path("config_examples/aircraft_trail_sample.yaml").read_text()
    )["params"]
    detectors = {method: detector_for(method, False) for method in METHOD_ORDER}
    params = {
        method: _build_runtime_params(settings, detector)
        for method, detector in detectors.items()
    }
    rng = np.random.default_rng(SEED)
    records = []
    checksums = {}
    for sequence in SEQUENCES:
        source, frames, _, checksum = inputs(sequence, args.raw_root)
        checksums[sequence] = checksum
        inventory = {path.name for path in source.glob("*.ORF")}
        labels = read_truth(source / "README.md", inventory, inventory)
        eligible = [
            i
            for i in range(1, len(frames))
            if not (labels[frames[i - 1]] or labels[frames[i]])
        ]
        for index in rng.choice(eligible, 3, replace=False):
            prior, current = [
                load_and_bin_raw_fast(str(source / name))
                for name in frames[index - 1 : index + 1]
            ]
            left = int(rng.integers(0, current.shape[1] - SIDE + 1))
            top = int(rng.integers(0, current.shape[0] - SIDE + 1))
            crop = np.s_[top : top + SIDE, left : left + SIDE]
            prior, current = (
                np.ascontiguousarray(image[crop]) for image in (prior, current)
            )
            residual = cv2.subtract(current, prior, dtype=cv2.CV_32F)
            tiles = noise_tiles(residual)
            roi = np.full(current.shape, 255, dtype=np.uint8)
            controls = _context_metrics(detectors, params, prior, current, roi)
            for length, target, peak, angle in _trials(rng):
                center = rng.uniform(78, SIDE - 78, size=2)
                direction = np.array((cos(angle), sin(angle))) * length / 2
                axis = np.array((center - direction, center + direction))
                modified, amplitude, snr = inject(
                    current, tiles, axis, target=target, peak_dn=peak
                )
                measured = _context_metrics(detectors, params, prior, modified, roi)
                records.append(
                    {
                        "series": sequence,
                        "frame": frames[index],
                        "previous": frames[index - 1],
                        "crop_origin": [left, top],
                        "axis": axis.tolist(),
                        "length_px": length,
                        "target_snr": target,
                        "peak_dn": amplitude,
                        "realized_snr": snr,
                        "control": _evidence(controls, axis),
                        "injected": _evidence(measured, axis),
                    }
                )
                if (
                    sequence == "late2024"
                    and peak
                    and not (args.output / "injection-example.png").exists()
                ):
                    _example(
                        args.output / "injection-example.png",
                        residual,
                        cv2.subtract(modified, prior, dtype=cv2.CV_32F),
                        axis,
                    )
            print(sequence, frames[index], len(records), flush=True)
    data = {
        "seed": SEED,
        "psf_sigma_px": PSF,
        "crop_px": SIDE,
        "input_sha256": checksums,
        "records": records,
    }
    (args.output / "injections.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
