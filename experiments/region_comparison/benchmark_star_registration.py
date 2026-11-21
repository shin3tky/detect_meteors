"""Compare rigid sky registration and translation before differencing the original RAWs."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from benchmark_v2 import METHODS, _lines, _record, detector_for, inputs
from meteor_core.image_io import load_and_bin_raw_fast
from meteor_core.pipeline import _build_runtime_params
from meteor_core.roi_selector import create_roi_mask_from_polygon
from meteor_core.schema import DetectionContext
from review_frt import _crop, _gray, _overlay
from review_metrics import event_matches, frame_metrics, read_truth
from star_registration import (
    IDENTITY,
    estimate_registration,
    match_centroids,
    star_centroids,
    transform_points,
    warp_reference,
)
from trail_measurement import profile_contrast

ROI_2024 = [(125, 85), (2440, 85), (2440, 1230), (125, 1230)]
SKY_ROIS = {
    "bundled": ROI_2024,
    "gemini2021": [
        (500, 235),
        (860, 100),
        (2410, 100),
        (2400, 700),
        (2270, 1070),
        (1990, 1370),
        (450, 1370),
        (330, 1110),
        (335, 600),
    ],
    "late2024": ROI_2024,
}
POLICIES = ("unaligned", "translation", "rigid")
FOCUS = "_C140521.ORF"
MARGIN = 16


def _contrast(label, residual, mask):
    a, b = np.asarray(label["axis"], dtype=np.float64)
    length = np.linalg.norm(b - a)
    u = (b - a) / length
    normal = np.array((-u[1], u[0]))
    positions = np.linspace(0, length, round(length) + 1)
    band = (
        a
        + positions[:, None] * u
        + np.arange(-label["half_width_px"], label["half_width_px"] + 1)[:, None, None]
        * normal
    )
    x, y = np.rint(band).astype(int).transpose(2, 0, 1)
    if not mask[y, x].all():
        raise ValueError(f"Sky ROI excludes part of reference trail {label['current']}")
    return profile_contrast(residual, mask, a, u, normal, positions, 6)


def _star_residual(residual, targets):
    if not len(targets):
        return None
    dx, dy = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    x, y = np.rint(targets).astype(int).T
    return float(
        np.median(
            np.mean(
                np.abs(residual[y[:, None, None] + dy, x[:, None, None] + dx]),
                axis=(1, 2),
            )
        )
    )


def _measurement(current, previous, roi, targets):
    residual = cv2.subtract(current, previous, dtype=cv2.CV_32F)
    sample = residual[::4, ::4][roi[::4, ::4]]
    center = np.median(sample)
    return residual, {
        "star_mae_dn": _star_residual(residual, targets),
        "sky_sigma_dn": float(1.4826 * np.median(np.abs(sample - center))),
        "sky_abs_p99_dn": float(np.percentile(np.abs(sample), 99)),
    }


def _figure(path, raw, roi, held_out, matrices, residuals, reference):
    """Show the shared sky ROI, stellar motion, and current-coordinate residuals."""
    dims = (1240, 740)
    canvas = Image.new("RGB", dims, "white")
    draw = ImageDraw.Draw(canvas)
    title, note = ImageFont.load_default(size=22), ImageFont.load_default(size=16)
    raw_low, raw_high = np.percentile(raw[::4, ::4], [6, 99])
    view = _gray(raw, raw_low, raw_high, (384, 286))
    overlay = ImageDraw.Draw(view)
    scale = np.array([384 / raw.shape[1], 286 / raw.shape[0]])
    points = (np.asarray(roi) * scale).tolist()
    overlay.line([tuple(v) for v in points + points[:1]], fill="#ffdb62", width=2)
    for source, target in zip(*held_out, strict=True):
        if np.linalg.norm(target - source) > 3:
            continue
        a, b = source * scale, (source + (target - source) * 35) * scale
        overlay.line([tuple(a), tuple(b)], fill="#55e2ff", width=1)
    canvas.paste(view, (24, 56))
    draw.text(
        (24, 21), "Sky ROI and held-out star motion (35x)", font=title, fill="#203146"
    )

    star_points = np.rint(held_out[1]).astype(int)
    dx, dy = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4))
    x, y = star_points.T
    aperture = raw[y[:, None, None] + dy, x[:, None, None] + dx]
    strength = aperture.max(axis=(1, 2)) - np.median(aperture, axis=(1, 2))
    plausible = star_points[strength >= np.percentile(strength, 90)]
    center = np.asarray(reference["axis"]).mean(axis=0)
    x, y = plausible[np.argmin(np.linalg.norm(plausible - center, axis=1))]
    star_crop = np.s_[y - 12 : y + 13, x - 12 : x + 13]
    trail_crop, origin, side = _crop(reference, raw.shape)
    draw.text((437, 21), f"Held-out star near ({x}, {y})", font=title, fill="#203146")
    draw.text(
        (437, 305),
        "Signed differences: star -28 to +28 DN; trail -8 to +20 DN",
        font=note,
        fill="#203146",
    )
    draw.text(
        (24, 367),
        f"{reference['current']} meteor trail (yellow: reference corridor)",
        font=title,
        fill="#203146",
    )
    names = ("Unaligned", "Translation", "Rotation + translation")
    for i, (name, residual) in enumerate(zip(names, residuals, strict=True)):
        star_x, trail_x = 437 + i * 253, 24 + i * 398
        canvas.paste(_gray(residual[star_crop], -28, 28, (212, 212)), (star_x, 73))
        draw.text((star_x, 49), name, font=note, fill="#203146")
        panel = _gray(residual[trail_crop], -8, 20, (298, 298))
        _overlay(panel, reference, origin, 298 / side)
        canvas.paste(panel, (trail_x, 428))
        theta = np.degrees(np.arctan2(matrices[i][1, 0], matrices[i][0, 0]))
        draw.text(
            (trail_x, 400), f"{name} ({theta:+.4f} deg)", font=note, fill="#203146"
        )
    canvas.save(path)


def benchmark(sequence, raw_root, output, *, focus_frt=True):
    source, frames, labels, digest = inputs(sequence, raw_root)
    checksums = source / ("SHA256SUMS" if sequence == "bundled" else "SHA256SUM")
    inventory = {
        line.split()[1].lstrip("*") for line in checksums.read_text().splitlines()
    }
    truth = read_truth(source / "README.md", frames, inventory)
    params = yaml.safe_load(
        Path("config_examples/aircraft_trail_sample.yaml").read_text()
    )["params"]
    methods = ["hough", "dog_lsd"] + (["frt"] if focus_frt else [])
    detectors = {key: detector_for(key, False) for key in methods}
    runtime = {
        key: _build_runtime_params(params, det) for key, det in detectors.items()
    }
    previous = load_and_bin_raw_fast(str(source / frames[0]))
    polygon = SKY_ROIS[sequence]
    roi = cv2.erode(
        create_roi_mask_from_polygon(polygon, previous.shape),
        np.ones((2 * MARGIN + 1, 2 * MARGIN + 1), np.uint8),
    )
    mask = roi != 0
    axes = {axis["current"]: axis for axis in labels}
    started = perf_counter()
    previous_stars = star_centroids(previous, mask)
    feature_seconds = perf_counter() - started
    pairs = {}
    decisions = {key: {method: {} for method in methods} for key in POLICIES}
    for index, name in enumerate(frames[1:], 1):
        current = load_and_bin_raw_fast(str(source / name))
        started = perf_counter()
        current_stars = star_centroids(current, mask)
        source_points, destination_points = match_centroids(
            previous_stars, current_stars
        )
        fit, validation = estimate_registration(source_points, destination_points)
        available = bool(fit)
        if fit:
            corners = np.asarray(polygon, dtype=np.float64)
            available = all(
                np.linalg.norm(transform_points(corners, t) - corners, axis=1).max()
                <= MARGIN - 2
                for t in fit[:2]
            )
        matrices = (IDENTITY, *fit[:2]) if available else (IDENTITY,) * 3
        pair = {
            "feature_and_fit_seconds": perf_counter() - started,
            "matched_stars": len(source_points),
            "validation_stars": len(validation[0]),
            "train_inliers": fit[2] if fit else 0,
            "registration_used": available,
            "policies": {},
        }
        references = [axes[f] for f in (name, frames[index - 1]) if f in axes]
        residuals = []
        for key, matrix in zip(POLICIES, matrices, strict=True):
            started = perf_counter()
            background = (
                warp_reference(previous, matrix)
                if key != "unaligned" and available
                else previous
            )
            warp_time = perf_counter() - started
            residual, metrics = _measurement(current, background, mask, validation[1])
            residuals.append(residual)
            errors = np.linalg.norm(
                transform_points(validation[0], matrix) - validation[1], axis=1
            )
            pair["policies"][key] = {
                "matrix": matrix.tolist(),
                "angle_deg": float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))),
                "held_out_median_error_px": float(np.median(errors))
                if errors.size
                else None,
                "warp_seconds": warp_time,
                **metrics,
                "appearance_contrast_dn": _contrast(axes[name], residual, mask)
                if name in axes
                else None,
            }
            for method in methods:
                if method == "frt" and not references:
                    continue
                context = DetectionContext(
                    current, background, roi, runtime[method], {"frame_index": index}
                )
                started = perf_counter()
                detected = detectors[method].detect(context)
                elapsed = perf_counter() - started
                lines = _lines(detected, METHODS[method][0]) if references else ()
                decisions[key][method][name] = {
                    **_record(detected.is_candidate, lines, references),
                    "detector_seconds": elapsed,
                }
        if name == FOCUS and len(validation[1]):
            _figure(
                output / "star-registration-c140521.png",
                current,
                polygon,
                validation,
                matrices,
                residuals,
                axes[name],
            )
        pairs[name] = pair
        previous, previous_stars = current, current_stars
        if index % 20 == 0:
            print(sequence, index, "/", len(frames) - 1, flush=True)
    return {
        "input_sha256": digest,
        "sky_polygon": polygon,
        "sky_mask_pixels": int(mask.sum()),
        "first_frame_feature_seconds": feature_seconds,
        "truth": truth,
        "pair_metrics": pairs,
        "results": {
            policy: {
                method: {
                    "pairs": scored,
                    "frame_metrics": frame_metrics(scored, truth),
                    "events": event_matches(scored, frames, labels),
                }
                for method, scored in by_method.items()
            }
            for policy, by_method in decisions.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    parser.add_argument(
        "--sequences", nargs="+", choices=SKY_ROIS, default=list(SKY_ROIS)
    )
    parser.add_argument("--no-focal-frt", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "opencv_threads": cv2.getNumThreads(),
        "numpy": np.__version__,
        "sequences": {},
    }
    for sequence in args.sequences:
        report["sequences"][sequence] = benchmark(
            sequence, args.raw_root, args.output, focus_frt=not args.no_focal_frt
        )
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Completed", sequence, flush=True)


if __name__ == "__main__":
    main()
