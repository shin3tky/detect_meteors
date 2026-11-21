"""Trace the final faint meteor through the FRT verification gates."""

import argparse
from dataclasses import dataclass
import json
from math import ceil
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark_frt_foreground import (
    GAPS,
    LEVELS,
    analysis_mask,
    interval_batches,
)
from meteor_core.detectors.frt_support import (
    noise_tiles,
    sample_profiles,
    smooth_profiles,
    support_runs,
    texture_map,
)
from meteor_core.image_io import load_and_bin_raw_fast
from review_frt import _crop, _gray, _overlay
from summarize import trail_coverage

SEQUENCE = "late2024"
FRAME = "_C140521.ORF"


class MatchingInterval(TypedDict):
    interval: list[float]
    proposal: list[float]
    length_px: float
    texture_p90: float
    frt_score: float
    axis_coverage: float
    gates: dict[str, bool]


@dataclass
class ReviewPair:
    residual: np.ndarray
    roi: np.ndarray
    texture: np.ndarray
    tiles: np.ndarray
    proposals: np.ndarray
    label: dict

    def measure(self, settings):
        matches: list[MatchingInterval] = []
        parameter_indices = (
            LEVELS.index(settings["bilateral_noise_min"]),
            GAPS.index(settings["max_gap_samples"]),
        )
        for indices, source, length, texture, lines in interval_batches(
            self.proposals, self.residual, self.roi, self.tiles, self.texture
        ):
            if indices != parameter_indices:
                continue
            for index, line in enumerate(lines):
                coverage = float(trail_coverage(self.label, [(line[:2], line[2:4])]))
                if coverage == 0:
                    continue
                proposal = self.proposals[source[index]]
                gates = {
                    "length": bool(length[index] >= settings["min_support_px"]),
                    "texture": bool(
                        texture[index] <= settings["maximum_prior_texture_noise"]
                    ),
                    "snr": bool(proposal[4] >= settings["frt_min_snr"]),
                }
                matches.append(
                    {
                        "interval": line.tolist(),
                        "proposal": proposal[:4].tolist(),
                        "length_px": float(length[index]),
                        "texture_p90": float(texture[index]),
                        "frt_score": float(proposal[4]),
                        "axis_coverage": coverage,
                        "gates": gates,
                    }
                )
        accepted = [
            (line["interval"][:2], line["interval"][2:])
            for line in matches
            if all(line["gates"].values())
        ]
        return {
            "settings": settings,
            "axis_coverage": float(trail_coverage(self.label, accepted)),
            "longest_axis_interval_px": max(
                (line["length_px"] for line in matches), default=0.0
            ),
            "matching_intervals": sorted(matches, key=lambda line: -line["length_px"]),
        }

    def proposal_profile(self, chord, level):
        line = np.asarray(chord)[None]
        count = ceil(np.linalg.norm(line[0, 2:4] - line[0, :2]))
        axis, spacing, valid, center, _, bilateral, sigma = sample_profiles(
            line, count, self.residual, self.roi, self.tiles
        )
        contrast = smooth_profiles(bilateral / sigma, valid)
        supported = valid & (center > 0) & (contrast >= level)
        groups = {}
        for max_gap in GAPS:
            _, starts, stops, lengths = support_runs(supported, valid, spacing, max_gap)
            groups[str(max_gap)] = [
                [int(a), int(b), float(length)]
                for a, b, length in zip(starts, stops, lengths, strict=True)
            ]
        return {
            "proposal_xyxy": chord,
            "axis_positions": axis[0].tolist(),
            "bilateral_noise": contrast[0].tolist(),
            "center_is_positive": (center[0] > 0).tolist(),
            "contrast_floor": level,
            "runs_by_gap": groups,
        }


def render(path, review, settings, results, profile):
    canvas = Image.new("RGB", (1040, 691), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    heading, subtitle, note = (ImageFont.load_default(size=n) for n in (24, 17, 15))
    ink = "#182b40"
    draw.text(
        (18, 13), f"{FRAME} | FRT support across brief gaps", font=heading, fill=ink
    )
    draw.text(
        (18, 45),
        "Signed RAW difference: black -22 DN, white +22 DN   |   Gold: meteor corridor   |   Cyan: accepted support",
        font=note,
        fill=ink,
    )
    crop, origin, side = _crop(review.label, review.residual.shape)
    for idx, (tag, header) in enumerate(
        (
            ("held_out", "Held out: strict continuity"),
            ("relax_gap_only", "Held out: gaps up to 3 samples"),
            ("all_labels", "All labels: selected setting"),
        )
    ):
        panel = _gray(review.residual[crop], -22, 22, (309, 309))
        _overlay(panel, review.label, origin, 309 / side)
        overlay = ImageDraw.Draw(panel)
        kept = [
            r for r in results[tag]["matching_intervals"] if all(r["gates"].values())
        ]
        for row in kept:
            xy = (np.asarray(row["interval"]).reshape(2, 2) - origin) * (309 / side)
            overlay.line([tuple(point) for point in xy], fill="#38d5ff", width=2)
        x = 18 + idx * 346
        draw.text((x, 77), header, font=subtitle, fill=ink)
        canvas.paste(panel, (x, 104))
        draw.text(
            (x, 422),
            f"{len(kept)} intervals  |  Axis coverage {results[tag]['axis_coverage'] * 100:.2f}%",
            font=note,
            fill=ink,
        )

    draw.text(
        (18, 465),
        "Bilateral contrast along the strongest retained interval's FRT chord",
        font=subtitle,
        fill=ink,
    )
    x0, x1, y0, y1 = 153, 1007, 511, 578
    values = np.asarray(profile["bilateral_noise"])
    span = max(1.25, float(np.abs(values).max()))
    xs = np.linspace(x0, x1, len(values))
    points = list(zip(xs, np.interp(values, (-span, span), (y1, y0)), strict=True))
    threshold = int(np.interp(settings["bilateral_noise_min"], (-span, span), (y1, y0)))
    draw.line((x0, threshold, x1, threshold), fill="#a07024", width=1)
    draw.text(
        (18, threshold - 10),
        f"{settings['bilateral_noise_min']} noise units",
        font=note,
        fill=ink,
    )
    draw.line(points, fill="#32528e", width=2)
    for index, gap in enumerate(("0", "3")):
        y = 597 + index * 27
        draw.text((18, y - 6), f"Max gap {gap}", font=note, fill=ink)
        draw.line((x0, y, x1, y), fill="#d2dbe4", width=1)
        for a, b, length in profile["runs_by_gap"][gap]:
            color = "#1585b2" if length >= settings["min_support_px"] else "#96a2b3"
            draw.line((xs[a], y, xs[b - 1], y), fill=color, width=7)
    draw.text(
        (153, 657),
        "Along-chord samples  |  Blue: at least 32 px  |  Gray: shorter support",
        font=note,
        fill=ink,
    )
    canvas.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frt_run", type=Path)
    parser.add_argument("foreground_run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw-root", type=Path, default=Path("rawfiles"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    source = json.loads((args.frt_run / "hough_frt_retry.json").read_text())[
        "sequences"
    ][SEQUENCE]
    report = json.loads(
        (args.foreground_run / "background_verification.json").read_text()
    )
    held = report["leave_one_sequence_out"][SEQUENCE]
    label = next(label for label in source["labels"] if label["current"] == FRAME)
    raw = args.raw_root / "2024GEMINI"
    prior, current = (
        load_and_bin_raw_fast(str(raw / label[key])) for key in ("previous", "current")
    )
    signed = cv2.subtract(current, prior, dtype=cv2.CV_32F)
    pair = ReviewPair(
        signed,
        analysis_mask(signed.shape, SEQUENCE) != 0,
        texture_map(prior),
        noise_tiles(signed),
        np.load(
            args.frt_run / "proposals" / SEQUENCE / f"{FRAME}.npy", allow_pickle=False
        ),
        label,
    )
    folded = held["floor_ten"]["settings"]
    settings = {
        "held_out": folded,
        "relax_gap_only": {**folded, "max_gap_samples": 3},
        "all_labels": report["all_labels"]["floor_ten"]["settings"],
        "score_margin": held["separation_margin"]["settings"],
        "relax_gap_only_with_margin_score": {
            **folded,
            "max_gap_samples": 3,
            "frt_min_snr": held["separation_margin"]["settings"]["frt_min_snr"],
        },
        "training_validated_gaps": held["validate_gap_relaxation"]["settings"],
    }
    results = {name: pair.measure(options) for name, options in settings.items()}
    accepted = next(
        line
        for line in results["relax_gap_only"]["matching_intervals"]
        if all(line["gates"].values())
    )
    profile = pair.proposal_profile(accepted["proposal"], folded["bilateral_noise_min"])
    (args.output / "c140521_gates.json").write_text(
        json.dumps(
            {"label": label, "results": results, "representative_profile": profile},
            indent=2,
        )
        + "\n"
    )
    render(args.output / "frt-c140521-support-gaps.png", pair, folded, results, profile)


if __name__ == "__main__":
    main()
