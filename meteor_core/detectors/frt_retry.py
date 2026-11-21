"""Verify Fast Radon proposals for pairs rejected by positive-difference Hough."""

from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter

import cv2
import numpy as np

from meteor_core.schema import DetectionContext, DetectionResult
from meteor_core.utils import ensure_numpy

from .frt_projection import FrtProjectionConfig, project_segments
from .frt_support import (
    noise_tiles,
    profiles,
    smooth_profiles,
    supported_intervals,
    texture_map,
)

PROJECTION = FrtProjectionConfig()


@dataclass(frozen=True)
class FrtRetryConfig:
    """Thresholds for accepting locally supported sections of FRT chords."""

    bilateral_noise_min: float = 0.3
    max_gap_samples: int = 3
    min_support_px: float = 32
    maximum_prior_texture_noise: float = 0.8
    frt_min_snr: float = 10.0

    def __post_init__(self) -> None:
        for name, minimum in (
            ("bilateral_noise_min", 0),
            ("min_support_px", 1),
            ("maximum_prior_texture_noise", 0),
            ("frt_min_snr", PROJECTION.min_snr),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or value < minimum
            ):
                raise ValueError(
                    f"frt_retry.{name} must be a finite number >= {minimum}"
                )
        if type(self.max_gap_samples) is not int or not 0 <= self.max_gap_samples <= 3:
            raise ValueError("frt_retry.max_gap_samples must be an integer from 0 to 3")


def verify_proposals(signed, prior, roi, proposals, config):
    """Return the supported subpixel intervals and their scores and texture."""
    proposals = proposals[proposals[:, 4] >= config.frt_min_snr]
    if not len(proposals):
        return []
    tiles = noise_tiles(signed)
    texture = texture_map(prior)
    output = []
    for ids, (axis, spacing, valid, center, _, bilateral, sigma) in profiles(
        proposals, signed, roi, tiles, config.min_support_px
    ):
        x, y = np.rint(axis).astype(np.intp).transpose(2, 0, 1)
        contrast = smooth_profiles(bilateral / sigma, valid)
        supported = valid & (center > 0) & (contrast >= config.bilateral_noise_min)
        row, lengths, roughness, lines = supported_intervals(
            axis,
            spacing,
            valid,
            supported,
            texture[y, x] / sigma,
            config.max_gap_samples,
            config.min_support_px,
        )
        for i in np.flatnonzero(roughness <= config.maximum_prior_texture_noise):
            output.append(
                {
                    "line_xyxy": lines[i].tolist(),
                    "length_px": float(lengths[i]),
                    "response": float(proposals[ids[row[i]], 4]),
                    "prior_texture_p90": float(roughness[i]),
                }
            )
    return output


def retry_rejected_pair(
    context: DetectionContext, hough: DetectionResult, config: FrtRetryConfig
) -> DetectionResult:
    """Keep the Hough rejection unless at least one FRT interval passes."""
    started = perf_counter()
    current = ensure_numpy(context.current_image)
    prior = ensure_numpy(context.previous_image)
    roi = ensure_numpy(context.roi_mask) != 0
    if current.dtype != prior.dtype or current.dtype not in (np.uint8, np.uint16):
        raise ValueError("frt_retry requires matching uint8 or uint16 frames")
    if "star_registration" in context.metadata:
        raise ValueError("frt_retry requires unaligned frames")

    signed = cv2.subtract(current, prior, dtype=cv2.CV_32F)
    segments, _ = project_segments(signed, roi, PROJECTION)
    proposals = np.asarray(
        [(*row["line_xyxy"], row["response"]) for row in segments if row["accepted"]],
        dtype=np.float64,
    ).reshape(-1, 5)
    intervals = verify_proposals(signed, prior, roi, proposals, config)
    extras = {
        **hough.extras,
        "frt_retry": {"hough_score": hough.score, "intervals": intervals},
    }
    retry_ms = (perf_counter() - started) * 1000
    metrics = {
        **hough.metrics,
        "duration_ms": hough.metrics["duration_ms"] + retry_ms,
        "frt_retry_ms": retry_ms,
    }
    if not intervals:
        return replace(hough, extras=extras, metrics=metrics)

    lines = [tuple(round(value) for value in item["line_xyxy"]) for item in intervals]
    positive = np.maximum(signed, 0)
    positive[~roi] = 0
    debug = cv2.cvtColor(
        cv2.convertScaleAbs(positive, alpha=255 / 32), cv2.COLOR_GRAY2BGR
    )
    for x1, y1, x2, y2 in lines:
        cv2.line(debug, (x1, y1), (x2, y2), (255, 255, 0), 1, cv2.LINE_AA)
    return DetectionResult(
        is_candidate=True,
        score=max(item["response"] for item in intervals),
        lines=lines,
        aspect_ratio=0.0,
        debug_image=debug,
        extras=extras,
        metrics=metrics,
    )
