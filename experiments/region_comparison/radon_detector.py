"""Experimental streak detection using bounded, discrete Radon projections."""

from dataclasses import dataclass
from math import ceil, hypot, isfinite
from time import perf_counter

import cv2
import numpy as np

from meteor_core.detectors import DataclassDetector
from meteor_core.schema import DetectionContext, DetectionResult
from trail_measurement import difference_of_gaussians, measure_segment, pair_residual


@dataclass(frozen=True)
class RadonConfig:
    tile_size_px: int = 256
    tile_padding_px: int = 32
    max_tiles: int = 8
    angle_step_deg: int = 2
    max_peaks_per_tile: int = 6
    min_length_px: float = 15.0
    min_contrast: float = 15.0
    background_offset_px: int = 6

    def __post_init__(self) -> None:
        for name, minimum in (
            ("tile_size_px", 1),
            ("tile_padding_px", 0),
            ("max_tiles", 1),
            ("angle_step_deg", 1),
            ("max_peaks_per_tile", 1),
            ("background_offset_px", 2),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.angle_step_deg >= 180:
            raise ValueError("angle_step_deg must be less than 180")
        for name, minimum in (("min_length_px", 1.0), ("min_contrast", 0.0)):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"{name} must be a finite number >= {minimum}")


def _rank_tiles(signal, config):
    """Rank nonoverlapping cores before adding a margin to each chosen window."""
    height, width = signal.shape
    size = config.tile_size_px
    padding = config.tile_padding_px
    cores = []
    for y in range(0, height, size):
        for x in range(0, width, size):
            right, bottom = min(x + size, width), min(y + size, height)
            strength = float(signal[y:bottom, x:right].sum(dtype=np.float64))
            if strength > 0:
                bounds = (
                    max(0, x - padding),
                    max(0, y - padding),
                    min(width, right + padding),
                    min(height, bottom + padding),
                )
                cores.append((strength, bounds))
    return sorted(cores, key=lambda core: core[0], reverse=True)[: config.max_tiles]


def _peak_segments(tile, angles, config):
    """Find peaks in the line integrals and recover their finite support."""
    height, width = tile.shape
    side = ceil(hypot(width, height))
    center = ((width - 1) / 2, (height - 1) / 2)
    shift = (side - 1) / 2 - np.array(center)
    transforms = []
    projections = np.empty((len(angles), side), dtype=np.float32)
    for index, angle in enumerate(angles):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        matrix[:, 2] += shift
        transforms.append(matrix)
        rotated = cv2.warpAffine(tile, matrix, (side, side))
        rows = np.pad(rotated.sum(axis=1), 1)
        projections[index] = rows[:-2] + rows[1:-1] + rows[2:]

    maxima = cv2.dilate(projections, np.ones((3, 7), dtype=np.uint8))
    angle_rows, offsets = np.nonzero((projections == maxima) & (projections > 0))
    strengths = projections[angle_rows, offsets]
    ranked = np.argsort(-strengths, kind="stable")[: config.max_peaks_per_tile]
    for index in ranked:
        angle_index, offset = int(angle_rows[index]), int(offsets[index])
        matrix = transforms[angle_index]
        rotated = cv2.warpAffine(tile, matrix, (side, side))
        band = rotated[max(0, offset - 1) : offset + 2]
        support = np.flatnonzero(band.max(axis=0) >= 6.0)
        # Bridge at most three absent samples and keep the two longest runs.
        runs = np.split(support, np.flatnonzero(np.diff(support) > 4) + 1)
        spans = [(int(run[0]), int(run[-1])) for run in runs if run.size]
        spans.sort(key=lambda span: span[1] - span[0], reverse=True)
        inverse = cv2.invertAffineTransform(matrix)
        for start, end in spans[:2]:
            if end - start < config.min_length_px:
                continue
            points = np.array(((start, offset, 1), (end, offset, 1))) @ inverse.T
            yield points, angles[angle_index], float(strengths[index])


class RadonDetector(DataclassDetector[RadonConfig]):
    """Fit line segments within the strongest residual tiles."""

    plugin_name = "radon"
    name = "Experimental Radon detector"
    version = "0.1.0"
    ConfigType = RadonConfig

    def compute_line_score(self, mask, hough_params):
        """The legacy scoring call has no intensity residual for a profile."""
        return 0.0, []

    def detect(self, context: DetectionContext) -> DetectionResult:
        started = perf_counter()
        residual, roi = pair_residual(context)
        signal = difference_of_gaussians(residual)
        signal -= 6.0
        np.clip(signal, 0, 70, out=signal)
        signal[~roi] = 0
        angles = range(0, 180, self.config.angle_step_deg)
        windows = _rank_tiles(signal, self.config)
        segments = []
        for _, (x, y, right, bottom) in windows:
            tile = signal[y:bottom, x:right]
            for points, angle, projection in _peak_segments(tile, angles, self.config):
                line = (points + (x, y)).reshape(4)
                evidence = measure_segment(
                    line,
                    residual,
                    roi,
                    min_length_px=self.config.min_length_px,
                    min_contrast=self.config.min_contrast,
                    background_offset_px=self.config.background_offset_px,
                )
                evidence.update(angle_deg=angle, projection=projection)
                segments.append(evidence)
        accepted = [item for item in segments if item["accepted"]]
        return DetectionResult(
            is_candidate=bool(accepted),
            score=max((item["contrast"] for item in accepted), default=0.0),
            lines=[
                tuple(round(value) for value in item["line_xyxy"]) for item in accepted
            ],
            aspect_ratio=0.0,
            debug_image=None,
            extras={
                "radon": {
                    "windows_xyxy": [list(bounds) for _, bounds in windows],
                    "segments": segments,
                    "segments_omitted": 0,
                    "accepted_segments": len(accepted),
                }
            },
            metrics={
                "duration_ms": (perf_counter() - started) * 1000.0,
                "num_projections": len(windows) * len(angles),
                "num_segments": len(segments),
            },
        )
