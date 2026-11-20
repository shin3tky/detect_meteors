"""Experimental finite-line search with a sparse, translated beamlet dictionary."""

from dataclasses import dataclass
from collections import Counter
from functools import lru_cache
from itertools import combinations
from math import isfinite
from time import perf_counter

import cv2
import numpy as np

from meteor_core.detectors import DataclassDetector
from meteor_core.schema import DetectionContext, DetectionResult
from trail_measurement import (
    difference_of_gaussians,
    measure_segment,
    pair_images,
    signed_segment_result,
)


@dataclass(frozen=True)
class BeamletConfig:
    tile_sizes_px: tuple[int, ...] = (32, 64, 128)
    min_response: float = 8.0
    min_length_px: float = 15.0
    min_contrast: float = 3.0
    max_measure_segments: int = 1536
    max_proposals_per_block: int = 6

    def __post_init__(self):
        if (
            not isinstance(self.tile_sizes_px, (list, tuple))
            or not self.tile_sizes_px
            or any(
                type(size) is not int or size not in (16, 32, 64, 128, 256)
                for size in self.tile_sizes_px
            )
        ):
            raise ValueError("tile_sizes_px must contain dyadic sizes from 16 to 256")
        object.__setattr__(
            self, "tile_sizes_px", tuple(dict.fromkeys(self.tile_sizes_px))
        )
        for name, minimum in (
            ("min_response", 0.01),
            ("min_length_px", 2),
            ("min_contrast", 0),
        ):
            value = getattr(self, name)
            if (
                type(value) not in (float, int)
                or not isfinite(value)
                or value < minimum
            ):
                raise ValueError(f"{name} must be a finite number >= {minimum}")
        for name in ("max_measure_segments", "max_proposals_per_block"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@lru_cache(maxsize=16)
def _dictionary(size, pitch, min_length):
    """Sample perimeter-to-perimeter segments on one square."""
    edge = size - 1
    positions = (*range(0, edge, size // 8), edge)
    rim = sorted(
        {(x, y) for i in positions for x, y in ((0, i), (i, 0), (edge, i), (i, edge))}
    )
    ends = np.array(list(combinations(rim, 2)), dtype=np.int32)
    delta = ends[:, 1] - ends[:, 0]
    keep = np.hypot(delta[:, 0], delta[:, 1]) >= min_length
    ends, delta = ends[keep], delta[keep]
    # Sample once per pixel of the dominant axis. Rows of equal size let
    # NumPy apply the same dictionary to a batch of tiles.
    steps = np.max(np.abs(delta), axis=1)
    axis = np.arange(size)
    fractions = np.minimum(axis, steps[:, None]) / steps[:, None]
    points = np.rint(ends[:, :1] + fractions[:, :, None] * delta[:, None]).astype(
        np.int32
    )
    offsets = points[..., 1] * pitch + points[..., 0]
    weights = (axis <= steps[:, None]) / np.sqrt(steps[:, None] + 1)
    return ends, offsets, weights.astype(np.float32)


def _proposals(response, config):
    height, width = response.shape
    pad = max(config.tile_sizes_px)
    padded = np.pad(response, pad)
    pitch = padded.shape[1]
    flat = padded.ravel()
    for size in config.tile_sizes_px:
        ends, offsets, weights = _dictionary(size, pitch, config.min_length_px)
        if not len(ends):
            continue
        hop = size // 2
        x, y = np.meshgrid(np.arange(-hop, width, hop), np.arange(-hop, height, hop))
        origins = np.column_stack((x.ravel(), y.ravel())).astype(np.int32)
        grid = np.arange(0, size, 4)
        noise_offsets = (grid[:, None] * pitch + grid).ravel().astype(np.int32)
        for start in range(0, len(origins), 16):
            xy = origins[start : start + 16]
            indices = (xy[:, 1] + pad) * pitch + xy[:, 0] + pad
            noise = flat[indices[:, None] + noise_offsets]
            medians = np.median(noise, axis=1)
            deviations = np.median(np.abs(noise - medians[:, None]), axis=1)
            noise_scale = np.maximum(1, 1.4826 * deviations)
            samples = flat[indices[:, None, None] + offsets]
            scores = np.einsum("btk,tk->bt", samples, weights)
            scores /= noise_scale[:, None]
            # Discard beams truncated by the padded image boundary before
            # ranking in the tile; the final ROI test also checks the interior.
            points = xy[:, None, None] + ends
            in_image = ((points >= 0) & (points < (width, height))).all(axis=(2, 3))
            scores[~in_image] = -np.inf
            tiles = np.arange(len(xy))
            for _ in range(2):
                best = scores.argmax(axis=1)
                values = scores[tiles, best]
                for tile in np.flatnonzero(values >= config.min_response):
                    yield (points[tile, best[tile]].ravel(), float(values[tile]), size)
                scores[tiles, best] = -np.inf


class BeamletDetector(DataclassDetector[BeamletConfig]):
    """Score a dictionary of finite segments on a signed DoG response."""

    plugin_name = "beamlet"
    name = "Experimental beamlet detector"
    version = "0.1.0"
    ConfigType = BeamletConfig

    def compute_line_score(self, mask, hough_params):
        return 0.0, []

    def detect(self, context: DetectionContext) -> DetectionResult:
        started = perf_counter()
        current, previous, roi = pair_images(context)
        signed = cv2.subtract(current, previous, dtype=cv2.CV_32F)
        response = difference_of_gaussians(signed)
        response[~roi] = 0
        proposals = sorted(
            _proposals(response, self.config), key=lambda p: p[1], reverse=True
        )
        # Give each 256 px block and scale a share of the measurement budget.
        # Otherwise bright foreground can occupy every entry in the global list.
        per_block = Counter()
        segments = []
        reserved = 0
        for line, score, size in proposals:
            x, y = (line[:2] + line[2:]) // 512
            key = (size, int(x), int(y))
            if per_block[key] >= self.config.max_proposals_per_block:
                continue
            per_block[key] += 1
            reserved += 1
            if len(segments) >= self.config.max_measure_segments:
                continue
            item = measure_segment(
                line,
                signed,
                roi,
                min_length_px=self.config.min_length_px,
                min_contrast=self.config.min_contrast,
                background_offset_px=6,
            )
            segments.append(dict(item, response=score, tile_size_px=size))
        return signed_segment_result(
            self.plugin_name,
            segments,
            started,
            self.config.max_measure_segments,
            proposals=len(proposals),
            block_pruned=len(proposals) - reserved,
            proposals_omitted=reserved - len(segments),
        )
