"""Full-ROI multiscale Fast Radon detector with local noise normalization."""

from dataclasses import dataclass
from functools import lru_cache
from math import ceil, isfinite

import cv2
import numpy as np


@dataclass(frozen=True)
class FrtProjectionConfig:
    min_snr: float = 8.0
    max_span: int = 256
    noise_tile_px: int = 64
    noise_floor_dn: float = 1.0

    def __post_init__(self):
        if type(self.max_span) is not int or self.max_span not in (32, 64, 128, 256):
            raise ValueError("max_span must be one of 32, 64, 128 or 256")
        for name, minimum in (("min_snr", 1), ("noise_floor_dn", 0.01)):
            v = getattr(self, name)
            if type(v) not in (int, float) or not isfinite(v) or v < minimum:
                raise ValueError(f"{name} must be a finite number >= {minimum}")
        if type(self.noise_tile_px) is not int or self.noise_tile_px < 8:
            raise ValueError("noise_tile_px must be an integer >= 8")


def _projection_input(signed, roi, config):
    """Filter only across the integration axis and estimate local background."""
    fine = cv2.getGaussianKernel(9, 1, cv2.CV_32F)
    broad = cv2.getGaussianKernel(25, 3, cv2.CV_32F)
    unit = np.ones(1, dtype=np.float32)
    filtered = cv2.sepFilter2D(signed, -1, fine, unit)
    np.subtract(filtered, cv2.sepFilter2D(signed, -1, broad, unit), out=filtered)
    height, width = filtered.shape
    size = config.noise_tile_px
    centers = np.zeros((ceil(height / size), ceil(width / size)), np.float32)
    sigma = np.full_like(centers, config.noise_floor_dn)
    for i, j in np.ndindex(centers.shape):
        box = np.s_[i * size : (i + 1) * size, j * size : (j + 1) * size]
        samples = filtered[box][roi[box]]
        if samples.size:
            center = np.median(samples)
            centers[i, j] = center
            sigma[i, j] = max(
                config.noise_floor_dn, 1.4826 * np.median(np.abs(samples - center))
            )

    # Interpolate actual tile centers, including partial tiles at the edges.
    def grid(count):
        starts = np.arange(0, count, size)
        centers_px = (starts + np.minimum(starts + size, count) - 1) / 2
        return np.interp(np.arange(count), centers_px, np.arange(len(starts))).astype(
            np.float32
        )

    x, y = np.meshgrid(grid(width), grid(height))
    centered = filtered - cv2.remap(centers, x, y, cv2.INTER_LINEAR)
    scale = cv2.remap(sigma, x, y, cv2.INTER_LINEAR)
    channels = np.empty((height, width, 3), np.float32)
    channels[..., 0] = np.where(roi, centered, 0)
    channels[..., 1] = np.where(roi, scale**2, 0)
    channels[..., 2] = roi
    return channels


def _half_shifts(delta):
    """Truncate the subline shift toward zero and round the join away from zero."""
    inner = (abs(delta) // 2) * (-1 if delta < 0 else 1)
    return inner, delta - inner


def _frt_stages(channels, max_span):
    """Yield (span, hop, sums[shift, band, intercept, channel])."""
    height, width, channel_count = channels.shape
    padded_height = ceil(height / max_span) * max_span
    padded = np.zeros((1, padded_height, width, channel_count), np.float32)
    padded[0, :height] = channels
    previous, hop = padded, 1
    for span in (2, 4, 8, 16, 32, 64, 128, 256):
        half = span // 2
        next_hop = half if span >= 16 else span
        step = next_hop // hop
        offset = half // hop
        bands = (padded_height - span) // next_hop + 1
        stop = bands * step
        result = np.empty((2 * span - 1, bands, width, channel_count), np.float32)
        for delta, dest in zip(range(1 - span, span), result, strict=True):
            inner, shift = _half_shifts(delta)
            sub = previous[inner + half - 1]
            dest[:] = sub[:stop:step]
            right = sub[offset : offset + stop : step]
            if 0 < shift < width:
                dest[:, :-shift] += right[:, shift:]
            elif -width < shift < 0:
                dest[:, -shift:] += right[:, :shift]
            elif shift == 0:
                dest += right
        previous, hop = result, next_hop
        if span >= 16:
            yield span, hop, result
        if span == max_span:
            break


@lru_cache(maxsize=2048)
def _digital_offsets(span, delta):
    if span == 1:
        return np.zeros(1, np.int32)
    inner, shift = _half_shifts(delta)
    offsets = _digital_offsets(span // 2, inner)
    return np.concatenate((offsets, offsets + shift))


def _band_peaks(band, span, floor):
    """Find local peaks across shift/intercept and keep three per 96 px of intercept."""
    summed, variance, support = band.transpose(2, 0, 1)
    score = summed / np.sqrt(np.maximum(variance, np.finfo(np.float32).tiny))
    # Require at least 3/4 of the digital path to have image and ROI support.
    score[support < span * 0.75] = 0
    maxima = cv2.dilate(score, np.ones((5, 5), dtype=np.uint8))
    shift_indices, x = np.nonzero((score >= floor) & (score == maxima))
    values = score[shift_indices, x]
    blocks = x // 96
    order = np.lexsort((-values, blocks))
    if not len(order):
        return []
    boundaries = np.r_[True, np.diff(blocks[order]) != 0]
    ranks = np.arange(len(order))
    ranks -= np.maximum.accumulate(np.where(boundaries, ranks, 0))
    selected = order[ranks < 3]
    return [
        (
            int(shift_indices[i] + 1 - span),
            int(x[i]),
            float(values[i]),
            float(summed[shift_indices[i], x[i]]),
            float(np.sqrt(variance[shift_indices[i], x[i]])),
        )
        for i in selected
    ]


def _segment(channels, signed, transposed, span, start, peak, min_snr):
    delta, intercept, score, signal, sigma = peak
    height, width = channels.shape[:2]
    y = start + np.arange(span)
    x = intercept + _digital_offsets(span, delta)
    inside = (y < height) & (x >= 0) & (x < width)
    y, x = y[inside], x[inside]
    samples = channels[y, x]
    valid = samples[:, 2] > 0
    indices = np.flatnonzero(valid)
    first, last = indices[0], indices[-1]
    # Use the tested digital path's true in-image endpoints. Interior ROI
    # holes are allowed in the transform but may not form a saved segment.
    continuous_roi = bool(valid[first : last + 1].all())
    # Use the original residual to avoid treating the filter's positive
    # side lobes around a disappearance as evidence of emergence.
    positive_fraction = float(np.mean(signed[y[valid], x[valid]] > 0))
    line = (
        [float(y[first]), float(x[first]), float(y[last]), float(x[last])]
        if transposed
        else [float(x[first]), float(y[first]), float(x[last]), float(y[last])]
    )
    length = float(np.hypot(line[2] - line[0], line[3] - line[1]))
    gates = {
        "snr": score >= min_snr,
        "length": length >= 15,
        "roi": continuous_roi,
        "positive_fraction": positive_fraction >= 0.65,
    }
    return {
        "line_xyxy": line,
        "length_px": length,
        "response": score,
        "integral_dn": signal,
        "integral_sigma_dn": sigma,
        "samples": int(len(indices)),
        "positive_fraction": positive_fraction,
        "span_px": span,
        "transposed": transposed,
        "gates": gates,
        "accepted": all(gates.values()),
    }


def project_segments(signed, roi, config):
    """Return signed digital chords in descending score order and the work count."""
    segments = []
    projected = 0
    for transposed in (False, True):
        values, mask = (signed.T, roi.T) if transposed else (signed, roi)
        channels = _projection_input(values, mask, config)
        for span, hop, sums in _frt_stages(channels, config.max_span):
            projected += sums.size // sums.shape[-1]
            for index, band in enumerate(sums.transpose(1, 0, 2, 3)):
                peaks = _band_peaks(band, span, min(5, config.min_snr))
                segments.extend(
                    _segment(
                        channels,
                        values,
                        transposed,
                        span,
                        index * hop,
                        peak,
                        config.min_snr,
                    )
                    for peak in peaks
                )
    segments.sort(key=lambda item: item["response"], reverse=True)
    return segments, projected
