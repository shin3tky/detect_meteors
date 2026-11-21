"""Measure signed local contrast and supported intervals on FRT chords."""

from math import ceil

import cv2
import numpy as np

TILE = 64
SIGMA_FLOOR = 1.0

OFFSETS = np.array([-8, -7, -6, -1, 0, 1, 6, 7, 8])
MIN_DENSITY = 0.75
WIDTH = 5
TEXTURE_QUANTILE = 0.9
TEXTURE_SIGMA_PX = 4.0
TEXTURE_WINDOW_PX = 13


def sample_profiles(lines, count, residual, roi, tiles):
    """Sample center and side bands along a batch of equally sampled chords."""
    start, end = lines[:, :2], lines[:, 2:4]
    delta = end - start
    length = np.linalg.norm(delta, axis=1)
    axis = (
        start[:, None, :]
        + np.linspace(0, 1, count + 1)[None, :, None] * delta[:, None, :]
    )
    normal = np.column_stack((-delta[:, 1], delta[:, 0])) / length[:, None]
    coordinates = np.rint(
        axis[:, :, None, :] + OFFSETS[None, None, :, None] * normal[:, None, None, :]
    ).astype(np.intp)
    x, y = coordinates[..., 0], coordinates[..., 1]
    height, width = residual.shape
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    np.clip(x, 0, width - 1, out=x)
    np.clip(y, 0, height - 1, out=y)
    valid = (inside & roi[y, x]).all(axis=-1)
    values = residual[y, x]
    center = np.median(values[..., 3:6], axis=-1)
    sides = np.median(values[..., [0, 1, 2, 6, 7, 8]], axis=-1)
    contrast = center - sides
    brighter_side = np.maximum(
        np.median(values[..., :3], axis=-1), np.median(values[..., 6:], axis=-1)
    )
    bilateral_contrast = center - brighter_side
    sigma = tiles[y[..., 4] // TILE, x[..., 4] // TILE]
    return axis, length / count, valid, center, contrast, bilateral_contrast, sigma


def smooth_profiles(value, valid):
    """Average along the axis using only locations with valid side samples."""
    kernel = (WIDTH, 1)
    weights = valid.astype(np.float32)
    summed = cv2.blur(value * weights, kernel, borderType=cv2.BORDER_REPLICATE)
    count = cv2.blur(weights, kernel, borderType=cv2.BORDER_REPLICATE)
    return np.divide(summed, count, out=np.zeros_like(summed), where=count > 0)


def support_runs(supported, valid, spacing, max_gap):
    """Return row, start, exclusive end and physical length of supported runs."""
    row, edge = np.nonzero(np.diff(np.pad(supported, ((0, 0), (1, 1))), axis=1))
    row, start, stop = row[::2], edge[::2], edge[1::2]
    if max_gap and row.size > 1:
        gaps = start[1:] - stop[:-1]
        join = (row[:-1] == row[1:]) & (gaps <= max_gap)
        for offset in range(max_gap):
            check = np.flatnonzero(join & (gaps > offset))
            join[check] &= valid[row[check], stop[check] + offset]
        first = np.flatnonzero(np.r_[True, ~join])
        last = np.r_[first[1:] - 1, len(row) - 1]
        counts = np.add.reduceat(stop - start, first)
        keep = counts >= MIN_DENSITY * (stop[last] - start[first])
        # A sparse merge must not erase the original continuous pieces.
        unmerged = np.repeat(~keep, last - first + 1)
        row = np.r_[row[first[keep]], row[unmerged]]
        start = np.r_[start[first[keep]], start[unmerged]]
        stop = np.r_[stop[last[keep]], stop[unmerged]]
    return row, start, stop, (stop - start - 1) * spacing[row]


def noise_tiles(residual):
    """Estimate robust RAW residual noise on 64-pixel tiles, with a 1 DN floor."""
    height, width = residual.shape
    output = np.empty((ceil(height / TILE), ceil(width / TILE)), dtype=np.float32)
    for i, j in np.ndindex(output.shape):
        values = residual[i * TILE : (i + 1) * TILE : 2, j * TILE : (j + 1) * TILE : 2]
        center = np.median(values)
        output[i, j] = max(SIGMA_FLOOR, 1.4826 * np.median(np.abs(values - center)))
    return output


def texture_map(previous):
    """Average the amplitude of spatial structure in the preceding RAW."""
    raw = previous.astype(np.float32)
    background = cv2.GaussianBlur(raw, (0, 0), TEXTURE_SIGMA_PX)
    return cv2.blur(np.abs(raw - background), (TEXTURE_WINDOW_PX, TEXTURE_WINDOW_PX))


def interval_texture(profile, row, start, stop):
    """Use a percentile of background texture over each supported interval."""
    counts = stop - start
    output = np.empty(len(row))
    for count in np.unique(counts):
        selected = np.flatnonzero(counts == count)
        values = profile[row[selected, None], start[selected, None] + np.arange(count)]
        output[selected] = np.quantile(values, TEXTURE_QUANTILE, axis=1)
    return output


def profiles(proposals, residual, roi, tiles, min_length=16):
    """Sample chords in batches grouped by their number of axial steps."""
    lengths = np.linalg.norm(proposals[:, 2:4] - proposals[:, :2], axis=1)
    counts = np.ceil(lengths).astype(int)
    for count in np.unique(counts):
        if count < min_length:
            continue
        indices = np.flatnonzero(counts == count)
        yield indices, sample_profiles(proposals[indices], count, residual, roi, tiles)


def supported_intervals(axis, spacing, valid, support, roughness, max_gap, min_length):
    """Return the source rows, lengths, textures and endpoints of supported intervals."""
    row, first, stop, length = support_runs(support, valid, spacing, max_gap)
    keep = length >= min_length
    row, first, stop, length = (values[keep] for values in (row, first, stop, length))
    texture = interval_texture(roughness, row, first, stop)
    lines = np.column_stack((axis[row, first], axis[row, stop - 1]))
    return row, length, texture, lines
