"""Estimate adjacent-frame sky motion using stellar centroids in a specified ROI."""

import cv2
import numpy as np

TILE_PX = 256
MAX_MATCH_PX = 12
IDENTITY = np.eye(2, 3, dtype=np.float64)


def star_centroids(image, mask):
    """Choose spatially distributed compact, bright peaks and refine their centers."""
    image = image.astype(np.float32)
    fine = cv2.GaussianBlur(image, (3, 3), 0.8)
    contrast = fine - cv2.GaussianBlur(image, (0, 0), 4)
    maxima = (contrast == cv2.dilate(contrast, np.ones((9, 9), np.uint8))) & mask
    height, width = image.shape
    points = []
    dy, dx = np.mgrid[-3:4, -3:4]
    for y0 in range(0, height, TILE_PX):
        for x0 in range(0, width, TILE_PX):
            area = np.s_[y0 : y0 + TILE_PX, x0 : x0 + TILE_PX]
            eligible = mask[area]
            if not eligible.any():
                continue
            response = contrast[area][eligible]
            center = np.median(response)
            sigma = 1.4826 * np.median(np.abs(response - center))
            y, x = np.nonzero(
                maxima[area] & (contrast[area] >= max(6.0, center + 5 * sigma))
            )
            x, y = x + x0, y + y0
            strengths = contrast[y, x]
            count = 0
            for index in np.argsort(-strengths, kind="stable"):
                a, b = int(x[index]), int(y[index])
                if not (3 <= a < width - 3 and 3 <= b < height - 3):
                    continue
                values = np.maximum(contrast[b - 3 : b + 4, a - 3 : a + 4], 0)
                total = values.sum()
                ux, uy = (values * dx).sum() / total, (values * dy).sum() / total
                vx = (values * (dx - ux) ** 2).sum() / total
                vy = (values * (dy - uy) ** 2).sum() / total
                cxy = (values * (dx - ux) * (dy - uy)).sum() / total
                minor, major = np.linalg.eigvalsh(((vx, cxy), (cxy, vy)))
                if minor < 0.25 or major > 2.8 * minor or max(abs(ux), abs(uy)) > 1.25:
                    continue
                points.append((a + ux, b + uy))
                count += 1
                if count == 12:
                    break
    return np.array(points, dtype=np.float32).reshape(-1, 2)


def match_centroids(previous, current):
    """Match only mutual nearest neighbors within the adjacent-exposure search radius."""
    if not (len(previous) and len(current)):
        return previous[:0], current[:0]
    distances = np.sum((previous[:, None] - current[None, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    index = np.arange(len(previous))
    mutual = np.argmin(distances, axis=0)[nearest] == index
    selected = mutual & (distances[index, nearest] <= MAX_MATCH_PX**2)
    return previous[selected], current[nearest[selected]]


def rigid_transform(previous, current):
    """Least-squares previous-to-current rigid transform, with scale fixed to one."""
    p, q = previous.mean(axis=0), current.mean(axis=0)
    u, _, vt = np.linalg.svd((previous - p).T @ (current - q))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return np.column_stack((rotation, q - rotation @ p))


def transform_points(points, matrix):
    return points @ matrix[:, :2].T + matrix[:, 2]


def estimate_registration(previous, current):
    """Fit on spatially interleaved training points and hold out matches for evaluation."""
    previous, current = previous.astype(np.float64), current.astype(np.float64)
    cells = np.floor(previous / TILE_PX).astype(int)
    validation = (cells[:, 0] + 2 * cells[:, 1]) % 4 == 0
    train, targets = previous[~validation], current[~validation]
    fitted = None
    if len(train) >= 16 and validation.sum() >= 6:
        # The initial similarity fit supplies outlier rejection only. Refit
        # the rigid transform below so the experiment never rescales an image.
        _, inlier_mask = cv2.estimateAffinePartial2D(
            train, targets, method=cv2.RANSAC, ransacReprojThreshold=0.85
        )
        if inlier_mask is not None:
            keep = inlier_mask.ravel().astype(bool)
            selected, expected = train[keep], targets[keep]
            if (
                keep.sum() >= max(14, len(train) // 2)
                and np.ptp(selected, axis=0).min() >= 420
            ):
                translation = IDENTITY.copy()
                translation[:, 2] = np.median(expected - selected, axis=0)
                rotation = rigid_transform(selected, expected)
                if np.abs(np.arctan2(rotation[1, 0], rotation[0, 0])) <= np.deg2rad(
                    0.5
                ):
                    fitted = (translation, rotation, int(keep.sum()))
    return fitted, (previous[validation], current[validation])


def warp_reference(image, matrix, *, interpolation=cv2.INTER_LINEAR):
    """Resample a previous exposure into current sensor coordinates."""
    height, width = image.shape
    return cv2.warpAffine(image, matrix, (width, height), flags=interpolation)
