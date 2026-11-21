"""Tests for stellar registration before adjacent RAW differencing."""

import unittest

import cv2
import numpy as np

from star_registration import (
    TILE_PX,
    estimate_registration,
    match_centroids,
    rigid_transform,
    star_centroids,
    transform_points,
    warp_reference,
)


class StarRegistrationTests(unittest.TestCase):
    def test_rigid_fit_maps_previous_to_current_at_unit_scale(self):
        previous = np.array([[2, 3], [90, 60], [5, 150], [82, 130]], dtype=float)
        theta = 0.004
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        expected = np.column_stack((rotation, [1.25, -0.75]))
        current = transform_points(previous, expected)
        np.testing.assert_allclose(
            rigid_transform(previous, current), expected, atol=1e-12
        )

    def test_matching_needs_both_neighbors_and_rejects_far_points(self):
        previous = np.array([[10, 10], [15, 10], [35, 35], [90, 90]])
        current = np.array([[11, 10], [70, 70], [90, 89]])
        source, matched = match_centroids(previous, current)
        np.testing.assert_array_equal(source, [[10, 10], [90, 90]])
        np.testing.assert_array_equal(matched, [[11, 10], [90, 89]])

    def test_held_out_stars_do_not_affect_fit(self):
        yy, xx = np.mgrid[43:970:55, 63:1250:65]
        points = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        expected = np.array([[1, 0, 0.8], [0, 1, -0.3]], dtype=float)
        current = transform_points(points, expected)
        tiles = np.floor(points / TILE_PX).astype(int)
        validation = (tiles[:, 0] + 2 * tiles[:, 1]) % 4 == 0
        current[validation] += [3, -4]
        current[np.flatnonzero(~validation)[::11]] += [-8, 5]
        fitted, _ = estimate_registration(points, current)
        self.assertIsNotNone(fitted)
        assert fitted is not None
        np.testing.assert_allclose(fitted[1], expected, atol=1e-10)
        failed, _ = estimate_registration(points[:6], current[:6])
        self.assertIsNone(failed)

    def test_warp_keeps_current_only_trail_and_reduces_stellar_residual(self):
        shape = (780, 840)
        impulses = np.zeros(shape, dtype=np.float32)
        rng = np.random.default_rng(4)
        for y in range(70, 700, 70):
            for x in range(65, 760, 70):
                impulses[y + rng.integers(-10, 10), x + rng.integers(-10, 10)] = 1000
        previous = np.uint16(400 + cv2.GaussianBlur(impulses, (0, 0), 1.05))
        expected = np.array([[1, 0, 0.79], [0, 1, 0.39]], dtype=float)
        current = warp_reference(previous, expected)
        current[375, 350:415] += 24
        mask = np.ones(shape, dtype=bool)
        mask[:32] = mask[-32:] = mask[:, :32] = mask[:, -32:] = False
        source, destination = match_centroids(
            star_centroids(previous, mask), star_centroids(current, mask)
        )
        fitted, validation = estimate_registration(source, destination)
        self.assertIsNotNone(fitted)
        assert fitted is not None
        matrix = fitted[1]
        aligned = warp_reference(previous, matrix)
        error_before = np.linalg.norm(validation[0] - validation[1], axis=1)
        error_after = np.linalg.norm(
            transform_points(validation[0], matrix) - validation[1], axis=1
        )
        self.assertLess(np.median(error_after), 0.28 * np.median(error_before))
        self.assertLess(
            np.mean(cv2.absdiff(current, aligned)[mask]),
            0.35 * np.mean(cv2.absdiff(current, previous)[mask]),
        )
        trail = current[375, 350:415].astype(int) - aligned[375, 350:415]
        self.assertEqual(np.median(trail), 24)


if __name__ == "__main__":
    unittest.main()
