"""Tests for causal alignment and false-positive budget replay."""

import unittest

import numpy as np

from benchmark_registered_median import cutoff, score_sequence
from benchmark_v2 import temporal_median
from registered_median import registered_median
from star_registration import transform_points, warp_reference


class RegistrationBackgroundTests(unittest.TestCase):
    @staticmethod
    def fixture():
        x, y = np.meshgrid(np.arange(60, 820, 59), np.arange(80, 750, 61))
        coordinates = np.column_stack((x.ravel(), y.ravel())).astype(float)
        yy, xx = np.indices((820, 860), dtype=np.float32)
        scene = np.rint(460 + 45 * np.sin(xx / 17) + 35 * np.cos(yy / 23)).astype(
            np.uint16
        )
        matrices = [
            np.array([[1, 0, dx], [0, 1, dy]], dtype=float)
            for dx, dy in ((-3, 2), (-2, 1), (-1, 0))
        ]
        images = [warp_reference(scene, matrix) for matrix in matrices]
        catalogues = [transform_points(coordinates, matrix) for matrix in matrices]
        return scene, images, catalogues, coordinates

    def test_direct_translation_recovers_reference_in_current_coordinates(self):
        expected, refs, catalogues, destination = self.fixture()
        refs[0][230:232, 180:240] += 80
        snapshots = [frame.copy() for frame in refs]
        result, quality = registered_median(refs, catalogues, destination, max_shift=10)
        self.assertTrue(quality["used_alignment"])
        self.assertEqual(
            [r["shift_xy"] for r in quality["references"]],
            [[3, -2], [2, -1], [1, 0]],
        )
        np.testing.assert_array_equal(result[15:-15, 15:-15], expected[15:-15, 15:-15])
        for image, snapshot in zip(refs, snapshots, strict=True):
            np.testing.assert_array_equal(image, snapshot)

    def test_weak_or_excessive_shift_falls_back_for_the_whole_window(self):
        _, refs, catalogues, dest = self.fixture()
        for sources, limit in (
            ([catalogues[0], catalogues[1][:0], catalogues[2]], 10),
            (catalogues, 2),
        ):
            with self.subTest(limit=limit):
                observed, quality = registered_median(
                    refs, sources, dest, max_shift=limit
                )
                self.assertFalse(quality["used_alignment"])
                np.testing.assert_array_equal(observed, temporal_median(refs))
        with self.assertRaisesRegex(ValueError, "three preceding"):
            registered_median(refs[:2], catalogues[:2], dest, max_shift=10)


class CalibrationTests(unittest.TestCase):
    def test_budget_resolves_boundary_ties_conservatively(self):
        negatives = [12, 10, 10, 5, 4]
        self.assertEqual(cutoff(negatives, 0.1), 12)
        self.assertEqual(cutoff(negatives, 0.4), 10)
        self.assertEqual(sum(score > cutoff(negatives, 0.4) for score in negatives), 1)
        self.assertEqual(cutoff([0, 0, 0], 0.5), 0)
        with self.assertRaises(ValueError):
            cutoff([], 0.1)

    def test_geometric_match_reapplies_cutoff_to_the_line(self):
        names = ["first", "earliest", "meteor", "next"]
        label = {
            "current": "meteor",
            "previous": "earliest",
            "axis": [[0, 0], [20, 0]],
            "half_width_px": 2,
        }

        def record(score, lines, *, warmup=False):
            return {
                "warmup": warmup,
                "policies": {"adjacent": {"score": score, "relevant_lines": lines}},
            }

        run = {
            "frames": names,
            "truth": {"first": False, "earliest": False, "meteor": True, "next": False},
            "annotations": [label],
            "records": {
                "earliest": record(12, {}, warmup=True),
                "meteor": record(35, {"meteor": [(14, [0, 0, 20, 0])]}),
                "next": record(0, {"meteor": []}),
            },
        }
        scored = score_sequence(run, "adjacent", 14)
        self.assertEqual(scored["all_pairs"]["tp"], ["meteor"])
        self.assertEqual(scored["mature_pairs"]["fp"], [])
        self.assertFalse(scored["events"]["meteor"]["event_match"])
        self.assertTrue(
            score_sequence(run, "adjacent", 13)["events"]["meteor"]["event_match"]
        )


if __name__ == "__main__":
    unittest.main()
