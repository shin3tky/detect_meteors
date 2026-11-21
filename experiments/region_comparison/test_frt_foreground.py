"""Verify foreground measurements and the training-only FRT score margin."""

from math import inf, nextafter, sqrt
import unittest

import numpy as np

from benchmark_frt_foreground import (
    GAPS,
    LENGTHS,
    LEVELS,
    SHAPE,
    TEXTURE_CAPS,
    highest_axis_cutoff,
    improves_training_coverage,
    measure,
    train_and_test,
)
from meteor_core.detectors.frt_support import interval_texture, noise_tiles, texture_map


class FrtForegroundTests(unittest.TestCase):
    def test_texture_is_zero_on_flat_raw_and_high_at_a_background_edge(self):
        raw = np.full((64, 64), 300, dtype=np.uint16)
        np.testing.assert_allclose(texture_map(raw), 0, atol=1e-4)
        raw[:, :32] = 150
        measured = texture_map(raw)
        self.assertGreater(measured[32, 31], 20)
        self.assertAlmostEqual(measured[32, 10], 0, places=3)
        profiles = np.array([[99, 0, 1, 2, 3, 99], [99, 2, 4, 99, 99, 99]])
        np.testing.assert_allclose(
            interval_texture(
                profiles, np.array([0, 1]), np.array([1, 1]), np.array([5, 3])
            ),
            [2.7, 3.8],
        )

    def test_texture_gate_rejects_supported_line_on_rough_background(self):
        residual = np.zeros((96, 120), dtype=np.float32)
        residual[29:32, 22:89] = 10
        residual[59:62, 22:89] = 10
        roi = np.ones_like(residual, dtype=bool)
        proposals = np.array(
            [[10, 30, 110, 30, 12, 0.9], [10, 60, 110, 60, 50, 0.9]], dtype=float
        )
        texture = np.full_like(residual, 0.4)
        texture[60, :] = 2.5
        base = (LEVELS.index(1), GAPS.index(0), LENGTHS.index(45))
        scores, _ = measure(
            proposals, residual, roi, noise_tiles(residual), texture, []
        )
        self.assertEqual(scores[(*base, TEXTURE_CAPS.index(1))], 12)
        self.assertEqual(scores[(*base, TEXTURE_CAPS.index(3))], 50)

    def test_axis_limit_uses_union_of_supported_geometry_and_inclusive_threshold(self):
        label = {"axis": [[0, 5], [100, 5]], "half_width_px": 2}
        lines = [
            (30, [0, 5, 32, 5]),
            (20, [32, 5, 52, 5]),
            (40, [0, 12, 100, 12]),
        ]
        self.assertEqual(highest_axis_cutoff(label, lines, 8), 20)
        self.assertIsNone(highest_axis_cutoff(label, lines, 21))

    def test_margin_comes_from_training_scores_only(self):
        trial = (0, 0, 0, 0)
        label = {
            "current": "meteor",
            "previous": "quiet",
            "axis": [[0, 5], [40, 5]],
            "half_width_px": 2,
        }
        train = {
            "frames": ["first", "quiet", "meteor"],
            "labels": [label],
            "truth": {"quiet": False, "meteor": True},
            "pairs": {name: {} for name in ("quiet", "meteor")},
        }
        test = {
            "frames": ["first", "bright"],
            "labels": [],
            "truth": {"bright": False},
            "pairs": {"bright": {}},
        }
        scores = [np.zeros(SHAPE) for _ in range(3)]
        for values, score in zip(scores, (10, 20, 18), strict=True):
            values[trial] = score
        verified = {
            "train": {
                "quiet": (scores[0], {}),
                "meteor": (
                    scores[1],
                    {"meteor": {(0, 0): [(0, 0, 20, [0, 5, 40, 5])]}},
                ),
            },
            "test": {"bright": (scores[2], {})},
        }
        result = train_and_test(
            {"train": train},
            {"test": test},
            verified,
            calibrate_snr=True,
            use_margin=True,
        )
        self.assertEqual(
            result["settings"]["frt_min_snr"], sqrt(nextafter(10, inf) * 20)
        )
        self.assertEqual(result["training"]["train"]["frames"]["fp"], [])
        self.assertTrue(
            result["training"]["train"]["events"]["meteor"]["appearance"]["match"]
        )
        self.assertEqual(result["evaluation"]["test"]["frames"]["fp"], ["bright"])

    def test_relaxation_requires_improved_training_coverage_and_no_new_fp(self):
        def result(coverage, false_positive=False):
            return {
                "train": {
                    "frames": {"fp": ["false"] if false_positive else []},
                    "events": {
                        name: {"appearance": {"coverage": value, "match": value >= 0.5}}
                        for name, value in coverage.items()
                    },
                }
            }

        original = result({"first": 0.55, "second": 0.80})
        self.assertTrue(
            improves_training_coverage(
                original, result({"first": 0.90, "second": 0.80})
            )
        )
        self.assertFalse(
            improves_training_coverage(
                original, result({"first": 0.90, "second": 0.80}, false_positive=True)
            )
        )
        self.assertFalse(
            improves_training_coverage(
                original, result({"first": 0.90, "second": 0.79})
            )
        )
        self.assertFalse(improves_training_coverage(original, original))
