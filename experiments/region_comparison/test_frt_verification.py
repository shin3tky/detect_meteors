"""Check signed support, gap handling and selection of verified FRT geometry."""

import json
import unittest

import numpy as np

from benchmark_frt_verification import CONTRASTS, GAPS, LENGTHS, measure, score_verified
from frt_support import sample_profiles, support_runs
from segment_verification import noise_tiles


class FrtVerificationTests(unittest.TestCase):
    def test_gap_join_requires_valid_samples_and_supported_density(self):
        supported = np.array(
            [
                [True] * 5 + [False] * 2 + [True] * 3,
                [True] * 5 + [False] * 2 + [True] * 3,
                [True] * 2 + [False] * 3 + [True] * 2 + [False] * 3,
            ]
        )
        valid = np.ones_like(supported)
        valid[1, 5] = False
        row, start, end, length = support_runs(supported, valid, np.ones(3), 3)
        self.assertEqual(
            sorted(zip(row.tolist(), start.tolist(), end.tolist(), length.tolist())),
            [(0, 0, 10, 9), (1, 0, 5, 4), (1, 7, 10, 2), (2, 0, 2, 1), (2, 5, 7, 1)],
        )

    def test_bilateral_contrast_uses_the_brighter_side(self):
        residual = np.zeros((70, 100), dtype=np.float32)
        residual[29:32, :] = 12
        residual[22:25, :] = -8
        residual[36:39, :] = 10
        roi = np.ones_like(residual, dtype=bool)
        _, _, valid, center, pooled, bilateral, _ = sample_profiles(
            np.array([[10, 30, 70, 30]], dtype=float),
            60,
            residual,
            roi,
            noise_tiles(residual),
        )
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(center, 12)
        np.testing.assert_array_equal(pooled, 11)
        np.testing.assert_array_equal(bilateral, 2)

    def test_accepts_long_supported_frt_geometry_and_rejects_brighter_short_spike(self):
        residual = np.zeros((96, 120), dtype=np.float32)
        residual[29:32, 22:89] = 10
        residual[59:62, 33:45] = 300
        roi = np.ones_like(residual, dtype=bool)
        meteor = {
            "current": "meteor",
            "previous": "prior",
            "axis": [[22, 30], [88, 30]],
            "half_width_px": 2,
        }
        proposals = np.array(
            [[10, 30, 110, 30, 12, 0.9], [10, 60, 110, 60, 500, 0.8]],
            dtype=float,
        )
        trial = (CONTRASTS.index(("dn", 5)), GAPS.index(0), LENGTHS.index(45))
        measured = measure(proposals, residual, roi, [meteor], noise_tiles(residual))
        self.assertEqual(measured["scores"][trial], 12)
        sequence = {
            "frames": ["prior", "meteor"],
            "labels": [meteor],
            "truth": {"meteor": True},
            "pairs": {
                "meteor": {"hough": {"candidate": False, "coverage": {"meteor": 0.0}}}
            },
        }
        result = score_verified(sequence, {"meteor": measured}, trial, 8)
        self.assertTrue(result["events"]["meteor"]["appearance"]["match"])
        self.assertEqual(result["frames"]["tp"], ["meteor"])
        self.assertEqual(json.loads(json.dumps(result)), result)
