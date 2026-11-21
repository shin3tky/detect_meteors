"""Check digital line indexing, propagated variance, and signed detection."""

import unittest

import cv2
import numpy as np

from frt_detector import FrtConfig, FrtDetector
from meteor_core.detectors.frt_projection import _digital_offsets, _frt_stages
from meteor_core.schema import DetectionContext
from summarize import line_trail_evidence


class TestFastRadonDetector(unittest.TestCase):
    def test_transform_matches_explicit_sums_and_preserves_padding(self):
        height, width = 29, 22
        rng = np.random.default_rng(6)
        channels = rng.integers(0, 8, size=(height, width, 3)).astype(np.float32)
        for span, hop, sums in _frt_stages(channels, 32):
            for delta in (1 - span, -5, -1, 0, 3, span - 1):
                offsets = _digital_offsets(span, delta)
                self.assertEqual((offsets[0], offsets[-1]), (0, delta))
                for band in range(sums.shape[1]):
                    y = band * hop + np.arange(span)
                    for intercept in (0, 7, width - 1):
                        x = intercept + offsets
                        ok = (y < height) & (x >= 0) & (x < width)
                        expected = channels[y[ok], x[ok]].sum(axis=0)
                        np.testing.assert_array_equal(
                            sums[delta + span - 1, band, intercept], expected
                        )

    def test_score_is_sum_divided_by_propagated_noise(self):
        span = 32
        channels = np.full((span, 10, 3), (5, 4, 1), np.float32)
        _, _, sums = tuple(_frt_stages(channels, span))[-1]
        signal, variance, support = sums[span - 1, 0, 0]
        self.assertEqual(support, span)
        self.assertAlmostEqual(
            signal / np.sqrt(variance), 5 * np.sqrt(span) / 2, places=5
        )

    def test_recovers_positive_trails_in_both_orientations(self):
        previous = np.full((120, 128), 100, np.uint16)
        roi = np.ones_like(previous, dtype=bool)
        detector = FrtDetector(FrtConfig(max_span=64))
        for axis in (((37, 20), (40, 77)), ((15, 72), (85, 46))):
            with self.subTest(axis=axis):
                current = previous.copy()
                cv2.line(current, *axis, 105, 2)
                label = {"axis": axis, "half_width_px": 3}
                result = detector.detect(
                    DetectionContext(current, previous, roi, {}, {})
                )
                self.assertTrue(result.is_candidate)
                evidence = line_trail_evidence(label, result.to_dict(), "frt")
                self.assertGreaterEqual(evidence["accepted_coverage"], 0.5)
                reverse = detector.detect(
                    DetectionContext(previous, current, roi, {}, {})
                )
                self.assertFalse(reverse.is_candidate)
