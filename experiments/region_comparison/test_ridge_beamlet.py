"""Geometry, sign, and diary coverage of the experimental line proposals."""

import unittest

import cv2
import numpy as np

from beamlet_detector import BeamletConfig, BeamletDetector, _dictionary
from meteor_core.schema import DetectionContext
from ridge_detector import RidgeConfig, RidgeDetector
from summarize import line_trail_evidence


class TestSignedTrailDetectors(unittest.TestCase):
    def test_localize_bright_trail_but_not_its_disappearance(self):
        previous = np.full((128, 128), 100, np.uint16)
        roi = np.full_like(previous, 255, dtype=np.uint8)
        cases = (((35, 60), (90, 60)), ((25, 25), (85, 85)))
        for start, end in cases:
            current = previous.copy()
            cv2.line(current, start, end, 160, 2)
            label = {"axis": (start, end), "half_width_px": 3}
            for detector in (
                RidgeDetector(RidgeConfig()),
                BeamletDetector(BeamletConfig()),
            ):
                with self.subTest(detector=detector.plugin_name, start=start):
                    result = detector.detect(
                        DetectionContext(current, previous, roi, {}, {})
                    )
                    evidence = line_trail_evidence(
                        label, result.to_dict(), detector.plugin_name
                    )
                    self.assertTrue(result.is_candidate)
                    self.assertGreaterEqual(evidence["accepted_coverage"], 0.5)
                    negative = detector.detect(
                        DetectionContext(previous, current, roi, {}, {})
                    )
                    self.assertFalse(negative.is_candidate)

    def test_do_not_return_a_trail_outside_the_roi(self):
        previous = np.full((96, 96), 100, np.uint16)
        current = previous.copy()
        cv2.line(current, (12, 28), (85, 28), 155, 2)
        roi = np.zeros_like(previous, dtype=np.uint8)
        roi[57:] = 255
        for detector in (
            RidgeDetector(RidgeConfig()),
            BeamletDetector(BeamletConfig()),
        ):
            with self.subTest(detector=detector.plugin_name):
                result = detector.detect(
                    DetectionContext(current, previous, roi, {}, {})
                )
                self.assertFalse(result.is_candidate)

    def test_beam_weights_count_each_sample_once(self):
        ends, offsets, weights = _dictionary(32, 64, 15)
        for beam, indices, coeff in zip(ends, offsets, weights, strict=True):
            points = indices[coeff > 0]
            self.assertEqual(len(points), np.max(np.abs(beam[0] - beam[1])) + 1)
            self.assertEqual(len(np.unique(points)), len(points))
            self.assertAlmostEqual(float(np.dot(coeff, coeff)), 1.0, places=6)
