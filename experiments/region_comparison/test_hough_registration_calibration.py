"""Check threshold sweep parity and the training recall constraint."""

import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from benchmark_hough_registration import (
    GRID,
    SHAPE,
    measure,
    positive_difference_pair,
    select_thresholds,
)
from meteor_core.detectors.hough_analysis import (
    CandidateThresholds,
    ContourMeasurement,
    HoughAnalysis,
)
from meteor_core.detectors.hough_default import HoughDetector
from meteor_core.pipeline import _build_runtime_params
from meteor_core.schema import DetectionContext, DetectionParams


class HoughRegistrationCalibrationTests(unittest.TestCase):
    def test_positive_difference_preserves_new_trail_and_suppresses_disappearance(self):
        previous = np.full((128, 128), 200, dtype=np.uint16)
        current = previous.copy()
        cv2.line(previous, (20, 40), (110, 40), 60000, 5)
        cv2.line(current, (20, 85), (110, 85), 350, 5)
        detector = HoughDetector()
        settings = _build_runtime_params(
            DetectionParams(diff_threshold=9, min_line_score=30).to_dict(), detector
        )
        context = DetectionContext(
            current, previous, np.full(current.shape, 255, np.uint8), settings, {}
        )
        absolute = detector.analyze(context)
        positive = detector.analyze(positive_difference_pair(context))

        self.assertGreater(cv2.countNonZero(absolute.mask[35:46]), 0)
        self.assertEqual(cv2.countNonZero(positive.mask[35:46]), 0)
        np.testing.assert_array_equal(positive.mask[80:91], absolute.mask[80:91])
        self.assertTrue(positive.evaluate().is_candidate)
        self.assertTrue(positive.lines)
        self.assertTrue(
            all(80 <= y1 <= 90 and 80 <= y2 <= 90 for _, y1, _, y2 in positive.lines)
        )
        self.assertEqual(current[85, 50], 350)
        self.assertEqual(previous[40, 50], 60000)

    def test_sweep_uses_strict_shape_and_inclusive_score_gates(self):
        image = np.zeros((16, 16), dtype=np.uint16)
        mask = np.full_like(image, 255, dtype=np.uint8)
        evidence = HoughAnalysis(
            mask=mask,
            score=30,
            lines=[],
            contours=(
                ContourMeasurement(3, ((5, 5), (2, 14), 0)),
                ContourMeasurement(10, ((7, 7), (3, 9), 0)),
                ContourMeasurement(20, ((8, 8), (0, 12), 0)),
            ),
            thresholds=CandidateThresholds(3, 3, 30),
        )
        detector = Mock(plugin_name="hough")
        detector.analyze.return_value = evidence
        context = DetectionContext(image, image, mask, {}, {})
        decisions, _ = measure(detector, context, {}, [])
        self.assertEqual(detector.analyze.call_count, len(GRID["diff_threshold"]))
        for area, ratio, score in np.ndindex(*SHAPE[1:]):
            thresholds = CandidateThresholds(
                GRID["min_area"][area],
                GRID["min_aspect_ratio"][ratio],
                GRID["min_line_score"][score],
            )
            expected = evidence.evaluate(thresholds).is_candidate
            np.testing.assert_array_equal(decisions[:, area, ratio, score], expected)

    def test_selection_requires_the_same_meteor_frame_and_appearance_match(self):
        sample = "_C140344.ORF"
        future = "_C140345.ORF"
        unrelated = "_C140346.ORF"
        no_coverage = np.zeros(SHAPE[0])
        good_coverage = no_coverage.copy()
        good_coverage[0] = good_coverage[2] = 1.0

        baseline = (0, 0, 0, 0)
        lost_appearance = (1, 0, 0, 0)
        valid = (2, 0, 0, 0)
        base_detection = np.zeros(SHAPE, dtype=bool)
        base_detection[baseline] = True
        adjusted_detection = np.zeros(SHAPE, dtype=bool)
        adjusted_detection[lost_appearance] = adjusted_detection[valid] = True
        negatives = np.zeros(SHAPE, dtype=bool)
        negatives[valid] = True
        empty = np.zeros(SHAPE, dtype=bool)
        sequence = {
            "frames": ["_C140343.ORF", sample, future, unrelated],
            "truth": {sample: True, future: False, unrelated: False},
            "labels": [{"current": sample}],
            "pairs": {
                "unaligned": {
                    sample: (base_detection, {sample: good_coverage}),
                    future: (empty, {sample: no_coverage}),
                    unrelated: (empty, {}),
                },
                "aligned": {
                    sample: (adjusted_detection, {sample: good_coverage}),
                    future: (empty, {sample: no_coverage}),
                    unrelated: (negatives, {}),
                },
            },
        }
        self.assertEqual(select_thresholds([sequence], "aligned", baseline), valid)
