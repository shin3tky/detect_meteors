"""Regression coverage for reusable Hough measurements and v1.x decisions."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from meteor_core.detectors.hough_analysis import CandidateThresholds
from meteor_core.detectors.hough_default import HoughDetector
from meteor_core.schema import DetectionContext, DetectionParams


class TestHoughAnalysis(unittest.TestCase):
    def setUp(self):
        self.detector = HoughDetector()
        self.params = DetectionParams(
            diff_threshold=10, min_area=3, min_aspect_ratio=3.0, min_line_score=30.0
        )

    def context(self, image, params=None, roi=None):
        return DetectionContext(
            current_image=image,
            previous_image=np.zeros_like(image),
            roi_mask=np.full(image.shape, 255, np.uint8) if roi is None else roi,
            runtime_params=self.detector.build_runtime_params(
                (params or self.params).to_dict()
            ),
            metadata={},
        )

    def test_normal_detection_preserves_geometry_and_output(self):
        image = np.zeros((96, 96), np.uint16)
        image[30:35, 10:80] = 50
        result = self.detector.detect(self.context(image))

        self.assertTrue(result.is_candidate)
        self.assertEqual(result.aspect_ratio, 17.25)
        self.assertGreater(result.score, self.params.min_line_score)
        self.assertEqual(result.metrics["mask_area"], 350)
        self.assertEqual(result.metrics["num_contours"], 1)
        self.assertEqual(result.metrics["hough_votes"], len(result.lines))
        self.assertEqual(result.extras, {})
        self.assertEqual(result.debug_image.shape, (96, 96, 3))

    def test_strong_wide_feature_can_be_rechecked_without_running_hough(self):
        image = np.zeros((96, 96), np.uint16)
        image[20:45, 10:70] = 1000
        context = self.context(image)
        with patch.object(
            self.detector, "compute_line_score", wraps=self.detector.compute_line_score
        ) as compute:
            analysis = self.detector.analyze(context)
            original = analysis.evaluate()
            adjusted = analysis.evaluate(
                replace(analysis.thresholds, min_aspect_ratio=2.0)
            )
            self.assertEqual(compute.call_count, 1)

        self.assertGreater(analysis.score, self.params.min_line_score)
        self.assertFalse(original.is_candidate)
        self.assertEqual(original.rejection_reasons, ("min_aspect_ratio",))
        self.assertTrue(adjusted.is_candidate)
        self.assertEqual(adjusted.rejection_reasons, ())
        self.assertFalse(analysis.evaluate().is_candidate)

        # Diagnostics can draw a rejected pair; normal detection still does not.
        self.assertEqual(analysis.render_debug(original).shape, (96, 96, 3))
        self.assertIsNone(self.detector.detect(context).debug_image)
        adjusted_params = replace(self.params, min_aspect_ratio=2.0)
        result = self.detector.detect(self.context(image, adjusted_params))
        self.assertTrue(result.is_candidate)
        self.assertEqual(result.score, analysis.score)
        self.assertEqual(result.aspect_ratio, adjusted.max_aspect_ratio)
        np.testing.assert_array_equal(
            result.debug_image, analysis.render_debug(adjusted)
        )

    def test_threshold_boundaries_match_existing_candidate_rule(self):
        image = np.zeros((96, 96), np.uint16)
        image[30:35, 10:80] = 50
        analysis = self.detector.analyze(self.context(image))
        area = analysis.contours[0].area
        ratio = analysis.contours[0].aspect_ratio
        thresholds = CandidateThresholds(area, ratio, analysis.score)

        self.assertEqual(analysis.evaluate(thresholds).rejection_reasons, ("min_area",))
        self.assertEqual(
            analysis.evaluate(replace(thresholds, min_area=area - 1)).rejection_reasons,
            ("min_aspect_ratio",),
        )
        at_score = replace(thresholds, min_area=area - 1, min_aspect_ratio=ratio - 0.1)
        self.assertTrue(analysis.evaluate(at_score).is_candidate)
        above_score = replace(
            at_score, min_line_score=float(np.nextafter(analysis.score, np.inf))
        )
        self.assertEqual(
            analysis.evaluate(above_score).rejection_reasons, ("min_line_score",)
        )

    def test_area_and_ratio_must_pass_for_the_same_contour(self):
        image = np.zeros((96, 96), np.uint16)
        image[10:13, 10:41] = 50  # Thin contour below the initial area threshold.
        image[60:81, 10:31] = 50  # Large square above the area threshold.
        params = replace(self.params, min_area=100)
        analysis = self.detector.analyze(self.context(image, params))
        decision = analysis.evaluate()

        self.assertFalse(decision.is_candidate)
        self.assertEqual(decision.max_aspect_ratio, 1.0)
        lowered = analysis.evaluate(replace(analysis.thresholds, min_area=3))
        self.assertTrue(lowered.is_candidate)
        self.assertEqual(lowered.max_aspect_ratio, 15.0)

    def test_roi_exclusion_and_absent_signal_do_not_suggest_a_shape_failure(self):
        image = np.zeros((96, 96), np.uint16)
        image[30:35, 10:80] = 50
        roi = np.zeros_like(image, dtype=np.uint8)
        analysis = self.detector.analyze(self.context(image, roi=roi))
        decision = analysis.evaluate()

        self.assertFalse(decision.is_candidate)
        self.assertEqual(analysis.score, 0.0)
        self.assertEqual(decision.max_aspect_ratio, 0.0)
        self.assertEqual(decision.rejection_reasons, ("no_contours", "min_line_score"))

    def test_namespaced_hough_settings_remain_supported(self):
        image = np.zeros((96, 96), np.uint16)
        image[30:35, 10:80] = 50
        flat_context = self.context(image)
        namespaced_context = self.context(image)
        params = self.params.to_dict()
        hough = params.pop("hough_threshold")
        length = params.pop("hough_min_line_length")
        gap = params.pop("hough_max_line_gap")
        namespaced_context.runtime_params = {
            "global": {**params, "min_aspect_ratio": 100.0},
            "detector": {
                "hough": {
                    "min_aspect_ratio": 3.0,
                    "hough": {
                        "threshold": hough,
                        "min_line_length": length,
                        "max_line_gap": gap,
                    },
                }
            },
        }
        original = self.detector.detect(flat_context)
        result = self.detector.detect(namespaced_context)
        self.assertEqual(result.is_candidate, original.is_candidate)
        self.assertEqual(result.score, original.score)
        self.assertEqual(result.lines, original.lines)
        self.assertEqual(result.aspect_ratio, original.aspect_ratio)
        np.testing.assert_array_equal(result.debug_image, original.debug_image)


if __name__ == "__main__":
    unittest.main()
