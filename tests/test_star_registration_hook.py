"""Exercise stellar registration with the normal pair and Hough entry points."""

from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from meteor_core.hooks import HookRegistry
from meteor_core.pipeline import process_image_batch
from meteor_core.schema import (
    DetectionContext,
    DetectionParams,
    HookConfig,
    InputContext,
)


class TestStarRegistrationHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shape = (780, 840)
        impulses = np.zeros(shape, dtype=np.float32)
        random = np.random.default_rng(4)
        for y in range(70, 720, 61):
            for x in range(65, 780, 64):
                impulses[y + random.integers(-10, 10), x + random.integers(-10, 10)] = (
                    1600
                )
        cls.previous = (400 + cv2.GaussianBlur(impulses, (0, 0), 1.05)).astype(
            np.uint16
        )
        cls.matrix = cv2.getRotationMatrix2D((0, 0), -0.21, 1)
        cls.matrix[:, 2] = (3.2, -1.6)
        cls.current = cv2.warpAffine(cls.previous, cls.matrix, shape[::-1])
        cls.roi = np.zeros(shape, dtype=np.uint8)
        cls.roi[30:750, 25:815] = 255

    def make_context(self):
        return DetectionContext(
            self.current.copy(), self.previous.copy(), self.roi.copy(), {}, {}
        )

    def test_rotation_translation_and_mask_for_uint16_and_normalized_raw(self):
        hook = HookRegistry.create("star_registration")
        for normalized in (False, True):
            with self.subTest(normalized=normalized):
                original = self.make_context()
                if normalized:
                    original = replace(
                        original,
                        current_image=original.current_image.astype(np.float32) / 65535,
                        previous_image=original.previous_image.astype(np.float32)
                        / 65535,
                    )
                aligned = hook.on_before_detection(original)
                diagnostics = aligned.metadata["star_registration"]
                self.assertEqual(diagnostics["status"], "applied")
                self.assertGreaterEqual(diagnostics["inliers"], 14)
                self.assertEqual(json.loads(json.dumps(diagnostics)), diagnostics)
                matrix = np.asarray(diagnostics["matrix"])
                np.testing.assert_allclose(
                    matrix[:, :2], self.matrix[:, :2], atol=0.0003
                )
                np.testing.assert_allclose(matrix[:, 2], self.matrix[:, 2], atol=0.2)
                self.assertIs(aligned.current_image, original.current_image)
                self.assertEqual(
                    aligned.previous_image.dtype, original.previous_image.dtype
                )
                np.testing.assert_array_equal(original.roi_mask, self.roi)
                self.assertEqual(original.metadata, {})

                self.assertEqual(aligned.roi_mask[200, 26], 0)
                self.assertEqual(aligned.roi_mask[200, 40], 255)
                self.assertFalse(np.any(aligned.roi_mask[self.roi == 0]))
                usable = aligned.roi_mask != 0
                before = cv2.absdiff(original.current_image, original.previous_image)
                after = cv2.absdiff(aligned.current_image, aligned.previous_image)
                self.assertLess(after[usable].mean(), before[usable].mean() * 0.20)

    def test_unusable_roi_falls_back_for_the_pair(self):
        roi = np.zeros_like(self.roi)
        roi[30:250, 25:250] = 255
        original = replace(self.make_context(), roi_mask=roi)
        actual = HookRegistry.create("star_registration").on_before_detection(original)
        self.assertEqual(actual.metadata["star_registration"]["status"], "fit_rejected")
        self.assertIs(actual.current_image, original.current_image)
        self.assertIs(actual.previous_image, original.previous_image)
        self.assertIs(actual.roi_mask, roi)

    def test_hough_detects_the_current_trail_in_original_coordinates(self):
        current = self.current.copy()
        cv2.line(current, (300, 375), (580, 375), 1500, 5)
        loader = Mock()
        loader.load.side_effect = [
            InputContext(current, "current.ORF"),
            InputContext(self.previous, "previous.ORF"),
        ]
        (pair,) = process_image_batch(
            [(1, "current.ORF", "previous.ORF")],
            self.roi,
            DetectionParams(diff_threshold=20, min_line_score=60).to_dict(),
            input_loader=loader,
            debug_image_enabled=True,
            hook_configs=[HookConfig("star_registration")],
        )
        result, context_payload = pair[7:]
        self.assertTrue(result.is_candidate)
        self.assertEqual(
            context_payload["metadata"]["star_registration"]["status"], "applied"
        )
        self.assertTrue(
            any(
                abs(y1 - 375) <= 4 and abs(y2 - 375) <= 4 and abs(x2 - x1) >= 200
                for x1, y1, x2, y2 in result.lines
            )
        )
        self.assertEqual(result.debug_image.shape[:2], self.current.shape)
