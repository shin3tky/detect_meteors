import unittest

import numpy as np

from meteor_core.schema import DetectionContext
from region_detector import RegionConfig, RegionDetector


class TestRefitContrast(unittest.TestCase):
    def test_explicit_threshold_applies_only_to_refit_and_keeps_polarity(self):
        previous = np.full((64, 80), 100, dtype=np.uint16)
        roi = np.ones_like(previous)
        for contrast in (9, 11):
            current = previous + 2
            current[30:33, 15:65] += contrast
            for threshold in (None, 20, 11, 9):
                detector = RegionDetector(
                    RegionConfig(
                        enable_ransac=True,
                        ransac_min_contrast=threshold,
                        reject_negative_axis_fraction=0.75,
                        min_positive_axis_fraction=0.75,
                    )
                )
                for before, after, positive in (
                    (previous, current, True),
                    (current, previous, False),
                ):
                    with self.subTest(
                        contrast=contrast, threshold=threshold, positive=positive
                    ):
                        result = detector.detect(
                            DetectionContext(after, before, roi, {}, {})
                        )
                        (region,) = result.extras["region"]["regions"]
                        ransac = region["ransac"]
                        self.assertFalse(ransac["original"]["gates"]["contrast"])
                        self.assertEqual(ransac["refit"]["contrast"], contrast)
                        passes = threshold is not None and contrast >= threshold
                        self.assertEqual(ransac["refit"]["accepted"], passes)
                        self.assertEqual(result.is_candidate, passes and positive)
