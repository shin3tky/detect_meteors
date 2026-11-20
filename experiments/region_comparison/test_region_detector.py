import unittest

import numpy as np

from meteor_core.schema import DetectionContext
from region_detector import RegionConfig, RegionDetector, _measure_axis


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


class TestRefitContiguousSupport(unittest.TestCase):
    def test_a_gap_splits_the_support(self):
        points = np.array([(x, 16) for x in (*range(10, 30), *range(50, 70))])
        residual = np.zeros((64, 80), dtype=np.uint16)
        for x, y in points:
            residual[y - 1 : y + 2, x] = 11
        axis = _measure_axis(
            points,
            np.array((0, 16)),
            np.array((1, 0)),
            residual,
            np.ones_like(residual, dtype=bool),
            RegionConfig(ransac_min_contrast=9, ransac_min_contiguous_support_px=40),
            refit=True,
        )
        self.assertEqual(axis["length_px"], 59)
        self.assertEqual(axis["max_contiguous_support_px"], 19.5)
        self.assertTrue(axis["gates"]["ransac_contrast"])
        self.assertFalse(axis["gates"]["contiguous_support"])
        self.assertFalse(axis["accepted"])

    def test_refit_support_threshold_leaves_ordinary_fits_available(self):
        previous = np.full((64, 80), 100, dtype=np.uint16)
        roi = np.ones_like(previous)
        for contrast, threshold, expected in (
            (11, 40, True),
            (11, 41, False),
            (16, 41, True),
        ):
            with self.subTest(contrast=contrast, threshold=threshold):
                current = previous.copy()
                current[30:33, 10:51] += contrast
                detector = RegionDetector(
                    RegionConfig(
                        enable_ransac=True,
                        ransac_min_contrast=9,
                        ransac_min_contiguous_support_px=threshold,
                    )
                )
                result = detector.detect(
                    DetectionContext(current, previous, roi, {}, {})
                )
                self.assertEqual(result.is_candidate, expected)
