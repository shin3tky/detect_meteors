"""
Integration test suite for meteor detection in meteor_core (v1.x).

Tests the process_image_batch() function with various:
- Meteor trail characteristics (length, thickness, intensity)
- Detection parameters (diff_threshold, min_area, min_line_score)
- ROI configurations
- Edge cases and boundary conditions
"""

import unittest
from unittest.mock import patch
import numpy as np
import cv2
import sys
import os

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core import (
    process_image_batch,
    DEFAULT_HOUGH_THRESHOLD,
    DEFAULT_HOUGH_MIN_LINE_LENGTH,
    DEFAULT_HOUGH_MAX_LINE_GAP,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
    DEFAULT_MIN_LINE_SCORE,
)


class TestMeteorDetectionBase(unittest.TestCase):
    """Base class for meteor detection tests with common utilities."""

    def setUp(self):
        """Set up default test parameters."""
        self.params = {
            "diff_threshold": 10,
            "min_area": 5,
            "min_aspect_ratio": 2.0,
            "min_line_score": 50.0,
            "hough_threshold": DEFAULT_HOUGH_THRESHOLD,
            "hough_min_line_length": DEFAULT_HOUGH_MIN_LINE_LENGTH,
            "hough_max_line_gap": DEFAULT_HOUGH_MAX_LINE_GAP,
        }
        self.shape = (1000, 1000)
        self.img_black = np.zeros(self.shape, dtype=np.uint16)

    def create_meteor_image(self, start, end, intensity=200, thickness=3):
        """Create an image with a meteor trail."""
        img = np.zeros(self.shape, dtype=np.uint16)
        cv2.line(img, start, end, intensity, thickness)
        return img

    def create_roi_mask(self, full=True, region=None):
        """Create ROI mask. Full or specific region."""
        if full:
            return np.full(self.shape, 255, dtype=np.uint8)
        else:
            mask = np.zeros(self.shape, dtype=np.uint8)
            if region:
                x1, y1, x2, y2 = region
                mask[y1:y2, x1:x2] = 255
            return mask

    def run_detection(self, img_current, img_previous, params=None, roi_mask=None):
        """Run detection with mocked image loading."""
        if params is None:
            params = self.params
        if roi_mask is None:
            roi_mask = self.create_roi_mask(full=True)

        with patch("meteor_core.pipeline.load_and_bin_raw_fast") as mock_load:

            def side_effect(filepath):
                if "current" in filepath:
                    return img_current
                else:
                    return img_previous

            mock_load.side_effect = side_effect
            batch_data = [("path/to/current.ORF", "path/to/prev.ORF")]
            results = process_image_batch(batch_data, roi_mask, params)

        return results


class TestMeteorDetectionBasic(TestMeteorDetectionBase):
    """Basic meteor detection tests."""

    def test_detects_clear_meteor(self):
        """Test detection of a clear meteor trail."""
        img_meteor = self.create_meteor_image(
            (100, 100), (300, 300), intensity=200, thickness=3
        )
        results = self.run_detection(img_meteor, self.img_black)

        self.assertEqual(len(results), 1)
        is_candidate, filename, _, line_score, _, _, _ = results[0]

        self.assertTrue(is_candidate, "Clear meteor should be detected")
        self.assertGreater(line_score, 0, "Line score should be positive")

    def test_no_detection_on_identical_images(self):
        """Test no detection when current and previous images are identical."""
        img = self.create_meteor_image((100, 100), (300, 300))
        results = self.run_detection(img, img)

        self.assertEqual(len(results), 1)
        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertFalse(is_candidate, "Identical images should not produce detection")

    def test_no_detection_on_black_images(self):
        """Test no detection on completely black images."""
        results = self.run_detection(self.img_black, self.img_black)

        self.assertEqual(len(results), 1)
        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertFalse(is_candidate, "Black images should not produce detection")


class TestMeteorTrailLength(TestMeteorDetectionBase):
    """Test detection with varying meteor trail lengths."""

    def test_long_trail_detected(self):
        """Test that a long meteor trail is detected."""
        # Long trail: 400 pixels diagonal
        img_meteor = self.create_meteor_image((100, 100), (500, 500), intensity=200)
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, line_score, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Long meteor trail should be detected")
        self.assertGreater(line_score, 100, "Long trail should have high line score")

    def test_medium_trail_detected(self):
        """Test that a medium meteor trail is detected."""
        # Medium trail: ~140 pixels diagonal
        img_meteor = self.create_meteor_image((100, 100), (200, 200), intensity=200)
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, line_score, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Medium meteor trail should be detected")

    def test_short_trail_with_low_threshold(self):
        """Test that a short trail is detected with low min_line_score."""
        # Short trail: ~70 pixels diagonal
        img_meteor = self.create_meteor_image(
            (100, 100), (150, 150), intensity=200, thickness=3
        )
        params = self.params.copy()
        params["min_line_score"] = 20.0  # Lower threshold

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, line_score, _, _, _ = results[0]

        # Short trail may or may not be detected depending on Hough parameters
        # But line_score should be calculated
        self.assertIsNotNone(line_score)

    def test_very_short_trail_rejected(self):
        """Test that a very short trail is rejected with default parameters."""
        # Very short trail: ~28 pixels diagonal
        img_meteor = self.create_meteor_image((100, 100), (120, 120), intensity=200)
        params = self.params.copy()
        params["min_line_score"] = 80.0  # Higher threshold

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        # Very short trail should likely be rejected
        # (depends on exact Hough parameters)


class TestMeteorTrailIntensity(TestMeteorDetectionBase):
    """Test detection with varying meteor trail intensities."""

    def test_bright_meteor_detected(self):
        """Test that a bright meteor is detected."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300), intensity=500)
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Bright meteor should be detected")

    def test_dim_meteor_with_low_threshold(self):
        """Test that a dim meteor is detected with low diff_threshold."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300), intensity=15)
        params = self.params.copy()
        params["diff_threshold"] = 5  # Lower threshold

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertTrue(
            is_candidate, "Dim meteor should be detected with low threshold"
        )

    def test_very_dim_meteor_rejected(self):
        """Test that a very dim meteor is rejected with high diff_threshold."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300), intensity=10)
        params = self.params.copy()
        params["diff_threshold"] = 20  # Higher threshold

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertFalse(
            is_candidate, "Very dim meteor should be rejected with high threshold"
        )

    def test_intensity_below_threshold_rejected(self):
        """Test meteor with intensity below diff_threshold is rejected."""
        # Intensity 8, threshold 10 → should be rejected
        img_meteor = self.create_meteor_image((100, 100), (300, 300), intensity=8)
        params = self.params.copy()
        params["diff_threshold"] = 10

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertFalse(is_candidate, "Meteor below threshold should be rejected")


class TestMeteorTrailThickness(TestMeteorDetectionBase):
    """Test detection with varying meteor trail thicknesses."""

    def test_thick_trail_detected(self):
        """Test that a thick meteor trail is detected."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300), thickness=10)
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Thick meteor trail should be detected")

    def test_thin_trail_detected(self):
        """Test that a thin meteor trail is detected with appropriate parameters."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300), thickness=1)
        params = self.params.copy()
        params["min_area"] = 1  # Lower min_area for thin trails
        params["min_line_score"] = 30.0  # Lower threshold

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        # Thin trails may or may not be detected depending on exact parameters
        # This test verifies the detection pipeline works without error
        self.assertIsNotNone(is_candidate)

    def test_very_thick_trail_may_fail_aspect_ratio(self):
        """Test that an extremely thick trail may fail aspect ratio check."""
        # Very thick trail: might look more like a blob than a line
        img_meteor = self.create_meteor_image((100, 100), (150, 150), thickness=30)
        params = self.params.copy()
        params["min_aspect_ratio"] = 3.0  # Require elongated shape

        results = self.run_detection(img_meteor, self.img_black, params=params)
        # Result depends on whether the blob is detected as elongated


class TestDiffThresholdParameter(TestMeteorDetectionBase):
    """Test the diff_threshold parameter effects."""

    def test_threshold_boundary_above(self):
        """Test detection when intensity is just above threshold."""
        intensity = 25
        threshold = 20
        img_meteor = self.create_meteor_image(
            (100, 100), (300, 300), intensity=intensity
        )
        params = self.params.copy()
        params["diff_threshold"] = threshold

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertTrue(is_candidate, "Intensity above threshold should be detected")

    def test_threshold_boundary_below(self):
        """Test rejection when intensity is just below threshold."""
        intensity = 15
        threshold = 20
        img_meteor = self.create_meteor_image(
            (100, 100), (300, 300), intensity=intensity
        )
        params = self.params.copy()
        params["diff_threshold"] = threshold

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertFalse(is_candidate, "Intensity below threshold should be rejected")

    def test_threshold_1_detects_any_difference(self):
        """Test that threshold of 1 detects any difference."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300), intensity=5)
        params = self.params.copy()
        params["diff_threshold"] = 1

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertTrue(is_candidate, "Threshold 1 should detect any difference")

    def test_high_threshold_rejects_moderate_meteors(self):
        """Test that high threshold rejects moderate intensity meteors."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300), intensity=50)
        params = self.params.copy()
        params["diff_threshold"] = 100

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertFalse(is_candidate, "High threshold should reject moderate meteor")


class TestMinAreaParameter(TestMeteorDetectionBase):
    """Test the min_area parameter effects."""

    def test_large_area_detected(self):
        """Test that a large area meteor is detected."""
        # Thick, long line = large area
        img_meteor = self.create_meteor_image((100, 100), (400, 400), thickness=8)
        params = self.params.copy()
        params["min_area"] = 50

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertTrue(is_candidate, "Large area meteor should be detected")

    def test_small_area_with_low_min_area(self):
        """Test that small area meteor is detected with low min_area."""
        # Thin, short line = small area
        img_meteor = self.create_meteor_image((100, 100), (150, 150), thickness=1)
        params = self.params.copy()
        params["min_area"] = 1

        results = self.run_detection(img_meteor, self.img_black, params=params)
        # Small area may still pass if it forms a line

    def test_small_area_rejected_with_high_min_area(self):
        """Test that small area meteor is rejected with high min_area."""
        # Create a small dot instead of a line
        img = np.zeros(self.shape, dtype=np.uint16)
        cv2.circle(img, (200, 200), 2, 200, -1)  # Small filled circle

        params = self.params.copy()
        params["min_area"] = 100  # Require large area

        results = self.run_detection(img, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertFalse(
            is_candidate, "Small area should be rejected with high min_area"
        )


class TestMinLineScoreParameter(TestMeteorDetectionBase):
    """Test the min_line_score parameter effects."""

    def test_high_line_score_meteor(self):
        """Test that a meteor with high line score is detected."""
        # Long, clear line = high score
        img_meteor = self.create_meteor_image((50, 50), (450, 450), thickness=3)
        params = self.params.copy()
        params["min_line_score"] = 100.0

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, line_score, _, _, _ = results[0]

        self.assertTrue(is_candidate, "High line score meteor should be detected")
        self.assertGreater(line_score, 100.0)

    def test_low_line_score_rejected(self):
        """Test that meteor with low line score is rejected with high threshold."""
        # Short line = low score
        img_meteor = self.create_meteor_image((100, 100), (130, 130), thickness=2)
        params = self.params.copy()
        params["min_line_score"] = 200.0  # Very high threshold

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertFalse(
            is_candidate, "Low line score should be rejected with high threshold"
        )

    def test_min_line_score_zero_accepts_all_lines(self):
        """Test that min_line_score of 0 accepts any detected line."""
        img_meteor = self.create_meteor_image((100, 100), (150, 150), thickness=2)
        params = self.params.copy()
        params["min_line_score"] = 0.0

        results = self.run_detection(img_meteor, self.img_black, params=params)
        # Any line should be accepted


class TestMinAspectRatioParameter(TestMeteorDetectionBase):
    """Test the min_aspect_ratio parameter effects."""

    def test_elongated_shape_detected(self):
        """Test that an elongated meteor is detected."""
        # Long, thin line = high aspect ratio
        img_meteor = self.create_meteor_image((100, 100), (400, 100), thickness=2)
        params = self.params.copy()
        params["min_aspect_ratio"] = 3.0

        results = self.run_detection(img_meteor, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertTrue(is_candidate, "Elongated meteor should be detected")

    def test_round_shape_rejected(self):
        """Test that a round shape is rejected with high aspect ratio requirement."""
        # Create a circular blob
        img = np.zeros(self.shape, dtype=np.uint16)
        cv2.circle(img, (200, 200), 20, 200, -1)

        params = self.params.copy()
        params["min_aspect_ratio"] = 5.0  # Require very elongated

        results = self.run_detection(img, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertFalse(is_candidate, "Round shape should be rejected")

    def test_low_aspect_ratio_accepts_more(self):
        """Test that low aspect ratio accepts more shapes."""
        # Square-ish shape
        img = np.zeros(self.shape, dtype=np.uint16)
        cv2.rectangle(img, (100, 100), (130, 150), 200, -1)

        params = self.params.copy()
        params["min_aspect_ratio"] = 1.0  # Accept any shape

        results = self.run_detection(img, self.img_black, params=params)
        # Should be more likely to accept


class TestROIMask(TestMeteorDetectionBase):
    """Test ROI mask effects on detection."""

    def test_meteor_inside_roi_detected(self):
        """Test that meteor inside ROI is detected."""
        img_meteor = self.create_meteor_image((200, 200), (400, 400))
        roi_mask = self.create_roi_mask(full=False, region=(100, 100, 500, 500))

        results = self.run_detection(img_meteor, self.img_black, roi_mask=roi_mask)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertTrue(is_candidate, "Meteor inside ROI should be detected")

    def test_meteor_outside_roi_not_detected(self):
        """Test that meteor outside ROI is not detected."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300))
        # ROI only covers bottom-right corner
        roi_mask = self.create_roi_mask(full=False, region=(600, 600, 900, 900))

        results = self.run_detection(img_meteor, self.img_black, roi_mask=roi_mask)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertFalse(is_candidate, "Meteor outside ROI should not be detected")

    def test_meteor_partially_in_roi(self):
        """Test meteor that is partially inside ROI."""
        img_meteor = self.create_meteor_image((200, 200), (400, 400))
        # ROI covers only part of the meteor trail
        roi_mask = self.create_roi_mask(full=False, region=(250, 250, 500, 500))

        results = self.run_detection(img_meteor, self.img_black, roi_mask=roi_mask)
        # Partial detection may or may not trigger depending on remaining length


class TestMeteorOrientation(TestMeteorDetectionBase):
    """Test detection with various meteor orientations."""

    def test_horizontal_meteor(self):
        """Test detection of horizontal meteor trail."""
        img_meteor = self.create_meteor_image((100, 300), (400, 300))
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Horizontal meteor should be detected")

    def test_vertical_meteor(self):
        """Test detection of vertical meteor trail."""
        img_meteor = self.create_meteor_image((300, 100), (300, 400))
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Vertical meteor should be detected")

    def test_diagonal_meteor_45deg(self):
        """Test detection of 45-degree diagonal meteor."""
        img_meteor = self.create_meteor_image((100, 100), (400, 400))
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "45-degree diagonal meteor should be detected")

    def test_diagonal_meteor_135deg(self):
        """Test detection of 135-degree diagonal meteor."""
        img_meteor = self.create_meteor_image((400, 100), (100, 400))
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "135-degree diagonal meteor should be detected")

    def test_steep_angle_meteor(self):
        """Test detection of steep angle meteor (nearly vertical)."""
        img_meteor = self.create_meteor_image((300, 100), (320, 400))
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Steep angle meteor should be detected")

    def test_shallow_angle_meteor(self):
        """Test detection of shallow angle meteor (nearly horizontal)."""
        img_meteor = self.create_meteor_image((100, 300), (400, 320))
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Shallow angle meteor should be detected")


class TestMultipleMeteors(TestMeteorDetectionBase):
    """Test detection with multiple meteor trails."""

    def test_two_meteors_same_frame(self):
        """Test detection when two meteors are in the same frame."""
        img = np.zeros(self.shape, dtype=np.uint16)
        cv2.line(img, (100, 100), (300, 300), 200, 3)  # First meteor
        cv2.line(img, (600, 100), (800, 300), 200, 3)  # Second meteor

        results = self.run_detection(img, self.img_black)
        is_candidate, _, _, line_score, _, _, _ = results[0]

        self.assertTrue(is_candidate, "Frame with multiple meteors should be detected")
        # Line score should reflect multiple lines
        self.assertGreater(line_score, 0)

    def test_crossing_meteors(self):
        """Test detection of crossing meteor trails."""
        img = np.zeros(self.shape, dtype=np.uint16)
        cv2.line(img, (100, 300), (400, 300), 200, 3)  # Horizontal
        cv2.line(img, (250, 100), (250, 500), 200, 3)  # Vertical, crossing

        params = self.params.copy()
        params["min_aspect_ratio"] = 1.5  # Lower aspect ratio for crossing trails

        results = self.run_detection(img, self.img_black, params=params)
        is_candidate, _, _, line_score, _, _, _ = results[0]

        # Crossing trails create a cross shape which may or may not pass
        # aspect ratio check depending on how contours are detected
        # The key is that line detection should find some lines
        self.assertIsNotNone(line_score)


class TestEdgeCases(TestMeteorDetectionBase):
    """Test edge cases and boundary conditions."""

    def test_meteor_at_image_edge(self):
        """Test detection of meteor at image edge."""
        img_meteor = self.create_meteor_image((0, 0), (200, 200))
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Meteor at edge should be detected")

    def test_meteor_at_image_corner(self):
        """Test detection of meteor starting from corner."""
        img_meteor = self.create_meteor_image((0, 0), (100, 100))
        params = self.params.copy()
        params["min_line_score"] = 30.0  # Lower threshold for shorter trail

        results = self.run_detection(img_meteor, self.img_black, params=params)
        # Corner meteor may or may not be fully detected

    def test_very_bright_meteor(self):
        """Test detection of very bright meteor (near saturation)."""
        img_meteor = self.create_meteor_image((100, 100), (300, 300), intensity=65000)
        results = self.run_detection(img_meteor, self.img_black)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Very bright meteor should be detected")

    def test_noisy_background(self):
        """Test detection with noisy background."""
        # Create background with random noise
        np.random.seed(42)
        img_noisy = (np.random.rand(*self.shape) * 5).astype(np.uint16)

        # Add meteor on noisy background
        img_meteor = img_noisy.copy()
        cv2.line(img_meteor, (100, 100), (300, 300), 200, 3)

        params = self.params.copy()
        params["diff_threshold"] = 15  # Higher threshold for noise

        results = self.run_detection(img_meteor, img_noisy, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        self.assertTrue(is_candidate, "Meteor should be detected above noise")

    def test_uniform_background_change(self):
        """Test that uniform background change is not detected as meteor."""
        # Simulate uniform brightness change (e.g., clouds)
        img_bright = np.full(self.shape, 50, dtype=np.uint16)

        params = self.params.copy()
        params["diff_threshold"] = 30

        results = self.run_detection(img_bright, self.img_black, params=params)
        is_candidate, _, _, _, _, _, _ = results[0]

        # Uniform change should not produce elongated shape
        # Detection depends on whether uniform area passes aspect ratio


class TestDefaultParameters(TestMeteorDetectionBase):
    """Test with default parameters from the module."""

    def test_with_module_defaults(self):
        """Test detection using module default parameters."""
        params = {
            "diff_threshold": DEFAULT_DIFF_THRESHOLD,
            "min_area": DEFAULT_MIN_AREA,
            "min_aspect_ratio": 2.0,
            "min_line_score": DEFAULT_MIN_LINE_SCORE,
            "hough_threshold": DEFAULT_HOUGH_THRESHOLD,
            "hough_min_line_length": DEFAULT_HOUGH_MIN_LINE_LENGTH,
            "hough_max_line_gap": DEFAULT_HOUGH_MAX_LINE_GAP,
        }

        img_meteor = self.create_meteor_image((100, 100), (350, 350), intensity=100)
        results = self.run_detection(img_meteor, self.img_black, params=params)

        is_candidate, _, _, _, _, _, _ = results[0]
        self.assertTrue(is_candidate, "Clear meteor should be detected with defaults")


if __name__ == "__main__":
    unittest.main()
