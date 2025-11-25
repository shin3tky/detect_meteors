import unittest
from unittest.mock import patch
import numpy as np
import cv2
import sys
import os

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detect_meteors_cli import (
    process_image_batch,
    DEFAULT_HOUGH_THRESHOLD,
    DEFAULT_HOUGH_MIN_LINE_LENGTH,
    DEFAULT_HOUGH_MAX_LINE_GAP,
)


class TestMeteorDetection(unittest.TestCase):

    def setUp(self):
        # Set parameters for testing
        self.params = {
            "diff_threshold": 10,
            "min_area": 5,
            "min_aspect_ratio": 2.0,
            "min_line_score": 50.0,
            "hough_threshold": DEFAULT_HOUGH_THRESHOLD,
            "hough_min_line_length": DEFAULT_HOUGH_MIN_LINE_LENGTH,
            "hough_max_line_gap": DEFAULT_HOUGH_MAX_LINE_GAP,
        }

        # Dummy image size (height, width)
        self.shape = (1000, 1000)

        # Create an all-black image (value 0)
        self.img_black = np.zeros(self.shape, dtype=np.uint16)

        # Create an image with a meteor (white line on black background)
        self.img_meteor = np.zeros(self.shape, dtype=np.uint16)
        # Draw a line from (100, 100) to (300, 300) (thickness 3, intensity 200)
        cv2.line(self.img_meteor, (100, 100), (300, 300), 200, 3)

    @patch("detect_meteors_cli.load_and_bin_raw_fast")
    def test_process_image_batch_detects_meteor(self, mock_load):
        """
        Mock file loading and test detection logic with in-memory images
        """

        # Configure mock to return prepared images when load_and_bin_raw_fast is called
        # Return img_meteor (current image) on 1st call, img_black (previous image) on 2nd call
        # Because process_image_batch loads in the order of curr, prev
        def side_effect(filepath):
            if "current.ORF" in filepath:
                return self.img_meteor
            else:
                return self.img_black

        mock_load.side_effect = side_effect

        # Execute test
        # Dummy file path list (not actually read)
        batch_data = [("path/to/current.ORF", "path/to/prev.ORF")]

        # ROI mask (255 for all to target the entire area)
        roi_mask = np.full(self.shape, 255, dtype=np.uint8)

        # Execute function
        results = process_image_batch(batch_data, roi_mask, self.params)

        # Verify
        self.assertEqual(len(results), 1)  # Should return one result

        is_candidate, filename, _, line_score, _, _, _ = results[0]

        # Expected results:
        # 1. Should be detected as a candidate (True)
        self.assertTrue(is_candidate, "Meteor image was not detected as a candidate")

        # 2. Line score should be greater than 0
        self.assertGreater(line_score, 0, "Line score was not calculated")

        print(f"\nTest successful: Detected line score = {line_score}")


if __name__ == "__main__":
    unittest.main()
