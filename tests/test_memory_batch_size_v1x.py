"""
Test suite for memory-based batch size adjustment (v1.0.3+).

Tests the estimate_batch_size() function that automatically adjusts
batch size based on available system memory.
"""

import unittest
import sys
import os
from unittest.mock import patch

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from detect_meteors_cli import (
    estimate_batch_size,
    AUTO_BATCH_MEMORY_FRACTION,
)


class TestMemoryBasedBatchSizeAdjustment(unittest.TestCase):
    """Test suite for memory-based worker/batch size adjustment."""

    def test_batch_size_with_abundant_memory(self):
        """Test that batch size is not reduced when memory is abundant."""
        # 16GB available memory
        available_mem = 16 * 1024 * 1024 * 1024
        image_shape = (3912, 5240)  # Typical MFT resolution
        num_workers = 4
        requested_batch_size = 10

        result = estimate_batch_size(
            requested_batch_size=requested_batch_size,
            image_shape=image_shape,
            num_workers=num_workers,
            available_mem=available_mem,
        )

        # With abundant memory, should return requested batch size
        self.assertEqual(result, requested_batch_size)

    def test_batch_size_with_limited_memory(self):
        """Test that batch size is reduced when memory is limited."""
        # 1GB available memory - should limit batch size
        available_mem = 1 * 1024 * 1024 * 1024
        image_shape = (3912, 5240)  # Typical MFT resolution
        num_workers = 8
        requested_batch_size = 50

        result = estimate_batch_size(
            requested_batch_size=requested_batch_size,
            image_shape=image_shape,
            num_workers=num_workers,
            available_mem=available_mem,
        )

        # Should reduce batch size
        self.assertLess(result, requested_batch_size)
        self.assertGreaterEqual(result, 1)

    def test_batch_size_minimum_is_one(self):
        """Test that batch size never goes below 1."""
        # Very limited memory
        available_mem = 100 * 1024 * 1024  # 100MB
        image_shape = (7000, 9000)  # Large sensor resolution
        num_workers = 16
        requested_batch_size = 100

        result = estimate_batch_size(
            requested_batch_size=requested_batch_size,
            image_shape=image_shape,
            num_workers=num_workers,
            available_mem=available_mem,
        )

        # Should be at least 1
        self.assertGreaterEqual(result, 1)

    def test_batch_size_with_no_memory_info(self):
        """Test that requested batch size is used when memory info unavailable."""
        requested_batch_size = 15
        image_shape = (3912, 5240)
        num_workers = 4

        # Mock get_available_memory_bytes to return None
        with patch("detect_meteors_cli.get_available_memory_bytes", return_value=None):
            result = estimate_batch_size(
                requested_batch_size=requested_batch_size,
                image_shape=image_shape,
                num_workers=num_workers,
                available_mem=None,  # Will call get_available_memory_bytes()
            )

            # Should return requested batch size when memory info is unavailable
            self.assertEqual(result, requested_batch_size)

    def test_batch_size_scales_with_workers(self):
        """Test that batch size scales inversely with number of workers."""
        available_mem = 4 * 1024 * 1024 * 1024  # 4GB
        image_shape = (3912, 5240)
        requested_batch_size = 50

        # More workers = less memory per worker = smaller batch size
        result_4_workers = estimate_batch_size(
            requested_batch_size=requested_batch_size,
            image_shape=image_shape,
            num_workers=4,
            available_mem=available_mem,
        )

        result_8_workers = estimate_batch_size(
            requested_batch_size=requested_batch_size,
            image_shape=image_shape,
            num_workers=8,
            available_mem=available_mem,
        )

        # 8 workers should have smaller or equal batch size compared to 4 workers
        self.assertLessEqual(result_8_workers, result_4_workers)

    def test_batch_size_scales_with_image_size(self):
        """Test that batch size scales inversely with image size."""
        available_mem = 4 * 1024 * 1024 * 1024  # 4GB
        num_workers = 4
        requested_batch_size = 50

        # Smaller image
        result_small = estimate_batch_size(
            requested_batch_size=requested_batch_size,
            image_shape=(2000, 3000),
            num_workers=num_workers,
            available_mem=available_mem,
        )

        # Larger image
        result_large = estimate_batch_size(
            requested_batch_size=requested_batch_size,
            image_shape=(6000, 8000),
            num_workers=num_workers,
            available_mem=available_mem,
        )

        # Larger images should result in smaller or equal batch size
        self.assertLessEqual(result_large, result_small)


if __name__ == "__main__":
    unittest.main()
