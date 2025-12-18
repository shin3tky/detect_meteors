"""
Infrastructure and Utility test suite for meteor_core (v1.x).

Covers:
- ROI polygon string parsing/formatting
- Parameter hashing (for progress tracking)
- Progress file I/O
- File collection logic
- Auto-estimation algorithms (using synthetic data)
"""

import unittest
import sys
import os
import tempfile
import shutil
import json
import numpy as np
import cv2
from unittest.mock import patch

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core import (  # noqa: E402
    MeteorLoadError,
    parse_roi_polygon_string,
    format_polygon_string,
    compute_params_hash,
    save_progress,
    load_progress,
    ProgressManager,
    collect_files,
    DEFAULT_DIFF_THRESHOLD,
    DEFAULT_MIN_AREA,
)

from meteor_core.pipeline import (  # noqa: E402
    estimate_diff_threshold_from_samples,
    estimate_min_area_from_samples,
)


class TestROIStringParsing(unittest.TestCase):
    """Test ROI polygon string parsing and formatting functions."""

    def test_parse_valid_string(self):
        """Test parsing a valid ROI string."""
        roi_str = "100,100;200,100;200,200;100,200"
        expected = [[100, 100], [200, 100], [200, 200], [100, 200]]
        result = parse_roi_polygon_string(roi_str)
        self.assertEqual(result, expected)

    def test_parse_string_with_spaces(self):
        """Test parsing a string with random spaces."""
        roi_str = " 100, 100 ; 200,100; 200, 200 "
        expected = [[100, 100], [200, 100], [200, 200]]
        result = parse_roi_polygon_string(roi_str)
        self.assertEqual(result, expected)

    def test_parse_invalid_format_raises_error(self):
        """Test that invalid formats raise ValueError."""
        invalid_strings = [
            "100,100",  # Only one point
            "100,100;200,200",  # Only two points (need 3+)
            "100-100;200-200;300-300",  # Wrong separator
            "abc,def;ghi,jkl;mno,pqr",  # Non-numeric
        ]
        for s in invalid_strings:
            with self.assertRaises(ValueError):
                parse_roi_polygon_string(s)

    def test_format_polygon_string(self):
        """Test formatting a polygon list back to string."""
        polygon = [[100, 100], [200, 100], [200, 200]]
        expected = "100,100;200,100;200,200"
        result = format_polygon_string(polygon)
        self.assertEqual(result, expected)


class TestParameterHashing(unittest.TestCase):
    """Test compute_params_hash function."""

    def test_hash_consistency(self):
        """Test that the same parameters produce the same hash."""
        params1 = {"a": 1, "b": 2.5, "c": [1, 2, 3]}
        params2 = {"a": 1, "b": 2.5, "c": [1, 2, 3]}
        self.assertEqual(compute_params_hash(params1), compute_params_hash(params2))

    def test_hash_change(self):
        """Test that changing a parameter changes the hash."""
        params1 = {"a": 1, "b": 2.5}
        params2 = {"a": 1, "b": 2.6}  # Changed value
        self.assertNotEqual(compute_params_hash(params1), compute_params_hash(params2))

    def test_hash_numpy_types(self):
        """Test that numpy types are handled correctly in hashing."""
        # JSON serialization usually fails with numpy types unless handled
        params = {
            "int_val": np.int64(10),
            "float_val": np.float32(5.5),
            "array_val": np.array([1, 2, 3]),
        }
        # Should not raise TypeError
        hash_val = compute_params_hash(params)
        self.assertIsInstance(hash_val, str)
        self.assertEqual(len(hash_val), 64)  # SHA-256 length


class TestProgressPersistence(unittest.TestCase):
    """Test save_progress and load_progress functions."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.progress_path = os.path.join(self.test_dir, "test_progress.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_save_and_load(self):
        """Test saving data and loading it back."""
        data = {
            "version": "1.0",
            "processed_files": ["a.ORF", "b.ORF"],
            "total_processed": 2,
        }
        save_progress(self.progress_path, data)

        loaded = load_progress(self.progress_path)
        self.assertEqual(loaded["processed_files"], data["processed_files"])
        self.assertEqual(loaded["total_processed"], 2)
        self.assertIn("last_updated", loaded)

    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist returns None."""
        result = load_progress(os.path.join(self.test_dir, "missing.json"))
        self.assertIsNone(result)

    def test_save_creates_directories(self):
        """Test that save_progress creates nested directories if needed."""
        nested_path = os.path.join(self.test_dir, "subdir", "progress.json")
        save_progress(nested_path, {"test": 1})
        self.assertTrue(os.path.exists(nested_path))

    def test_progress_manager_load_invalid_structure(self):
        """ProgressManager.load should not raise on non-dict JSON."""

        with open(self.progress_path, "w", encoding="utf-8") as fp:
            fp.write('["not", "a", "dict"]')

        manager = ProgressManager(self.progress_path)

        loaded = manager.load()

        self.assertFalse(loaded)
        self.assertEqual(manager.get_total_processed(), 0)
        self.assertEqual(manager.get_total_detected(), 0)

    def test_progress_manager_load_sanitizes_invalid_fields(self):
        """ProgressManager.load should sanitize unexpected field types."""

        invalid_payload = {
            "processed_files": 123,
            "detected_files": ["kept.ORF", {"unexpected": "dict"}],
            "detected_details": [
                {"filename": "kept.ORF", "score": "50"},
                "skip-me",
                {"filename": ""},
            ],
            "processing_params": "not-a-dict",
            "roi": 42,
            "params_hash": 9876,
        }
        with open(self.progress_path, "w", encoding="utf-8") as fp:
            json.dump(invalid_payload, fp)

        manager = ProgressManager(self.progress_path)
        loaded = manager.load()

        self.assertTrue(loaded)
        self.assertEqual(manager.progress_data["processed_files"], [])
        self.assertEqual(
            manager.progress_data["detected_files"],
            ["kept.ORF", "{'unexpected': 'dict'}"],
        )
        self.assertEqual(
            manager.progress_data["detected_details"],
            [{"filename": "kept.ORF", "score": "50"}],
        )
        self.assertEqual(manager.progress_data["processing_params"], {})
        self.assertEqual(manager.progress_data["roi"], "full_image")
        self.assertEqual(manager.progress_data["params_hash"], "9876")
        self.assertEqual(manager.get_total_processed(), 0)
        self.assertEqual(manager.get_total_detected(), 2)

    def test_load_progress_returns_none_on_non_mapping(self):
        """Standalone load_progress should return None for invalid shapes."""

        with open(self.progress_path, "w", encoding="utf-8") as fp:
            fp.write("42")

        result = load_progress(self.progress_path)

        self.assertIsNone(result)


class TestFileCollection(unittest.TestCase):
    """Test collect_files function."""

    @patch("glob.glob")
    @patch("os.path.isdir")
    @patch("os.path.exists")
    def test_collect_files_success(self, mock_exists, mock_isdir, mock_glob):
        """Test successful file collection."""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        # Mock finding some ORF and ARW files
        mock_glob.side_effect = lambda p: (
            ["img1.ORF", "img2.ORF"]
            if "*.ORF" in p
            else ["img3.ARW"] if "*.ARW" in p else []
        )

        files = collect_files("/fake/path")
        self.assertEqual(len(files), 3)
        self.assertEqual(files, ["img1.ORF", "img2.ORF", "img3.ARW"])

    @patch("os.path.exists")
    def test_collect_files_not_found(self, mock_exists):
        """Test error when directory doesn't exist."""
        mock_exists.return_value = False
        with self.assertRaises(MeteorLoadError):
            collect_files("/missing/path")

    @patch("glob.glob")
    @patch("os.path.isdir")
    @patch("os.path.exists")
    def test_collect_files_empty(self, mock_exists, mock_isdir, mock_glob):
        """Test error when no supported files are found."""
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_glob.return_value = []  # No files found

        with self.assertRaises(MeteorLoadError):
            collect_files("/empty/path")


class TestAutoEstimationLogic(unittest.TestCase):
    """Test auto-estimation algorithms using synthetic data."""

    def setUp(self):
        self.shape = (100, 100)
        self.roi_mask = np.full(self.shape, 255, dtype=np.uint8)

    def test_estimate_diff_threshold(self):
        """Test diff_threshold estimation based on noise levels."""
        # Scenario: create 5 images with random noise
        # Mean diff should be around 10
        np.random.seed(42)
        images = []
        base_img = np.zeros(self.shape, dtype=np.uint16)

        for i in range(5):
            noise = np.random.normal(0, 10, self.shape).astype(np.int16)
            img = np.clip(base_img + noise, 0, 65535).astype(np.uint16)
            images.append(img)

        loader = _IteratorLoader(images)

        # Run estimation
        files = ["f1", "f2", "f3", "f4", "f5"]
        threshold = estimate_diff_threshold_from_samples(
            files, self.roi_mask, sample_size=5, input_loader=loader
        )

        # Threshold should be greater than average noise (which is ~10-15 difference)
        # The logic calculates mean/std/percentiles.
        # With std=10, diffs will be around 10-15.
        # The function clamps between 3 and 18.
        self.assertGreaterEqual(threshold, 3)
        self.assertLessEqual(threshold, 25)

    def test_estimate_min_area(self):
        """Test min_area estimation based on star sizes."""
        # Scenario: Create images with "stars" of specific sizes
        images = []
        for _ in range(3):
            img = np.zeros(self.shape, dtype=np.uint16)
            # Add stars of size ~20 pixels
            cv2.circle(img, (20, 20), 3, 255, -1)  # Area ~28
            cv2.circle(img, (50, 50), 3, 255, -1)  # Area ~28
            cv2.circle(img, (80, 80), 2, 255, -1)  # Area ~12
            images.append(img)

        loader = _IteratorLoader(images)

        # Use a low threshold so stars are detected
        files = ["f1", "f2", "f3"]
        min_area = estimate_min_area_from_samples(
            files, self.roi_mask, diff_threshold=5, sample_size=3, input_loader=loader
        )

        # Logic uses 75th percentile * 2.0.
        # Stars are around 28px and 12px.
        # Should detect meaningful area size, not default
        self.assertGreater(min_area, 5)

    def test_estimate_defaults_on_error(self):
        """Test that estimation falls back to defaults on loading error."""
        loader = _ErrorLoader()

        files = ["bad_file"]

        threshold = estimate_diff_threshold_from_samples(
            files, self.roi_mask, input_loader=loader
        )
        self.assertEqual(threshold, DEFAULT_DIFF_THRESHOLD)

        area = estimate_min_area_from_samples(
            files, self.roi_mask, 10, input_loader=loader
        )
        self.assertEqual(area, DEFAULT_MIN_AREA)


class _IteratorLoader:
    def __init__(self, images):
        self._images = iter(images)

    def load(self, filepath):
        return next(self._images)


class _ErrorLoader:
    def load(self, filepath):
        raise Exception("Load error")


if __name__ == "__main__":
    unittest.main()
