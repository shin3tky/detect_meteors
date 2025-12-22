"""Tests for ROI parsing and params hashing utilities."""

import os
import sys
import unittest

import numpy as np

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core.utils import (
    compute_params_hash,
    format_polygon_string,
    parse_roi_polygon_string,
)


class TestRoiPolygonHelpers(unittest.TestCase):
    def test_parse_roi_polygon_string_strips_spaces(self):
        roi_str = " 10,20 ; 30,40;50,60 "
        self.assertEqual(
            parse_roi_polygon_string(roi_str), [[10, 20], [30, 40], [50, 60]]
        )

    def test_parse_roi_polygon_string_requires_three_vertices(self):
        with self.assertRaises(ValueError):
            parse_roi_polygon_string("10,20;30,40")

    def test_parse_roi_polygon_string_validates_pairs(self):
        with self.assertRaises(ValueError):
            parse_roi_polygon_string("10,20;30,40;bad")

    def test_format_polygon_string(self):
        polygon = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(format_polygon_string(polygon), "1,2;3,4;5,6")


class TestComputeParamsHash(unittest.TestCase):
    def test_compute_params_hash_normalizes_numpy_types(self):
        params_numpy = {
            "count": np.int32(5),
            "score": np.float64(1.25),
            "roi": np.array([[1, 2], [3, 4]]),
            "roi_polygon": [
                [np.int64(10), np.int64(20)],
                [np.int64(30), np.int64(40)],
            ],
        }
        params_native = {
            "count": 5,
            "score": 1.25,
            "roi": [[1, 2], [3, 4]],
            "roi_polygon": [[10, 20], [30, 40]],
        }
        self.assertEqual(
            compute_params_hash(params_numpy), compute_params_hash(params_native)
        )


if __name__ == "__main__":
    unittest.main()
