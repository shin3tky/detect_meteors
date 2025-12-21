"""Tests for ROI selector helper functions."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
import numpy as np


missing_deps = [
    name for name in ("numpy", "cv2") if importlib.util.find_spec(name) is None
]
if missing_deps:
    raise unittest.SkipTest(f"Missing dependencies: {', '.join(missing_deps)}")

if importlib.util.find_spec("yaml") is None:

    class _YamlError(Exception):
        """Fallback YAML error type for tests."""

    sys.modules["yaml"] = types.SimpleNamespace(  # type: ignore[assignment]
        safe_load=lambda _: {},
        YAMLError=_YamlError,
    )

roi_selector = importlib.import_module("meteor_core.roi_selector")
create_full_roi_mask = roi_selector.create_full_roi_mask
create_roi_mask_from_polygon = roi_selector.create_roi_mask_from_polygon


class TestRoiSelectorHelpers(unittest.TestCase):
    """Coverage for ROI mask helper functions."""

    def test_create_full_roi_mask_fills_image(self) -> None:
        mask = create_full_roi_mask((3, 4))
        self.assertEqual(mask.shape, (3, 4))
        self.assertTrue(np.all(mask == 255))

    def test_create_roi_mask_from_polygon_fills_polygon(self) -> None:
        polygon = [[1, 1], [3, 1], [3, 3], [1, 3]]
        mask = create_roi_mask_from_polygon(polygon, (5, 5))
        self.assertEqual(mask.shape, (5, 5))
        self.assertEqual(mask[2, 2], 255)
        self.assertEqual(mask[0, 0], 0)
