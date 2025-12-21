"""Tests for ROI selector UI flow."""

import importlib.util
import sys
import unittest
from unittest.mock import patch

_NUMPY_MODULE = sys.modules.get("numpy")
_NUMPY_SPEC = (
    getattr(_NUMPY_MODULE, "__spec__", None)
    if _NUMPY_MODULE is not None
    else importlib.util.find_spec("numpy")
)
if _NUMPY_SPEC:
    import numpy as np
else:
    np = None

_CV2_MODULE = sys.modules.get("cv2")
_CV2_SPEC = (
    getattr(_CV2_MODULE, "__spec__", None)
    if _CV2_MODULE is not None
    else importlib.util.find_spec("cv2")
)
if _CV2_SPEC:
    import cv2
    from meteor_core.roi_selector import select_roi
else:
    cv2 = None
    select_roi = None


@unittest.skipUnless(_CV2_SPEC and _NUMPY_SPEC, "cv2 or numpy is not installed")
class TestRoiSelectorUi(unittest.TestCase):
    """Simulate ROI selection without opening UI windows."""

    def test_select_roi_completes_polygon(self) -> None:
        image = np.ones((10, 10), dtype=np.uint16)
        callbacks = []
        points = [(1, 1), (6, 1), (6, 6), (1, 1)]

        def fake_set_mouse_callback(_window, callback):
            callbacks.append(callback)

        def fake_wait_key(_delay):
            if callbacks and fake_wait_key.calls < len(points):
                x, y = points[fake_wait_key.calls]
                callbacks[0](cv2.EVENT_LBUTTONDOWN, x, y, None, None)
            fake_wait_key.calls += 1
            return 0

        fake_wait_key.calls = 0

        with (
            patch("cv2.namedWindow"),
            patch("cv2.imshow"),
            patch("cv2.destroyWindow"),
            patch("cv2.waitKey", side_effect=fake_wait_key),
            patch("cv2.setMouseCallback", side_effect=fake_set_mouse_callback),
        ):
            selection = select_roi(image, locale="en")

        self.assertIsNotNone(selection)
        self.assertEqual(selection["mask"].shape, image.shape)
        self.assertEqual(len(selection["polygon"]), 3)
        self.assertEqual(selection["bounding_rect"][:2], (1, 1))
        self.assertGreaterEqual(selection["bounding_rect"][2], 5)
        self.assertGreaterEqual(selection["bounding_rect"][3], 5)
        self.assertEqual(selection["mask"][3, 3], 255)

    def test_select_roi_cancelled(self) -> None:
        image = np.ones((8, 8), dtype=np.uint16)

        with (
            patch("cv2.namedWindow"),
            patch("cv2.imshow"),
            patch("cv2.destroyWindow"),
            patch("cv2.waitKey", return_value=ord("q")),
            patch("cv2.setMouseCallback"),
        ):
            selection = select_roi(image, locale="en")

        self.assertIsNone(selection)
