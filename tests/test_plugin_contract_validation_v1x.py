"""Contract validation parity tests for plugin helpers."""

from typing import Any, Dict, List, Tuple

import numpy as np
import sys
import types
import unittest

# Stub cv2 to avoid optional GUI dependency during import.
sys.modules.setdefault("cv2", types.SimpleNamespace())

from meteor_core.detectors.base import BaseDetector, _is_valid_detector  # noqa: E402
from meteor_core.inputs.base import (  # noqa: E402
    BaseInputLoader,
    _is_valid_input_loader,
)
from meteor_core.outputs.base import (  # noqa: E402
    BaseOutputHandler,
    _is_valid_output_handler,
)


class ValidInputLoader(BaseInputLoader[str]):
    """Concrete input loader for validator tests."""

    plugin_name = "valid_input"

    def load(self, filepath: str) -> str:
        return filepath


class ValidOutputHandler(BaseOutputHandler):
    """Concrete output handler for validator tests."""

    plugin_name = "valid_output"

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: np.ndarray | None = None,
        roi_polygon: List[Tuple[int, int]] | None = None,
    ) -> bool:
        return True

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: List[Tuple[int, int]] | None = None,
    ) -> str:
        return filename


class ValidDetector(BaseDetector):
    """Concrete detector for validator tests."""

    plugin_name = "valid_detector"

    def detect(
        self,
        current_image: np.ndarray,
        previous_image: np.ndarray,
        roi_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[bool, float, List[Tuple[int, int, int, int]], float, np.ndarray | None]:
        return False, 0.0, [], 0.0, None

    def compute_line_score(
        self, mask: np.ndarray, hough_params: Dict[str, int]
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        return 0.0, []


class TestPluginContractValidation(unittest.TestCase):
    """unittest-compatible validation tests for plugin helpers."""

    def test_validators_accept_valid_plugins(self) -> None:
        """All validator helpers accept well-formed plugin classes."""

        self.assertTrue(_is_valid_input_loader(ValidInputLoader))
        self.assertTrue(_is_valid_output_handler(ValidOutputHandler))
        self.assertTrue(_is_valid_detector(ValidDetector))

    def test_validators_reject_missing_or_nonstring_plugin_name(self) -> None:
        """Validators require a non-empty string plugin_name for all plugin kinds."""

        class MissingInputName(BaseInputLoader[str]):
            plugin_name: Any = None

            def load(self, filepath: str) -> str:
                return filepath

        class MissingOutputName(BaseOutputHandler):
            plugin_name: Any = True

            def save_candidate(
                self,
                source_path: str,
                filename: str,
                debug_image=None,
                roi_polygon=None,
            ):
                return True

            def save_debug_image(self, debug_image, filename, roi_polygon=None):
                return filename

        class MissingDetectorName(BaseDetector):
            plugin_name: Any = 0

            def detect(self, current_image, previous_image, roi_mask, params):
                return False, 0.0, [], 0.0, None

            def compute_line_score(self, mask, hough_params):
                return 0.0, []

        self.assertFalse(_is_valid_input_loader(MissingInputName))
        self.assertFalse(_is_valid_output_handler(MissingOutputName))
        self.assertFalse(_is_valid_detector(MissingDetectorName))

    def test_validators_reject_non_types_and_non_subclasses(self) -> None:
        """Validators reject non-type objects and unrelated types consistently."""

        class NotAPlugin:
            plugin_name = "something"

        self.assertFalse(_is_valid_input_loader("not a type"))
        self.assertFalse(_is_valid_output_handler(123))
        self.assertFalse(_is_valid_detector(None))

        self.assertFalse(_is_valid_input_loader(NotAPlugin))
        self.assertFalse(_is_valid_output_handler(NotAPlugin))
        self.assertFalse(_is_valid_detector(NotAPlugin))
