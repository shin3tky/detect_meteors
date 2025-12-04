#!/usr/bin/env python
#
# Detect Meteors CLI - Detector Base Class
# © 2025 Shinichi Morita (shin3tky)
#

"""
Abstract base class for meteor detectors.
Provides a plugin interface for different detection algorithms.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional, Dict
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for meteor detection algorithms.

    This class defines the interface that all detector implementations must follow.
    Subclasses should implement the `detect` method to perform the actual detection.

    Attributes:
        name: Human-readable name of the detector
        version: Version string of the detector
    """

    name: str = "BaseDetector"
    version: str = "1.0.0"

    @abstractmethod
    def detect(
        self,
        current_image: np.ndarray,
        previous_image: np.ndarray,
        roi_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[
        bool, float, List[Tuple[int, int, int, int]], float, Optional[np.ndarray]
    ]:
        """
        Detect meteor candidates by comparing two consecutive frames.

        Args:
            current_image: Current frame (uint16 grayscale)
            previous_image: Previous frame (uint16 grayscale)
            roi_mask: Binary mask for region of interest (uint8)
            params: Detection parameters dictionary containing:
                - diff_threshold: int
                - min_area: int
                - min_aspect_ratio: float
                - hough_threshold: int
                - hough_min_line_length: int
                - hough_max_line_gap: int
                - min_line_score: float

        Returns:
            Tuple of:
                - is_candidate: bool - Whether a meteor candidate was detected
                - line_score: float - Hough line detection score
                - line_segments: List of (x1, y1, x2, y2) tuples
                - max_aspect_ratio: float - Maximum aspect ratio of detected contours
                - debug_image: Optional numpy array (BGR) for visualization
        """
        pass

    @abstractmethod
    def compute_line_score(
        self, mask: np.ndarray, hough_params: Dict[str, int]
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """
        Compute line detection score using Hough transform or similar.

        Args:
            mask: Binary mask of detected changes (uint8)
            hough_params: Dictionary with Hough transform parameters:
                - threshold: int
                - min_line_length: int
                - max_line_gap: int

        Returns:
            Tuple of:
                - score: float - Total length of detected lines
                - line_segments: List of (x1, y1, x2, y2) tuples
        """
        pass

    def get_info(self) -> Dict[str, str]:
        """
        Get information about the detector.

        Returns:
            Dictionary with detector metadata
        """
        return {
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate detection parameters.

        Args:
            params: Detection parameters dictionary

        Returns:
            True if parameters are valid, False otherwise

        Raises:
            ValueError: If parameters are invalid (optional, for detailed error info)
        """
        required_keys = [
            "diff_threshold",
            "min_area",
            "min_aspect_ratio",
            "hough_threshold",
            "hough_min_line_length",
            "hough_max_line_gap",
            "min_line_score",
        ]

        for key in required_keys:
            if key not in params:
                return False

        return True
