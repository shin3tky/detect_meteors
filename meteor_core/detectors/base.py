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
from dataclasses import is_dataclass
import importlib.util
from typing import Tuple, List, Any, Optional, Dict, Generic, Type, TypeVar

from meteor_core.plugin_contract import require_config_type, require_plugin_name

ConfigType = TypeVar("ConfigType")

_PYDANTIC_SPEC = importlib.util.find_spec("pydantic")
if _PYDANTIC_SPEC:
    from pydantic import BaseModel
else:
    BaseModel = None
import numpy as np  # noqa: E402


class BaseDetector(ABC):
    """
    Abstract base class for meteor detection algorithms.

    This class defines the interface that all detector implementations must follow.
    Subclasses should implement the `detect` method to perform the actual detection.

    See :doc:`PLUGIN_AUTHOR_GUIDE` for lifecycle details shared across
    plugin kinds (discovery order, config coercion, and hooks).

    Attributes:
        plugin_name: Unique identifier for the detector plugin (used in registry)
        name: Human-readable name of the detector
        version: Version string of the detector
    """

    plugin_name: str = ""  # Must be overridden by subclasses
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
            "plugin_name": self.plugin_name,
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


class DataclassDetector(BaseDetector, Generic[ConfigType]):
    """Base class for detectors configured by dataclasses."""

    ConfigType: Type[ConfigType]
    config: ConfigType

    def __init__(self, config: ConfigType) -> None:
        require_plugin_name(self.__class__, kind="detector")

        config_type = require_config_type(self.__class__)
        if config_type is not None:
            if not is_dataclass(config_type):
                raise TypeError(
                    "ConfigType must be a dataclass type for DataclassDetector."
                )
            if not isinstance(config, config_type):
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config


class PydanticDetector(BaseDetector, Generic[ConfigType]):
    """Base class for detectors configured by Pydantic models."""

    ConfigType: Type[ConfigType]
    config: ConfigType

    def __init__(self, config: ConfigType) -> None:
        if BaseModel is None:
            raise ImportError(
                "pydantic is required to use PydanticDetector. Install pydantic first."
            )

        require_plugin_name(self.__class__, kind="detector")

        config_type = require_config_type(self.__class__)
        if config_type is not None:
            if not issubclass(config_type, BaseModel):
                raise TypeError(
                    "ConfigType must be a pydantic BaseModel for PydanticDetector."
                )
            if not isinstance(config, config_type):
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config


def _is_valid_detector(cls: type) -> bool:
    """Check if a class is a valid detector implementation.

    Args:
        cls: Class to check.

    Returns:
        True if the class is a valid BaseDetector subclass with plugin_name.
    """
    if not isinstance(cls, type):
        return False

    if not issubclass(cls, BaseDetector):
        return False

    # Must have a non-empty plugin_name
    plugin_name = getattr(cls, "plugin_name", "")
    if not plugin_name:
        return False

    return True
