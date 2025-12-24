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
import logging
from typing import Tuple, List, Any, Dict, Generic, Type, TypeVar

from meteor_core.plugin_contract import (
    forbid_unknown_keys as _forbid_unknown_keys,
    require_config_type,
    require_plugin_name,
)
from meteor_core.schema import (
    DEFAULT_DETECTOR_NAME,
    DetectionContext,
    DetectionResult,
    RUNTIME_PARAMS_SCHEMA_VERSION,
    RuntimeParams,
)

ConfigType = TypeVar("ConfigType")

_PYDANTIC_SPEC = importlib.util.find_spec("pydantic")
if _PYDANTIC_SPEC:
    from pydantic import BaseModel
else:
    BaseModel = None
import numpy as np  # noqa: E402


logger = logging.getLogger(__name__)


forbid_unknown_keys = _forbid_unknown_keys


class BaseDetector(ABC, Generic[ConfigType]):
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
        context: DetectionContext,
    ) -> DetectionResult:
        """
        Detect meteor candidates by comparing two consecutive frames.

        Args:
            context: DetectionContext containing image data, ROI mask,
                runtime parameters, and metadata.

        Returns:
            DetectionResult with detection outcome, score, line segments,
            aspect ratio, debug visualization, and extras.
        """
        pass

    def detect_legacy(
        self,
        current_image: np.ndarray,
        previous_image: np.ndarray,
        roi_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> DetectionResult:
        """Backward-compatible adapter for the legacy detect signature."""
        logger.warning(
            "Deprecated detect(current_image, previous_image, roi_mask, params) "
            "signature used. Please migrate to detect(DetectionContext)."
        )
        runtime_params = self.build_runtime_params(params)
        context = DetectionContext(
            current_image=current_image,
            previous_image=previous_image,
            roi_mask=roi_mask,
            runtime_params=runtime_params,
            metadata={},
        )
        return self.detect(context)

    def build_runtime_params(
        self, params: Dict[str, Any] | RuntimeParams
    ) -> Dict[str, Any]:
        """Wrap flat params into the namespaced runtime_params structure."""
        if isinstance(params, RuntimeParams):
            return params.to_dict(include_schema_version=False)
        detector_name = self.plugin_name or DEFAULT_DETECTOR_NAME
        return RuntimeParams(
            global_params=params,
            detector={detector_name: params},
            schema_version=RUNTIME_PARAMS_SCHEMA_VERSION,
        ).to_dict(include_schema_version=False)

    def split_runtime_params(
        self, runtime_params: Dict[str, Any] | RuntimeParams
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (global_params, detector_params) from runtime_params."""
        if isinstance(runtime_params, RuntimeParams):
            runtime_params = runtime_params.to_dict()
        if not isinstance(runtime_params, dict):
            logger.error("runtime_params must be a dictionary.")
            raise TypeError("runtime_params must be a dictionary.")

        global_params = runtime_params.get("global", runtime_params)
        detector_params: Dict[str, Any] = {}
        detector_group = runtime_params.get("detector")
        if isinstance(detector_group, dict):
            detector_params = detector_group.get(
                self.plugin_name or DEFAULT_DETECTOR_NAME, {}
            )
        if not isinstance(global_params, dict) or not isinstance(detector_params, dict):
            logger.error("Runtime params namespaces must be dictionaries.")
            raise TypeError("runtime_params namespaces must be dictionaries.")
        return global_params, detector_params

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

        Subclasses are responsible for validating detector-specific keys.

        Args:
            params: Detection parameters dictionary

        Returns:
            True if parameters are valid, False otherwise

        Raises:
            TypeError: If parameters are not provided as a dictionary
            ValueError: If parameters are invalid (optional, for detailed error info)
        """
        if not isinstance(params, dict):
            logger.error("Detection parameters must be provided as a dictionary.")
            raise TypeError("params must be a dictionary.")
        return True


class DataclassDetector(BaseDetector[ConfigType], Generic[ConfigType]):
    """Base class for detectors configured by dataclasses.

    Subclasses must define ``plugin_name`` and ``ConfigType`` (a dataclass
    type). During initialization, ``require_plugin_name`` verifies the
    subclass declares a non-empty plugin identifier for registry discovery
    while ``require_config_type`` fetches the declared configuration type so
    that the incoming ``config`` instance can be type-checked.

    The validation sequence mirrors :class:`meteor_core.inputs.base.DataclassInputLoader`
    and :class:`meteor_core.outputs.base.DataclassOutputHandler`:

    1. Verify the plugin name is present.
    2. Fetch ``ConfigType`` and ensure it is a dataclass type.
    3. Confirm the provided ``config`` instance matches ``ConfigType``.
    """

    ConfigType: Type[ConfigType]
    config: ConfigType

    def __init__(self, config: ConfigType) -> None:
        require_plugin_name(self.__class__, kind="detector")

        config_type = require_config_type(self.__class__)
        if config_type is not None:
            if not is_dataclass(config_type):
                logger.error(
                    "ConfigType %s is not a dataclass for detector %s",
                    config_type,
                    self.__class__.__name__,
                )
                raise TypeError(
                    "ConfigType must be a dataclass type for DataclassDetector."
                )
            if not isinstance(config, config_type):
                logger.error(
                    "Config instance %s does not match dataclass %s for detector %s",
                    type(config).__name__,
                    config_type.__name__,
                    self.__class__.__name__,
                )
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config
        logger.debug(
            "%s initialized with dataclass config %s",
            self.__class__.__name__,
            config,
        )


class PydanticDetector(BaseDetector[ConfigType], Generic[ConfigType]):
    """Base class for detectors configured by Pydantic models.

    Subclasses must define ``plugin_name`` and ``ConfigType`` (a
    :class:`pydantic.BaseModel`). The initializer follows the same
    validation order used by the inputs/outputs plugin bases:

    1. Ensure Pydantic is available.
    2. Verify ``plugin_name`` is declared for discovery.
    3. Fetch ``ConfigType`` and confirm it inherits ``BaseModel``.
    4. Validate that the provided ``config`` instance matches ``ConfigType``.
    """

    ConfigType: Type[ConfigType]
    config: ConfigType

    def __init__(self, config: ConfigType) -> None:
        if BaseModel is None:
            logger.error(
                "PydanticDetector requires pydantic but it is not installed (%s)",
                self.__class__.__name__,
            )
            raise ImportError(
                "pydantic is required to use PydanticDetector. Install pydantic first."
            )

        require_plugin_name(self.__class__, kind="detector")

        config_type = require_config_type(self.__class__)
        if config_type is not None:
            if not issubclass(config_type, BaseModel):
                logger.error(
                    "ConfigType %s is not a pydantic BaseModel for detector %s",
                    config_type,
                    self.__class__.__name__,
                )
                raise TypeError(
                    "ConfigType must be a pydantic BaseModel for PydanticDetector."
                )
            if not isinstance(config, config_type):
                logger.error(
                    "Config instance %s does not match Pydantic model %s for detector %s",
                    type(config).__name__,
                    config_type.__name__,
                    self.__class__.__name__,
                )
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config
        logger.debug(
            "%s initialized with pydantic config %s",
            self.__class__.__name__,
            config,
        )


def _is_valid_detector(cls: type) -> bool:
    """Check if a class is a valid detector implementation.

    Args:
        cls: Class to check.

    Returns:
        True if the class is a valid BaseDetector subclass with plugin_name.
    """
    if not isinstance(cls, type):
        logger.debug(
            "_is_valid_detector: %r is not a type",
            cls,
        )
        return False

    if not issubclass(cls, BaseDetector):
        logger.debug(
            "_is_valid_detector: %s does not inherit from BaseDetector",
            cls.__name__,
        )
        return False

    # Must have a non-empty string plugin_name
    plugin_name = getattr(cls, "plugin_name", None)
    if not (
        plugin_name is not None and isinstance(plugin_name, str) and plugin_name != ""
    ):
        logger.debug(
            "_is_valid_detector: %s has invalid plugin_name: %r",
            cls.__name__,
            plugin_name,
        )
        return False
    return True
