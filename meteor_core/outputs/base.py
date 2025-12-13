#!/usr/bin/env python
#
# Detect Meteors CLI - Output Handler Base Class
# © 2025 Shinichi Morita (shin3tky)
#

"""
Abstract base class for output handler plugins.

This module provides the base classes and utilities for creating
output handler plugins that can be discovered and used by the
detection pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import Dict, Generic, List, Optional, Type, TypeVar

import numpy as np

ConfigType = TypeVar("ConfigType")


class BaseOutputHandler(ABC):
    """Abstract base class for output handling plugins.

    All output handlers must inherit from this class to be discoverable
    and usable by the detection pipeline.

    Subclasses must define:
        - plugin_name: str - Unique identifier for the handler
        - save_candidate: Save a meteor candidate file
        - save_debug_image: Save a debug visualization image

    Attributes:
        plugin_name: Unique identifier for the handler plugin (used in registry).
        name: Human-readable name of the handler.
        version: Version string of the handler.

    Example:
        >>> class MyHandler(BaseOutputHandler):
        ...     plugin_name = "my_handler"
        ...     name = "My Custom Handler"
        ...     version = "1.0.0"
        ...
        ...     def save_candidate(self, source_path, filename, debug_image=None, roi_polygon=None):
        ...         # Implementation here
        ...         return True
        ...
        ...     def save_debug_image(self, debug_image, filename, roi_polygon=None):
        ...         # Implementation here
        ...         return "/path/to/debug.png"
    """

    #: Unique name identifying this handler plugin
    plugin_name: str = ""

    #: Human-readable name of the handler
    name: str = "BaseOutputHandler"

    #: Version string of the handler
    version: str = "1.0.0"

    @abstractmethod
    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> bool:
        """Save a meteor candidate and optional debug visualization.

        Args:
            source_path: Path to the source file.
            filename: Output filename.
            debug_image: Optional debug visualization image (BGR).
            roi_polygon: Optional ROI polygon to draw on debug image.

        Returns:
            True when the candidate was persisted, False when skipped.
        """
        pass

    @abstractmethod
    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Persist a debug visualization and return its path.

        Args:
            debug_image: Debug visualization image (BGR).
            filename: Base filename for the debug image.
            roi_polygon: Optional ROI polygon to draw.

        Returns:
            Path to the saved debug image.
        """
        pass

    def get_info(self) -> Dict[str, str]:
        """Get information about the handler.

        Returns:
            Dictionary with handler metadata.
        """
        return {
            "plugin_name": self.plugin_name,
            "name": self.name,
            "version": self.version,
            "class": self.__class__.__name__,
        }

    # === Progress notification hooks (optional overrides) ===

    def on_candidate_detected(
        self,
        filename: str,
        saved: bool,
        score: float = 0.0,
        aspect_ratio: float = 0.0,
    ) -> None:
        """Hook called when a candidate is detected.

        Override this method to add custom notifications (Slack, webhook, etc.).

        Args:
            filename: Name of the detected file.
            saved: Whether the file was saved (False if skipped due to existing).
            score: Line score of the detection.
            aspect_ratio: Aspect ratio of the detected contour.
        """
        pass

    def on_batch_complete(
        self,
        processed_count: int,
        detected_count: int,
        batch_size: int,
    ) -> None:
        """Hook called when a batch of images is processed.

        Override for batch-level progress notifications.

        Args:
            processed_count: Total files processed so far.
            detected_count: Total candidates detected so far.
            batch_size: Number of files in this batch.
        """
        pass

    def on_pipeline_complete(
        self,
        total_processed: int,
        total_detected: int,
        elapsed_seconds: float,
    ) -> None:
        """Hook called when the pipeline completes.

        Override for final summary notifications.

        Args:
            total_processed: Total files processed.
            total_detected: Total candidates detected.
            elapsed_seconds: Total processing time.
        """
        pass


class DataclassOutputHandler(BaseOutputHandler, Generic[ConfigType]):
    """Abstract base class for handlers configured by dataclasses.

    This base class provides common functionality for handlers that use
    dataclasses for their configuration.

    Subclasses must define:
        - plugin_name: str - Unique identifier for the handler
        - ConfigType: Type - The dataclass type for configuration
        - save_candidate: Save a meteor candidate file
        - save_debug_image: Save a debug visualization image

    Example:
        >>> @dataclass
        ... class MyConfig:
        ...     output_folder: str
        ...     debug_folder: str
        ...
        >>> class MyHandler(DataclassOutputHandler[MyConfig]):
        ...     plugin_name = "my_handler"
        ...     ConfigType = MyConfig
        ...
        ...     def save_candidate(self, source_path, filename, debug_image=None, roi_polygon=None):
        ...         return save_to_folder(source_path, self.config.output_folder)
    """

    ConfigType: Type[ConfigType]
    config: ConfigType

    def __init__(self, config: ConfigType) -> None:
        """Initialize the handler with configuration.

        Args:
            config: Configuration instance (must match ConfigType).

        Raises:
            ValueError: If plugin_name is not defined.
            TypeError: If ConfigType is not a dataclass or config type mismatches.
        """
        plugin_name = getattr(self.__class__, "plugin_name", "")
        if not plugin_name:
            raise ValueError("Subclasses must define a non-empty 'plugin_name' string.")

        config_type = getattr(self.__class__, "ConfigType", None)
        if config_type is not None:
            if not is_dataclass(config_type):
                raise TypeError(
                    "ConfigType must be a dataclass type for DataclassOutputHandler."
                )
            if not isinstance(config, config_type):
                raise TypeError(
                    f"config must be an instance of {config_type.__name__}."
                )
        self.config = config


def _is_valid_output_handler(cls: type) -> bool:
    """Check if a class is a valid output handler implementation.

    Args:
        cls: Class to check.

    Returns:
        True if the class is a valid BaseOutputHandler subclass with plugin_name.
    """
    if not isinstance(cls, type):
        return False

    if not issubclass(cls, BaseOutputHandler):
        return False

    # Must have a non-empty plugin_name
    plugin_name = getattr(cls, "plugin_name", "")
    if not plugin_name:
        return False

    return True


__all__ = [
    "ConfigType",
    "BaseOutputHandler",
    "DataclassOutputHandler",
    "_is_valid_output_handler",
]
