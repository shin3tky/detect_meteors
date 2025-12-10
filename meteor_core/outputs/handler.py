#!/usr/bin/env python
#
# Detect Meteors CLI - Output Handler Base Class
# © 2025 Shinichi Morita (shin3tky)
#

"""Abstract base class for output handling."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BaseOutputHandler(ABC):
    """Abstract base class for handling detection outputs.

    Subclasses must implement:
        - save_candidate: Save a meteor candidate file
        - save_debug_image: Save a debug visualization image

    Example:
        >>> class MyOutputHandler(BaseOutputHandler):
        ...     def save_candidate(self, source_path, filename, debug_image=None, roi_polygon=None):
        ...         # Implementation here
        ...         return True
        ...
        ...     def save_debug_image(self, debug_image, filename, roi_polygon=None):
        ...         # Implementation here
        ...         return "/path/to/debug.png"
    """

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
