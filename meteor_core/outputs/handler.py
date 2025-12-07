#!/usr/bin/env python
#
# Detect Meteors CLI - Output Handler Protocol
# © 2025 Shinichi Morita (shin3tky)
#

"""Protocol definitions for output handling."""

from typing import List, Optional, Protocol

import numpy as np


class OutputHandler(Protocol):
    """Protocol for handling detection outputs."""

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> bool:
        """
        Save a meteor candidate and optional debug visualization.

        Returns:
            ``True`` when the candidate was persisted, ``False`` when skipped.
        """

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Persist a debug visualization and return its path."""
