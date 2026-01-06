#!/usr/bin/env python
#
# Detect Meteors CLI - Output Writer (Backward Compatibility)
# © 2025 Shinichi Morita (shin3tky)
#

"""
Backward compatibility module for output writing.

This module provides aliases and wrapper classes for backward compatibility
with code that uses the legacy OutputWriter and ProgressManager APIs.

For new code, use:
- OutputHandlerRegistry.create("file", config) instead of OutputWriter
- ProgressManager from meteor_core.outputs.progress

.. deprecated::
    OutputWriter is deprecated. Use FileOutputHandler via OutputHandlerRegistry instead.
"""

import logging
import warnings
from typing import List, Optional

import numpy as np

from .file_handler import FileOutputHandler, FileOutputConfig
from .progress import ProgressManager, load_progress, save_progress
from ..schema import OutputResult

# Module-level logger
logger = logging.getLogger(__name__)


class OutputWriter(FileOutputHandler):
    """
    Backward-compatible wrapper for FileOutputHandler.

    .. deprecated::
        Use FileOutputHandler via OutputHandlerRegistry instead::

            from meteor_core.outputs import OutputHandlerRegistry

            handler = OutputHandlerRegistry.create("file", {
                "output_folder": output_folder,
                "debug_folder": debug_folder,
                "output_overwrite": output_overwrite,
            })

    This class wraps FileOutputHandler to provide the legacy OutputWriter API.
    The progress_file parameter is accepted but not used internally;
    use ProgressManager separately for progress tracking.
    """

    def __init__(
        self,
        output_folder: str,
        debug_folder: str,
        progress_file: str,
        output_overwrite: bool = False,
    ):
        """
        Initialize the output writer.

        Args:
            output_folder: Directory for candidate files.
            debug_folder: Directory for debug images.
            progress_file: Path to progress JSON file (stored for compatibility,
                but ProgressManager should be used separately).
            output_overwrite: Whether to overwrite existing files.
        """
        logger.debug(
            "OutputWriter.__init__: output_folder=%s, debug_folder=%s, "
            "progress_file=%s, output_overwrite=%s (deprecated)",
            output_folder,
            debug_folder,
            progress_file,
            output_overwrite,
        )
        warnings.warn(
            "OutputWriter is deprecated. Use OutputHandlerRegistry.create('file', ...) "
            "or FileOutputHandler directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        config = FileOutputConfig(
            output_folder=output_folder,
            debug_folder=debug_folder,
            output_overwrite=output_overwrite,
        )
        super().__init__(config)

        # Store progress_file for compatibility (not used internally)
        self.progress_file = progress_file
        logger.debug("OutputWriter initialized (deprecated wrapper)")

    # Legacy property accessors for backward compatibility
    @property
    def output_folder(self) -> str:
        """Get output folder path."""
        return self.config.output_folder

    @property
    def debug_folder(self) -> str:
        """Get debug folder path."""
        return self.config.debug_folder

    @property
    def output_overwrite(self) -> bool:
        """Get output overwrite setting."""
        return self.config.output_overwrite

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> OutputResult:
        """Save a meteor candidate file.

        Returns the full OutputResult. For legacy boolean checks, inspect
        ``result.saved``.
        """
        return super().save_candidate(
            source_path,
            filename,
            debug_image=debug_image,
            roi_polygon=roi_polygon,
        )


__all__ = [
    # Backward-compatible classes
    "OutputWriter",
    "ProgressManager",
    # Backward-compatible functions
    "load_progress",
    "save_progress",
]
