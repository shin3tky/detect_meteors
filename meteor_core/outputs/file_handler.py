#!/usr/bin/env python
#
# Detect Meteors CLI - File Output Handler
# © 2025 Shinichi Morita (shin3tky)
#

"""
File-based output handler (default implementation).

This module provides the default file-based output handler that saves
meteor candidate RAW files and debug visualization images to disk.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .base import DataclassOutputHandler
from ..schema import DEFAULT_DEBUG_FOLDER, DEFAULT_OUTPUT_FOLDER
from ..exceptions import MeteorWriteError

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class FileOutputConfig:
    """Configuration for FileOutputHandler.

    Attributes:
        output_folder: Directory for candidate files.
        debug_folder: Directory for debug images.
        output_overwrite: Whether to overwrite existing files.
    """

    #: Directory for candidate files
    output_folder: str = DEFAULT_OUTPUT_FOLDER

    #: Directory for debug images
    debug_folder: str = DEFAULT_DEBUG_FOLDER

    #: Whether to overwrite existing files
    output_overwrite: bool = False


class FileOutputHandler(DataclassOutputHandler[FileOutputConfig]):
    """Default file-based output handler.

    Saves meteor candidate RAW files by copying them to the output folder,
    and saves debug visualization images to the debug folder.

    Attributes:
        plugin_name: "file" - the unique identifier for this handler.
        name: "File Output Handler" - human-readable name.
        version: Version string of this handler.
        config: FileOutputConfig instance with folder and overwrite settings.

    Example:
        >>> config = FileOutputConfig(
        ...     output_folder="./candidates",
        ...     debug_folder="./debug",
        ...     output_overwrite=False,
        ... )
        >>> handler = FileOutputHandler(config)
        >>> saved = handler.save_candidate("/path/to/source.CR2", "source.CR2")
        >>> print(saved)
        True
    """

    plugin_name = "file"
    name = "File Output Handler"
    version = "1.0.0"
    ConfigType = FileOutputConfig

    def __init__(self, config: FileOutputConfig) -> None:
        """Initialize the file output handler.

        Creates output and debug directories if they don't exist.

        Args:
            config: Configuration instance.
        """
        super().__init__(config)
        logger.debug(
            "FileOutputHandler initializing: output_folder=%s, debug_folder=%s, "
            "output_overwrite=%s",
            config.output_folder,
            config.debug_folder,
            config.output_overwrite,
        )
        try:
            os.makedirs(config.output_folder, exist_ok=True)
            os.makedirs(config.debug_folder, exist_ok=True)
        except Exception as exc:
            logger.error(
                "Failed to create output directories %s and %s: %s: %s",
                config.output_folder,
                config.debug_folder,
                type(exc).__name__,
                exc,
            )
            raise MeteorWriteError(
                "Failed to create output directories",
                original_error=exc,
                destination_path=config.output_folder,
                operation="mkdir",
                context={
                    "error_category": "directory_creation_failed",
                    "debug_folder": config.debug_folder,
                },
            ) from exc
        logger.debug("FileOutputHandler initialized: directories created/verified")

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> bool:
        """Save a meteor candidate file and optional debug image.

        Args:
            source_path: Path to the source RAW file.
            filename: Output filename.
            debug_image: Optional debug visualization image (BGR).
            roi_polygon: Optional ROI polygon to draw on debug image.

        Returns:
            True if file was saved, False if skipped (already exists).
        """
        logger.debug(
            "save_candidate: source_path=%s, filename=%s, has_debug_image=%s, "
            "has_roi_polygon=%s",
            source_path,
            filename,
            debug_image is not None,
            roi_polygon is not None,
        )
        output_path = os.path.join(self.config.output_folder, filename)

        # Check if file exists
        if os.path.exists(output_path) and not self.config.output_overwrite:
            logger.debug(
                "save_candidate: skipped (file exists and overwrite disabled): %s",
                output_path,
            )
            return False

        # Copy the RAW file
        try:
            shutil.copy(source_path, output_path)
            logger.debug("save_candidate: copied %s -> %s", source_path, output_path)
        except Exception as exc:
            logger.error(
                "Failed to copy candidate %s -> %s: %s: %s",
                source_path,
                output_path,
                type(exc).__name__,
                exc,
            )
            raise MeteorWriteError(
                f"Failed to copy candidate file to {output_path}",
                filepath=source_path,
                original_error=exc,
                destination_path=output_path,
                operation="copy",
                context={"error_category": "copy_failed"},
            ) from exc

        # Save debug image if provided
        if debug_image is not None:
            try:
                self._save_debug_with_roi(debug_image, filename, roi_polygon)
            except MeteorWriteError:
                # Already wrapped, re-raise as-is
                raise
            except Exception as exc:
                logger.error(
                    "Failed to save debug image for %s to %s: %s: %s",
                    filename,
                    self.config.debug_folder,
                    type(exc).__name__,
                    exc,
                )
                debug_path = os.path.join(
                    self.config.debug_folder, f"mask_{filename}.png"
                )
                raise MeteorWriteError(
                    f"Failed to save debug image for {filename}",
                    filepath=source_path,
                    original_error=exc,
                    destination_path=debug_path,
                    operation="save_image",
                    context={"error_category": "image_write_failed"},
                ) from exc

        logger.debug("save_candidate: completed successfully")
        return True

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Save a debug visualization image.

        Args:
            debug_image: Debug visualization image (BGR).
            filename: Base filename (will be prefixed with 'mask_').
            roi_polygon: Optional ROI polygon to draw.

        Returns:
            Path to saved debug image.
        """
        logger.debug(
            "save_debug_image: filename=%s, image_shape=%s, has_roi_polygon=%s",
            filename,
            debug_image.shape,
            roi_polygon is not None,
        )
        try:
            return self._save_debug_with_roi(debug_image, filename, roi_polygon)
        except MeteorWriteError:
            # Already wrapped, re-raise as-is
            raise
        except Exception as exc:
            logger.error(
                "Failed to save debug image for %s to %s: %s: %s",
                filename,
                self.config.debug_folder,
                type(exc).__name__,
                exc,
            )
            debug_path = os.path.join(self.config.debug_folder, f"mask_{filename}.png")
            raise MeteorWriteError(
                f"Failed to save debug image for {filename}",
                original_error=exc,
                destination_path=debug_path,
                operation="save_image",
                context={"error_category": "image_write_failed"},
            ) from exc

    def _save_debug_with_roi(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Internal method to save debug image with optional ROI overlay.

        Args:
            debug_image: Debug visualization image (BGR).
            filename: Base filename.
            roi_polygon: Optional ROI polygon to draw.

        Returns:
            Path to saved debug image.
        """
        if roi_polygon:
            logger.debug(
                "_save_debug_with_roi: drawing ROI polygon with %d points",
                len(roi_polygon),
            )
            try:
                cv2.polylines(
                    debug_image,
                    [np.array(roi_polygon, dtype=np.int32)],
                    True,
                    (0, 255, 0),
                    2,
                )
            except Exception as exc:
                debug_path = os.path.join(
                    self.config.debug_folder, f"mask_{filename}.png"
                )
                logger.error(
                    "Failed to draw ROI polygon for %s: %s: %s",
                    filename,
                    type(exc).__name__,
                    exc,
                )
                raise MeteorWriteError(
                    f"Failed to draw ROI polygon for {filename}",
                    original_error=exc,
                    destination_path=debug_path,
                    operation="draw_roi",
                    context={"error_category": "image_write_failed"},
                ) from exc
        debug_path = os.path.join(self.config.debug_folder, f"mask_{filename}.png")
        try:
            success = cv2.imwrite(debug_path, debug_image)
        except Exception as exc:
            logger.error(
                "Failed to write debug image %s: %s: %s",
                debug_path,
                type(exc).__name__,
                exc,
            )
            raise MeteorWriteError(
                f"Failed to write debug image to {debug_path}",
                original_error=exc,
                destination_path=debug_path,
                operation="save_image",
                context={"error_category": "image_write_failed"},
            ) from exc

        if not success:
            logger.error("OpenCV failed to write debug image to %s", debug_path)
            raise MeteorWriteError(
                f"Failed to write debug image to {debug_path}",
                destination_path=debug_path,
                operation="save_image",
                context={"error_category": "image_write_failed"},
            )

        logger.debug("_save_debug_with_roi: saved debug image to %s", debug_path)
        return debug_path


def create_file_handler(
    output_folder: str,
    debug_folder: str,
    output_overwrite: bool = False,
) -> FileOutputHandler:
    """Factory helper to create a FileOutputHandler.

    Args:
        output_folder: Directory for candidate files.
        debug_folder: Directory for debug images.
        output_overwrite: Whether to overwrite existing files.

    Returns:
        Configured FileOutputHandler instance.

    Example:
        >>> handler = create_file_handler("./candidates", "./debug")
        >>> handler = create_file_handler("./candidates", "./debug", output_overwrite=True)
    """
    logger.debug(
        "create_file_handler: output_folder=%s, debug_folder=%s, output_overwrite=%s",
        output_folder,
        debug_folder,
        output_overwrite,
    )
    config = FileOutputConfig(
        output_folder=output_folder,
        debug_folder=debug_folder,
        output_overwrite=output_overwrite,
    )
    return FileOutputHandler(config)


__all__ = [
    "FileOutputConfig",
    "FileOutputHandler",
    "create_file_handler",
]
