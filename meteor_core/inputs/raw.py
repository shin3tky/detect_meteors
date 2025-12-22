"""RAW image loader plugin."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..exceptions import (
    MeteorLoadError,
    MeteorUnsupportedFormatError,
    MeteorValidationError,
)
from ..image_io import extract_exif_metadata, load_and_bin_raw_fast
from ..schema import InputContext
from .base import BaseMetadataExtractor, DataclassInputLoader

# Module-level logger
logger = logging.getLogger(__name__)


@dataclass
class RawLoaderConfig:
    """Configuration for :class:`RawImageLoader`.

    Attributes:
        binning: Binning factor for image reduction. Currently only 2x2 (value=2)
            is supported. Default is 2.
        normalize: Whether to normalize the output image to [0, 1] float32 range.
            Default is False (returns uint16).
    """

    #: Binning factor (currently only 2x2 is supported)
    binning: int = 2

    #: Whether to normalize output to [0, 1] float32
    normalize: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            MeteorValidationError: If binning is not a supported value.
        """
        if self.binning not in (2,):
            logger.error(
                "Invalid binning factor: %d (supported: 2)",
                self.binning,
            )
            raise MeteorValidationError(
                "Invalid binning factor for RawLoaderConfig",
                parameter_name="binning",
                provided_value=self.binning,
                expected="2 (only 2x2 binning is currently supported)",
            )


class RawImageLoader(DataclassInputLoader[RawLoaderConfig], BaseMetadataExtractor):
    """Loader that wraps the existing RAW helpers for plugin use.

    This loader uses rawpy to load RAW image files and applies 2x2 binning
    for efficient processing. It supports various RAW formats including
    CR2, CR3, NEF, ARW, ORF, RW2, and many others.

    Attributes:
        plugin_name: "raw" - the unique identifier for this loader.
        name: "RAW Image Loader" - human-readable name.
        version: Version string of this loader.
        config: RawLoaderConfig instance with binning and normalization settings.

    Example:
        >>> loader = RawImageLoader(RawLoaderConfig())
        >>> context = loader.load("photo.CR2")
        >>> print(context.image_data.shape, context.image_data.dtype)
        (3000, 4000) uint16

        >>> loader_normalized = RawImageLoader(RawLoaderConfig(normalize=True))
        >>> context = loader_normalized.load("photo.CR2")
        >>> print(context.image_data.shape, context.image_data.dtype)
        (3000, 4000) float32
    """

    plugin_name = "raw"
    name = "RAW Image Loader"
    version = "1.0.0"
    ConfigType = RawLoaderConfig

    def load(self, filepath: str) -> InputContext:
        """Load a RAW image applying the configured binning and normalization.

        Args:
            filepath: Path to the RAW image file.

        Returns:
            InputContext with loaded image data and metadata.

        Raises:
            MeteorUnsupportedFormatError: If the file format is not supported.
            MeteorLoadError: If the file cannot be loaded (missing file,
                corrupted data, I/O errors, etc.)
        """
        logger.debug(
            "RawImageLoader.load: filepath=%s, binning=%d, normalize=%s",
            filepath,
            self.config.binning,
            self.config.normalize,
        )

        # Validation is done in RawLoaderConfig.__post_init__
        # MeteorLoadError/MeteorUnsupportedFormatError raised by load_and_bin_raw_fast
        try:
            image = load_and_bin_raw_fast(filepath)
        except (MeteorLoadError, MeteorUnsupportedFormatError):
            logger.error(
                "Failed to load RAW image %s (binning=%d, normalize=%s)",
                filepath,
                self.config.binning,
                self.config.normalize,
            )
            raise
        except Exception as exc:
            logger.error(
                "Failed to load RAW image %s (binning=%d, normalize=%s): %s: %s",
                filepath,
                self.config.binning,
                self.config.normalize,
                type(exc).__name__,
                exc,
            )
            raise

        if self.config.normalize:
            logger.debug("Normalizing image to float32 [0, 1] range")
            image = image.astype(np.float32) / np.iinfo(np.uint16).max

        metadata = self.extract_metadata(filepath)
        logger.debug(
            "RawImageLoader.load: completed, shape=%s, dtype=%s",
            image.shape,
            image.dtype,
        )
        return InputContext(
            image_data=image,
            filepath=filepath,
            metadata=metadata,
            loader_info=self.get_info(),
        )

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract EXIF metadata for the RAW file.

        Args:
            filepath: Path to the RAW image file.

        Returns:
            Dictionary containing EXIF metadata with keys:
            - focal_length: Actual focal length in mm
            - focal_length_35mm: 35mm equivalent focal length in mm
            - iso: ISO sensitivity
            - exposure_time: Exposure time in seconds
            - f_number: Aperture value
            - camera_make: Camera manufacturer
            - camera_model: Camera model name
            - lens_model: Lens model name
            - image_width: Image width in pixels
            - image_height: Image height in pixels

            Values may be None if metadata is not available.

        Note:
            This method does not raise exceptions on failure; it returns
            a dictionary with None values for unavailable metadata.
        """
        logger.debug("RawImageLoader.extract_metadata: filepath=%s", filepath)

        try:
            metadata = extract_exif_metadata(filepath)
        except Exception as exc:  # pragma: no cover - unexpected metadata failure
            logger.warning(
                "Failed to extract EXIF metadata from %s: %s: %s",
                filepath,
                type(exc).__name__,
                exc,
            )
            return {}

        # Log extracted metadata summary
        available_fields = [k for k, v in metadata.items() if v is not None]
        if available_fields:
            logger.debug(
                "Extracted metadata fields: %s",
                ", ".join(available_fields),
            )
        else:
            logger.debug("No EXIF metadata available for: %s", filepath)

        return metadata


def create_raw_loader(config: Optional[RawLoaderConfig] = None) -> RawImageLoader:
    """Factory helper to create a :class:`RawImageLoader` with defaults.

    Args:
        config: Optional configuration. If None, uses default RawLoaderConfig.

    Returns:
        Configured RawImageLoader instance.

    Example:
        >>> loader = create_raw_loader()  # Use defaults
        >>> loader = create_raw_loader(RawLoaderConfig(normalize=True))
    """
    return RawImageLoader(config or RawLoaderConfig())
