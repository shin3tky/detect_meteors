"""RAW image loader plugin."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from ..image_io import extract_exif_metadata, load_and_bin_raw_fast
from .base import BaseMetadataExtractor, DataclassInputLoader


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
            ValueError: If binning is not a supported value.
        """
        if self.binning not in (2,):
            raise ValueError(
                f"RawLoaderConfig.binning must be 2 (got {self.binning}). "
                "Other binning factors are not yet supported."
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
        >>> image = loader.load("photo.CR2")
        >>> print(image.shape, image.dtype)
        (3000, 4000) uint16

        >>> loader_normalized = RawImageLoader(RawLoaderConfig(normalize=True))
        >>> image = loader_normalized.load("photo.CR2")
        >>> print(image.shape, image.dtype)
        (3000, 4000) float32
    """

    plugin_name = "raw"
    name = "RAW Image Loader"
    version = "1.0.0"
    ConfigType = RawLoaderConfig

    def load(self, filepath: str) -> np.ndarray:
        """Load a RAW image applying the configured binning and normalization.

        Args:
            filepath: Path to the RAW image file.

        Returns:
            Loaded image as numpy array. Shape is (height, width) after 2x2 binning.
            dtype is uint16 if normalize=False, float32 if normalize=True.

        Raises:
            Exception: If the file cannot be loaded (invalid format, missing file, etc.)
        """
        # Validation is done in RawLoaderConfig.__post_init__
        image = load_and_bin_raw_fast(filepath)

        if self.config.normalize:
            image = image.astype(np.float32) / np.iinfo(np.uint16).max

        return image

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
        """
        return extract_exif_metadata(filepath)


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
