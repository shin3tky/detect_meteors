"""RAW image loader plugin."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np

from ..image_io import extract_exif_metadata, load_and_bin_raw_fast
from .base import DataclassInputLoader


@dataclass
class RawLoaderConfig:
    """Configuration for :class:`RawImageLoader`."""

    binning: int = 2
    normalize: bool = False


class RawImageLoader(DataclassInputLoader[RawLoaderConfig]):
    """Loader that wraps the existing RAW helpers for plugin use."""

    plugin_name = "raw"
    ConfigType = RawLoaderConfig

    def load(self, filepath: str) -> np.ndarray:
        """Load a RAW image applying the configured binning and normalization."""

        if self.config.binning not in (None, 2):
            raise ValueError("RawImageLoader currently supports only 2x2 binning.")

        image = load_and_bin_raw_fast(filepath)

        if self.config.normalize:
            image = image.astype(np.float32) / np.iinfo(image.dtype).max

        return image

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract EXIF metadata for the RAW file."""

        return extract_exif_metadata(filepath)


def create_raw_loader(config: Optional[RawLoaderConfig] = None) -> RawImageLoader:
    """Factory helper to create a :class:`RawImageLoader` with defaults."""

    return RawImageLoader(config or RawLoaderConfig())
