#!/usr/bin/env python
#
# Detect Meteors CLI - Detectors Package
# © 2025 Shinichi Morita (shin3tky)
#

"""
Detector implementations for meteor detection.
"""

from .base import BaseDetector, DataclassDetector, PydanticDetector, _is_valid_detector
from .hough_default import HoughDetector, compute_line_score_fast
from .simple_threshold import SimpleThresholdConfig, SimpleThresholdDetector
from .registry import DetectorRegistry
from .discovery import PLUGIN_DIR, PLUGIN_GROUP

# Deprecated: use DetectorRegistry.discover() instead
from .discovery import discover_detectors

__all__ = [
    # Base class
    "BaseDetector",
    "DataclassDetector",
    "PydanticDetector",
    "_is_valid_detector",
    # Built-in detector
    "HoughDetector",
    "SimpleThresholdDetector",
    "SimpleThresholdConfig",
    "compute_line_score_fast",
    # Registry (recommended)
    "DetectorRegistry",
    # Discovery constants
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
    # Discovery function (deprecated)
    "discover_detectors",
]
