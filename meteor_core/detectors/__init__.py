#!/usr/bin/env python
#
# Detect Meteors CLI - Detectors Package
# © 2025 Shinichi Morita (shin3tky)
#

"""
Detector implementations for meteor detection.
"""

from .base import BaseDetector, _is_valid_detector
from .hough_default import HoughDetector, compute_line_score_fast
from .discovery import discover_detectors, PLUGIN_DIR, PLUGIN_GROUP

__all__ = [
    "BaseDetector",
    "HoughDetector",
    "compute_line_score_fast",
    "_is_valid_detector",
    "discover_detectors",
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
]
