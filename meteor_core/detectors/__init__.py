#!/usr/bin/env python
#
# Detect Meteors CLI - Detectors Package
# © 2025 Shinichi Morita (shin3tky)
#

"""
Detector implementations for meteor detection.
"""

from .base import BaseDetector
from .hough_default import HoughDetector, compute_line_score_fast

__all__ = [
    "BaseDetector",
    "HoughDetector",
    "compute_line_score_fast",
]
