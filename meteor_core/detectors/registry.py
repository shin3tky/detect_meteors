#!/usr/bin/env python
#
# Detect Meteors CLI - Detector Registry
# © 2025 Shinichi Morita (shin3tky)
#

"""
Registry for detector plugins with discovery, registration, and instantiation.

This module provides a centralized registry for managing detectors,
supporting both automatic discovery (via entry points and plugin directory)
and runtime registration for testing and dynamic plugins.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Type

from ..schema import DEFAULT_DETECTOR_NAME
from .base import BaseDetector, _is_valid_detector


class DetectorRegistry:
    """Detector registry with discovery, registration, and instantiation.

    This registry provides:
    - Lazy discovery of detectors from entry points and plugin directory
    - Runtime registration/unregistration for testing and dynamic plugins
    - Factory method for creating detector instances

    The registry uses class-level state for caching discovered detectors
    and storing runtime-registered detectors.

    Example:
        >>> # Get a detector class
        >>> detector_cls = DetectorRegistry.get("hough")
        >>> print(detector_cls.plugin_name)
        'hough'

        >>> # Create an instance
        >>> detector = DetectorRegistry.create("hough")
        >>> result = detector.detect(current_img, prev_img, roi_mask, params)

        >>> # Register a custom detector for testing
        >>> DetectorRegistry.register(MyCustomDetector)
        >>> DetectorRegistry.unregister("my_custom")
    """

    # Discovered detectors (from entry points + plugin dir), lazily initialized
    _discovered: Optional[Dict[str, Type[BaseDetector]]] = None

    # Runtime-registered detectors (for testing/dynamic plugins)
    _custom: Dict[str, Type[BaseDetector]] = {}

    # ========================================
    # Discovery & Registration
    # ========================================

    @classmethod
    def discover(cls, force: bool = False) -> Dict[str, Type[BaseDetector]]:
        """Discover available detectors (cached, lazy).

        Discovers detectors from:
        1. Built-in detectors (e.g., HoughDetector)
        2. Entry points (detect_meteors.detector group)
        3. Plugin directory (~/.detect_meteors/detector_plugins)

        Args:
            force: If True, re-discover even if already cached.

        Returns:
            Dict mapping plugin_name to detector class.
            Does not include runtime-registered detectors; use get() or
            list_available() for the complete list.
        """
        if cls._discovered is None or force:
            # Import here to avoid circular dependency
            from .discovery import _discover_detectors_internal

            cls._discovered = _discover_detectors_internal()
        return cls._discovered

    @classmethod
    def get(cls, name: str) -> Type[BaseDetector]:
        """Get detector class by name.

        Lookup priority:
        1. Runtime-registered detectors (_custom)
        2. Discovered detectors (_discovered)

        Args:
            name: Detector plugin_name.

        Returns:
            Detector class.

        Raises:
            KeyError: If detector not found.
        """
        # Normalize name to lowercase for case-insensitive lookup
        name_lower = name.lower()

        # 1. Custom (runtime-registered) takes priority
        if name_lower in cls._custom:
            return cls._custom[name_lower]

        # 2. Discovered detectors
        discovered = cls.discover()
        if name_lower in discovered:
            return discovered[name_lower]

        # 3. Not found - provide helpful error message
        available = cls.list_available()
        available_str = ", ".join(sorted(available)) if available else "none"
        raise KeyError(f"Unknown detector '{name}'. Available: {available_str}")

    @classmethod
    def register(cls, detector_cls: Type[BaseDetector]) -> None:
        """Register a detector class at runtime.

        Runtime-registered detectors take priority over discovered detectors
        with the same name. This is useful for testing and dynamic plugins.

        Args:
            detector_cls: Detector class with plugin_name attribute.

        Raises:
            ValueError: If plugin_name is empty or class is invalid.

        Example:
            >>> class MockDetector(BaseDetector):
            ...     plugin_name = "mock"
            ...     def detect(self, ...): ...
            ...     def compute_line_score(self, ...): ...
            >>> DetectorRegistry.register(MockDetector)
        """
        if not _is_valid_detector(detector_cls):
            raise ValueError(
                f"Invalid detector class: {detector_cls}. "
                "Must inherit from BaseDetector and have non-empty plugin_name."
            )

        name = detector_cls.plugin_name
        if not name:
            raise ValueError("Detector must have non-empty plugin_name")

        # Normalize to lowercase
        name_lower = name.lower()

        # Warn on overwrite (but allow it for testing purposes)
        if name_lower in cls._custom:
            warnings.warn(
                f"Overwriting existing runtime-registered detector '{name}'",
                stacklevel=2,
            )

        cls._custom[name_lower] = detector_cls

    @classmethod
    def unregister(cls, name: str) -> bool:
        """Unregister a detector by name.

        Only removes from runtime-registered detectors. Discovered detectors
        cannot be unregistered (they will be re-discovered).

        Args:
            name: Detector plugin_name to remove.

        Returns:
            True if removed, False if not found in runtime registry.
        """
        name_lower = name.lower()
        if name_lower in cls._custom:
            del cls._custom[name_lower]
            return True
        return False

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available detector names.

        Returns:
            Sorted list of available detector plugin_names,
            including both discovered and runtime-registered detectors.
        """
        discovered = cls.discover()
        all_names = set(discovered.keys()) | set(cls._custom.keys())
        return sorted(all_names)

    # ========================================
    # Instantiation
    # ========================================

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseDetector:
        """Create a detector instance.

        Args:
            name: Detector plugin_name.
            config: Configuration for the detector (reserved for future use).
                   Currently, detectors don't have configuration support,
                   but this parameter is included for API consistency.

        Returns:
            Detector instance.

        Raises:
            KeyError: If detector not found.

        Example:
            >>> detector = DetectorRegistry.create("hough")
            >>> detector = DetectorRegistry.create("hough", {"some_option": True})
        """
        detector_cls = cls.get(name)
        # TODO: Pass config to detector constructor when supported
        return detector_cls()

    @classmethod
    def create_default(cls) -> BaseDetector:
        """Create the default detector.

        Returns:
            Default detector instance (currently "hough" detector).
        """
        return cls.create(DEFAULT_DETECTOR_NAME)

    # ========================================
    # Internal Methods
    # ========================================

    @classmethod
    def _reset(cls) -> None:
        """Reset registry state.

        This method is intended for testing only. It clears both
        the discovered cache and runtime-registered detectors.
        """
        cls._discovered = None
        cls._custom = {}


__all__ = ["DetectorRegistry"]
