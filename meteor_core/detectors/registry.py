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

Developer guidance
------------------
Use ``create`` when providing explicit configuration that may need to be
coerced into the detector's ``ConfigType``. ``create_default`` is reserved for
the built-in default detector and assumes its ``ConfigType`` can be
instantiated with no arguments to supply a full default configuration; the
method raises when that contract is not met.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, Union

from ..plugin_registry import _PLUGIN_KIND_DETECTOR
from ..plugin_registry_base import PluginRegistryBase
from ..schema import DEFAULT_DETECTOR_NAME
from .base import BaseDetector, _is_valid_detector


class DetectorRegistry(PluginRegistryBase[BaseDetector]):
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

    _plugin_kind = _PLUGIN_KIND_DETECTOR

    @classmethod
    def _discover_internal(cls) -> Dict[str, Type[BaseDetector]]:
        # Import here to avoid circular dependency
        # Keep plugin directory path in sync with meteor_core.detectors.discovery.PLUGIN_DIR
        from .discovery import _discover_handlers_internal

        return _discover_handlers_internal()

    @classmethod
    def _is_valid_plugin(cls, detector_cls: Type[BaseDetector]) -> bool:
        return _is_valid_detector(detector_cls)

    # ========================================
    # Instantiation
    # ========================================

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[Dict[str, Any], Any]] = None,
    ) -> BaseDetector:
        """Create a detector instance.

        Select this method when you need to override default settings; the
        registry will coerce dictionaries into ``ConfigType`` instances where
        possible before constructing the detector.

        Args:
            name: Detector plugin_name.
            config: Configuration for the detector. Can be:
                - None: Uses default config (ConfigType() if available)
                - Dict: Coerced to ConfigType (dataclass or Pydantic model)
                - ConfigType instance: Used as-is

        Returns:
            Detector instance.

        Raises:
            KeyError: If detector not found.

        Example:
            >>> detector = DetectorRegistry.create("hough")
            >>> detector = DetectorRegistry.create("hough", {"some_option": True})
        """
        detector_cls = cls.get(name)
        coerced_config = cls._coerce_config(detector_cls, config)
        return detector_cls(coerced_config)

    @classmethod
    def create_default(
        cls, config: Optional[Union[Dict[str, Any], Any]] = None
    ) -> BaseDetector:
        """Create the default detector using its default configuration.

        The default detector must define a zero-argument ``ConfigType`` that
        returns a fully populated configuration. Callers can optionally supply a
        config object or mapping, which will be coerced using :meth:`create`
        semantics.

        Returns:
            Default detector instance (currently "hough" detector).
        """
        detector_cls = cls.get(DEFAULT_DETECTOR_NAME)
        coerced_config = cls._coerce_config(detector_cls, config)
        if coerced_config is None:
            raise TypeError(
                "Default detector does not define ConfigType; cannot create default."
            )

        return detector_cls(coerced_config)


__all__ = ["DetectorRegistry"]
