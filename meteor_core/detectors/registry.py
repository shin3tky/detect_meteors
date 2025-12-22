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

import logging
from typing import Any, Dict, Optional, Type, Union

from ..plugin_registry import _PLUGIN_KIND_DETECTOR
from ..plugin_registry_base import PluginRegistryBase
from ..schema import DEFAULT_DETECTOR_NAME
from .base import BaseDetector, _is_valid_detector

# Module-level logger for registry operations
logger = logging.getLogger(__name__)


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
        >>> context = DetectionContext(
        ...     current_image=current_img,
        ...     previous_image=prev_img,
        ...     roi_mask=roi_mask,
        ...     runtime_params=params,
        ...     metadata={},
        ... )
        >>> result = detector.detect(context)

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
            TypeError: If config type is incompatible or defaults cannot be built.
            ValueError: If config validation fails.

        Example:
            >>> detector = DetectorRegistry.create("hough")
            >>> detector = DetectorRegistry.create("hough", {"some_option": True})
        """
        logger.debug("Creating detector '%s' with config %r", name, config)
        detector_cls = cls.get(name)
        try:
            coerced_config = cls._coerce_config(detector_cls, config)
        except Exception as exc:
            logger.error(
                "Failed to coerce config for detector '%s': %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
            raise

        try:
            instance = detector_cls(coerced_config)
        except Exception as exc:
            logger.error(
                "Failed to instantiate detector '%s' (%s): %s: %s",
                name,
                detector_cls.__name__,
                type(exc).__name__,
                exc,
            )
            raise

        logger.debug(
            "Created detector '%s' (%s) with config type %s",
            name,
            detector_cls.__name__,
            type(coerced_config).__name__ if coerced_config is not None else None,
        )
        return instance

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

        Raises:
            KeyError: If the default detector is not found.
            TypeError: If ConfigType is missing or config type is incompatible.
            ValueError: If config validation fails.
        """
        # Detectors rely solely on their ConfigType defaults; unlike output handlers
        # there are no registry-level path overrides to apply here.
        logger.debug(
            "Creating default detector '%s' with override config %r",
            DEFAULT_DETECTOR_NAME,
            config,
        )

        detector_cls = cls.get(DEFAULT_DETECTOR_NAME)
        try:
            coerced_config = cls._coerce_config(detector_cls, config)
        except Exception as exc:
            logger.error(
                "Failed to coerce config for default detector '%s': %s: %s",
                DEFAULT_DETECTOR_NAME,
                type(exc).__name__,
                exc,
            )
            raise

        if coerced_config is None:
            logger.error(
                "Default detector '%s' does not define ConfigType; cannot create default instance",
                DEFAULT_DETECTOR_NAME,
            )
            raise TypeError(
                "Default detector does not define ConfigType; cannot create default."
            )

        try:
            instance = detector_cls(coerced_config)
        except Exception as exc:
            logger.error(
                "Failed to instantiate default detector '%s' (%s): %s: %s",
                DEFAULT_DETECTOR_NAME,
                detector_cls.__name__,
                type(exc).__name__,
                exc,
            )
            raise

        logger.debug(
            "Created default detector '%s' (%s) with config type %s",
            DEFAULT_DETECTOR_NAME,
            detector_cls.__name__,
            type(coerced_config).__name__,
        )
        return instance


__all__ = ["DetectorRegistry"]
