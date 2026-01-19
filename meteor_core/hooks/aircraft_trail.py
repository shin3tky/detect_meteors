"""Hook for aircraft light trail metadata."""

from __future__ import annotations

from .base import DataclassHook


class AircraftTrailHook(DataclassHook[object]):
    """Hook placeholder for aircraft light trail analysis."""

    plugin_name = "aircraft_trail"
    name = "Aircraft Trail"
    version = "1.0.0"
