"""Hook for aircraft light trail metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .base import DataclassHook


@dataclass(frozen=True)
class AircraftTrailConfig:
    """Configuration defaults for aircraft light trail analysis."""

    min_track_frames: int = 3
    max_start_distance_px: float = 6.0
    max_end_distance_px: float = 6.0
    max_angle_diff_deg: float = 3.0
    max_speed_variance: float = 0.3
    likelihood_threshold: float = 0.7
    track_ttl_frames: int = 5


class AircraftTrailHook(DataclassHook[AircraftTrailConfig]):
    """Hook placeholder for aircraft light trail analysis."""

    plugin_name = "aircraft_trail"
    name = "Aircraft Trail"
    version = "1.0.0"
    ConfigType = AircraftTrailConfig
