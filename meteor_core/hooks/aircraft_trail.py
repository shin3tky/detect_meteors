"""Hook for aircraft light trail metadata.

This module aggregates heuristics for tracking consecutive line detections that
look like aircraft light trails. The hook maintains lightweight state across
frames and annotates detection results with likelihood evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

from .base import DataclassHook
from ..schema import DetectionContext, DetectionResult


@dataclass
class _TrackState:
    """Track state for a candidate aircraft trail.

    Attributes:
        track_id: Unique identifier for the track.
        last_start: The most recent line start point (x, y).
        last_end: The most recent line end point (x, y).
        last_angle_deg: The most recent line angle in degrees.
        last_midpoint: The most recent line midpoint (x, y).
        last_frame_index: Frame index of the most recent observation.
        frames: Number of frames matched to this track.
        last_speed: Last observed midpoint speed in pixels per frame.
        last_start_distance: Last start-point distance to the previous line.
        last_end_distance: Last end-point distance to the previous line.
        last_angle_diff: Last angular difference in degrees to the previous line.
        last_speed_variance: Last normalized speed variance to the previous line.
    """

    track_id: str
    last_start: Tuple[float, float]
    last_end: Tuple[float, float]
    last_angle_deg: float
    last_midpoint: Tuple[float, float]
    last_frame_index: int
    frames: int = 1
    last_speed: Optional[float] = None
    last_start_distance: Optional[float] = None
    last_end_distance: Optional[float] = None
    last_angle_diff: Optional[float] = None
    last_speed_variance: Optional[float] = None


@dataclass(frozen=True)
class AircraftTrailConfig:
    """Configuration defaults for aircraft light trail analysis.

    Attributes:
        min_track_frames: Minimum frames before continuity reaches full weight.
        max_start_distance_px: Max start-point distance allowed for matching.
        max_end_distance_px: Max end-point distance allowed for matching.
        max_angle_diff_deg: Max angle difference allowed for matching.
        max_speed_variance: Max normalized speed variance allowed for matching.
        likelihood_threshold: Minimum likelihood considered as aircraft trail.
        track_ttl_frames: Frames to keep inactive tracks before expiration.
    """

    min_track_frames: int = 3
    max_start_distance_px: float = 6.0
    max_end_distance_px: float = 6.0
    max_angle_diff_deg: float = 3.0
    max_speed_variance: float = 0.3
    likelihood_threshold: float = 0.7
    track_ttl_frames: int = 5


class AircraftTrailHook(DataclassHook[AircraftTrailConfig]):
    """Annotate detections with aircraft trail likelihood metadata."""

    plugin_name = "aircraft_trail"
    name = "Aircraft Trail"
    version = "1.0.0"
    ConfigType = AircraftTrailConfig

    def __init__(self, config: Optional[AircraftTrailConfig] = None) -> None:
        """Initialize the hook state.

        Args:
            config: Configuration values for track matching and scoring.
                If None, default configuration is used.
        """
        super().__init__(config)
        self._tracks: Dict[str, _TrackState] = {}
        self._track_counter = 0

    def on_detection_complete(
        self,
        result: DetectionResult,
        context: DetectionContext,
    ) -> DetectionResult:
        """Attach aircraft trail metadata to a detection result.

        Args:
            result: Detection output containing line candidates.
            context: Context metadata for the detection pass.

        Returns:
            The updated detection result with aircraft trail metadata.
        """
        frame_index = None
        if context.metadata and isinstance(context.metadata, dict):
            current_meta = context.metadata.get("current", {})
            frame_index = current_meta.get("frame_index")

        if not result.lines:
            result.extras["aircraft"] = {
                "likelihood": 0.0,
                "track_id": None,
                "evidence": {
                    "track_frames": 0,
                    "angle_diff_deg": None,
                    "start_distance_px": None,
                    "end_distance_px": None,
                    "speed_consistency": None,
                },
            }
            return result

        line = _select_primary_line(result.lines)
        start, end, angle_deg, midpoint = _line_geometry(line)

        if frame_index is None:
            result.extras["aircraft"] = {
                "likelihood": 0.0,
                "track_id": None,
                "evidence": {
                    "track_frames": 1,
                    "angle_diff_deg": None,
                    "start_distance_px": None,
                    "end_distance_px": None,
                    "speed_consistency": None,
                },
            }
            return result

        self._expire_tracks(frame_index)
        track = self._match_track(start, end, angle_deg, midpoint, frame_index)
        if track is None:
            track = self._create_track(start, end, angle_deg, midpoint, frame_index)

        likelihood = _compute_likelihood(track, self.config)
        result.extras["aircraft"] = {
            "likelihood": likelihood,
            "track_id": track.track_id,
            "evidence": {
                "track_frames": track.frames,
                "angle_diff_deg": track.last_angle_diff,
                "start_distance_px": track.last_start_distance,
                "end_distance_px": track.last_end_distance,
                "speed_consistency": _speed_consistency(track, self.config),
            },
        }
        return result

    def _expire_tracks(self, frame_index: int) -> None:
        """Remove tracks that are too old for the current frame.

        Args:
            frame_index: Current frame index for TTL evaluation.
        """
        ttl = self.config.track_ttl_frames
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame_index > ttl
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _match_track(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        angle_deg: float,
        midpoint: Tuple[float, float],
        frame_index: int,
    ) -> Optional[_TrackState]:
        """Find the best matching track for the current line observation.

        Args:
            start: Line start point (x, y).
            end: Line end point (x, y).
            angle_deg: Line angle in degrees.
            midpoint: Line midpoint (x, y).
            frame_index: Current frame index.

        Returns:
            The matched track, or None if no track qualifies.
        """
        best_track: Optional[_TrackState] = None
        best_score = float("inf")
        for track in self._tracks.values():
            start_distance, end_distance = _endpoint_distance(
                track.last_start,
                track.last_end,
                start,
                end,
            )
            angle_diff = _angle_diff_deg(track.last_angle_deg, angle_deg)
            if (
                start_distance > self.config.max_start_distance_px
                or end_distance > self.config.max_end_distance_px
                or angle_diff > self.config.max_angle_diff_deg
            ):
                continue

            speed_variance, speed = _speed_variance(
                track,
                midpoint,
                frame_index,
            )
            if (
                speed_variance is not None
                and speed_variance > self.config.max_speed_variance
            ):
                continue

            score = start_distance + end_distance + angle_diff
            if score < best_score:
                best_score = score
                best_track = track
                best_track.last_start_distance = start_distance
                best_track.last_end_distance = end_distance
                best_track.last_angle_diff = angle_diff
                best_track.last_speed_variance = speed_variance

        if best_track is None:
            return None

        speed_variance, speed = _speed_variance(
            best_track,
            midpoint,
            frame_index,
        )
        best_track.frames += 1
        best_track.last_start = start
        best_track.last_end = end
        best_track.last_angle_deg = angle_deg
        best_track.last_midpoint = midpoint
        best_track.last_frame_index = frame_index
        if speed is not None:
            best_track.last_speed = speed
        best_track.last_speed_variance = speed_variance
        return best_track

    def _create_track(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        angle_deg: float,
        midpoint: Tuple[float, float],
        frame_index: int,
    ) -> _TrackState:
        """Create a new track from the current line observation.

        Args:
            start: Line start point (x, y).
            end: Line end point (x, y).
            angle_deg: Line angle in degrees.
            midpoint: Line midpoint (x, y).
            frame_index: Current frame index.

        Returns:
            The newly created track.
        """
        self._track_counter += 1
        track_id = f"air-{self._track_counter:04d}"
        track = _TrackState(
            track_id=track_id,
            last_start=start,
            last_end=end,
            last_angle_deg=angle_deg,
            last_midpoint=midpoint,
            last_frame_index=frame_index,
            frames=1,
            last_speed=None,
        )
        self._tracks[track_id] = track
        return track


def _select_primary_line(
    lines: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    """Select the longest line from a list of line candidates.

    Args:
        lines: Sequence of line endpoints (x1, y1, x2, y2).

    Returns:
        The line with the greatest length.
    """
    return max(lines, key=_line_length)


def _line_length(line: Tuple[int, int, int, int]) -> float:
    """Compute the Euclidean length of a line.

    Args:
        line: Line endpoints (x1, y1, x2, y2).

    Returns:
        The line length in pixels.
    """
    x1, y1, x2, y2 = line
    return math.hypot(x2 - x1, y2 - y1)


def _line_geometry(
    line: Tuple[int, int, int, int],
) -> Tuple[Tuple[float, float], Tuple[float, float], float, Tuple[float, float]]:
    """Return geometric properties derived from a line.

    Args:
        line: Line endpoints (x1, y1, x2, y2).

    Returns:
        A tuple containing the start point, end point, angle in degrees, and
        midpoint.
    """
    x1, y1, x2, y2 = line
    start = (float(x1), float(y1))
    end = (float(x2), float(y2))
    angle_deg = _normalize_angle_deg(math.degrees(math.atan2(y2 - y1, x2 - x1)))
    midpoint = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return start, end, angle_deg, midpoint


def _normalize_angle_deg(angle_deg: float) -> float:
    """Normalize an angle into the [0, 180) range.

    Args:
        angle_deg: Angle in degrees.

    Returns:
        Normalized angle in degrees.
    """
    angle = angle_deg % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def _angle_diff_deg(angle_a: float, angle_b: float) -> float:
    """Compute the minimal angle difference between two angles.

    Args:
        angle_a: First angle in degrees.
        angle_b: Second angle in degrees.

    Returns:
        The smallest absolute difference in degrees.
    """
    diff = abs(angle_a - angle_b)
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def _endpoint_distance(
    prev_start: Tuple[float, float],
    prev_end: Tuple[float, float],
    curr_start: Tuple[float, float],
    curr_end: Tuple[float, float],
) -> Tuple[float, float]:
    """Compute the best-matching endpoint distances between two lines.

    Args:
        prev_start: Previous line start point (x, y).
        prev_end: Previous line end point (x, y).
        curr_start: Current line start point (x, y).
        curr_end: Current line end point (x, y).

    Returns:
        A tuple of distances (start_distance, end_distance) using the endpoint
        pairing that minimizes total distance.
    """
    direct_start = math.hypot(
        curr_start[0] - prev_start[0], curr_start[1] - prev_start[1]
    )
    direct_end = math.hypot(curr_end[0] - prev_end[0], curr_end[1] - prev_end[1])
    swap_start = math.hypot(curr_end[0] - prev_start[0], curr_end[1] - prev_start[1])
    swap_end = math.hypot(curr_start[0] - prev_end[0], curr_start[1] - prev_end[1])
    if swap_start + swap_end < direct_start + direct_end:
        return swap_start, swap_end
    return direct_start, direct_end


def _speed_variance(
    track: _TrackState,
    midpoint: Tuple[float, float],
    frame_index: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute normalized speed variance from the previous track observation.

    Args:
        track: Track state being updated.
        midpoint: Current line midpoint (x, y).
        frame_index: Current frame index.

    Returns:
        A tuple containing the normalized speed variance and the computed speed.
    """
    frame_delta = frame_index - track.last_frame_index
    if frame_delta <= 0:
        return None, None
    distance = math.hypot(
        midpoint[0] - track.last_midpoint[0],
        midpoint[1] - track.last_midpoint[1],
    )
    speed = distance / frame_delta
    if track.last_speed is None:
        return 0.0, speed
    baseline = max(track.last_speed, speed, 1e-6)
    variance = abs(speed - track.last_speed) / baseline
    return variance, speed


def _speed_consistency(
    track: _TrackState, config: AircraftTrailConfig
) -> Optional[float]:
    """Convert speed variance into a normalized consistency score.

    Args:
        track: Track state with speed variance metadata.
        config: Configuration with maximum speed variance threshold.

    Returns:
        Consistency score in the range [0, 1], or None if unavailable.
    """
    if track.last_speed_variance is None:
        return None
    if config.max_speed_variance <= 0:
        return 1.0
    return max(0.0, 1.0 - (track.last_speed_variance / config.max_speed_variance))


def _compute_likelihood(track: _TrackState, config: AircraftTrailConfig) -> float:
    """Compute the likelihood that a track is an aircraft trail.

    Args:
        track: Track state to score.
        config: Configuration values for weighting thresholds.

    Returns:
        Likelihood score in the range [0, 1].
    """
    continuity = min(1.0, track.frames / max(1, config.min_track_frames))
    angle_score = 0.0
    start_score = 0.0
    end_score = 0.0
    if track.last_angle_diff is not None and config.max_angle_diff_deg > 0:
        angle_score = max(
            0.0, 1.0 - (track.last_angle_diff / config.max_angle_diff_deg)
        )
    if track.last_start_distance is not None and config.max_start_distance_px > 0:
        start_score = max(
            0.0,
            1.0 - (track.last_start_distance / config.max_start_distance_px),
        )
    if track.last_end_distance is not None and config.max_end_distance_px > 0:
        end_score = max(
            0.0,
            1.0 - (track.last_end_distance / config.max_end_distance_px),
        )
    speed_score = _speed_consistency(track, config) or 0.0
    likelihood = (
        continuity * 0.5
        + angle_score * 0.2
        + ((start_score + end_score) / 2.0) * 0.2
        + speed_score * 0.1
    )
    return max(0.0, min(1.0, likelihood))
