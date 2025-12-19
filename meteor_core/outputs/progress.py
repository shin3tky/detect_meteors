#!/usr/bin/env python
#
# Detect Meteors CLI - Progress Manager
# © 2025 Shinichi Morita (shin3tky)
#

"""
Progress tracking for resumable meteor detection processing.

This module provides the ProgressManager class for tracking processing
progress, enabling resumption of interrupted detection runs.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from ..exceptions import MeteorProgressError
from ..schema import VERSION
from ..i18n import log_warning

# Module-level logger for progress tracking diagnostics
logger = logging.getLogger(__name__)


def _coerce_string_list(value: Any, *, field_name: str, filepath: str) -> List[str]:
    """Ensure list values are safe lists of strings.

    Args:
        value: Raw value from the progress file.
        field_name: Name of the field being coerced.
        filepath: Path to the progress file (for logging).

    Returns:
        A list of stringified entries. Non-list inputs are reset to an empty list.
    """

    if not isinstance(value, list):
        log_warning(
            logger,
            "log.progress.invalid_field",
            path=filepath,
            field=field_name,
            expected="list",
        )
        return []

    safe_list: List[str] = []
    for entry in value:
        try:
            safe_list.append(str(entry))
        except Exception:  # pragma: no cover - extremely defensive
            log_warning(
                logger,
                "log.progress.unconvertible_entry",
                path=filepath,
                field=field_name,
                entry=entry,
            )
    return safe_list


def _normalize_detected_details(value: Any, *, filepath: str) -> List[Dict[str, Any]]:
    """Normalize detected_details to a safe list of dictionaries."""

    if not isinstance(value, list):
        log_warning(
            logger,
            "log.progress.invalid_field",
            path=filepath,
            field="detected_details",
            expected="list",
        )
        return []

    normalized: List[Dict[str, Any]] = []
    for entry in value:
        if not isinstance(entry, dict):
            log_warning(
                logger,
                "log.progress.detected_details.non_dict",
                path=filepath,
                entry=entry,
            )
            continue

        filename = entry.get("filename")
        if not filename:
            log_warning(
                logger,
                "log.progress.detected_details.missing_filename",
                path=filepath,
                entry=entry,
            )
            continue

        normalized_entry = dict(entry)
        normalized_entry["filename"] = str(filename)
        normalized.append(normalized_entry)

    return normalized


def _normalize_progress_data(
    raw_data: Any, *, filepath: str
) -> Optional[Dict[str, Any]]:
    """Validate and sanitize loaded progress JSON data.

    Args:
        raw_data: Data loaded from JSON.
        filepath: Path to the progress file (for logging).

    Returns:
        Sanitized progress data dictionary, or None if data is unusable.
    """

    if not isinstance(raw_data, dict):
        error = MeteorProgressError(
            "Progress file format is invalid (expected JSON object)",
            filepath=filepath,
            operation="load",
            context={"error_category": "invalid_format"},
        )
        logger.warning("%s", error)
        return None

    normalized = dict(raw_data)

    normalized["processed_files"] = _coerce_string_list(
        raw_data.get("processed_files", []),
        field_name="processed_files",
        filepath=filepath,
    )
    normalized["detected_files"] = _coerce_string_list(
        raw_data.get("detected_files", []),
        field_name="detected_files",
        filepath=filepath,
    )
    normalized["detected_details"] = _normalize_detected_details(
        raw_data.get("detected_details", []), filepath=filepath
    )

    processing_params = raw_data.get("processing_params", {})
    if not isinstance(processing_params, dict):
        log_warning(
            logger,
            "log.progress.invalid_field",
            path=filepath,
            field="processing_params",
            expected="dict",
        )
        processing_params = {}
    normalized["processing_params"] = processing_params

    roi_value = raw_data.get("roi", "full_image")
    if not isinstance(roi_value, (str, list, dict)):
        log_warning(
            logger,
            "log.progress.unexpected_type",
            path=filepath,
            field="roi",
            actual_type=type(roi_value).__name__,
            fallback="full_image",
        )
        roi_value = "full_image"
    normalized["roi"] = roi_value

    params_hash = raw_data.get("params_hash", "")
    normalized["params_hash"] = str(params_hash) if params_hash is not None else ""

    # Recompute totals from the sanitized lists
    normalized["total_processed"] = len(normalized["processed_files"])
    normalized["total_detected"] = len(normalized["detected_files"])

    return normalized


class ProgressManager:
    """
    Manages progress tracking for resumable processing.

    This class tracks which files have been processed and detected,
    allowing detection runs to be resumed after interruption.

    Attributes:
        progress_file: Path to progress JSON file.
        progress_data: Current progress data dictionary.
        processed_set: Set of processed filenames (for fast lookup).
        detected_set: Set of detected filenames (for fast lookup).

    Example:
        >>> manager = ProgressManager("progress.json")
        >>> loaded = manager.load()
        >>> if not loaded:
        ...     manager.reset()
        >>> manager.record_result("image001.CR2", True, score=150.0)
        >>> manager.save()
    """

    def __init__(self, progress_file: str):
        """
        Initialize the progress manager.

        Args:
            progress_file: Path to progress JSON file.
        """
        self.progress_file = progress_file
        self.progress_data: Dict[str, Any] = {
            "version": VERSION,
            "params_hash": "",
            "params": "",
            "roi": "full_image",
            "processing_params": {},
            "processed_files": [],
            "detected_files": [],
            "detected_details": [],
            "total_processed": 0,
            "total_detected": 0,
        }
        self.processed_set: Set[str] = set()
        self.detected_set: Set[str] = set()
        self._detected_details_map: Dict[str, Dict[str, Any]] = {}
        logger.debug(
            "ProgressManager initialized with progress file: %s",
            self.progress_file,
        )

    def load(self) -> bool:
        """
        Load progress from file.

        Returns:
            True if progress was loaded successfully, False otherwise.
        """
        if not os.path.exists(self.progress_file):
            logger.debug(
                "Progress file does not exist: %s",
                self.progress_file,
            )
            return False

        logger.debug("Loading progress from %s", self.progress_file)
        try:
            with open(self.progress_file, encoding="utf-8") as fp:
                loaded = json.load(fp)
                normalized = _normalize_progress_data(
                    loaded, filepath=self.progress_file
                )
                if normalized is None:
                    return False

                self.progress_data.update(normalized)
                self.processed_set = set(self.progress_data.get("processed_files", []))
                self.detected_set = set(self.progress_data.get("detected_files", []))
                self._sync_detected_details_map()
                self.progress_data["total_processed"] = len(self.processed_set)
                self.progress_data["total_detected"] = len(self.detected_set)
                logger.info(
                    "Loaded progress: %d processed, %d detected from %s",
                    len(self.processed_set),
                    len(self.detected_set),
                    self.progress_file,
                )
                return True
        except json.JSONDecodeError as exc:
            error = MeteorProgressError(
                f"Failed to parse progress file: {exc}",
                filepath=self.progress_file,
                original_error=exc,
                operation="parse",
                context={"error_category": "parse_failed"},
            )
            logger.warning("%s", error)
            return False
        except OSError as exc:
            error = MeteorProgressError(
                f"Failed to read progress file: {exc}",
                filepath=self.progress_file,
                original_error=exc,
                operation="load",
                context={"error_category": "read_failed"},
            )
            logger.warning("%s", error)
            return False

    def save(self) -> None:
        """Persist progress JSON to disk."""
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.progress_data.setdefault("created_at", now_iso)
        self.progress_data["last_updated"] = now_iso

        logger.debug(
            "Saving progress to %s (%d processed, %d detected)",
            self.progress_file,
            len(self.processed_set),
            len(self.detected_set),
        )
        try:
            os.makedirs(os.path.dirname(self.progress_file) or ".", exist_ok=True)
            with open(self.progress_file, "w", encoding="utf-8") as fp:
                json.dump(self.progress_data, fp, ensure_ascii=False, indent=2)
            logger.debug("Progress saved successfully to %s", self.progress_file)
        except OSError as exc:
            error = MeteorProgressError(
                f"Failed to write progress file: {exc}",
                filepath=self.progress_file,
                original_error=exc,
                operation="save",
                context={"error_category": "write_failed"},
            )
            logger.warning("%s", error)
        except (TypeError, ValueError) as exc:
            # JSON serialization errors
            error = MeteorProgressError(
                f"Failed to serialize progress data: {exc}",
                filepath=self.progress_file,
                original_error=exc,
                operation="serialize",
                context={"error_category": "serialize_failed"},
            )
            logger.warning("%s", error)

    def set_params_hash(self, params_hash: str) -> None:
        """Set the parameters hash for this session.

        Args:
            params_hash: Hash string representing current parameters.
        """
        self.progress_data["params_hash"] = params_hash

    def set_params(self, params: Any) -> None:
        """Record the user-specified parameters.

        Args:
            params: Parameter data to record.
        """
        self.progress_data["params"] = params

    def set_roi(self, roi: Any) -> None:
        """Record the ROI selection (polygon or 'full_image').

        Args:
            roi: ROI data (polygon list or 'full_image' string).
        """
        self.progress_data["roi"] = roi

    def set_processing_params(self, processing_params: Dict[str, Any]) -> None:
        """Record the final processing parameters used.

        Args:
            processing_params: Dictionary of processing parameters.
        """
        self.progress_data["processing_params"] = processing_params

    def get_params_hash(self) -> str:
        """Get the stored parameters hash.

        Returns:
            The stored parameters hash string.
        """
        return self.progress_data.get("params_hash", "")

    def is_processed(self, filename: str) -> bool:
        """Check if a file has been processed.

        Args:
            filename: Name of the file to check.

        Returns:
            True if the file has been processed.
        """
        return filename in self.processed_set

    def record_result(
        self,
        filename: str,
        is_candidate: bool,
        score: float = 0.0,
        lines: int = 0,
        ratio: float = 0.0,
    ) -> int:
        """
        Record the result of processing a file.

        Args:
            filename: Name of the processed file.
            is_candidate: Whether the file was detected as a candidate.
            score: Line score of detected meteor candidate.
            lines: Number of lines detected.
            ratio: Aspect ratio of detected meteor candidate.

        Returns:
            Current total detected count.
        """
        self.processed_set.add(filename)
        if filename not in self.progress_data["processed_files"]:
            self.progress_data["processed_files"].append(filename)

        if is_candidate:
            self.detected_set.add(filename)
            if filename not in self.progress_data["detected_files"]:
                self.progress_data["detected_files"].append(filename)
            entry = self._detected_details_map.get(filename)
            if not entry:
                entry = {"filename": filename}
                self._detected_details_map[filename] = entry

            entry.update(score=score, lines=lines, ratio=ratio)
            self._sync_detected_details_list()
            logger.debug(
                "Recorded candidate: %s (score=%.2f, lines=%d, ratio=%.2f)",
                filename,
                score,
                lines,
                ratio,
            )
        else:
            logger.debug("Recorded non-candidate: %s", filename)

        self.progress_data["total_processed"] = len(self.processed_set)
        self.progress_data["total_detected"] = len(self.detected_set)
        self.save()
        return self.progress_data["total_detected"]

    def filter_existing_files(self, existing_basenames: Set[str]) -> None:
        """
        Filter progress data to only include files that still exist.

        Args:
            existing_basenames: Set of filenames that currently exist.
        """
        original_processed = len(self.progress_data.get("processed_files", []))
        original_detected = len(self.progress_data.get("detected_files", []))

        self.progress_data["processed_files"] = [
            name
            for name in self.progress_data.get("processed_files", [])
            if name in existing_basenames
        ]
        self.progress_data["detected_files"] = [
            name
            for name in self.progress_data.get("detected_files", [])
            if name in existing_basenames
        ]

        # Filter detected_details to only include existing files
        if "detected_details" in self.progress_data:
            self.progress_data["detected_details"] = [
                detail
                for detail in self.progress_data["detected_details"]
                if detail.get("filename") in existing_basenames
            ]
            self._sync_detected_details_map()
        else:
            self._detected_details_map = {}

        self.processed_set = set(self.progress_data["processed_files"])
        self.detected_set = set(self.progress_data["detected_files"])

        self.progress_data["total_processed"] = len(self.processed_set)
        self.progress_data["total_detected"] = len(self.detected_set)

        removed_processed = original_processed - len(self.processed_set)
        removed_detected = original_detected - len(self.detected_set)
        if removed_processed > 0 or removed_detected > 0:
            logger.info(
                "Filtered progress: removed %d processed, %d detected "
                "(now %d processed, %d detected)",
                removed_processed,
                removed_detected,
                len(self.processed_set),
                len(self.detected_set),
            )

    def get_total_processed(self) -> int:
        """Get the total number of processed files.

        Returns:
            Total number of processed files.
        """
        return self.progress_data.get("total_processed", 0)

    def get_total_detected(self) -> int:
        """Get the total number of detected candidates.

        Returns:
            Total number of detected candidates.
        """
        return self.progress_data.get("total_detected", 0)

    def reset(self) -> None:
        """Reset progress data while preserving configuration."""
        logger.info(
            "Resetting progress (was %d processed, %d detected)",
            len(self.processed_set),
            len(self.detected_set),
        )
        self.progress_data = {
            "version": VERSION,
            "params_hash": self.progress_data.get("params_hash", ""),
            "params": self.progress_data.get("params", ""),
            "roi": self.progress_data.get("roi", "full_image"),
            "processing_params": self.progress_data.get("processing_params", {}),
            "processed_files": [],
            "detected_files": [],
            "detected_details": [],
            "total_processed": 0,
            "total_detected": 0,
        }
        self.processed_set = set()
        self.detected_set = set()
        self._detected_details_map = {}
        logger.debug("Progress reset completed")

    def _sync_detected_details_map(self) -> None:
        """Synchronize the detected details map from the list representation."""
        self._detected_details_map = {}
        for detail in self.progress_data.get("detected_details", []):
            filename = detail.get("filename")
            if filename:
                self._detected_details_map[filename] = detail

    def _sync_detected_details_list(self) -> None:
        """Synchronize the detected details list from the map representation."""
        self.progress_data["detected_details"] = list(
            self._detected_details_map.values()
        )


# Standalone functions for backward compatibility
def load_progress(progress_path: str) -> Optional[Dict]:
    """Load progress JSON if it exists.

    Args:
        progress_path: Path to the progress file.

    Returns:
        Progress data dictionary, or None if file doesn't exist or is invalid.
    """
    if not os.path.exists(progress_path):
        logger.debug("Progress file does not exist: %s", progress_path)
        return None

    logger.debug("Loading progress from %s", progress_path)
    try:
        with open(progress_path, encoding="utf-8") as fp:
            loaded = json.load(fp)
            data = _normalize_progress_data(loaded, filepath=progress_path)
            if data is None:
                return None
            logger.debug(
                "Loaded progress data with %d processed, %d detected",
                len(data.get("processed_files", [])),
                len(data.get("detected_files", [])),
            )
            return data
    except json.JSONDecodeError as exc:
        error = MeteorProgressError(
            f"Failed to parse progress file: {exc}",
            filepath=progress_path,
            original_error=exc,
            operation="parse",
            context={"error_category": "parse_failed"},
        )
        logger.warning("%s", error)
        return None
    except OSError as exc:
        error = MeteorProgressError(
            f"Failed to read progress file: {exc}",
            filepath=progress_path,
            original_error=exc,
            operation="load",
            context={"error_category": "read_failed"},
        )
        logger.warning("%s", error)
        return None


def save_progress(progress_path: str, progress_data: Dict) -> None:
    """Persist progress JSON to disk.

    Args:
        progress_path: Path to save the progress file.
        progress_data: Progress data dictionary to save.
    """
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    progress_data.setdefault("created_at", now_iso)
    progress_data["last_updated"] = now_iso

    logger.debug(
        "Saving progress to %s (%d processed, %d detected)",
        progress_path,
        len(progress_data.get("processed_files", [])),
        len(progress_data.get("detected_files", [])),
    )
    try:
        os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as fp:
            json.dump(progress_data, fp, ensure_ascii=False, indent=2)
        logger.debug("Progress saved successfully to %s", progress_path)
    except OSError as exc:
        error = MeteorProgressError(
            f"Failed to write progress file: {exc}",
            filepath=progress_path,
            original_error=exc,
            operation="save",
            context={"error_category": "write_failed"},
        )
        logger.warning("%s", error)
    except (TypeError, ValueError) as exc:
        # JSON serialization errors
        error = MeteorProgressError(
            f"Failed to serialize progress data: {exc}",
            filepath=progress_path,
            original_error=exc,
            operation="serialize",
            context={"error_category": "serialize_failed"},
        )
        logger.warning("%s", error)


__all__ = [
    "ProgressManager",
    "load_progress",
    "save_progress",
]
