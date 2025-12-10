#!/usr/bin/env python
#
# Detect Meteors CLI - Output Writer
# © 2025 Shinichi Morita (shin3tky)
#

"""
Output writing and progress management for meteor detection.
Handles file saving, debug images, and progress tracking.
"""

import os
import json
import shutil
import cv2
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set

from ..schema import VERSION
from .handler import BaseOutputHandler


class OutputWriter(BaseOutputHandler):
    """
    Manages output files, debug images, and progress tracking.

    Attributes:
        output_folder: Directory for candidate files
        debug_folder: Directory for debug images
        progress_file: Path to progress JSON file
        output_overwrite: Whether to overwrite existing files
    """

    def __init__(
        self,
        output_folder: str,
        debug_folder: str,
        progress_file: str,
        output_overwrite: bool = False,
    ):
        """
        Initialize the output writer.

        Args:
            output_folder: Directory for candidate files
            debug_folder: Directory for debug images
            progress_file: Path to progress JSON file
            output_overwrite: Whether to overwrite existing files
        """
        self.output_folder = output_folder
        self.debug_folder = debug_folder
        self.progress_file = progress_file
        self.output_overwrite = output_overwrite

        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(debug_folder, exist_ok=True)

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> bool:
        """
        Save a meteor candidate file and optional debug image.

        Args:
            source_path: Path to the source RAW file
            filename: Output filename
            debug_image: Optional debug visualization image (BGR)
            roi_polygon: Optional ROI polygon to draw on debug image

        Returns:
            True if file was saved, False if skipped (already exists)
        """
        output_path = os.path.join(self.output_folder, filename)

        # Check if file exists
        if os.path.exists(output_path) and not self.output_overwrite:
            return False

        # Copy the RAW file
        shutil.copy(source_path, output_path)

        # Save debug image if provided
        if debug_image is not None:
            if roi_polygon:
                cv2.polylines(
                    debug_image,
                    [np.array(roi_polygon, dtype=np.int32)],
                    True,
                    (0, 255, 0),
                    2,
                )
            debug_path = os.path.join(self.debug_folder, f"mask_{filename}.png")
            cv2.imwrite(debug_path, debug_image)

        return True

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """
        Save a debug visualization image.

        Args:
            debug_image: Debug visualization image (BGR)
            filename: Base filename (will be prefixed with 'mask_')
            roi_polygon: Optional ROI polygon to draw

        Returns:
            Path to saved debug image
        """
        if roi_polygon:
            cv2.polylines(
                debug_image,
                [np.array(roi_polygon, dtype=np.int32)],
                True,
                (0, 255, 0),
                2,
            )
        debug_path = os.path.join(self.debug_folder, f"mask_{filename}.png")
        cv2.imwrite(debug_path, debug_image)
        return debug_path


class ProgressManager:
    """
    Manages progress tracking for resumable processing.

    Attributes:
        progress_file: Path to progress JSON file
        progress_data: Current progress data
        processed_set: Set of processed filenames (for fast lookup)
        detected_set: Set of detected filenames (for fast lookup)
    """

    def __init__(self, progress_file: str):
        """
        Initialize the progress manager.

        Args:
            progress_file: Path to progress JSON file
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
            "total_processed": 0,
            "total_detected": 0,
        }
        self.processed_set: Set[str] = set()
        self.detected_set: Set[str] = set()

    def load(self) -> bool:
        """
        Load progress from file.

        Returns:
            True if progress was loaded successfully, False otherwise
        """
        if not os.path.exists(self.progress_file):
            return False

        try:
            with open(self.progress_file, encoding="utf-8") as fp:
                loaded = json.load(fp)
                self.progress_data.update(loaded)
                self.processed_set = set(self.progress_data.get("processed_files", []))
                self.detected_set = set(self.progress_data.get("detected_files", []))
                return True
        except Exception as exc:
            print(f"Failed to read progress file {self.progress_file}: {exc}")
            return False

    def save(self) -> None:
        """Persist progress JSON to disk."""
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.progress_data.setdefault("created_at", now_iso)
        self.progress_data["last_updated"] = now_iso

        try:
            os.makedirs(os.path.dirname(self.progress_file) or ".", exist_ok=True)
            with open(self.progress_file, "w", encoding="utf-8") as fp:
                json.dump(self.progress_data, fp, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"Failed to write progress file {self.progress_file}: {exc}")

    def set_params_hash(self, params_hash: str) -> None:
        """Set the parameters hash for this session."""
        self.progress_data["params_hash"] = params_hash

    def set_params(self, params: Any) -> None:
        """Record the user-specified parameters."""
        self.progress_data["params"] = params

    def set_roi(self, roi: Any) -> None:
        """Record the ROI selection (polygon or 'full_image')."""
        self.progress_data["roi"] = roi

    def set_processing_params(self, processing_params: Dict[str, Any]) -> None:
        """Record the final processing parameters used."""
        self.progress_data["processing_params"] = processing_params

    def get_params_hash(self) -> str:
        """Get the stored parameters hash."""
        return self.progress_data.get("params_hash", "")

    def is_processed(self, filename: str) -> bool:
        """Check if a file has been processed."""
        return filename in self.processed_set

    def record_result(self, filename: str, is_candidate: bool) -> None:
        """
        Record the result of processing a file.

        Args:
            filename: Name of the processed file
            is_candidate: Whether the file was detected as a candidate
        """
        self.processed_set.add(filename)
        if filename not in self.progress_data["processed_files"]:
            self.progress_data["processed_files"].append(filename)

        if is_candidate:
            self.detected_set.add(filename)
            if filename not in self.progress_data["detected_files"]:
                self.progress_data["detected_files"].append(filename)

        self.progress_data["total_processed"] = len(self.processed_set)
        self.progress_data["total_detected"] = len(self.detected_set)
        self.save()

    def filter_existing_files(self, existing_basenames: Set[str]) -> None:
        """
        Filter progress data to only include files that still exist.

        Args:
            existing_basenames: Set of filenames that currently exist
        """
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

        self.processed_set = set(self.progress_data["processed_files"])
        self.detected_set = set(self.progress_data["detected_files"])

        self.progress_data["total_processed"] = len(self.processed_set)
        self.progress_data["total_detected"] = len(self.detected_set)

    def get_total_processed(self) -> int:
        """Get the total number of processed files."""
        return self.progress_data.get("total_processed", 0)

    def get_total_detected(self) -> int:
        """Get the total number of detected candidates."""
        return self.progress_data.get("total_detected", 0)

    def reset(self) -> None:
        """Reset progress data."""
        self.progress_data = {
            "version": VERSION,
            "params_hash": self.progress_data.get("params_hash", ""),
            "params": self.progress_data.get("params", ""),
            "roi": self.progress_data.get("roi", "full_image"),
            "processing_params": self.progress_data.get("processing_params", {}),
            "processed_files": [],
            "detected_files": [],
            "total_processed": 0,
            "total_detected": 0,
        }
        self.processed_set = set()
        self.detected_set = set()


# Standalone functions for backward compatibility
def load_progress(progress_path: str) -> Optional[Dict]:
    """Load progress JSON if it exists."""
    if not os.path.exists(progress_path):
        return None

    try:
        with open(progress_path, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        print(f"Failed to read progress file {progress_path}: {exc}")
        return None


def save_progress(progress_path: str, progress_data: Dict) -> None:
    """Persist progress JSON to disk."""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    progress_data.setdefault("created_at", now_iso)
    progress_data["last_updated"] = now_iso

    try:
        os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
        with open(progress_path, "w", encoding="utf-8") as fp:
            json.dump(progress_data, fp, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"Failed to write progress file {progress_path}: {exc}")
