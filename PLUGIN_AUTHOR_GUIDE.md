# Plugin Author Guide

> ⚠️ **Experimental**: The plugin architecture is under active development and **may undergo breaking changes before the v2.0 stable release**.
>
> **Current status (v1.5.x)**:
> - ✅ Registry system and base classes are stable
> - ✅ Input Loaders, Detectors, Output Handlers work as documented
> - ⚠️ Method signatures may change (especially `detect` parameters)
> - ⚠️ Configuration coercion behavior may be refined

This guide provides comprehensive instructions for developing custom plugins for Detect Meteors CLI.

---

## Table of Contents

1. [Application Lifecycle](#1-application-lifecycle)
2. [Extension Points](#2-extension-points)
3. [Plugin Architecture](#3-plugin-architecture)
4. [Sample Code](#4-sample-code)
5. [Best Practices](#5-best-practices)
6. [Step-by-Step Tutorial](#6-step-by-step-tutorial)

---

## 1. Application Lifecycle

Understanding the detection pipeline lifecycle is essential for effective plugin development.

### 1.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Detection Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ 1. Initialize│───▶│ 2. Collect   │───▶│ 3. ROI Selection     │  │
│  │    Pipeline  │    │    Files     │    │    (if enabled)      │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                      │              │
│                                                      ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ 6. Finalize  │◀───│ 5. Save      │◀───│ 4. Process Batches   │  │
│  │    & Report  │    │    Results   │    │    (parallel/seq)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Processing Flow Detail

```
For each image pair (current, previous):

    ┌─────────────────────────────────────────────────────────────┐
    │                    Input Loader                             │
    │  • Load current image  ───▶  np.ndarray (uint16 grayscale)  │
    │  • Load previous image ───▶  np.ndarray (uint16 grayscale)  │
    │  • Extract metadata (optional)                              │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                      Detector                               │
    │  • Compute frame difference                                 │
    │  • Apply ROI mask                                           │
    │  • Detect meteor candidates                                 │
    │  • Generate debug visualization                             │
    │                                                             │
    │  Returns: (is_candidate, score, lines, aspect_ratio, debug) │
    └─────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   Output Handler                            │
    │  • save_candidate() ── Save detected meteor image           │
    │  • save_debug_image() ── Save debug visualization           │
    │                                                             │
    │  Lifecycle Hooks (called by pipeline):                      │
    │  • on_candidate_detected() ── After each detection          │
    │  • on_batch_complete() ── After each batch                  │
    │  • on_pipeline_complete() ── When pipeline finishes         │
    └─────────────────────────────────────────────────────────────┘
```

### 1.3 Lifecycle Events (Output Handler Only)

Only Output Handlers receive lifecycle events from the pipeline. These hooks enable real-time notifications, progress tracking, and post-processing.

| Event | When Called | Use Case |
|-------|-------------|----------|
| `on_candidate_detected` | After each meteor detection | Real-time notifications (Slack, webhook) |
| `on_batch_complete` | After each batch finishes | Progress reporting, metrics collection |
| `on_pipeline_complete` | When entire pipeline completes | Final summary, cleanup, reporting |

**Important**: Input Loaders and Detectors do **not** receive lifecycle events.

---

## 2. Extension Points

The plugin system provides three extension points:

### 2.1 Input Loaders

**Purpose**: Load images from various file formats

**When to create**:
- Support a new image format (TIFF, FITS, etc.)
- Apply pre-processing during load (debayer, normalize)
- Extract custom metadata

**Required methods**:
| Method | Signature | Description |
|--------|-----------|-------------|
| `load` | `(filepath: str) -> np.ndarray` | Load image as uint16 grayscale |

**Optional features**:
- Implement `BaseMetadataExtractor` for EXIF-like metadata extraction
- Define `name`, `version` attributes for plugin info

**BaseMetadataExtractor mixin**:

Implement this optional interface to extract metadata (EXIF, timestamps, camera info) from image files:

```python
class MyLoader(DataclassInputLoader[MyConfig], BaseMetadataExtractor):
    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        # Return metadata dictionary
        return {"timestamp": ..., "camera": ..., "exposure": ...}
```

**When is `extract_metadata` called?**
- The pipeline calls it automatically when metadata logging is enabled (`--log-metadata`)
- You can also call it manually for custom processing
- If not implemented, metadata extraction is silently skipped

### 2.2 Detectors

**Purpose**: Implement meteor detection algorithms

**When to create**:
- Use different detection approach (ML-based, morphological)
- Optimize for specific conditions (bright meteors, fireball detection)
- Add custom scoring logic

**Required methods**:
| Method | Signature | Description |
|--------|-----------|-------------|
| `detect` | `(current, previous, roi_mask, params) -> DetectionResult` | Main detection (see below) |
| `compute_line_score` | `(mask, hough_params) -> Tuple[float, List]` | Line scoring (called internally by `detect`) |

**DetectionResult type** (return value of `detect`):
```python
Tuple[
    bool,                              # is_candidate: Whether a meteor was detected
    float,                             # score: Detection confidence score
    List[Tuple[int, int, int, int]],   # lines: Line segments as (x1, y1, x2, y2)
    float,                             # aspect_ratio: Contour aspect ratio
    Optional[np.ndarray],              # debug_image: BGR visualization (or None)
]
```

**Detection parameters** (`params` argument):

The `params` dict contains CLI-configured detection settings:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `diff_threshold` | `int` | `8` | Frame difference threshold (0-255 scale) |
| `min_line_score` | `float` | `30.0` | Minimum score to classify as candidate |
| `min_aspect_ratio` | `float` | `2.0` | Minimum contour aspect ratio |
| `hough_threshold` | `int` | `50` | Hough transform vote threshold |
| `hough_min_line_length` | `int` | `50` | Minimum line length in pixels |
| `hough_max_line_gap` | `int` | `10` | Maximum gap between line segments |

**Note**: `compute_line_score` is a helper method typically called within your `detect` implementation. The pipeline calls only `detect`.

### 2.3 Output Handlers

**Purpose**: Save results and send notifications

**When to create**:
- Upload to cloud storage (S3, GCS)
- Send notifications (Slack, Discord, email)
- Store in database
- Generate custom reports

**Required methods**:
| Method | Signature | Description |
|--------|-----------|-------------|
| `save_candidate` | `(source_path, filename, ...) -> bool` | Save meteor candidate |
| `save_debug_image` | `(debug_image, filename, ...) -> str` | Save debug image |

**Lifecycle hooks (optional)**:
| Hook | Signature |
|------|-----------|
| `on_candidate_detected` | `(filename, saved, score, aspect_ratio) -> None` |
| `on_batch_complete` | `(processed_count, detected_count, batch_size) -> None` |
| `on_pipeline_complete` | `(total_processed, total_detected, elapsed_seconds) -> None` |

---

## 3. Plugin Architecture

### 3.1 Registry System

Each plugin type has its own registry:

```python
from meteor_core.inputs import LoaderRegistry
from meteor_core.detectors import DetectorRegistry
from meteor_core.outputs import OutputHandlerRegistry
```

**Registry Operations**:
```python
# Register a plugin class
LoaderRegistry.register(MyLoader)

# Get plugin class by name (case-insensitive)
loader_cls = LoaderRegistry.get("my_loader")
loader_cls = LoaderRegistry.get("MY_LOADER")  # Same result

# Create configured instance
loader = LoaderRegistry.create("my_loader", {"option": "value"})

# List available plugins
names = LoaderRegistry.list_available()  # ["raw", "my_loader", ...]

# Trigger discovery (automatic on first use)
LoaderRegistry.discover()
```

### 3.2 Base Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                      Input Loaders                              │
├─────────────────────────────────────────────────────────────────┤
│  BaseInputLoader (ABC)                                          │
│  ├── DataclassInputLoader[ConfigType] ── Dataclass config       │
│  │   └── RawImageLoader ── Built-in RAW loader                  │
│  └── PydanticInputLoader[ConfigType] ── Pydantic config         │
│                                                                 │
│  BaseMetadataExtractor (ABC) ── Optional mixin for metadata     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        Detectors                                │
├─────────────────────────────────────────────────────────────────┤
│  BaseDetector (ABC)                                             │
│  ├── DataclassDetector[ConfigType] ── Dataclass config          │
│  │   ├── HoughDetector ── Built-in Hough detector               │
│  │   └── SimpleThresholdDetector ── Built-in threshold detector │
│  └── PydanticDetector[ConfigType] ── Pydantic config            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Output Handlers                             │
├─────────────────────────────────────────────────────────────────┤
│  BaseOutputHandler (ABC) ── Includes lifecycle hooks            │
│  ├── DataclassOutputHandler[ConfigType] ── Dataclass config     │
│  │   └── FileOutputHandler ── Built-in file handler             │
│  └── PydanticOutputHandler[ConfigType] ── Pydantic config       │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Configuration Management (ConfigType)

Plugins can define a `ConfigType` for typed configuration. See [5.1 Choosing ConfigType](#51-choosing-configtype) for guidance on when to use dataclass vs Pydantic.

**Coercion Rules**:
| Input | ConfigType | Result |
|-------|------------|--------|
| `None` | Defined | `ConfigType()` with defaults |
| `None` | Not defined | `None` |
| ConfigType instance | — | Used as-is |
| `dict` | Dataclass | `ConfigType(**dict)` |
| `dict` | Pydantic v2 | `ConfigType.model_validate(dict)` |
| `dict` | Pydantic v1 | `ConfigType.parse_obj(dict)` |
| Other | — | Passed as-is |

**Error Handling**:
- `TypeError`: Missing required fields, wrong type
- `ValueError`: Validation failed (Pydantic)

### 3.4 Plugin Discovery

Plugins are discovered in this order (duplicates warn but don't overwrite):

1. **Built-in plugins** (RawImageLoader, HoughDetector, FileOutputHandler)
2. **Entry points** (sorted alphabetically by name)
3. **Plugin directories** (sorted alphabetically by filename)
4. **Runtime registrations** via `Registry.register()`

**Plugin Directories**:
| Plugin Type | Directory |
|-------------|-----------|
| Input Loaders | `~/.detect_meteors/input_plugins/` |
| Detectors | `~/.detect_meteors/detector_plugins/` |
| Output Handlers | `~/.detect_meteors/output_plugins/` |

**How plugin directory discovery works**:
1. Place your `.py` file in the appropriate directory (create it if needed)
2. Your plugin class **must** call `Registry.register()` at module level (see examples)
3. All `.py` files are loaded alphabetically on first registry access
4. No special naming convention required, but descriptive names help organization

Example file structure:
```
~/.detect_meteors/
└── input_plugins/
    ├── fits_loader.py      # Contains: LoaderRegistry.register(FitsLoader)
    └── tiff_loader.py      # Contains: LoaderRegistry.register(TiffLoader)
```

**Entry Points** (in `pyproject.toml`):
```toml
[project.entry-points."detect_meteors.input"]
my_loader = "my_package.loaders:MyLoader"

[project.entry-points."detect_meteors.detector"]
my_detector = "my_package.detectors:MyDetector"

[project.entry-points."detect_meteors.output"]
my_handler = "my_package.handlers:MyHandler"
```

---

## 4. Sample Code

### 4.1 Input Loader (Complete Example)

```python
"""Custom TIFF image loader with metadata extraction."""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from meteor_core.inputs import (
    DataclassInputLoader,
    BaseMetadataExtractor,
    LoaderRegistry,
)


@dataclass
class TiffLoaderConfig:
    """Configuration for TIFF loader."""
    normalize: bool = False
    bit_depth: int = 16


class TiffImageLoader(DataclassInputLoader[TiffLoaderConfig], BaseMetadataExtractor):
    """Load TIFF images with optional normalization."""

    plugin_name = "tiff"           # Required: unique identifier
    name = "TIFF Image Loader"     # Optional: human-readable name
    version = "1.0.0"              # Optional: version string
    ConfigType = TiffLoaderConfig  # Optional: configuration class

    def load(self, filepath: str) -> np.ndarray:
        """Load a TIFF image file.

        Args:
            filepath: Path to the TIFF file.

        Returns:
            Image as uint16 grayscale numpy array.
        """
        import tifffile

        image = tifffile.imread(filepath)

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)

        # Normalize if configured
        if self.config.normalize:
            max_val = 2 ** self.config.bit_depth - 1
            image = (image / max_val * 65535).astype(np.uint16)
        else:
            image = image.astype(np.uint16)

        return image

    def extract_metadata(self, filepath: str) -> Dict[str, Any]:
        """Extract metadata from TIFF file.

        Args:
            filepath: Path to the TIFF file.

        Returns:
            Dictionary with metadata.
        """
        import tifffile

        with tifffile.TiffFile(filepath) as tif:
            page = tif.pages[0]
            return {
                "width": page.shape[1] if len(page.shape) > 1 else page.shape[0],
                "height": page.shape[0],
                "dtype": str(page.dtype),
                "compression": page.compression.name if page.compression else None,
            }


# Register the plugin
LoaderRegistry.register(TiffImageLoader)
```

### 4.2 Detector (Complete Example)

```python
"""Simple threshold-based detector for bright meteors."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2

from meteor_core.detectors import DataclassDetector, DetectorRegistry


@dataclass
class ThresholdDetectorConfig:
    """Configuration for threshold detector."""
    brightness_multiplier: float = 1.5
    min_contour_area: int = 50


class ThresholdDetector(DataclassDetector[ThresholdDetectorConfig]):
    """Detect bright meteors using simple thresholding."""

    plugin_name = "threshold"
    name = "Threshold Detector"
    version = "1.0.0"
    ConfigType = ThresholdDetectorConfig

    def detect(
        self,
        current_image: np.ndarray,
        previous_image: np.ndarray,
        roi_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> Tuple[bool, float, List[Tuple[int, int, int, int]], float, Optional[np.ndarray]]:
        """Detect meteor candidates using threshold-based approach.

        Args:
            current_image: Current frame (uint16 grayscale).
            previous_image: Previous frame (uint16 grayscale).
            roi_mask: Binary ROI mask (uint8, 255=inside).
            params: Detection parameters from CLI.

        Returns:
            Tuple of (is_candidate, score, lines, aspect_ratio, debug_image).
        """
        # Compute absolute difference
        diff = cv2.absdiff(current_image, previous_image)

        # Apply ROI mask
        diff = cv2.bitwise_and(diff, diff, mask=roi_mask)

        # Get threshold from params
        threshold = params.get("diff_threshold", 8) * 256

        # Apply threshold
        _, binary = cv2.threshold(
            diff, 
            int(threshold * self.config.brightness_multiplier),
            255,
            cv2.THRESH_BINARY
        )
        binary = binary.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by area
        valid_contours = [
            c for c in contours 
            if cv2.contourArea(c) >= self.config.min_contour_area
        ]

        if not valid_contours:
            return False, 0.0, [], 0.0, None

        # Compute metrics
        max_area = max(cv2.contourArea(c) for c in valid_contours)
        score = float(max_area) / 100.0

        # Compute aspect ratios
        aspect_ratios = []
        for c in valid_contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratios.append(max(w, h) / max(min(w, h), 1))
        max_aspect_ratio = max(aspect_ratios) if aspect_ratios else 0.0

        # Create debug image
        debug_image = cv2.cvtColor(
            (current_image // 256).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        cv2.drawContours(debug_image, valid_contours, -1, (0, 255, 0), 2)

        # Simple line representation (bounding box diagonal)
        lines = []
        for c in valid_contours:
            x, y, w, h = cv2.boundingRect(c)
            lines.append((x, y, x + w, y + h))

        is_candidate = score >= params.get("min_line_score", 30.0)
        return is_candidate, score, lines, max_aspect_ratio, debug_image

    def compute_line_score(
        self,
        mask: np.ndarray,
        hough_params: Dict[str, int],
    ) -> Tuple[float, List[Tuple[int, int, int, int]]]:
        """Compute line score (simplified for threshold detector).

        Args:
            mask: Binary mask of detected regions.
            hough_params: Hough transform parameters (unused here).

        Returns:
            Tuple of (score, line_segments).
        """
        # Find contours and compute score
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0, []

        total_area = sum(cv2.contourArea(c) for c in contours)
        return float(total_area) / 100.0, []


# Register the plugin
DetectorRegistry.register(ThresholdDetector)
```

### 4.3 Output Handler with Lifecycle Hooks (Complete Example)

```python
"""Slack notification handler with full lifecycle support."""
from dataclasses import dataclass
from typing import List, Optional
import os
import shutil
import json
import urllib.request

import numpy as np
import cv2

from meteor_core.outputs import DataclassOutputHandler, OutputHandlerRegistry


@dataclass
class SlackOutputConfig:
    """Configuration for Slack notification handler."""
    output_folder: str = "./candidates"
    debug_folder: str = "./debug_masks"
    webhook_url: str = ""
    notify_on_detection: bool = True
    notify_on_complete: bool = True
    channel: str = "#meteor-alerts"


class SlackNotificationHandler(DataclassOutputHandler[SlackOutputConfig]):
    """Output handler with Slack notifications."""

    plugin_name = "slack"
    name = "Slack Notification Handler"
    version = "1.0.0"
    ConfigType = SlackOutputConfig

    def __init__(self, config: SlackOutputConfig):
        super().__init__(config)
        self._detection_count = 0
        os.makedirs(self.config.output_folder, exist_ok=True)
        if self.config.debug_folder:
            os.makedirs(self.config.debug_folder, exist_ok=True)

    def save_candidate(
        self,
        source_path: str,
        filename: str,
        debug_image: Optional[np.ndarray] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> bool:
        """Save a meteor candidate file.

        Args:
            source_path: Path to source RAW file.
            filename: Output filename.
            debug_image: Optional debug visualization.
            roi_polygon: Optional ROI polygon.

        Returns:
            True if saved, False if skipped.
        """
        dest_path = os.path.join(self.config.output_folder, filename)

        # Skip if exists
        if os.path.exists(dest_path):
            return False

        # Copy the file
        shutil.copy2(source_path, dest_path)

        # Save debug image if provided
        if debug_image is not None and self.config.debug_folder:
            debug_filename = os.path.splitext(filename)[0] + "_debug.png"
            self.save_debug_image(debug_image, debug_filename, roi_polygon)

        return True

    def save_debug_image(
        self,
        debug_image: np.ndarray,
        filename: str,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> str:
        """Save a debug visualization.

        Args:
            debug_image: Debug image (BGR).
            filename: Output filename.
            roi_polygon: Optional ROI polygon to draw.

        Returns:
            Path to saved debug image.
        """
        if not self.config.debug_folder:
            return ""

        # Draw ROI if provided
        if roi_polygon and len(roi_polygon) >= 3:
            pts = np.array(roi_polygon, dtype=np.int32)
            cv2.polylines(debug_image, [pts], True, (255, 0, 0), 2)

        path = os.path.join(self.config.debug_folder, filename)
        cv2.imwrite(path, debug_image)
        return path

    # ========== LIFECYCLE HOOKS ==========

    def on_candidate_detected(
        self,
        filename: str,
        saved: bool,
        score: float = 0.0,
        aspect_ratio: float = 0.0,
    ) -> None:
        """Called after each meteor detection.

        Use this for real-time notifications.

        Args:
            filename: Detected file name.
            saved: Whether file was saved (False if skipped).
            score: Detection score.
            aspect_ratio: Contour aspect ratio.
        """
        self._detection_count += 1

        if self.config.notify_on_detection and self.config.webhook_url and saved:
            self._send_slack(
                f"🌠 *Meteor Detected!*\n"
                f"• File: `{filename}`\n"
                f"• Score: {score:.1f}\n"
                f"• Aspect Ratio: {aspect_ratio:.2f}"
            )

    def on_batch_complete(
        self,
        processed_count: int,
        detected_count: int,
        batch_size: int,
    ) -> None:
        """Called after each batch completes.

        Use for progress tracking.

        Args:
            processed_count: Total processed so far.
            detected_count: Total detected so far.
            batch_size: Files in this batch.
        """
        # Could implement periodic progress updates here
        pass

    def on_pipeline_complete(
        self,
        total_processed: int,
        total_detected: int,
        elapsed_seconds: float,
    ) -> None:
        """Called when pipeline finishes.

        Use for final summary notifications.

        Args:
            total_processed: Total files processed.
            total_detected: Total candidates detected.
            elapsed_seconds: Total time in seconds.
        """
        if self.config.notify_on_complete and self.config.webhook_url:
            minutes = elapsed_seconds / 60
            rate = total_processed / elapsed_seconds if elapsed_seconds > 0 else 0
            self._send_slack(
                f"✅ *Detection Complete*\n"
                f"• Processed: {total_processed} images\n"
                f"• Detected: {total_detected} candidates\n"
                f"• Time: {minutes:.1f} minutes\n"
                f"• Rate: {rate:.2f} images/sec"
            )

    def _send_slack(self, message: str) -> None:
        """Send a message to Slack webhook.

        Args:
            message: Message text (supports Slack markdown).
        """
        if not self.config.webhook_url:
            return

        payload = {
            "channel": self.config.channel,
            "text": message,
            "mrkdwn": True,
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.config.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            # Don't fail pipeline on notification errors
            pass


# Register the plugin
OutputHandlerRegistry.register(SlackNotificationHandler)
```

---

## 5. Best Practices

### 5.1 Choosing ConfigType

Use this decision tree to select the right configuration approach:

```
Need configuration?
        │
       No ──────────▶ Don't define ConfigType
        │              (accept None in __init__)
       Yes
        │
        ▼
Need validation?
(ranges, patterns,
 custom rules)
        │
       No ──────────▶ Use @dataclass
        │              (simple, built-in, no deps)
       Yes
        │
        ▼
Use Pydantic BaseModel
(rich validation, type coercion)
```

**Dataclass (Recommended for most cases)**:
```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    output_folder: str = "./output"  # Always provide defaults
    threshold: float = 0.5
    enabled: bool = True
```

**Pydantic (For complex validation)**:
```python
from pydantic import BaseModel, Field, field_validator

class MyConfig(BaseModel):
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    url: str = ""

    @field_validator("url")
    @classmethod
    def validate_url(cls, v):
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    model_config = {"extra": "forbid"}  # Reject unknown keys
```

### 5.2 Required Attributes

| Attribute | Required | Type | Description |
|-----------|----------|------|-------------|
| `plugin_name` | ✅ Yes | `str` | Unique identifier (case-insensitive) |
| `name` | ❌ No | `str` | Human-readable name |
| `version` | ❌ No | `str` | Version string |
| `ConfigType` | ❌ No | `type` | Configuration class |

### 5.3 Error Handling Best Practices

**In load/detect methods**:
```python
def load(self, filepath: str) -> np.ndarray:
    try:
        # Your loading logic
        return image
    except FileNotFoundError:
        raise  # Let pipeline handle missing files
    except Exception as e:
        # Log and re-raise with context
        print(f"Error loading {filepath}: {e}")
        raise
```

**In lifecycle hooks** (don't fail the pipeline):
```python
def on_candidate_detected(self, filename, saved, score, aspect_ratio):
    try:
        self._send_notification(filename)
    except Exception:
        # Log but don't raise - pipeline should continue
        pass
```

### 5.4 Performance Considerations

**Input Loaders**:
- Return uint16 grayscale arrays for consistency
- Avoid unnecessary copies (`image.astype()` creates a copy)
- Consider memory-mapped files for very large images

**Detectors**:
- Use NumPy vectorized operations over Python loops
- Pre-allocate arrays when possible
- Consider using OpenCV's optimized functions

**Output Handlers**:
- Make lifecycle hooks non-blocking (use timeouts)
- Buffer notifications for batch sending if needed
- Use async I/O for network operations (advanced)

### 5.5 Thread Safety

The pipeline may process images in parallel. Ensure your plugins are thread-safe:

```python
class MyHandler(DataclassOutputHandler[MyConfig]):
    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.Lock()
        self._count = 0

    def on_candidate_detected(self, filename, saved, score, aspect_ratio):
        with self._lock:
            self._count += 1
```

---

## 6. Step-by-Step Tutorial

### 6.1 Step-by-Step: Creating a Plugin

#### Step 1: Choose Plugin Type and Base Class

```python
# For input loaders with dataclass config
from meteor_core.inputs import DataclassInputLoader, BaseMetadataExtractor

# For detectors with dataclass config
from meteor_core.detectors import DataclassDetector

# For output handlers with dataclass config
from meteor_core.outputs import DataclassOutputHandler
```

#### Step 2: Define Configuration (Optional)

```python
from dataclasses import dataclass

@dataclass
class MyPluginConfig:
    option1: str = "default"
    option2: int = 10
```

#### Step 3: Implement the Plugin Class

```python
class MyPlugin(DataclassInputLoader[MyPluginConfig]):
    plugin_name = "my_plugin"  # Required
    name = "My Plugin"         # Optional
    version = "1.0.0"          # Optional
    ConfigType = MyPluginConfig

    def load(self, filepath: str) -> np.ndarray:
        # Implementation
        pass
```

#### Step 4: Register the Plugin

**Option A: Runtime registration**
```python
from meteor_core.inputs import LoaderRegistry
LoaderRegistry.register(MyPlugin)
```

**Option B: Entry point (for packages)**
```toml
# pyproject.toml
[project.entry-points."detect_meteors.input"]
my_plugin = "my_package:MyPlugin"
```

**Option C: Plugin directory (for user plugins)**
```bash
# Save as ~/.detect_meteors/input_plugins/my_plugin.py
```

### 6.2 Testing Your Plugin

#### Unit Test Example

```python
import unittest
import numpy as np
from my_plugin import MyPlugin, MyPluginConfig


class TestMyPlugin(unittest.TestCase):
    def test_load_returns_correct_shape(self):
        config = MyPluginConfig(option1="test")
        plugin = MyPlugin(config)

        image = plugin.load("test_image.tiff")

        self.assertEqual(len(image.shape), 2)  # Grayscale
        self.assertEqual(image.dtype, np.uint16)

    def test_default_config(self):
        # Test with default configuration
        from meteor_core.inputs import LoaderRegistry
        LoaderRegistry.register(MyPlugin)

        loader = LoaderRegistry.create("my_plugin")  # Uses defaults
        self.assertIsNotNone(loader)


if __name__ == "__main__":
    unittest.main()
```

#### Integration Test

```python
def test_plugin_in_pipeline():
    from meteor_core.inputs import LoaderRegistry
    from my_plugin import MyPlugin

    LoaderRegistry.register(MyPlugin)

    # Verify registration
    assert "my_plugin" in LoaderRegistry.list_available()

    # Create instance
    loader = LoaderRegistry.create("my_plugin", {"option1": "test"})
    assert loader.config.option1 == "test"
```

### 6.3 Debugging Tips

**Enable verbose logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check plugin info**:
```python
from meteor_core.inputs import LoaderRegistry

LoaderRegistry.discover()
for name in LoaderRegistry.list_available():
    cls = LoaderRegistry.get(name)
    instance = cls(None)
    print(instance.get_info())
```

**Verify configuration coercion**:
```python
# Test that dict config works
handler = OutputHandlerRegistry.create("my_handler", {"option": "value"})

# Test that ConfigType instance works
config = MyConfig(option="value")
handler = OutputHandlerRegistry.create("my_handler", config)

# Test default config
handler = OutputHandlerRegistry.create("my_handler")  # Uses ConfigType()
```

---

## See Also

**Documentation**:
- [INSTALL_DEV.md](INSTALL_DEV.md) — Development environment setup
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [README.md](README.md) — User documentation

**Built-in plugin implementations** (reference code):
- [`meteor_core/inputs/raw_image_loader.py`](meteor_core/inputs/raw_image_loader.py) — RawImageLoader
- [`meteor_core/detectors/hough_detector.py`](meteor_core/detectors/hough_detector.py) — HoughDetector
- [`meteor_core/outputs/file_output_handler.py`](meteor_core/outputs/file_output_handler.py) — FileOutputHandler
