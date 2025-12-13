#!/usr/bin/env python
#
# Detect Meteors CLI - Outputs Package
# © 2025 Shinichi Morita (shin3tky)
#

"""
Output handling for meteor detection.

This package provides output handlers for saving detection results,
including candidate files, debug images, and progress tracking.

Recommended usage::

    from meteor_core.outputs import OutputHandlerRegistry

    # Create a handler using the registry
    handler = OutputHandlerRegistry.create("file", {
        "output_folder": "./candidates",
        "debug_folder": "./debug",
    })

    # Save a candidate
    saved = handler.save_candidate("/path/to/source.CR2", "source.CR2")

Available handlers:
    - "file": FileOutputHandler - Default file-based output (copies files to disk)

For progress tracking::

    from meteor_core.outputs import ProgressManager

    manager = ProgressManager("progress.json")
    manager.load()
    manager.record_result("image001.CR2", is_candidate=True, score=150.0)
    manager.save()
"""

# Base classes
from .base import (
    BaseOutputHandler,
    DataclassOutputHandler,
    _is_valid_output_handler,
)

# File handler (default implementation)
from .file_handler import (
    FileOutputConfig,
    FileOutputHandler,
    create_file_handler,
)

# Registry (recommended)
from .registry import OutputHandlerRegistry

# Discovery
from .discovery import PLUGIN_DIR, PLUGIN_GROUP

# Deprecated: use OutputHandlerRegistry.discover() instead
from .discovery import discover_handlers

# Progress tracking
from .progress import (
    ProgressManager,
    load_progress,
    save_progress,
)

# Backward compatibility (deprecated)
from .writer import OutputWriter

__all__ = [
    # Base classes
    "BaseOutputHandler",
    "DataclassOutputHandler",
    "_is_valid_output_handler",
    # File handler (default implementation)
    "FileOutputConfig",
    "FileOutputHandler",
    "create_file_handler",
    # Registry (recommended)
    "OutputHandlerRegistry",
    # Discovery constants
    "PLUGIN_DIR",
    "PLUGIN_GROUP",
    # Discovery function (deprecated)
    "discover_handlers",
    # Progress tracking
    "ProgressManager",
    "load_progress",
    "save_progress",
    # Backward compatibility (deprecated)
    "OutputWriter",
]
