#!/usr/bin/env python
#
# Detect Meteors CLI - Custom Exceptions
# © 2025 Shinichi Morita (shin3tky)
#

"""
Custom exception classes for meteor detection.

This module provides a hierarchy of exceptions for structured error handling
throughout the meteor_core package. Each exception includes diagnostic
information that can be used for troubleshooting and bug reporting.

Exception Hierarchy:
    MeteorError (base)
    ├── MeteorLoadError (image loading failures)
    │   └── MeteorUnsupportedFormatError (unsupported file formats)
    ├── MeteorOutputError (output operation failures)
    │   ├── MeteorWriteError (file write failures)
    │   └── MeteorProgressError (progress tracking errors)
    ├── MeteorValidationError (parameter/input validation)
    └── MeteorConfigError (configuration errors)

Example:
    >>> try:
    ...     image = load_and_bin_raw_fast("corrupted.CR2")
    ... except MeteorLoadError as e:
    ...     print(e.get_diagnostic_info())
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import datetime, timezone
from importlib import resources
from string import Template
from typing import Any, Dict, List, Optional

from .i18n import DEFAULT_LOCALE, get_message
from .schema import VERSION

# Key dependencies to include in diagnostic reports
_KEY_DEPENDENCIES = [
    "numpy",
    "opencv-python",
    "rawpy",
    "pillow",
    "psutil",
    "pydantic",
]

_ISSUE_URL = "https://github.com/shin3tky/detect_meteors/issues/new"


def _get_package_versions() -> Dict[str, str]:
    """Collect versions of key dependencies.

    Returns:
        Dictionary mapping package names to version strings.
        Returns "not installed" for missing packages.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: Dict[str, str] = {}
    for pkg in _KEY_DEPENDENCIES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not installed"
    return versions


@lru_cache(maxsize=1)
def _load_diagnostic_template() -> Template:
    """Load the diagnostic report template."""
    template_path = resources.files(__package__).joinpath(
        "templates", "diagnostic_report.md.j2"
    )
    content = template_path.read_text(encoding="utf-8")
    return Template(content)


def _format_section(title: str, lines: List[str]) -> str:
    """Format a section with a title and a code block."""
    if not lines:
        return ""

    section_lines = [title, "", "```"]
    section_lines.extend(lines)
    section_lines.extend(["```", ""])
    return "\n".join(section_lines)


def _build_report_header(locale: str) -> str:
    """Build the report header text from localized messages."""
    header_lines = [
        get_message("ui.diagnostic.report.title", locale=locale),
        "",
        get_message("ui.diagnostic.report.description", locale=locale),
        get_message("ui.diagnostic.report.instructions", locale=locale),
        get_message(
            "ui.diagnostic.report.issue_link",
            locale=locale,
            issue_url=_ISSUE_URL,
        ),
        "",
        "---",
        "",
    ]
    return "\n".join(header_lines)


def _render_diagnostic_report(
    diagnostic: "DiagnosticInfo", *, locale: str, include_header: bool
) -> str:
    """Render a diagnostic report using the template and localized messages."""
    label_version = get_message("ui.diagnostic.label.version", locale=locale)
    label_python = get_message("ui.diagnostic.label.python_version", locale=locale)
    label_platform = get_message("ui.diagnostic.label.platform", locale=locale)
    label_timestamp = get_message("ui.diagnostic.label.timestamp", locale=locale)

    general_section = _format_section(
        get_message("ui.diagnostic.section.heading", locale=locale),
        [
            f"{label_version}: {diagnostic.version}",
            f"{label_python}: {diagnostic.python_version}",
            f"{label_platform}: {diagnostic.platform}",
            f"{label_timestamp}: {diagnostic.timestamp}",
        ],
    )

    file_section = ""
    if diagnostic.filepath:
        label_file = get_message("ui.diagnostic.label.filepath", locale=locale)
        label_exists = get_message("ui.diagnostic.label.file_exists", locale=locale)
        file_lines = [
            f"{label_file}: {diagnostic.filepath}",
            f"{label_exists}: {diagnostic.file_exists}",
        ]
        if diagnostic.file_size is not None:
            label_size = get_message("ui.diagnostic.label.file_size", locale=locale)
            bytes_label = get_message("ui.diagnostic.label.bytes", locale=locale)
            file_lines.append(f"{label_size}: {diagnostic.file_size:,} {bytes_label}")

        file_section = _format_section(
            get_message("ui.diagnostic.section.file", locale=locale),
            file_lines,
        )

    label_error_type = get_message("ui.diagnostic.label.error_type", locale=locale)
    label_error_message = get_message(
        "ui.diagnostic.label.error_message", locale=locale
    )
    error_lines = [
        f"{label_error_type}: {diagnostic.error_type}",
        f"{label_error_message}: {diagnostic.error_message}",
    ]

    if diagnostic.original_error_type:
        label_original_type = get_message(
            "ui.diagnostic.label.original_error_type", locale=locale
        )
        error_lines.append(f"{label_original_type}: {diagnostic.original_error_type}")
    if diagnostic.original_error_message:
        label_original_message = get_message(
            "ui.diagnostic.label.original_error_message", locale=locale
        )
        error_lines.append(
            f"{label_original_message}: {diagnostic.original_error_message}"
        )

    error_section = _format_section(
        get_message("ui.diagnostic.section.error", locale=locale),
        error_lines,
    )

    context_section = ""
    if diagnostic.context:
        context_lines = [f"{key}: {value}" for key, value in diagnostic.context.items()]
        context_section = _format_section(
            get_message("ui.diagnostic.section.context", locale=locale),
            context_lines,
        )

    dependencies_section = ""
    if diagnostic.dependencies:
        dependency_lines = [
            f"{pkg}: {ver}" for pkg, ver in sorted(diagnostic.dependencies.items())
        ]
        dependencies_section = _format_section(
            get_message("ui.diagnostic.section.dependencies", locale=locale),
            dependency_lines,
        )

    report_header = _build_report_header(locale) if include_header else ""

    template = _load_diagnostic_template()
    return template.safe_substitute(
        report_header=report_header,
        diagnostic_section=general_section,
        file_section=file_section,
        error_section=error_section,
        context_section=context_section,
        dependencies_section=dependencies_section,
    )


@dataclass
class DiagnosticInfo:
    """Structured diagnostic information for error reporting.

    This dataclass collects system and context information that helps
    diagnose issues when filing bug reports or troubleshooting.

    Attributes:
        version: meteor_core version string.
        python_version: Python interpreter version.
        platform: Operating system and architecture.
        timestamp: ISO format timestamp when the error occurred.
        filepath: Path to the file that caused the error (if applicable).
        file_exists: Whether the file exists at the given path.
        file_size: Size of the file in bytes (if exists).
        error_type: Name of the exception class.
        error_message: The error message.
        original_error_type: Type of the wrapped original exception.
        original_error_message: Message from the wrapped original exception.
        context: Additional context-specific information.
    """

    version: str = ""
    python_version: str = ""
    platform: str = ""
    timestamp: str = ""
    filepath: Optional[str] = None
    file_exists: Optional[bool] = None
    file_size: Optional[int] = None
    error_type: str = ""
    error_message: str = ""
    original_error_type: Optional[str] = None
    original_error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of diagnostic info.
        """
        return {
            "version": self.version,
            "python_version": self.python_version,
            "platform": self.platform,
            "timestamp": self.timestamp,
            "filepath": self.filepath,
            "file_exists": self.file_exists,
            "file_size": self.file_size,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "original_error_type": self.original_error_type,
            "original_error_message": self.original_error_message,
            "context": self.context,
            "dependencies": self.dependencies,
        }

    def format_for_issue(
        self, *, locale: str = DEFAULT_LOCALE, include_header: bool = False
    ) -> str:
        """Format diagnostic info using the report template.

        Args:
            locale: Locale code for localized messages.
            include_header: Whether to include the report header section.

        Returns:
            Markdown-formatted string suitable for pasting into GitHub issues.
        """
        return _render_diagnostic_report(
            self, locale=locale, include_header=include_header
        )


def _collect_file_info(filepath: Optional[str]) -> tuple[Optional[bool], Optional[int]]:
    """Collect file existence and size information.

    Args:
        filepath: Path to the file to check.

    Returns:
        Tuple of (file_exists, file_size). file_size is None if file doesn't exist.
    """
    if filepath is None:
        return None, None

    try:
        exists = os.path.exists(filepath)
        if exists:
            size = os.path.getsize(filepath)
            return exists, size
        return exists, None
    except OSError:
        return None, None


class MeteorError(Exception):
    """Base exception for all meteor_core errors.

    This is the root of the exception hierarchy for meteor_core.
    All custom exceptions in this package inherit from this class.

    Attributes:
        message: Human-readable error message.
        filepath: Path to the related file (if applicable).
        original_error: The original exception that was caught (if wrapping).
        context: Additional context information as key-value pairs.

    Example:
        >>> raise MeteorError("Something went wrong", context={"step": "detection"})
    """

    def __init__(
        self,
        message: str,
        *,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MeteorError.

        Args:
            message: Human-readable error message.
            filepath: Path to the related file (if applicable).
            original_error: The original exception being wrapped.
            context: Additional context information.
        """
        self.message = message
        self.filepath = filepath
        self.original_error = original_error
        self.context = context or {}

        # Build the full message
        full_message = message
        if filepath:
            full_message = f"{message} (file: {filepath})"
        if original_error:
            full_message = (
                f"{full_message}: {type(original_error).__name__}: {original_error}"
            )

        super().__init__(full_message)

    def get_diagnostic_info(self) -> DiagnosticInfo:
        """Generate diagnostic information for this error.

        Returns:
            DiagnosticInfo instance with system and error details.
        """
        file_exists, file_size = _collect_file_info(self.filepath)

        original_type = None
        original_message = None
        if self.original_error:
            original_type = type(self.original_error).__name__
            original_message = str(self.original_error)

        return DiagnosticInfo(
            version=VERSION,
            python_version=sys.version,
            platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            filepath=self.filepath,
            file_exists=file_exists,
            file_size=file_size,
            error_type=type(self).__name__,
            error_message=self.message,
            original_error_type=original_type,
            original_error_message=original_message,
            context=self.context,
            dependencies=_get_package_versions(),
        )

    def format_for_issue(
        self, *, locale: str = DEFAULT_LOCALE, include_header: bool = False
    ) -> str:
        """Format this error as a GitHub issue-ready report.

        Args:
            locale: Locale code for localized messages.
            include_header: Whether to include the report header section.

        Returns:
            Markdown-formatted string suitable for pasting into GitHub issues.
        """
        return self.get_diagnostic_info().format_for_issue(
            locale=locale, include_header=include_header
        )


class MeteorLoadError(MeteorError):
    """Exception raised when loading an image file fails.

    This exception is raised when rawpy or other image loading
    operations fail due to file corruption, I/O errors, or
    other loading-related issues.

    Example:
        >>> try:
        ...     with rawpy.imread(filepath) as raw:
        ...         data = raw.raw_image
        ... except Exception as e:
        ...     raise MeteorLoadError(
        ...         "Failed to load RAW file",
        ...         filepath=filepath,
        ...         original_error=e,
        ...     )
    """

    def __init__(
        self,
        message: str = "Failed to load image file",
        *,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MeteorLoadError.

        Args:
            message: Human-readable error message.
            filepath: Path to the file that failed to load.
            original_error: The original exception from the loading library.
            context: Additional context information.
        """
        super().__init__(
            message,
            filepath=filepath,
            original_error=original_error,
            context=context,
        )


class MeteorUnsupportedFormatError(MeteorLoadError):
    """Exception raised when a file format is not supported.

    This is a subclass of MeteorLoadError specifically for cases
    where the file format is recognized but not supported, or
    when the file extension doesn't match a known RAW format.

    Attributes:
        detected_format: The detected file format (if available).
        supported_formats: List of supported formats.

    Example:
        >>> raise MeteorUnsupportedFormatError(
        ...     "JPEG files are not supported",
        ...     filepath="image.jpg",
        ...     detected_format="JPEG",
        ... )
    """

    def __init__(
        self,
        message: str = "Unsupported file format",
        *,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        detected_format: Optional[str] = None,
        supported_formats: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MeteorUnsupportedFormatError.

        Args:
            message: Human-readable error message.
            filepath: Path to the unsupported file.
            original_error: The original exception (if any).
            detected_format: The detected file format.
            supported_formats: List of formats that are supported.
            context: Additional context information.
        """
        self.detected_format = detected_format
        self.supported_formats = supported_formats

        # Add format info to context
        ctx = context.copy() if context else {}
        if detected_format:
            ctx["detected_format"] = detected_format
        if supported_formats:
            ctx["supported_formats"] = ", ".join(supported_formats)

        super().__init__(
            message,
            filepath=filepath,
            original_error=original_error,
            context=ctx,
        )


class MeteorValidationError(MeteorError):
    """Exception raised when validation of parameters or inputs fails.

    This exception is used for invalid detection parameters,
    ROI specifications, or other user-provided values that
    don't meet the required constraints.

    Attributes:
        parameter_name: Name of the invalid parameter.
        provided_value: The value that was provided.
        expected: Description of what was expected.

    Example:
        >>> raise MeteorValidationError(
        ...     "Invalid diff_threshold value",
        ...     parameter_name="diff_threshold",
        ...     provided_value=-5,
        ...     expected="positive integer",
        ... )
    """

    def __init__(
        self,
        message: str = "Validation failed",
        *,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        parameter_name: Optional[str] = None,
        provided_value: Any = None,
        expected: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MeteorValidationError.

        Args:
            message: Human-readable error message.
            filepath: Path to related file (if applicable).
            original_error: The original exception (if any).
            parameter_name: Name of the invalid parameter.
            provided_value: The value that was provided.
            expected: Description of what was expected.
            context: Additional context information.
        """
        self.parameter_name = parameter_name
        self.provided_value = provided_value
        self.expected = expected

        # Add validation info to context
        ctx = context.copy() if context else {}
        if parameter_name:
            ctx["parameter_name"] = parameter_name
        if provided_value is not None:
            ctx["provided_value"] = repr(provided_value)
        if expected:
            ctx["expected"] = expected

        super().__init__(
            message,
            filepath=filepath,
            original_error=original_error,
            context=ctx,
        )


class MeteorHookError(MeteorError):
    """Exception raised when a pipeline hook fails.

    This exception wraps unexpected errors from hook callbacks so callers can
    distinguish hook failures from other pipeline errors.
    """

    def __init__(
        self,
        message: str = "Hook execution failed",
        *,
        hook_name: Optional[str] = None,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        ctx = context.copy() if context else {}
        if hook_name:
            ctx["hook_name"] = hook_name
        super().__init__(
            message,
            filepath=filepath,
            original_error=original_error,
            context=ctx,
        )


class MeteorConfigError(MeteorError):
    """Exception raised when configuration is invalid.

    This exception is used for plugin configuration errors,
    pipeline configuration issues, or other setup-related problems.

    Attributes:
        config_key: The configuration key that has an issue.
        plugin_name: Name of the plugin (if plugin-related).

    Example:
        >>> raise MeteorConfigError(
        ...     "Invalid detector configuration",
        ...     config_key="threshold",
        ...     plugin_name="hough",
        ... )
    """

    def __init__(
        self,
        message: str = "Configuration error",
        *,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        config_key: Optional[str] = None,
        plugin_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MeteorConfigError.

        Args:
            message: Human-readable error message.
            filepath: Path to configuration file (if applicable).
            original_error: The original exception (if any).
            config_key: The configuration key with the issue.
            plugin_name: Name of the related plugin.
            context: Additional context information.
        """
        self.config_key = config_key
        self.plugin_name = plugin_name

        # Add config info to context
        ctx = context.copy() if context else {}
        if config_key:
            ctx["config_key"] = config_key
        if plugin_name:
            ctx["plugin_name"] = plugin_name

        super().__init__(
            message,
            filepath=filepath,
            original_error=original_error,
            context=ctx,
        )


class MeteorOutputError(MeteorError):
    """Base exception for output operation failures.

    This exception serves as the base for all output-related errors,
    including file saving, directory creation, and progress tracking failures.

    Attributes:
        destination_path: Path where the output was attempted (if applicable).
        operation: Type of operation being performed (e.g., "save", "copy").

    Example:
        >>> raise MeteorOutputError(
        ...     "Output operation failed",
        ...     destination_path="./candidates/image.CR2",
        ...     operation="copy",
        ... )
    """

    def __init__(
        self,
        message: str = "Output operation failed",
        *,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        destination_path: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MeteorOutputError.

        Args:
            message: Human-readable error message.
            filepath: Source file path (if applicable).
            original_error: The original exception (if any).
            destination_path: Destination path for the output.
            operation: Type of operation being performed.
            context: Additional context information.
        """
        self.destination_path = destination_path
        self.operation = operation

        # Add output info to context
        ctx = context.copy() if context else {}
        if destination_path:
            ctx["destination_path"] = destination_path
        if operation:
            ctx["operation"] = operation

        super().__init__(
            message,
            filepath=filepath,
            original_error=original_error,
            context=ctx,
        )


class MeteorWriteError(MeteorOutputError):
    """Exception raised when writing files fails.

    This exception is used for file copy failures, debug image saving errors,
    directory creation failures, and other write-related issues.

    Common error categories (stored in context["error_category"]):
        - "copy_failed": Failed to copy candidate file
        - "image_write_failed": Failed to save debug image
        - "directory_creation_failed": Failed to create output directory
        - "permission_denied": Write permission denied
        - "disk_full": Insufficient disk space

    Example:
        >>> raise MeteorWriteError(
        ...     "Failed to copy candidate file",
        ...     filepath="/source/image.CR2",
        ...     destination_path="/output/image.CR2",
        ...     operation="copy",
        ...     context={"error_category": "copy_failed"},
        ... )
    """

    def __init__(
        self,
        message: str = "Failed to write file",
        *,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        destination_path: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MeteorWriteError.

        Args:
            message: Human-readable error message.
            filepath: Source file path.
            original_error: The original exception from the I/O operation.
            destination_path: Destination path where write was attempted.
            operation: Type of write operation (e.g., "copy", "save_image").
            context: Additional context information.
        """
        super().__init__(
            message,
            filepath=filepath,
            original_error=original_error,
            destination_path=destination_path,
            operation=operation,
            context=context,
        )


class MeteorProgressError(MeteorOutputError):
    """Exception raised when progress tracking fails.

    This exception is used for progress file read/write errors,
    JSON parsing failures, serialization errors, and progress data
    validation errors.

    Common error categories (stored in context["error_category"]):
        - "read_failed": Failed to read progress file
        - "write_failed": Failed to write progress file
        - "parse_failed": Failed to parse progress JSON
        - "serialize_failed": Failed to serialize progress JSON
        - "validation_failed": Progress data validation error

    Attributes:
        progress_file: Path to the progress file (stored in `filepath`).

    Example:
        >>> raise MeteorProgressError(
        ...     "Failed to parse progress file",
        ...     filepath="progress.json",
        ...     context={"error_category": "parse_failed"},
        ... )
    """

    def __init__(
        self,
        message: str = "Progress tracking error",
        *,
        filepath: Optional[str] = None,
        original_error: Optional[BaseException] = None,
        destination_path: Optional[str] = None,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize MeteorProgressError.

        Args:
            message: Human-readable error message.
            filepath: Path to the progress file.
            original_error: The original exception (if any).
            destination_path: Not typically used for progress errors.
            operation: Type of operation (e.g., "load", "save", "parse", "serialize").
            context: Additional context information.
        """
        super().__init__(
            message,
            filepath=filepath,
            original_error=original_error,
            destination_path=destination_path,
            operation=operation,
            context=context,
        )


# =============================================================================
# CLI Helper Functions
# =============================================================================


def format_error_for_user(
    error: MeteorError, *, verbose: bool = False, locale: str = DEFAULT_LOCALE
) -> str:
    """Format an error message for CLI display.

    Provides a user-friendly error message with optional verbose diagnostics.

    Args:
        error: The MeteorError to format.
        verbose: If True, include full diagnostic information.
        locale: Locale code for message localization.

    Returns:
        Formatted error message string.

    Example:
        >>> try:
        ...     load_and_bin_raw_fast("bad.CR2")
        ... except MeteorError as e:
        ...     print(format_error_for_user(e, verbose=True))
    """
    # NOTE: ui.error/ui.diagnostic strings are intentionally English-only across locales.
    lines: List[str] = [
        "",
        "=" * 60,
        get_message("ui.error.header", locale=locale, message=error.message),
        "=" * 60,
    ]

    if error.filepath:
        lines.append(
            get_message("ui.error.filepath", locale=locale, filepath=error.filepath)
        )

    if error.original_error:
        lines.append(
            get_message(
                "ui.error.cause",
                locale=locale,
                error_type=type(error.original_error).__name__,
                error_message=error.original_error,
            )
        )

    if verbose:
        lines.extend(
            [
                "",
                get_message("ui.diagnostic.info", locale=locale),
                "-" * 60,
                error.format_for_issue(locale=locale),
            ]
        )
    else:
        lines.extend(
            [
                "",
                get_message("ui.diagnostic.hint.verbose", locale=locale),
                get_message("ui.diagnostic.hint.save_option", locale=locale),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def save_diagnostic_report(
    error: MeteorError,
    output_path: Optional[str] = None,
    *,
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Save diagnostic information to a file.

    Creates a markdown file with full diagnostic information that can be
    attached to GitHub issues.

    Args:
        error: The MeteorError to generate diagnostics for.
        output_path: Path for the output file. If None, generates a
            timestamped filename in the current directory.
        locale: Locale code for localized messages.

    Returns:
        Path to the saved diagnostic file.

    Example:
        >>> try:
        ...     load_and_bin_raw_fast("bad.CR2")
        ... except MeteorError as e:
        ...     path = save_diagnostic_report(e)
        ...     print(f"Diagnostic saved to: {path}")
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"meteor_diagnostic_{timestamp}.md"

    diagnostic = error.get_diagnostic_info()
    full_content = diagnostic.format_for_issue(
        locale=locale,
        include_header=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    return output_path


def create_diagnostic_from_exception(
    exc: BaseException,
    *,
    filepath: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> DiagnosticInfo:
    """Create diagnostic info from any exception.

    This is useful for wrapping non-MeteorError exceptions with
    diagnostic information.

    Args:
        exc: Any exception to create diagnostics for.
        filepath: Optional filepath related to the error.
        context: Optional additional context.

    Returns:
        DiagnosticInfo instance with system and error details.

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     diag = create_diagnostic_from_exception(e, filepath="data.txt")
        ...     print(diag.format_for_issue())
    """
    file_exists, file_size = _collect_file_info(filepath)

    return DiagnosticInfo(
        version=VERSION,
        python_version=sys.version,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        filepath=filepath,
        file_exists=file_exists,
        file_size=file_size,
        error_type=type(exc).__name__,
        error_message=str(exc),
        original_error_type=None,
        original_error_message=None,
        context=context or {},
        dependencies=_get_package_versions(),
    )


__all__ = [
    # Diagnostic info
    "DiagnosticInfo",
    # Exception classes
    "MeteorError",
    "MeteorLoadError",
    "MeteorUnsupportedFormatError",
    "MeteorValidationError",
    "MeteorConfigError",
    "MeteorOutputError",
    "MeteorWriteError",
    "MeteorProgressError",
    # Helper functions
    "format_error_for_user",
    "save_diagnostic_report",
    "create_diagnostic_from_exception",
]
