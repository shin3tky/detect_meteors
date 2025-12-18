#!/usr/bin/env python
#
# Detect Meteors CLI - Exception Tests
# © 2025 Shinichi Morita (shin3tky)
#

"""
Unit tests for custom exception classes and exception wrapping in image_io.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from meteor_core.exceptions import (
    DiagnosticInfo,
    MeteorConfigError,
    MeteorError,
    MeteorLoadError,
    MeteorOutputError,
    MeteorProgressError,
    MeteorUnsupportedFormatError,
    MeteorValidationError,
    MeteorWriteError,
)
from meteor_core.messages import get_message
from meteor_core.schema import VERSION


class TestMeteorError(unittest.TestCase):
    """Test cases for the base MeteorError exception."""

    def test_basic_creation(self):
        """Test creating a basic MeteorError."""
        err = MeteorError("Something went wrong")
        self.assertEqual(err.message, "Something went wrong")
        self.assertIsNone(err.filepath)
        self.assertIsNone(err.original_error)
        self.assertEqual(err.context, {})

    def test_with_filepath(self):
        """Test MeteorError with filepath."""
        err = MeteorError("Load failed", filepath="/path/to/file.CR2")
        self.assertEqual(err.filepath, "/path/to/file.CR2")
        self.assertIn("/path/to/file.CR2", str(err))

    def test_with_original_error(self):
        """Test MeteorError wrapping another exception."""
        original = ValueError("Original message")
        err = MeteorError("Wrapped error", original_error=original)
        self.assertIs(err.original_error, original)
        self.assertIn("ValueError", str(err))
        self.assertIn("Original message", str(err))

    def test_with_context(self):
        """Test MeteorError with context dictionary."""
        err = MeteorError("Error", context={"step": "detection", "count": 42})
        self.assertEqual(err.context["step"], "detection")
        self.assertEqual(err.context["count"], 42)

    def test_full_message_format(self):
        """Test that full message includes all components."""
        original = OSError("Disk full")
        err = MeteorError(
            "Write failed",
            filepath="/output/result.txt",
            original_error=original,
        )
        msg = str(err)
        self.assertIn("Write failed", msg)
        self.assertIn("/output/result.txt", msg)
        self.assertIn("OSError", msg)
        self.assertIn("Disk full", msg)

    def test_exception_hierarchy(self):
        """Test that MeteorError inherits from Exception."""
        self.assertTrue(issubclass(MeteorError, Exception))


class TestMeteorLoadError(unittest.TestCase):
    """Test cases for MeteorLoadError."""

    def test_default_message(self):
        """Test MeteorLoadError with default message."""
        err = MeteorLoadError()
        self.assertEqual(err.message, "Failed to load image file")

    def test_is_subclass_of_meteor_error(self):
        """Test that MeteorLoadError is subclass of MeteorError."""
        self.assertTrue(issubclass(MeteorLoadError, MeteorError))

    def test_can_catch_as_meteor_error(self):
        """Test that MeteorLoadError can be caught as MeteorError."""
        with self.assertRaises(MeteorError):
            raise MeteorLoadError("Test")

    def test_with_all_attributes(self):
        """Test MeteorLoadError with all attributes."""
        original = OSError("File not found")
        err = MeteorLoadError(
            "Cannot read file",
            filepath="/data/image.NEF",
            original_error=original,
            context={"attempt": 3},
        )
        self.assertEqual(err.message, "Cannot read file")
        self.assertEqual(err.filepath, "/data/image.NEF")
        self.assertIs(err.original_error, original)
        self.assertEqual(err.context["attempt"], 3)


class TestMeteorUnsupportedFormatError(unittest.TestCase):
    """Test cases for MeteorUnsupportedFormatError."""

    def test_default_message(self):
        """Test MeteorUnsupportedFormatError with default message."""
        err = MeteorUnsupportedFormatError()
        self.assertEqual(err.message, "Unsupported file format")

    def test_is_subclass_of_load_error(self):
        """Test that MeteorUnsupportedFormatError is subclass of MeteorLoadError."""
        self.assertTrue(issubclass(MeteorUnsupportedFormatError, MeteorLoadError))

    def test_can_catch_as_load_error(self):
        """Test that MeteorUnsupportedFormatError can be caught as MeteorLoadError."""
        with self.assertRaises(MeteorLoadError):
            raise MeteorUnsupportedFormatError("Test")

    def test_with_format_info(self):
        """Test MeteorUnsupportedFormatError with format information."""
        err = MeteorUnsupportedFormatError(
            "JPEG not supported",
            filepath="photo.jpg",
            detected_format="JPEG",
            supported_formats=[".CR2", ".NEF", ".ARW"],
        )
        self.assertEqual(err.detected_format, "JPEG")
        self.assertEqual(err.supported_formats, [".CR2", ".NEF", ".ARW"])
        self.assertEqual(err.context["detected_format"], "JPEG")
        self.assertIn(".CR2", err.context["supported_formats"])

    def test_format_info_in_context(self):
        """Test that format info is added to context."""
        err = MeteorUnsupportedFormatError(
            "Bad format",
            detected_format="PNG",
            supported_formats=["RAW"],
            context={"extra": "info"},
        )
        # Original context should be preserved
        self.assertEqual(err.context["extra"], "info")
        # Format info should be added
        self.assertEqual(err.context["detected_format"], "PNG")


class TestMeteorValidationError(unittest.TestCase):
    """Test cases for MeteorValidationError."""

    def test_default_message(self):
        """Test MeteorValidationError with default message."""
        err = MeteorValidationError()
        self.assertEqual(err.message, "Validation failed")

    def test_is_subclass_of_meteor_error(self):
        """Test that MeteorValidationError is subclass of MeteorError."""
        self.assertTrue(issubclass(MeteorValidationError, MeteorError))

    def test_with_parameter_info(self):
        """Test MeteorValidationError with parameter information."""
        err = MeteorValidationError(
            "Invalid threshold",
            parameter_name="diff_threshold",
            provided_value=-5,
            expected="positive integer",
        )
        self.assertEqual(err.parameter_name, "diff_threshold")
        self.assertEqual(err.provided_value, -5)
        self.assertEqual(err.expected, "positive integer")

    def test_parameter_info_in_context(self):
        """Test that parameter info is added to context."""
        err = MeteorValidationError(
            "Bad value",
            parameter_name="min_area",
            provided_value="not_a_number",
            expected="integer >= 1",
        )
        self.assertEqual(err.context["parameter_name"], "min_area")
        self.assertIn("not_a_number", err.context["provided_value"])
        self.assertEqual(err.context["expected"], "integer >= 1")


class TestMeteorConfigError(unittest.TestCase):
    """Test cases for MeteorConfigError."""

    def test_default_message(self):
        """Test MeteorConfigError with default message."""
        err = MeteorConfigError()
        self.assertEqual(err.message, "Configuration error")

    def test_is_subclass_of_meteor_error(self):
        """Test that MeteorConfigError is subclass of MeteorError."""
        self.assertTrue(issubclass(MeteorConfigError, MeteorError))

    def test_with_config_info(self):
        """Test MeteorConfigError with configuration information."""
        err = MeteorConfigError(
            "Invalid detector config",
            config_key="threshold",
            plugin_name="hough",
        )
        self.assertEqual(err.config_key, "threshold")
        self.assertEqual(err.plugin_name, "hough")

    def test_config_info_in_context(self):
        """Test that config info is added to context."""
        err = MeteorConfigError(
            "Missing required field",
            config_key="output_folder",
            plugin_name="file",
            context={"severity": "critical"},
        )
        self.assertEqual(err.context["config_key"], "output_folder")
        self.assertEqual(err.context["plugin_name"], "file")
        self.assertEqual(err.context["severity"], "critical")


class TestMeteorOutputError(unittest.TestCase):
    """Test cases for MeteorOutputError."""

    def test_default_message(self):
        """Test MeteorOutputError with default message."""
        err = MeteorOutputError()
        self.assertEqual(err.message, "Output operation failed")

    def test_is_subclass_of_meteor_error(self):
        """Test that MeteorOutputError is subclass of MeteorError."""
        self.assertTrue(issubclass(MeteorOutputError, MeteorError))

    def test_with_output_info(self):
        """Test MeteorOutputError with output information."""
        err = MeteorOutputError(
            "Failed to save file",
            destination_path="/output/file.CR2",
            operation="copy",
        )
        self.assertEqual(err.destination_path, "/output/file.CR2")
        self.assertEqual(err.operation, "copy")

    def test_output_info_in_context(self):
        """Test that output info is added to context."""
        err = MeteorOutputError(
            "Operation failed",
            destination_path="/output/dir",
            operation="mkdir",
            context={"permissions": "755"},
        )
        self.assertEqual(err.context["destination_path"], "/output/dir")
        self.assertEqual(err.context["operation"], "mkdir")
        self.assertEqual(err.context["permissions"], "755")


class TestMeteorWriteError(unittest.TestCase):
    """Test cases for MeteorWriteError."""

    def test_default_message(self):
        """Test MeteorWriteError with default message."""
        err = MeteorWriteError()
        self.assertEqual(err.message, "Failed to write file")

    def test_is_subclass_of_output_error(self):
        """Test that MeteorWriteError is subclass of MeteorOutputError."""
        self.assertTrue(issubclass(MeteorWriteError, MeteorOutputError))
        self.assertTrue(issubclass(MeteorWriteError, MeteorError))

    def test_can_catch_as_output_error(self):
        """Test catching MeteorWriteError as MeteorOutputError."""
        try:
            raise MeteorWriteError("Copy failed")
        except MeteorOutputError as e:
            self.assertIsInstance(e, MeteorWriteError)

    def test_with_write_info(self):
        """Test MeteorWriteError with write operation information."""
        original = IOError("Disk full")
        err = MeteorWriteError(
            "Failed to copy candidate file",
            filepath="/source/image.CR2",
            destination_path="/output/image.CR2",
            operation="copy",
            original_error=original,
            context={"error_category": "copy_failed"},
        )
        self.assertEqual(err.filepath, "/source/image.CR2")
        self.assertEqual(err.destination_path, "/output/image.CR2")
        self.assertEqual(err.operation, "copy")
        self.assertEqual(err.original_error, original)
        self.assertEqual(err.context["error_category"], "copy_failed")


class TestMeteorProgressError(unittest.TestCase):
    """Test cases for MeteorProgressError."""

    def test_default_message(self):
        """Test MeteorProgressError with default message."""
        err = MeteorProgressError()
        self.assertEqual(err.message, "Progress tracking error")

    def test_is_subclass_of_output_error(self):
        """Test that MeteorProgressError is subclass of MeteorOutputError."""
        self.assertTrue(issubclass(MeteorProgressError, MeteorOutputError))
        self.assertTrue(issubclass(MeteorProgressError, MeteorError))

    def test_can_catch_as_output_error(self):
        """Test catching MeteorProgressError as MeteorOutputError."""
        try:
            raise MeteorProgressError("Parse failed")
        except MeteorOutputError as e:
            self.assertIsInstance(e, MeteorProgressError)

    def test_with_progress_info(self):
        """Test MeteorProgressError with progress file information."""
        original = ValueError("Invalid JSON")
        err = MeteorProgressError(
            "Failed to parse progress file",
            filepath="progress.json",
            operation="parse",
            original_error=original,
            context={"error_category": "parse_failed"},
        )
        self.assertEqual(err.filepath, "progress.json")
        self.assertEqual(err.operation, "parse")
        self.assertEqual(err.original_error, original)
        self.assertEqual(err.context["error_category"], "parse_failed")

    def test_diagnostic_info_generation(self):
        """Test that MeteorProgressError can generate diagnostic info."""
        err = MeteorProgressError(
            "Failed to save progress",
            filepath="progress.json",
            operation="save",
            context={"error_category": "write_failed"},
        )
        diag = err.get_diagnostic_info()
        self.assertIsInstance(diag, DiagnosticInfo)
        self.assertEqual(diag.error_type, "MeteorProgressError")
        self.assertEqual(diag.filepath, "progress.json")


class TestDiagnosticInfo(unittest.TestCase):
    """Test cases for DiagnosticInfo dataclass."""

    def test_default_values(self):
        """Test DiagnosticInfo with default values."""
        info = DiagnosticInfo()
        self.assertEqual(info.version, "")
        self.assertIsNone(info.filepath)
        self.assertEqual(info.context, {})

    def test_to_dict(self):
        """Test DiagnosticInfo.to_dict() method."""
        info = DiagnosticInfo(
            version="1.0.0",
            python_version="3.12.0",
            platform="Darwin",
            timestamp="2025-01-01T00:00:00+00:00",
            error_type="MeteorLoadError",
            error_message="Test error",
        )
        d = info.to_dict()
        self.assertEqual(d["version"], "1.0.0")
        self.assertEqual(d["error_type"], "MeteorLoadError")
        self.assertIsNone(d["filepath"])

    def test_format_for_issue_basic(self):
        """Test DiagnosticInfo.format_for_issue() basic output."""
        info = DiagnosticInfo(
            version="1.5.11",
            python_version="3.12.0",
            platform="Darwin 24.0 (arm64)",
            timestamp="2025-01-01T12:00:00+00:00",
            error_type="MeteorLoadError",
            error_message="Test error message",
        )
        output = info.format_for_issue()
        self.assertIn("## Diagnostic Information", output)
        self.assertIn("meteor_core version: 1.5.11", output)
        self.assertIn("### Error Details", output)
        self.assertIn("MeteorLoadError", output)

    def test_format_for_issue_with_file_info(self):
        """Test DiagnosticInfo.format_for_issue() with file information."""
        info = DiagnosticInfo(
            version="1.5.11",
            python_version="3.12.0",
            platform="Linux",
            timestamp="2025-01-01T12:00:00+00:00",
            filepath="/path/to/image.CR2",
            file_exists=True,
            file_size=12345678,
            error_type="MeteorLoadError",
            error_message="Corrupted file",
        )
        output = info.format_for_issue()
        self.assertIn("### File Information", output)
        self.assertIn("/path/to/image.CR2", output)
        self.assertIn("Exists: True", output)
        self.assertIn("12,345,678 bytes", output)

    def test_format_for_issue_with_original_error(self):
        """Test DiagnosticInfo.format_for_issue() with original error."""
        info = DiagnosticInfo(
            version="1.5.11",
            python_version="3.12.0",
            platform="Windows",
            timestamp="2025-01-01T12:00:00+00:00",
            error_type="MeteorLoadError",
            error_message="Load failed",
            original_error_type="LibRawIOError",
            original_error_message="Cannot open file",
        )
        output = info.format_for_issue()
        self.assertIn("Original Error Type: LibRawIOError", output)
        self.assertIn("Original Error Message: Cannot open file", output)

    def test_format_for_issue_with_context(self):
        """Test DiagnosticInfo.format_for_issue() with context."""
        info = DiagnosticInfo(
            version="1.5.11",
            python_version="3.12.0",
            platform="Darwin",
            timestamp="2025-01-01T12:00:00+00:00",
            error_type="MeteorError",
            error_message="Error",
            context={"step": "detection", "image_index": 42},
        )
        output = info.format_for_issue()
        self.assertIn("### Additional Context", output)
        self.assertIn("step: detection", output)
        self.assertIn("image_index: 42", output)

    def test_format_for_issue_with_header_and_locale(self):
        """Test DiagnosticInfo.format_for_issue() with header and locale support."""
        info = DiagnosticInfo(
            version="1.5.11",
            python_version="3.12.0",
            platform="Darwin",
            timestamp="2025-01-01T12:00:00+00:00",
            error_type="MeteorError",
            error_message="Error",
        )
        output = info.format_for_issue(locale="ja", include_header=True)
        self.assertIn(get_message("diagnostic.report.title", locale="ja"), output)
        self.assertIn(get_message("diagnostic.section.heading", locale="ja"), output)
        self.assertIn(get_message("diagnostic.label.version", locale="ja"), output)


class TestMeteorErrorDiagnosticInfo(unittest.TestCase):
    """Test cases for MeteorError.get_diagnostic_info() method."""

    def test_get_diagnostic_info_basic(self):
        """Test get_diagnostic_info() returns correct info."""
        err = MeteorLoadError("Test error", filepath="/test/file.CR2")
        info = err.get_diagnostic_info()

        self.assertEqual(info.version, VERSION)
        self.assertEqual(info.error_type, "MeteorLoadError")
        self.assertEqual(info.error_message, "Test error")
        self.assertEqual(info.filepath, "/test/file.CR2")
        self.assertIn(".", info.python_version)  # Version contains dots like "3.12.0"

    def test_get_diagnostic_info_with_original_error(self):
        """Test get_diagnostic_info() captures original error."""
        original = ValueError("Original")
        err = MeteorError("Wrapped", original_error=original)
        info = err.get_diagnostic_info()

        self.assertEqual(info.original_error_type, "ValueError")
        self.assertEqual(info.original_error_message, "Original")

    def test_format_for_issue_method(self):
        """Test MeteorError.format_for_issue() convenience method."""
        err = MeteorLoadError("Test", filepath="/test.CR2")
        output = err.format_for_issue()

        self.assertIn("## Diagnostic Information", output)
        self.assertIn("MeteorLoadError", output)


class TestLoadAndBinRawFastExceptionWrapping(unittest.TestCase):
    """Test cases for exception wrapping in load_and_bin_raw_fast."""

    def test_file_not_found_raises_meteor_load_error(self):
        """Test that non-existent file raises MeteorLoadError."""
        from meteor_core.image_io import load_and_bin_raw_fast

        with self.assertRaises(MeteorLoadError) as ctx:
            load_and_bin_raw_fast("/nonexistent/path/to/file.CR2")

        err = ctx.exception
        self.assertIn("not found", err.message.lower())
        self.assertEqual(err.filepath, "/nonexistent/path/to/file.CR2")
        self.assertEqual(err.context.get("error_category"), "file_not_found")

    def test_permission_denied_raises_meteor_load_error(self):
        """Test that unreadable file raises MeteorLoadError."""
        from meteor_core.image_io import load_and_bin_raw_fast

        # Create a temporary file with no read permissions
        with tempfile.NamedTemporaryFile(delete=False, suffix=".CR2") as f:
            temp_path = f.name

        try:
            # Remove read permission
            os.chmod(temp_path, 0o000)

            with self.assertRaises(MeteorLoadError) as ctx:
                load_and_bin_raw_fast(temp_path)

            err = ctx.exception
            self.assertIn("permission", err.message.lower())
            self.assertEqual(err.context.get("error_category"), "permission_denied")
        finally:
            # Restore permissions and clean up
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)

    @patch("meteor_core.image_io.rawpy.imread")
    def test_libraw_file_unsupported_raises_unsupported_format_error(self, mock_imread):
        """Test that LibRawFileUnsupportedError is wrapped correctly."""
        import rawpy

        from meteor_core.image_io import load_and_bin_raw_fast

        # Create a temp file so file existence check passes
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            temp_path = f.name

        try:
            mock_imread.side_effect = rawpy.LibRawFileUnsupportedError("Not a RAW file")

            with self.assertRaises(MeteorUnsupportedFormatError) as ctx:
                load_and_bin_raw_fast(temp_path)

            err = ctx.exception
            self.assertIsInstance(err.original_error, rawpy.LibRawFileUnsupportedError)
            self.assertIn(".JPG", err.detected_format.upper())
            self.assertIsNotNone(err.supported_formats)
        finally:
            os.unlink(temp_path)

    @patch("meteor_core.image_io.rawpy.imread")
    def test_libraw_io_error_raises_meteor_load_error(self, mock_imread):
        """Test that LibRawIOError is wrapped correctly."""
        import rawpy

        from meteor_core.image_io import load_and_bin_raw_fast

        with tempfile.NamedTemporaryFile(delete=False, suffix=".CR2") as f:
            temp_path = f.name

        try:
            mock_imread.side_effect = rawpy.LibRawIOError("I/O Error")

            with self.assertRaises(MeteorLoadError) as ctx:
                load_and_bin_raw_fast(temp_path)

            err = ctx.exception
            self.assertIsInstance(err.original_error, rawpy.LibRawIOError)
            self.assertEqual(err.context.get("error_category"), "io_error")
        finally:
            os.unlink(temp_path)

    @patch("meteor_core.image_io.rawpy.imread")
    def test_libraw_data_error_raises_meteor_load_error(self, mock_imread):
        """Test that LibRawDataError is wrapped correctly."""
        import rawpy

        from meteor_core.image_io import load_and_bin_raw_fast

        with tempfile.NamedTemporaryFile(delete=False, suffix=".CR2") as f:
            temp_path = f.name

        try:
            mock_imread.side_effect = rawpy.LibRawDataError("Data corrupted")

            with self.assertRaises(MeteorLoadError) as ctx:
                load_and_bin_raw_fast(temp_path)

            err = ctx.exception
            self.assertIn("corrupt", err.message.lower())
            self.assertEqual(err.context.get("error_category"), "data_corruption")
        finally:
            os.unlink(temp_path)

    @patch("meteor_core.image_io.rawpy.imread")
    def test_generic_libraw_error_raises_meteor_load_error(self, mock_imread):
        """Test that generic LibRawError is wrapped correctly."""
        import rawpy

        from meteor_core.image_io import load_and_bin_raw_fast

        with tempfile.NamedTemporaryFile(delete=False, suffix=".CR2") as f:
            temp_path = f.name

        try:
            # Use a generic LibRawError subclass
            mock_imread.side_effect = rawpy.LibRawError("Generic error")

            with self.assertRaises(MeteorLoadError) as ctx:
                load_and_bin_raw_fast(temp_path)

            err = ctx.exception
            self.assertEqual(err.context.get("error_category"), "libraw_error")
        finally:
            os.unlink(temp_path)

    @patch("meteor_core.image_io.rawpy.imread")
    def test_unexpected_error_raises_meteor_load_error(self, mock_imread):
        """Test that unexpected exceptions are wrapped correctly."""
        from meteor_core.image_io import load_and_bin_raw_fast

        with tempfile.NamedTemporaryFile(delete=False, suffix=".CR2") as f:
            temp_path = f.name

        try:
            mock_imread.side_effect = RuntimeError("Unexpected!")

            with self.assertRaises(MeteorLoadError) as ctx:
                load_and_bin_raw_fast(temp_path)

            err = ctx.exception
            self.assertIsInstance(err.original_error, RuntimeError)
            self.assertEqual(err.context.get("error_category"), "unexpected")
            self.assertIn("RuntimeError", err.message)
        finally:
            os.unlink(temp_path)

    def test_diagnostic_info_can_be_generated(self):
        """Test that diagnostic info can be generated from load errors."""
        from meteor_core.image_io import load_and_bin_raw_fast

        try:
            load_and_bin_raw_fast("/nonexistent/file.CR2")
        except MeteorLoadError as e:
            info = e.get_diagnostic_info()
            self.assertEqual(info.error_type, "MeteorLoadError")
            self.assertEqual(info.filepath, "/nonexistent/file.CR2")
            self.assertFalse(info.file_exists)

            # Test format_for_issue
            report = e.format_for_issue()
            self.assertIn("## Diagnostic Information", report)
            self.assertIn("MeteorLoadError", report)


class TestExceptionChaining(unittest.TestCase):
    """Test exception chaining with __cause__."""

    def test_exception_chain_preserved(self):
        """Test that exception chaining is preserved with 'from'."""
        import rawpy

        from meteor_core.image_io import load_and_bin_raw_fast

        with tempfile.NamedTemporaryFile(delete=False, suffix=".CR2") as f:
            temp_path = f.name

        try:
            with patch("meteor_core.image_io.rawpy.imread") as mock_imread:
                original = rawpy.LibRawDataError("Original error")
                mock_imread.side_effect = original

                with self.assertRaises(MeteorLoadError) as ctx:
                    load_and_bin_raw_fast(temp_path)

                err = ctx.exception
                # Check __cause__ is set (from ... from e)
                self.assertIs(err.__cause__, original)
        finally:
            os.unlink(temp_path)


class TestDependencyVersions(unittest.TestCase):
    """Test cases for dependency version collection."""

    def test_diagnostic_info_includes_dependencies(self):
        """Test that DiagnosticInfo includes dependency versions."""
        err = MeteorLoadError("Test error")
        info = err.get_diagnostic_info()

        self.assertIsInstance(info.dependencies, dict)
        self.assertIn("numpy", info.dependencies)
        self.assertIn("rawpy", info.dependencies)

    def test_dependencies_in_format_for_issue(self):
        """Test that dependencies are included in format_for_issue output."""
        err = MeteorLoadError("Test error")
        output = err.format_for_issue()

        self.assertIn("### Dependencies", output)
        self.assertIn("numpy:", output)

    def test_to_dict_includes_dependencies(self):
        """Test that to_dict includes dependencies."""
        err = MeteorLoadError("Test error")
        info = err.get_diagnostic_info()
        d = info.to_dict()

        self.assertIn("dependencies", d)
        self.assertIsInstance(d["dependencies"], dict)


class TestFormatErrorForUser(unittest.TestCase):
    """Test cases for format_error_for_user helper function."""

    def test_basic_format(self):
        """Test basic error formatting."""
        from meteor_core.exceptions import format_error_for_user

        err = MeteorLoadError("File corrupted", filepath="/test/image.CR2")
        output = format_error_for_user(err)

        self.assertIn("ERROR:", output)
        self.assertIn("File corrupted", output)
        self.assertIn("/test/image.CR2", output)

    def test_verbose_format(self):
        """Test verbose error formatting includes diagnostics."""
        from meteor_core.exceptions import format_error_for_user

        err = MeteorLoadError("File corrupted", filepath="/test/image.CR2")
        output = format_error_for_user(err, verbose=True)

        self.assertIn("## Diagnostic Information", output)
        self.assertIn("meteor_core version:", output)

    def test_with_original_error(self):
        """Test formatting with original error."""
        from meteor_core.exceptions import format_error_for_user

        original = ValueError("Bad value")
        err = MeteorLoadError("Load failed", original_error=original)
        output = format_error_for_user(err)

        self.assertIn("Cause:", output)
        self.assertIn("ValueError", output)

    def test_locale_specific_format(self):
        """Test that locale parameter affects the rendered message."""
        from meteor_core.exceptions import format_error_for_user

        err = MeteorLoadError("ファイルが壊れています", filepath="/test/image.CR2")
        output = format_error_for_user(err, locale="ja")

        self.assertIn("エラー", output)
        self.assertIn("ファイル", output)


class TestSaveDiagnosticReport(unittest.TestCase):
    """Test cases for save_diagnostic_report helper function."""

    def test_save_to_file(self):
        """Test saving diagnostic report to file."""
        from meteor_core.exceptions import save_diagnostic_report

        err = MeteorLoadError("Test error", filepath="/test/file.CR2")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "diagnostic.md")
            result_path = save_diagnostic_report(err, output_path)

            self.assertEqual(result_path, output_path)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, encoding="utf-8") as f:
                content = f.read()

            self.assertIn("# Meteor Detection - Diagnostic Report", content)
            self.assertIn("## Diagnostic Information", content)
            self.assertIn("MeteorLoadError", content)

    def test_auto_generated_filename(self):
        """Test auto-generated filename when path not specified."""
        from meteor_core.exceptions import save_diagnostic_report

        err = MeteorLoadError("Test error")

        # Change to temp directory to avoid polluting current dir
        original_dir = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                result_path = save_diagnostic_report(err)

                self.assertTrue(result_path.startswith("meteor_diagnostic_"))
                self.assertTrue(result_path.endswith(".md"))
                self.assertTrue(os.path.exists(result_path))
            finally:
                os.chdir(original_dir)

    def test_save_to_file_respects_locale(self):
        """Test that diagnostic report rendering honors locale selection."""
        from meteor_core.exceptions import save_diagnostic_report

        err = MeteorLoadError("テストエラー", filepath="/test/file.CR2")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "diagnostic_ja.md")
            result_path = save_diagnostic_report(err, output_path, locale="ja")

            self.assertEqual(result_path, output_path)
            with open(output_path, encoding="utf-8") as f:
                content = f.read()

        self.assertIn(get_message("diagnostic.report.title", locale="ja"), content)
        self.assertIn(
            get_message("diagnostic.report.instructions", locale="ja"), content
        )
        self.assertIn(get_message("diagnostic.section.heading", locale="ja"), content)


class TestCreateDiagnosticFromException(unittest.TestCase):
    """Test cases for create_diagnostic_from_exception helper function."""

    def test_from_standard_exception(self):
        """Test creating diagnostic from standard exception."""
        from meteor_core.exceptions import create_diagnostic_from_exception

        exc = ValueError("Something went wrong")
        info = create_diagnostic_from_exception(exc)

        self.assertEqual(info.error_type, "ValueError")
        self.assertEqual(info.error_message, "Something went wrong")
        self.assertIn("numpy", info.dependencies)

    def test_with_filepath_and_context(self):
        """Test with filepath and context."""
        from meteor_core.exceptions import create_diagnostic_from_exception

        exc = IOError("File error")
        info = create_diagnostic_from_exception(
            exc,
            filepath="/test/file.txt",
            context={"operation": "read"},
        )

        self.assertEqual(info.filepath, "/test/file.txt")
        self.assertEqual(info.context["operation"], "read")

    def test_format_for_issue_works(self):
        """Test that format_for_issue works on created diagnostic."""
        from meteor_core.exceptions import create_diagnostic_from_exception

        exc = RuntimeError("Unexpected")
        info = create_diagnostic_from_exception(exc)
        output = info.format_for_issue()

        self.assertIn("## Diagnostic Information", output)
        self.assertIn("RuntimeError", output)


class TestMessageCatalog(unittest.TestCase):
    """Test cases for localized message lookups."""

    def test_fallback_to_english(self):
        """Nonexistent locale should fall back to English messages."""
        from meteor_core.messages import get_message

        message = get_message("diagnostic.hint.save", locale="fr")

        self.assertIn("--save-diagnostic", message)


class TestImportFromPackage(unittest.TestCase):
    """Test that all new functions are exported from meteor_core."""

    def test_import_helper_functions(self):
        """Test importing helper functions from meteor_core."""
        from meteor_core import (
            create_diagnostic_from_exception,
            format_error_for_user,
            save_diagnostic_report,
        )

        self.assertTrue(callable(format_error_for_user))
        self.assertTrue(callable(save_diagnostic_report))
        self.assertTrue(callable(create_diagnostic_from_exception))


if __name__ == "__main__":
    unittest.main()
