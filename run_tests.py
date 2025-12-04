#!/usr/bin/env python3
"""
Version-aware test runner for detect_meteors_cli
Runs different test suites based on the major version number
"""

import sys
import os
import re
import unittest

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def get_version():
    """Extract version from detect_meteors_cli.py"""
    try:
        with open("meteor_core/schema.py", "r") as f:
            content = f.read()
            match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
    except FileNotFoundError:
        print("Error: detect_meteors_cli.py not found")
        sys.exit(1)

    print("Error: Could not find VERSION in detect_meteors_cli.py")
    sys.exit(1)


def get_major_version(version_string):
    """Extract major version number from version string (e.g., '1.4.2' -> 1)"""
    match = re.match(r"^(\d+)\.", version_string)
    if match:
        return int(match.group(1))
    return None


def run_tests_for_version(major_version):
    """Run appropriate tests based on major version"""

    if major_version == 1:
        print(f"Running tests for version 1.x")
        # Discover and run tests in tests/ directory (for version 1.x)
        loader = unittest.TestLoader()
        suite = loader.discover("tests", pattern="test_*.py")

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return 0 if result.wasSuccessful() else 1

    elif major_version == 2:
        print(f"Running tests for version 2.x")
        # For version 2.x, use different test directory or patterns
        # This is a placeholder - adjust based on actual 2.x structure
        loader = unittest.TestLoader()
        suite = loader.discover("tests_v2", pattern="test_*.py")

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        return 0 if result.wasSuccessful() else 1

    else:
        print(f"Error: Unsupported major version {major_version}")
        return 1


def main():
    version = get_version()
    print(f"Detected version: {version}")

    major_version = get_major_version(version)
    if major_version is None:
        print(f"Error: Could not parse major version from '{version}'")
        sys.exit(1)

    print(f"Major version: {major_version}")

    exit_code = run_tests_for_version(major_version)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
