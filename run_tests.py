#!/usr/bin/env python3
"""
Run the shared CLI and compatibility tests for supported major versions.
"""

import sys
import os
import re
import unittest

# Add project root directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def get_version():
    """Extract the application version from meteor_core/schema.py."""
    try:
        with open("meteor_core/schema.py", "r") as f:
            content = f.read()
            match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
    except FileNotFoundError:
        print("Error: meteor_core/schema.py not found")
        sys.exit(1)

    print("Error: Could not find VERSION in meteor_core/schema.py")
    sys.exit(1)


def get_major_version(version_string):
    """Extract major version number from version string (e.g., '1.4.2' -> 1)"""
    match = re.match(r"^(\d+)\.", version_string)
    if match:
        return int(match.group(1))
    return None


def run_tests_for_version(major_version):
    """Use the same regression tests for the 1.x and 2.x workflows."""
    if major_version not in (1, 2):
        print(f"Error: Unsupported major version {major_version}")
        return 1

    print(f"Running tests for version {major_version}.x")
    suite = unittest.TestLoader().discover("tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


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
