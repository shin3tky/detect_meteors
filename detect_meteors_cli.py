"""Compatibility wrapper for the legacy detect_meteors CLI entry point."""

from detect_meteors import default_plugin as _default_plugin
from detect_meteors.default_plugin import *  # noqa: F401,F403

# Explicitly bind version for tooling that parses this file directly.
VERSION = _default_plugin.VERSION

if __name__ == "__main__":
    main()
