"""Deprecated message helpers.

This module remains for backward compatibility. Message definitions now live in
YAML files under ``meteor_core/locales`` and helpers are implemented in
``meteor_core.i18n``.
"""

from .i18n import DEFAULT_LOCALE, get_message, log_warning

__all__ = ["DEFAULT_LOCALE", "get_message", "log_warning"]
