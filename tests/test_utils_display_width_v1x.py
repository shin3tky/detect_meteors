"""Tests for unicode display width helpers in meteor_core.utils."""

import os
import sys
import unittest
from unittest.mock import patch

# Add project root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meteor_core.utils import _display_width, _pad_label, _use_wide_ambiguous_chars


class TestUnicodeDisplayWidth(unittest.TestCase):
    """Ensure display width logic handles wide/ambiguous characters."""

    def test_use_wide_ambiguous_chars_env(self):
        with patch.dict(os.environ, {"DETECT_METEORS_LOCALE": "ja_JP"}, clear=False):
            self.assertTrue(_use_wide_ambiguous_chars())

        with patch.dict(
            os.environ,
            {"DETECT_METEORS_LOCALE": "en_US", "LC_ALL": "en_US"},
            clear=False,
        ):
            self.assertFalse(_use_wide_ambiguous_chars())

    def test_display_width_with_ambiguous_chars(self):
        ambiguous = "·"
        with patch("meteor_core.utils._use_wide_ambiguous_chars", return_value=False):
            self.assertEqual(_display_width(ambiguous), 1)

        with patch("meteor_core.utils._use_wide_ambiguous_chars", return_value=True):
            self.assertEqual(_display_width(ambiguous), 2)

    def test_pad_label_accounts_for_wide_chars(self):
        label = "界"
        padded = _pad_label(label, 4)
        self.assertEqual(_display_width(padded), 4)
        self.assertTrue(padded.startswith(label))


if __name__ == "__main__":
    unittest.main()
