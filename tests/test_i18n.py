import sys
import unittest


class _StubModule:
    def __getattr__(self, name):  # pragma: no cover - defensive
        return self

    def __call__(self, *args, **kwargs):  # pragma: no cover - defensive
        return self


class _RawPyStub(_StubModule):
    class LibRawFileUnsupportedError(Exception):
        pass

    class LibRawIOError(Exception):
        pass

    class LibRawError(Exception):
        pass


sys.modules.setdefault("cv2", _StubModule())
sys.modules.setdefault("rawpy", _RawPyStub())

from meteor_core.i18n import DEFAULT_LOCALE, get_message  # noqa: E402


class TestI18nMessages(unittest.TestCase):
    """Tests for localized message loading and formatting."""

    def test_basic_lookup_and_locale(self):
        message_en = get_message(
            "ui.error.header", locale="en", params={"message": "Boom"}
        )
        self.assertEqual(message_en, "ERROR: Boom")

        message_ja = get_message(
            "ui.error.header", locale="ja", params={"message": "異常終了"}
        )
        self.assertEqual(message_ja, "ERROR: 異常終了")

    def test_plural_rendering(self):
        self.assertEqual(
            get_message("ui.run.summary", locale="en", count=0),
            "Complete! No candidates extracted",
        )
        self.assertEqual(
            get_message("ui.run.summary", locale="en", count=1),
            "Complete! 1 candidate extracted",
        )
        self.assertEqual(
            get_message("ui.run.summary", locale="en", count=5),
            "Complete! 5 candidates extracted",
        )
        self.assertEqual(
            get_message("ui.run.summary", locale="ja", count=2),
            "完了！候補を 2 件抽出しました",
        )

    def test_fallback_to_default_locale(self):
        message = get_message(
            "log.progress.invalid_field",
            locale="ja",
            path="/tmp/file.json",
            field="items",
            expected="list",
        )
        self.assertIn("Progress file", message)
        self.assertIn("items", message)
        self.assertTrue(message.startswith("Progress"))

    def test_missing_key_returns_key(self):
        key = "ui.does.not.exist"
        self.assertEqual(get_message(key, locale="fr"), key)

    def test_missing_params_are_left_in_template(self):
        message = get_message("ui.error.header", locale=DEFAULT_LOCALE)
        self.assertEqual(message, "ERROR: {message}")


if __name__ == "__main__":
    unittest.main()
