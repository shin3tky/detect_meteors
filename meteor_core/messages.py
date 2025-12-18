"""Message catalog and localization helpers for meteor_core.

This module provides a simple dictionary-based message catalog that supports
multiple locales with graceful fallback to English. Messages are formatted
using ``str.format`` by default and fall back to ``string.Template`` safe
substitution when formatting keys are missing.
"""

from __future__ import annotations

from string import Template
from typing import Dict

DEFAULT_LOCALE = "en"


MESSAGE_CATALOG: Dict[str, Dict[str, str]] = {
    "en": {
        "error.header": "ERROR: {message}",
        "error.filepath": "File: {filepath}",
        "error.cause": "Cause: {error_type}: {error_message}",
        "error.unexpected": "Unexpected error: {error_type}: {error_message}",
        "error.unexpected.hint": "Run with --verbose for full traceback, or report this issue.",
        "diagnostic.info": "For bug reports, please include the following diagnostic information:",
        "diagnostic.hint.verbose": "Run with --verbose flag for detailed diagnostic information.",
        "diagnostic.hint.save_option": "Or use --save-diagnostic to save a diagnostic report file.",
        "diagnostic.hint.save": "Tip: Use --save-diagnostic to save a report for bug reporting.",
        "diagnostic.report.saved": "Diagnostic report saved to: {path}",
        "interrupt.generic": "Interrupted by user.",
        "interrupt.progress": "Interrupted by user. Progress saved to {progress_file}.",
        "progress.removed": "Removed progress file: {path}",
        "progress.not_found": "Progress file not found: {path}",
    },
    "ja": {
        "error.header": "エラー: {message}",
        "error.filepath": "ファイル: {filepath}",
        "error.cause": "原因: {error_type}: {error_message}",
        "error.unexpected": "予期しないエラー: {error_type}: {error_message}",
        "error.unexpected.hint": "詳細なトレースは --verbose を付けて実行するか、不具合として報告してください。",
        "diagnostic.info": "バグ報告には次の診断情報を含めてください:",
        "diagnostic.hint.verbose": "--verbose を付けて実行すると詳細な診断情報を表示します。",
        "diagnostic.hint.save_option": "--save-diagnostic で診断レポートを保存できます。",
        "diagnostic.hint.save": "バグ報告用のレポートを保存するには --save-diagnostic を利用してください。",
        "diagnostic.report.saved": "診断レポートを保存しました: {path}",
        "interrupt.generic": "処理がユーザーにより中断されました。",
        "interrupt.progress": "処理が中断されました。進捗は {progress_file} に保存済みです。",
        "progress.removed": "進捗ファイルを削除しました: {path}",
        "progress.not_found": "進捗ファイルが見つかりません: {path}",
    },
}


def _normalize_locale(locale: str | None) -> str:
    normalized = (locale or DEFAULT_LOCALE).strip()
    normalized = normalized.replace("_", "-")
    return normalized.lower() or DEFAULT_LOCALE


def get_message(key: str, locale: str = DEFAULT_LOCALE, **kwargs) -> str:
    """Retrieve a localized message by key.

    Falls back to the default locale (English) when the requested locale or key
    is missing. Formatting placeholders are resolved with ``str.format`` and
    fall back to ``string.Template.safe_substitute`` if formatting keys are
    incomplete.

    Args:
        key: Message key, e.g., ``"diagnostic.hint.save"``.
        locale: Locale code such as ``"en"`` or ``"ja"``.
        **kwargs: Values for template placeholders.

    Returns:
        Localized and formatted message string.
    """

    normalized_locale = _normalize_locale(locale)
    locale_candidates = [normalized_locale]

    base_language = normalized_locale.split("-")[0]
    if base_language not in locale_candidates:
        locale_candidates.append(base_language)

    if DEFAULT_LOCALE not in locale_candidates:
        locale_candidates.append(DEFAULT_LOCALE)

    template = None
    for candidate in locale_candidates:
        messages = MESSAGE_CATALOG.get(candidate)
        if messages and key in messages:
            template = messages[key]
            break

    if template is None:
        template = key

    try:
        return template.format(**kwargs)
    except Exception:
        return Template(template).safe_substitute(**kwargs)


__all__ = ["get_message", "MESSAGE_CATALOG", "DEFAULT_LOCALE"]
