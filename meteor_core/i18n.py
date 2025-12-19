"""Internationalization helpers for meteor_core.

This module loads ICU-style message templates from locale-specific YAML files
and renders them with a lightweight formatter that supports plural rules and
safe placeholder substitution.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources
from logging import Logger
from numbers import Number
from string import Template
from typing import Any, Dict, Mapping

import yaml

DEFAULT_LOCALE = "en"
_LOCALES_PACKAGE = "meteor_core.locales"


class _SafeDict(dict):
    """Dictionary that leaves unknown format keys untouched."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _normalize_locale(locale: str | None) -> str:
    normalized = (locale or DEFAULT_LOCALE).strip().replace("_", "-").lower()
    return normalized or DEFAULT_LOCALE


def _candidate_locales(locale: str) -> list[str]:
    normalized = _normalize_locale(locale)
    language = normalized.split("-")[0]

    candidates = [normalized]
    if language not in candidates:
        candidates.append(language)
    if DEFAULT_LOCALE not in candidates:
        candidates.append(DEFAULT_LOCALE)
    return candidates


def _flatten_messages(node: Mapping[str, Any], prefix: str = "") -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for key, value in node.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten_messages(value, prefix=full_key))
        else:
            flat[full_key] = str(value)
    return flat


def _parse_yaml_like(text: str) -> Dict[str, Any]:
    """Parse YAML locale catalogs."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


@lru_cache(maxsize=None)
def _load_catalog(locale: str) -> Dict[str, str]:
    """Load and flatten the locale catalog."""
    try:
        base = resources.files(_LOCALES_PACKAGE)
    except ModuleNotFoundError:
        return {}

    path = base.joinpath(locale, "messages.yaml")
    if not path.is_file():
        return {}

    content = path.read_text(encoding="utf-8")
    raw_catalog = _parse_yaml_like(content)
    if not isinstance(raw_catalog, Mapping):
        return {}
    return _flatten_messages(raw_catalog)


def _coerce_number(value: Any) -> Number | None:
    if isinstance(value, Number):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _plural_category(locale: str, count: Number) -> str:
    language = _normalize_locale(locale).split("-")[0]
    if language == "ja":
        return "other"
    return "one" if count == 1 else "other"


def _extract_braced(text: str, start_index: int) -> tuple[str, int] | None:
    """Extract content within balanced braces starting at start_index."""
    if start_index >= len(text) or text[start_index] != "{":
        return None

    depth = 0
    for idx in range(start_index, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index + 1 : idx], idx
    return None


def _parse_plural_block(
    template: str, index: int, params: Mapping[str, Any], locale: str
) -> tuple[str, int] | None:
    parsed = _extract_braced(template, index)
    if not parsed:
        return None

    content, end_index = parsed
    parts = [part.strip() for part in content.split(",", 2)]
    if len(parts) < 2 or parts[1] != "plural":
        return None

    variable = parts[0]
    options_source = parts[2] if len(parts) == 3 else ""

    options: Dict[str, str] = {}
    cursor = 0
    while cursor < len(options_source):
        while cursor < len(options_source) and options_source[cursor].isspace():
            cursor += 1
        if cursor >= len(options_source):
            break

        key_match = re.match(r"[a-zA-Z0-9_=]+", options_source[cursor:])
        if not key_match:
            break

        key = key_match.group(0)
        cursor += len(key)
        while cursor < len(options_source) and options_source[cursor].isspace():
            cursor += 1
        if cursor >= len(options_source) or options_source[cursor] != "{":
            break

        extracted = _extract_braced(options_source, cursor)
        if not extracted:
            break

        value, cursor = extracted
        cursor += 1  # move past closing brace
        options[key] = value

    if not options:
        return None

    count = _coerce_number(params.get(variable))
    if count is None:
        selected = options.get("other")
    else:
        explicit_key = f"={count}"
        if explicit_key in options:
            selected = options[explicit_key]
        else:
            category = _plural_category(locale, count)
            selected = options.get(category) or options.get("other")

    if selected is None:
        return None

    return selected.replace("#", str(count if count is not None else "")), end_index


def _render_plurals(template: str, params: Mapping[str, Any], locale: str) -> str:
    rendered = []
    index = 0
    while index < len(template):
        if template[index] == "{":
            parsed = _parse_plural_block(template, index, params, locale)
            if parsed:
                replacement, end_index = parsed
                rendered.append(replacement)
                index = end_index + 1
                continue
        rendered.append(template[index])
        index += 1
    return "".join(rendered)


def _format_template(template: str, params: Mapping[str, Any], locale: str) -> str:
    rendered = _render_plurals(template, params, locale)
    try:
        return rendered.format_map(_SafeDict(params))
    except Exception:
        return Template(rendered).safe_substitute(**params)


def get_message(
    key: str,
    *,
    locale: str = DEFAULT_LOCALE,
    params: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Retrieve a localized message by key with ICU-style formatting."""

    merged_params: Dict[str, Any] = {}
    if params:
        merged_params.update(params)
    merged_params.update(kwargs)

    template: str | None = None
    for candidate in _candidate_locales(locale):
        catalog = _load_catalog(candidate)
        if key in catalog:
            template = catalog[key]
            break

    if template is None:
        template = key

    return _format_template(template, merged_params, locale)


def log_warning(
    logger: Logger,
    key: str,
    *,
    locale: str = DEFAULT_LOCALE,
    params: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Log a localized warning message."""

    message = get_message(key, locale=locale, params=params, **kwargs)
    logger.warning(message)


__all__ = ["DEFAULT_LOCALE", "get_message", "log_warning"]
