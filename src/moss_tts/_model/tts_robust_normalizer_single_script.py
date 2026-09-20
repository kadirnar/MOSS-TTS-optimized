#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust TTS input normalization (without semantic text normalization).

Goals
-----
1. Clean up input without expanding numbers, units, dates, or currency amounts.
2. Protect sensitive tokens such as `.map`, `app.js.map`, `v2.3.1`, URLs,
   email addresses, @mentions, and #hashtags.
3. Preserve the contents of [] and {} for sound events and control tags.
4. Replace structural symbols with readable boundaries:
   - Structural brackets become sentence boundaries.
   - Unwrap standalone title brackets; preserve titles embedded in a sentence.
   - Long dashes and repeated hyphens become sentence boundaries.
   - Flow arrows such as ->, =>, and → become Chinese commas.
5. Lightly normalize social-media punctuation noise:
   - Repeated periods and ellipses become a Chinese full stop.
   - Repeated question/exclamation marks become a single pair or mark.
6. Handle whitespace according to the writing system:
   - Collapse repeated spaces inside Latin text.
   - Remove spaces inside Han and Japanese kana text.
   - Preserve or insert one space between Han/kana and Latin/protected tokens.
   - Do not force a space between Han/kana and digits.
7. Apply lightweight Markdown and newline cleanup:
   - Convert [text](url) to text url.
   - Remove heading, quote, and list prefixes.
   - Convert newlines into Chinese sentence boundaries.

Non-goals
---------
1. Decide how text should be pronounced.
2. Remove the contents of [] or {}.
3. Interpret HTML, SSML, or semantic tags.
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------
# Basic constants and regular expressions
# ---------------------------

# Writing systems that do not require word-separating spaces: Han and Japanese kana.
_CJK_CHARS = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"
_CJK = f"[{_CJK_CHARS}]"

# Protected-span placeholders
_PROT = r"___PROT\d+___"

# Sensitive tokens that must be protected
_URL_RE = re.compile(r"https?://[^\s\u3000，。！？；、）】》〉」』]+")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}")
_REDDIT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#(?!\s)[^\s#]+")

# `.map` / `.env` / `.gitignore`
_DOT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+")

# Examples: `app.js.map`, `index.d.ts`, `v2.3.1`, and `foo/bar-baz.py`.
_FILELIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Za-z0-9._/+:-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/+:-]*[._/+:-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9._/+:-]*[A-Za-z0-9])?"
    r"(?![A-Za-z0-9_])"
)

# Mixed-script spacing tokens must contain a Latin letter or be protected tokens.
_LATINISH = rf"(?:{_PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)"

# Zero-width characters
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")


# ---------------------------
# Main function
# ---------------------------

def normalize_tts_text(text: str) -> str:
    """Normalize TTS input for robustness."""
    text = _base_cleanup(text)
    text = _normalize_markdown_and_lines(text)
    text, protected = _protect_spans(text)

    text = _normalize_spaces(text)
    text = _normalize_structural_punctuation(text)
    text = _normalize_repeated_punctuation(text)
    text = _normalize_spaces(text)

    text = _restore_spans(text, protected)
    return text.strip()


# ---------------------------
# Normalization rules
# ---------------------------

def _base_cleanup(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = _ZERO_WIDTH_RE.sub("", text)

    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch in "\n\t " or not cat.startswith("C"):
            cleaned.append(ch)
    return "".join(cleaned)


def _normalize_markdown_and_lines(text: str) -> str:
    # Markdown links: [text](url) -> text url
    text = re.sub(r"\[([^\[\]]+?)\]\((https?://[^)\s]+)\)", r"\1 \2", text)

    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        line = re.sub(r"^#{1,6}\s+", "", line)   # Heading
        line = re.sub(r"^>\s+", "", line)        # Quote
        line = re.sub(r"^[-*+]\s+", "", line)    # Unordered list
        line = re.sub(r"^\d+[.)]\s+", "", line)  # Ordered list
        lines.append(line)

    return "。".join(lines) if lines else ""


def _protect_spans(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def repl(match: re.Match[str]) -> str:
        idx = len(protected)
        protected.append(match.group(0))
        return f"___PROT{idx}___"

    for pattern in (
        _URL_RE,
        _EMAIL_RE,
        _MENTION_RE,
        _REDDIT_RE,
        _HASHTAG_RE,
        _DOT_TOKEN_RE,
        _FILELIKE_RE,
    ):
        text = pattern.sub(repl, text)

    return text, protected


def _restore_spans(text: str, protected: list[str]) -> str:
    for idx, original in enumerate(protected):
        text = text.replace(f"___PROT{idx}___", original)
    return text


def _normalize_spaces(text: str) -> str:
    # Normalize whitespace.
    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    # Remove spaces within Han and Japanese text.
    text = re.sub(rf"({_CJK})\s+(?={_CJK})", r"\1", text)

    # Remove spaces between Han/Japanese text and digits.
    text = re.sub(rf"({_CJK})\s+(?=\d)", r"\1", text)
    text = re.sub(rf"(\d)\s+(?={_CJK})", r"\1", text)

    # Preserve or insert a space between Han/Japanese text and Latin/protected tokens.
    text = re.sub(rf"({_CJK})(?=({_LATINISH}))", r"\1 ", text)
    text = re.sub(rf"(({_LATINISH}))(?={_CJK})", r"\1 ", text)

    # Collapse repeated spaces again.
    text = re.sub(r" {2,}", " ", text)

    # Remove spaces around Chinese punctuation.
    text = re.sub(r"\s+([，。！？；：、”’」』】）》])", r"\1", text)
    text = re.sub(r"([（【「『《“‘])\s+", r"\1", text)
    text = re.sub(r"([，。！？；：、])\s*", r"\1", text)

    # Remove spaces before ASCII punctuation; retain normal English spacing after it.
    text = re.sub(r"\s+([,.;!?])", r"\1", text)

    return re.sub(r" {2,}", " ", text).strip()


def _normalize_structural_punctuation(text: str) -> str:
    # Unwrap brackets in structural positions and convert them into sentence boundaries.
    # Use two passes to handle adjacent blocks.
    for _ in range(2):
        text = re.sub(
            r"(^|[。！？!?；;]\s*)[【〖『「]([^】〗』」]+)[】〗』」]\s*",
            r"\1\2。",
            text,
        )

    # Unwrap only standalone title brackets, preserving embedded titles.
    # For example, a bracketed heading followed by a dash becomes a separate sentence.
    text = re.sub(
        r"(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))",
        r"\1\2",
        text,
    )

    # Convert flow/mapping arrows into Chinese commas, preserving the sequence for TTS.
    text = re.sub(
        r"\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*",
        "，",
        text,
    )

    # Convert long dashes and repeated hyphens into sentence boundaries.
    text = re.sub(r"\s*(?:—|–|―|-){2,}\s*", "。", text)

    return text


def _normalize_repeated_punctuation(text: str) -> str:
    # Ellipses and repeated periods
    text = re.sub(r"(?:\.{3,}|…{2,}|……+)", "。", text)

    # Repeated punctuation of the same type
    text = re.sub(r"[。．]{2,}", "。", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[!！]{2,}", "！", text)
    text = re.sub(r"[?？]{2,}", "？", text)

    # Collapse mixed question/exclamation marks into one pair.
    def _mixed_qe(match: re.Match[str]) -> str:
        s = match.group(0)
        has_q = any(ch in s for ch in "?？")
        has_e = any(ch in s for ch in "!！")
        if has_q and has_e:
            return "？！"
        return "？" if has_q else "！"

    text = re.sub(r"[!?！？]{2,}", _mixed_qe, text)
    return text


# ---------------------------
# Tests
