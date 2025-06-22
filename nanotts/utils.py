"""Utility functions for nanoTTS."""

from __future__ import annotations

import re
import unicodedata


# Markdown cleaning patterns
MARKDOWN_PATTERNS = [
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),  # **bold** → text
    (re.compile(r"\*([^*]+)\*"), r"\1"),  # *italic* → text
    (re.compile(r"`([^`]+)`"), r"\1"),  # `code` → text
    (re.compile(r"#{1,6}\s*(.+)"), r"\1"),  # ### Header → Header
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),  # [text](url) → text
    (re.compile(r">\s*(.+)"), r"\1"),  # > quote → quote
    (re.compile(r"^\s*[-*+]\s+", re.MULTILINE), ""),  # - list → text
    (re.compile(r"^\s*\d+\.\s+", re.MULTILINE), ""),  # 1. list → text
]


def normalize_text(text: str) -> str:
    """Normalize unicode text for consistent processing."""
    if not text:
        return text

    # Unicode normalization (NFC form) - preserve all whitespace
    normalized = unicodedata.normalize("NFC", text)

    # Only normalize excessive whitespace, but preserve single spaces and structure
    normalized = re.sub(
        r"  +", " ", normalized
    )  # Multiple spaces → single space (but keep single spaces)
    normalized = re.sub(
        r"\n\s*\n\s*\n+", "\n\n", normalized
    )  # Multiple newlines → double newline max

    return normalized  # Don't strip - preserve leading/trailing spaces!


def clean_markdown(text: str) -> str:
    """Remove markdown formatting while preserving text content."""
    if not text:
        return text

    cleaned = text

    # Apply all markdown cleaning patterns
    for pattern, replacement in MARKDOWN_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)

    # Only normalize excessive whitespace, preserve structure
    cleaned = re.sub(r"  +", " ", cleaned)  # Multiple spaces → single space

    return cleaned  # Don't strip - preserve leading/trailing spaces


def preprocess_text(text: str) -> str:
    """Complete text preprocessing pipeline - applied to input chunks."""
    if not text:
        return text

    # Only normalize unicode for input chunks
    # Markdown cleaning will be applied to final segments
    normalized = normalize_text(text)

    return normalized
