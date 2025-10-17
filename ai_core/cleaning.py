"""Utilities for cleaning and merging extracted text fields.

This module keeps text normalization small and predictable; callers can build
more advanced pipelines on top.
"""
from __future__ import annotations

import re


def compact_whitespace(s: str) -> str:
    """Collapse consecutive whitespace and trim.

    Args:
        s: Input string.
    Returns:
        A string with all runs of whitespace replaced by a single space,
        preserving newlines around section headers.
    """
    if not s:
        return ""
    # Keep newlines but collapse runs of spaces/tabs.
    s = re.sub(r"[\t\x0b\x0c\r ]+", " ", s)
    # Trim lines, drop excessive blank lines (max 2)
    lines = [ln.strip() for ln in s.split("\n")]
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln:
            blank_run = 0
            out.append(ln)
        else:
            blank_run += 1
            if blank_run <= 2:
                out.append("")
    return "\n".join(out).strip()


def strip_artifacts(s: str) -> str:
    """Remove common OCR/PDF artifacts without changing semantics much.

    This keeps implementation conservative so it doesn't delete meaningful content.
    """
    if not s:
        return ""
    # Remove repeated hyphenation at line breaks: e.g., "exam-\nple" -> "example"
    s = re.sub(r"-\n\s*", "", s)
    # Remove excessive page headers/footers markers (simple heuristic)
    s = re.sub(r"\n\s*Page\s+\d+\s*(/\s*\d+)?\s*\n", "\n", s, flags=re.I)
    return s


def merge_fields(raw: str, ocr: str, caps: list[str]) -> str:
    """Merge fields into a single page_context string.

    Format prioritizes captions for better visibility to LLM.
    """
    raw = raw or ""
    ocr = ocr or ""
    caps = caps or []
    
    # Build the merged context with captions more prominent
    parts = []
    
    if raw:
        parts.append(f"TEXT:\n{raw}")
    
    if caps:
        caps_joined = "\n".join(f"- {c}" for c in caps if c)
        parts.append(f"FIGURES/IMAGES:\n{caps_joined}")
    
    if ocr and ocr.strip() not in raw:  # Only add OCR if it's different from raw
        parts.append(f"OCR:\n{ocr}")
    
    merged = "\n\n".join(parts)
    merged = strip_artifacts(merged)
    merged = compact_whitespace(merged)
    return merged
