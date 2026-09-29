"""Conservative preprocessing for Indonesian IPO prospectus text.

The rules intentionally preserve line structure, Markdown, Unicode letters,
punctuation, and financial symbols. Only demonstrable extraction artefacts and
repeated page-edge noise are removed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import replace
from typing import Any, Iterable, List, Tuple

import config


_ZERO_WIDTH = {"\u00ad", "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
_PAGE_NUMBER_RE = re.compile(
    r"^(?:(?:halaman|hal\.?|page)\s*)?(?:[-–—]\s*)?\d+(?:\s*[-–—])?$",
    re.IGNORECASE,
)
_FINANCIAL_OR_STRUCTURAL_RE = re.compile(
    r"(?:^#{1,6}\s|^BAB\s+[IVXLCDM\d]+|\||\bRp\.?\s*\d|%|"
    r"\bdalam\s+(?:ribuan|jutaan|miliar)|^\s*[-*+]\s+)",
    re.IGNORECASE,
)


def get_preprocessing_config() -> dict:
    """Return the cache-relevant preprocessing configuration."""
    return {
        "version": config.PREPROCESSING_VERSION,
        "enabled": config.PREPROCESSING_ENABLED,
        "remove_repeated_edges": config.PREPROCESS_REMOVE_REPEATED_EDGES,
        "edge_lines": config.PREPROCESS_EDGE_LINES,
        "repeat_ratio": config.PREPROCESS_REPEAT_RATIO,
        "min_repeat_pages": config.PREPROCESS_MIN_REPEAT_PAGES,
        "max_blank_lines": config.PREPROCESS_MAX_BLANK_LINES,
    }


def _remove_invisible_and_control_chars(text: str) -> Tuple[str, int]:
    output: List[str] = []
    removed = 0
    for char in text:
        if char in _ZERO_WIDTH:
            removed += 1
            continue
        category = unicodedata.category(char)
        if category.startswith("C") and char not in {"\n", "\t"}:
            removed += 1
            continue
        output.append(char)
    return "".join(output), removed


def normalize_text(text: str, *, max_blank_lines: int | None = None) -> Tuple[str, int]:
    """Normalize extraction artefacts without deleting meaningful symbols."""
    if not text:
        return "", 0

    cleaned = unicodedata.normalize("NFC", str(text))
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\u00a0", " ").replace("\u202f", " ")
    cleaned, removed_controls = _remove_invisible_and_control_chars(cleaned)

    lines = []
    for line in cleaned.split("\n"):
        # Horizontal whitespace is presentation noise; line boundaries remain.
        lines.append(re.sub(r"[\t \f\v]+", " ", line).strip())

    limit = (
        config.PREPROCESS_MAX_BLANK_LINES
        if max_blank_lines is None
        else max(1, max_blank_lines)
    )
    output: List[str] = []
    blank_run = 0
    for line in lines:
        if line:
            blank_run = 0
            output.append(line)
        else:
            blank_run += 1
            if blank_run <= limit:
                output.append("")

    return "\n".join(output).strip(), removed_controls


def preprocess_query(question: str) -> str:
    """Apply safe Unicode/whitespace cleanup to a user query."""
    cleaned, _ = normalize_text(question, max_blank_lines=1)
    return " ".join(line for line in cleaned.splitlines() if line).strip()


def _edge_lines(text: str, count: int) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    lines = text.splitlines()
    populated = [(index, line) for index, line in enumerate(lines) if line.strip()]
    return populated[:count], populated[-count:]


def _canonical_edge_line(line: str) -> str:
    canonical = re.sub(r"\s+", " ", line).strip().casefold()
    if _PAGE_NUMBER_RE.fullmatch(canonical):
        return "<page-number>"
    return canonical


def _safe_to_remove(line: str) -> bool:
    stripped = line.strip()
    if _PAGE_NUMBER_RE.fullmatch(stripped):
        return True
    return not _FINANCIAL_OR_STRUCTURAL_RE.search(stripped)


def _repeated_edge_keys(pages: Iterable[Any]) -> Tuple[set[str], set[str], int]:
    pages = list(pages)
    if len(pages) < config.PREPROCESS_MIN_REPEAT_PAGES:
        return set(), set(), 0

    threshold = max(
        config.PREPROCESS_MIN_REPEAT_PAGES,
        math.ceil(len(pages) * config.PREPROCESS_REPEAT_RATIO),
    )
    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    for page in pages:
        top, bottom = _edge_lines(page.text_content, config.PREPROCESS_EDGE_LINES)
        # Count a candidate at most once per page.
        top_counter.update({_canonical_edge_line(line) for _, line in top if line.strip()})
        bottom_counter.update({_canonical_edge_line(line) for _, line in bottom if line.strip()})

    top_keys = {key for key, count in top_counter.items() if count >= threshold}
    bottom_keys = {key for key, count in bottom_counter.items() if count >= threshold}
    return top_keys, bottom_keys, threshold


def _remove_repeated_edges(
    text: str, top_keys: set[str], bottom_keys: set[str]
) -> Tuple[str, List[str], List[str]]:
    lines = text.splitlines()
    top, bottom = _edge_lines(text, config.PREPROCESS_EDGE_LINES)
    top_indexes = {
        index
        for index, line in top
        if _canonical_edge_line(line) in top_keys and _safe_to_remove(line)
    }
    bottom_indexes = {
        index
        for index, line in bottom
        if _canonical_edge_line(line) in bottom_keys and _safe_to_remove(line)
    }
    removed_headers = [lines[index] for index in sorted(top_indexes)]
    removed_footers = [lines[index] for index in sorted(bottom_indexes - top_indexes)]
    removed_indexes = top_indexes | bottom_indexes
    retained = [line for index, line in enumerate(lines) if index not in removed_indexes]
    cleaned, _ = normalize_text("\n".join(retained))
    return cleaned, removed_headers, removed_footers


def preprocess_pages(pages: List[Any]) -> Tuple[List[Any], dict]:
    """Preprocess parsed pages and return immutable replacements plus audit data."""
    config_snapshot = get_preprocessing_config()
    chars_before = sum(
        len(page.text_content or "") + sum(len(table or "") for table in page.tables)
        for page in pages
    )

    if not config.PREPROCESSING_ENABLED:
        report = {
            **config_snapshot,
            "page_count": len(pages),
            "normalized_pages": 0,
            "removed_control_characters": 0,
            "removed_header_lines": 0,
            "removed_footer_lines": 0,
            "repeat_threshold_pages": 0,
            "characters_before": chars_before,
            "characters_after": chars_before,
        }
        return pages, report

    normalized_pages: List[Any] = []
    total_controls = 0
    for page in pages:
        normalized_text, control_count = normalize_text(page.text_content or "")
        normalized_tables: List[str] = []
        for table in page.tables:
            normalized_table, table_controls = normalize_text(table or "")
            normalized_tables.append(normalized_table)
            control_count += table_controls
        total_controls += control_count
        normalized_pages.append(
            replace(
                page,
                text_content=normalized_text,
                tables=normalized_tables,
                removed_header_footer=[],
                preprocessing_stats={"removed_control_characters": control_count},
            )
        )

    top_keys: set[str] = set()
    bottom_keys: set[str] = set()
    threshold = 0
    if config.PREPROCESS_REMOVE_REPEATED_EDGES:
        top_keys, bottom_keys, threshold = _repeated_edge_keys(normalized_pages)

    processed_pages: List[Any] = []
    header_total = 0
    footer_total = 0
    for page in normalized_pages:
        cleaned_text, headers, footers = _remove_repeated_edges(
            page.text_content, top_keys, bottom_keys
        )
        header_total += len(headers)
        footer_total += len(footers)
        page_stats = dict(page.preprocessing_stats)
        page_stats.update(
            {
                "removed_header_lines": len(headers),
                "removed_footer_lines": len(footers),
            }
        )
        processed_pages.append(
            replace(
                page,
                text_content=cleaned_text,
                removed_header_footer=headers + footers,
                preprocessing_stats=page_stats,
            )
        )

    chars_after = sum(
        len(page.text_content or "") + sum(len(table or "") for table in page.tables)
        for page in processed_pages
    )
    report = {
        **config_snapshot,
        "page_count": len(processed_pages),
        "normalized_pages": len(processed_pages),
        "removed_control_characters": total_controls,
        "removed_header_lines": header_total,
        "removed_footer_lines": footer_total,
        "repeat_threshold_pages": threshold,
        "characters_before": chars_before,
        "characters_after": chars_after,
    }
    return processed_pages, report
