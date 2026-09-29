"""Dependency-free BM25 tokenization for Indonesian financial documents."""

from __future__ import annotations

import re
import unicodedata
from typing import List


def _normalize_numbers(text: str) -> str:
    """Create searchable tokens while retaining Indonesian number semantics."""

    def _financial_number(value: str) -> str:
        return value.replace(".", "").replace(",", "_")

    text = re.sub(
        r"[Rr][Pp]\.?\s*([\d][\d.]*(?:,\d+)?)",
        lambda match: (
            "rp" + _financial_number(match.group(1)) + " "
            + _financial_number(match.group(1))
        ),
        text,
    )
    text = re.sub(
        r"(\d+)[,.](\d+)\s*%",
        lambda match: match.group(1) + "_" + match.group(2) + "_persen",
        text,
    )
    text = re.sub(
        r"(\d+)\s*%", lambda match: match.group(1) + "_persen", text
    )
    return re.sub(
        r"\b(\d{1,3})(\.\d{3})+\b",
        lambda match: match.group(0).replace(".", ""),
        text,
    )


def _tokenize(text: str) -> List[str]:
    """Tokenize Unicode text and emit normalized plus original number forms."""
    source = unicodedata.normalize("NFC", text).casefold()
    normalized = _normalize_numbers(source)
    tokens_normalized = re.findall(r"\w+", normalized, flags=re.UNICODE)
    tokens_original = re.findall(r"\w+", source, flags=re.UNICODE)

    seen = set()
    result: List[str] = []
    for token in tokens_normalized + tokens_original:
        if token and token not in seen:
            seen.add(token)
            result.append(token)
    return result
