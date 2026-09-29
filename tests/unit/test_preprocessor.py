"""Regression tests for conservative prospectus preprocessing."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.ingestion.parser import ParsedPage
from src.ingestion.preprocessor import normalize_text, preprocess_pages, preprocess_query


def _page(number: int, text: str, tables=None) -> ParsedPage:
    return ParsedPage(
        page_number=number,
        text_content=text,
        tables=tables or [],
        raw_text=text,
    )


def test_normalization_removes_only_extraction_artifacts():
    source = "Cafe\u0301\u200b\u00a0  Rp 1.250.000 (18,7%)\r\n\r\n\r\n\r\nBAB I"
    cleaned, removed = normalize_text(source, max_blank_lines=2)

    assert "Café" in cleaned
    assert "\u200b" not in cleaned
    assert "Rp 1.250.000 (18,7%)" in cleaned
    assert "\n\n\n\n" not in cleaned
    assert removed == 1


def test_table_symbols_and_line_structure_are_preserved():
    table = "| Keterangan | 2025 |\r\n| --- | ---: |\r\n| Laba (rugi) | (Rp 25.000) |"
    pages, report = preprocess_pages([_page(1, "# BAB I\nIsi", [table])])

    assert pages[0].tables[0].splitlines() == [
        "| Keterangan | 2025 |",
        "| --- | ---: |",
        "| Laba (rugi) | (Rp 25.000) |",
    ]
    assert report["removed_header_lines"] == 0


def test_repeated_headers_and_page_numbers_are_removed_but_bab_is_kept(monkeypatch):
    import config

    monkeypatch.setattr(config, "PREPROCESS_MIN_REPEAT_PAGES", 3)
    monkeypatch.setattr(config, "PREPROCESS_REPEAT_RATIO", 0.60)
    pages = []
    for number in range(1, 5):
        text = (
            "PROSPEKTUS AWAL PT CONTOH\n"
            "BAB I PENDAHULUAN\n\n"
            f"Isi unik halaman {number} dengan data Rp {number}.000.\n\n"
            f"Catatan penting unik halaman {number}\n"
            f"Halaman {number}"
        )
        pages.append(_page(number, text))

    processed, report = preprocess_pages(pages)

    assert report["removed_header_lines"] == 4
    assert report["removed_footer_lines"] == 4
    assert all("PROSPEKTUS AWAL PT CONTOH" not in page.text_content for page in processed)
    assert all("BAB I PENDAHULUAN" in page.text_content for page in processed)
    assert all(page.raw_text.startswith("PROSPEKTUS AWAL") for page in processed)
    assert all(len(page.removed_header_footer) == 2 for page in processed)


def test_small_documents_are_not_cleaned_as_repetition(monkeypatch):
    import config

    monkeypatch.setattr(config, "PREPROCESS_MIN_REPEAT_PAGES", 3)
    pages, report = preprocess_pages([
        _page(1, "HEADER\nIsi halaman pertama\n1"),
        _page(2, "HEADER\nIsi halaman kedua\n2"),
    ])

    assert report["removed_header_lines"] == 0
    assert report["removed_footer_lines"] == 0
    assert pages[0].text_content.startswith("HEADER")


def test_query_preprocessing_is_conservative():
    query = preprocess_query("  Berapa\u200b laba\u00a0  bersih 18,7%?\r\n")
    assert query == "Berapa laba bersih 18,7%?"
