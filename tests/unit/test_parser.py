"""
tests/unit/test_parser.py
Uji fungsi PDFParser & helper metadata dari src/ingestion/parser.py
"""
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.parser import (
    PDFParser,
    ParsedPage,
    _cache_path,
    _load_parse_cache,
    _save_parse_cache,
    _source_digest,
    get_document_metadata,
)
import config

def test_metadata_extraction():
    meta = get_document_metadata("BACH_2026.pdf")
    assert meta["file_name"] == "BACH_2026.pdf"


def test_parse_cache_round_trip(tmp_path, monkeypatch):
    pdf = tmp_path / "SAMPLE.pdf"
    pdf.write_bytes(b"stable-pdf-content")
    cache_dir = tmp_path / "parsed"
    monkeypatch.setattr(config, "PARSED_CACHE_DIR", cache_dir)
    digest = _source_digest(pdf)
    pages = [ParsedPage(1, "Narasi", ["| A |\n|---|\n| 1 |"], "Raw")]

    path = _save_parse_cache(
        pdf, "strict_llamaparse", "llamaparse", digest, pages
    )
    cached = _load_parse_cache(pdf, "strict_llamaparse", digest)

    assert path.exists()
    assert cached is not None
    loaded_pages, parser_used = cached
    assert loaded_pages == pages
    assert parser_used == "llamaparse"


def test_parser_rejects_unknown_mode(tmp_path):
    pdf = tmp_path / "SAMPLE.pdf"
    pdf.write_bytes(b"pdf")
    import pytest

    with pytest.raises(ValueError, match="PARSER_MODE tidak valid"):
        PDFParser(pdf, parser_mode="unknown")


def test_legacy_v2_cache_is_preprocessed_and_migrated(tmp_path, monkeypatch):
    pdf = tmp_path / "LEGACY.pdf"
    pdf.write_bytes(b"legacy-pdf-content")
    cache_dir = tmp_path / "parsed"
    monkeypatch.setattr(config, "PARSED_CACHE_DIR", cache_dir)
    digest = _source_digest(pdf)
    legacy_path = _cache_path(pdf, "pdfplumber", digest, cache_version="v2")
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_text(
        json.dumps(
            {
                "cache_version": "v2",
                "source_sha256": digest,
                "parser_mode": "pdfplumber",
                "parser_used": "pdfplumber",
                "pages": [
                    {
                        "page_number": 1,
                        "text_content": "Laba\u200b\u00a0  Rp 1.000",
                        "tables": [],
                        "raw_text": "Laba\u200b\u00a0  Rp 1.000",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    parser = PDFParser(pdf, parser_mode="pdfplumber")
    pages = parser.parse()

    assert parser.cache_hit is True
    assert parser.cache_migrated is True
    assert parser.cache_path is not None and parser.cache_path.name.endswith("_v3.json")
    assert pages[0].text_content == "Laba Rp 1.000"
    assert pages[0].raw_text == "Laba\u200b\u00a0  Rp 1.000"

if __name__ == "__main__":
    test_metadata_extraction()
    print("All parser unit tests PASSED")
