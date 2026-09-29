"""
tests/unit/test_chunker.py
Uji ProspectusChunker dari src/ingestion/chunker.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.ingestion.chunker import ProspectusChunker, _extract_financial_unit, _is_orphan_table
from src.ingestion.parser import ParsedPage

def make_page(text="", tables=None, page_num=1):
    p = ParsedPage.__new__(ParsedPage)
    p.page_number = page_num
    p.text_content = text
    p.tables = tables or []
    return p

def test_min_length_filter():
    chunker = ProspectusChunker("test.pdf")
    page = make_page("ok")  # too short
    chunks = chunker._chunk_narrative(page)
    assert len(chunks) == 0

def test_financial_unit_detection():
    assert _extract_financial_unit("(dalam jutaan Rupiah)") != ""
    assert _extract_financial_unit("(dalam ribuan Rupiah)") != ""
    assert _extract_financial_unit("teks biasa") == ""

def test_orphan_table_detection():
    orphan = "|---|---|---|\n| data | data |"
    normal = "| Header1 | Header2 |\n|---|---|\n| data | data |"
    assert _is_orphan_table(orphan) == True
    assert _is_orphan_table(normal) == False

def test_chunk_has_metadata():
    chunker = ProspectusChunker("TEST.pdf")
    long_text = "BAB V — ANALISIS\n\n" + ("Pendapatan usaha meningkat. " * 40)
    page = make_page(text=long_text)
    chunks = chunker._chunk_narrative(page)
    assert len(chunks) > 0
    for c in chunks:
        assert c.doc_source == "TEST.pdf"
        assert c.page_number == 1
        assert c.chunk_type == "narrative"

def test_chunk_ids_are_unique_between_documents_and_stable():
    first = ProspectusChunker("ABDI_2025.pdf")
    second = ProspectusChunker("BACH_2026.pdf")
    first_again = ProspectusChunker("ABDI_2025.pdf")

    first_id = first._new_id("narr")
    second_id = second._new_id("narr")
    assert first_id != second_id
    assert first_id == first_again._new_id("narr")
    assert first_id.startswith("abdi_2025_")

def test_heading_and_chapter_are_inherited_across_pages():
    chunker = ProspectusChunker("TEST.pdf", min_length=20)
    pages = [
        make_page("BAB V — ANALISIS\n\n" + ("Kinerja meningkat. " * 10), page_num=1),
        make_page(("Pendapatan tumbuh berkelanjutan. " * 10), page_num=2),
    ]

    chunks = chunker.chunk_pages(pages)
    page_two = [chunk for chunk in chunks if chunk.page_number == 2]

    assert page_two
    assert all(chunk.section_heading == "BAB V — ANALISIS" for chunk in page_two)
    assert all(chunk.chapter.upper() == "BAB V" for chunk in page_two)

if __name__ == "__main__":
    test_min_length_filter(); test_financial_unit_detection()
    test_orphan_table_detection(); test_chunk_has_metadata()
    print("All chunker tests PASSED")
