"""Tests for issuer-scoped BM25 statistics and retrieval."""

from src.ingestion.chunker import Chunk
from src.retrieval.indexer import DualIndexer


def make_chunk(chunk_id: str, content: str, doc_source: str) -> Chunk:
    return Chunk(chunk_id, content, "narrative", 1, "", doc_source)


def make_indexer_without_external_services() -> DualIndexer:
    indexer = object.__new__(DualIndexer)
    indexer._bm25_by_document = {}
    indexer._corpus_chunks_by_document = {}
    indexer._corpus_chunks = []
    return indexer


def test_bm25_is_built_independently_for_each_document():
    indexer = make_indexer_without_external_services()
    chunks = [
        make_chunk("a1", "total aset tahun 2025", "A.pdf"),
        make_chunk("a2", "liabilitas dan ekuitas", "A.pdf"),
        make_chunk("b1", "total aset perusahaan lain", "B.pdf"),
    ]

    indexer._rebuild_bm25_by_document(chunks)

    assert set(indexer._bm25_by_document) == {"A.pdf", "B.pdf"}
    assert len(indexer._corpus_chunks_by_document["A.pdf"]) == 2
    assert len(indexer._corpus_chunks_by_document["B.pdf"]) == 1
    assert indexer._bm25_by_document["A.pdf"].corpus_size == 2
    assert indexer._bm25_by_document["B.pdf"].corpus_size == 1


def test_sparse_search_never_returns_another_issuer():
    indexer = make_indexer_without_external_services()
    issuer_a_chunks = [
        make_chunk("a1", "total aset tahun 2025", "A.pdf"),
        *[
            make_chunk(f"a{number}", f"informasi prospektus bagian {number}", "A.pdf")
            for number in range(2, 11)
        ],
    ]
    indexer._rebuild_bm25_by_document([
        *issuer_a_chunks,
        make_chunk("b1", "total aset tahun 2025", "B.pdf"),
    ])

    results = indexer.sparse_search(
        "total aset 2025",
        k=10,
        doc_source_filter="A.pdf",
    )

    assert results
    assert all(chunk.doc_source == "A.pdf" for chunk, _ in results)
