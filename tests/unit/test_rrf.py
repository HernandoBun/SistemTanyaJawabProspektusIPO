"""Tests for the production Reciprocal Rank Fusion implementation."""
from src.ingestion.chunker import Chunk
from src.retrieval.retriever import HybridRetriever


def make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(chunk_id, content, "narrative", 1, "", "TEST.pdf")


def test_rrf_merges_duplicate_chunk_and_adds_both_scores():
    shared = make_chunk("shared", "Konten yang sama dan cukup panjang")
    dense_only = make_chunk("dense", "Hanya ditemukan dense dan cukup panjang")
    sparse_only = make_chunk("sparse", "Hanya ditemukan sparse dan cukup panjang")
    retriever = object.__new__(HybridRetriever)

    results = retriever._rrf_merge(
        [(shared, 0.9), (dense_only, 0.8)],
        [(shared, 8.0), (sparse_only, 7.0)],
        rrf_k=60,
    )

    assert len(results) == 3
    assert results[0].chunk.content == shared.content
    assert results[0].rrf_score == 2 / 61
    assert results[0].dense_rank == 1
    assert results[0].sparse_rank == 1


def test_rrf_returns_descending_scores():
    first = make_chunk("first", "Konten pertama yang unik dan cukup panjang")
    second = make_chunk("second", "Konten kedua yang unik dan cukup panjang")
    retriever = object.__new__(HybridRetriever)
    results = retriever._rrf_merge([(first, 0.9), (second, 0.8)], [], rrf_k=60)
    assert [item.chunk.chunk_id for item in results] == ["first", "second"]
    assert results[0].rrf_score > results[1].rrf_score


def test_rrf_keeps_different_chunks_with_identical_prefixes():
    first = make_chunk("a", "Judul berulang " * 30 + "A")
    second = make_chunk("b", "Judul berulang " * 30 + "B")
    retriever = object.__new__(HybridRetriever)
    results = retriever._rrf_merge([(first, 0.9)], [(second, 1.2)])
    assert [r.chunk.chunk_id for r in results] == ["a", "b"]
    assert all(r.rrf_score == 1 / 61 for r in results)
