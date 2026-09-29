"""
tests/integration/test_pipeline.py
Uji integrasi: PDF → parser → chunker → indexer → retriever
Jalankan dengan: python -m pytest tests/integration/ -v
CATATAN: Test ini membutuhkan CEREBRAS_API_KEY dan file PDF.
"""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def test_retrieve_requires_active_document():
    from src.services.pipeline import RAGPipeline

    pipeline = object.__new__(RAGPipeline)
    pipeline._is_ready = True
    pipeline._retriever = object()
    pipeline._active_doc = None
    with pytest.raises(RuntimeError, match="Dokumen aktif belum dipilih"):
        pipeline.retrieve("Berapa total aset?")


def test_retrieval_keeps_query_without_added_synonyms():
    from src.services.pipeline import RAGPipeline

    from unittest.mock import Mock
    pipeline = object.__new__(RAGPipeline)
    pipeline._is_ready = True
    pipeline._active_doc = "TEST.pdf"
    pipeline._retriever = Mock()
    pipeline.retrieve("Apa rencana penggunaan dana hasil IPO?")
    assert pipeline._retriever.retrieve.call_args.kwargs["query"] == "Apa rencana penggunaan dana hasil IPO?"
    assert pipeline._retriever.retrieve.call_args.kwargs["doc_source_filter"] == "TEST.pdf"


def test_retrieve_preprocesses_query_before_search():
    from src.services.pipeline import RAGPipeline

    class RetrieverStub:
        received_query = None

        def retrieve(self, *, query, **kwargs):
            self.received_query = query
            return []

    pipeline = object.__new__(RAGPipeline)
    pipeline._is_ready = True
    pipeline._retriever = RetrieverStub()
    pipeline._active_doc = "TEST.pdf"

    assert pipeline.retrieve("  Berapa\u200b laba\u00a0 18,7%?\r\n") == []
    assert pipeline._retriever.received_query == "Berapa laba 18,7%?"


def test_retrieve_rejects_empty_cleaned_query():
    from src.services.pipeline import RAGPipeline

    pipeline = object.__new__(RAGPipeline)
    pipeline._is_ready = True
    pipeline._retriever = object()
    pipeline._active_doc = "TEST.pdf"

    with pytest.raises(ValueError, match="Pertanyaan kosong"):
        pipeline.retrieve("\u200b  \r\n")


@pytest.mark.skip(reason="Memerlukan indeks fixture lokal dan API Cerebras")
def test_full_pipeline_query_with_fixture():
    """Placeholder only for the external-resource smoke test."""
