"""Dense results must retain IDs so the same chunk can receive both RRF votes."""
from unittest.mock import Mock

import numpy as np

from src.ingestion.chunker import Chunk
from src.retrieval.indexer import DualIndexer
from src.retrieval.retriever import HybridRetriever


def test_dense_identity_fuses_with_bm25_and_keeps_document_filter():
    indexer = object.__new__(DualIndexer)
    indexer._collection = Mock()
    indexer._collection.count.return_value = 1
    indexer._collection.query.return_value = {
        'ids': [['rans_narr_001']],
        'documents': [['Direktur utama perusahaan']],
        'metadatas': [[{'doc_source':'RANS_2026.pdf','page_number':15,'chunk_type':'narrative'}]],
        'distances': [[0.2]],
    }
    dense = indexer.dense_search(np.ones(1024), doc_source_filter='RANS_2026.pdf')
    assert dense[0][0].chunk_id == 'rans_narr_001'
    assert indexer._collection.query.call_args.kwargs['where'] == {'doc_source':'RANS_2026.pdf'}
    sparse = [(Chunk('rans_narr_001','Direktur utama perusahaan','narrative',15,'','RANS_2026.pdf'), 3.5)]
    result = object.__new__(HybridRetriever)._rrf_merge(dense, sparse)
    assert len(result) == 1
    assert result[0].rrf_score == 2 / 61
    assert result[0].rrf_score >= 0.02
