import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import config
from src.generation.generator import RAGGenerator
from src.ingestion.chunker import Chunk
from src.retrieval.indexer import DualIndexer
from src.retrieval import indexer as indexer_module
from src.services import pipeline as pipeline_module
from src.services.pipeline import RAGPipeline
from evaluation.validate_dataset import validate


def evidence(page, content='Pendapatan 100.'):
    return SimpleNamespace(chunk=Chunk(str(page), content, 'narrative', page, '', 'TEST.pdf'), rrf_score=.03)


@pytest.mark.parametrize('answer,pages', [
    ('Pendapatan 100 [S2].', [2]),
    ('Pendapatan 100 [S99].', []),
    ('Informasi tersebut tidak tersedia di dokumen prospektus ini.', []),
])
def test_sources_are_selected_not_all_retrieved(answer, pages):
    generator = object.__new__(RAGGenerator)
    generator.llm = Mock()
    generator.llm.invoke.return_value = SimpleNamespace(content=answer)
    response = generator.generate('Berapa pendapatan?', [evidence(1), evidence(2)])
    assert response.source_pages == pages
    assert '[S' not in response.answer
    assert bool(response.retrieval_candidates) == (not pages)


def test_unrelated_letterhead_does_not_veto_clear_board_section():
    generator = object.__new__(RAGGenerator)
    generator.llm = Mock()
    generator.llm.invoke.return_value = SimpleNamespace(content='Direktur Utama Perseroan adalah Budi [S1].')
    result = generator.generate('siapa direktur?', [
        evidence(1, 'No. Ref.: 123\nDireksi anak perusahaan'),
        evidence(2, 'Susunan Direksi Perseroan\nDirektur Utama: Budi'),
    ])
    assert result.source_pages == [2]
    assert 'anak perusahaan' not in generator.llm.invoke.call_args.args[0][1].content


def test_changed_pdf_is_not_skipped(tmp_path, monkeypatch):
    pdf = tmp_path / 'TEST.pdf'
    pdf.write_bytes(b'original')
    pipeline = object.__new__(RAGPipeline)
    pipeline._registry = {'TEST.pdf': {'index_schema_version': config.INDEX_SCHEMA_VERSION,
        'content_sha256': hashlib.sha256(b'original').hexdigest()}}
    assert pipeline.ingest(pdf)['skipped']
    pdf.write_bytes(b'changed')
    parser = Mock(side_effect=RuntimeError('parsing invoked'))
    monkeypatch.setattr(pipeline_module, 'PDFParser', parser)
    with pytest.raises(RuntimeError, match='parsing invoked'):
        pipeline.ingest(pdf)


def test_official_roster_is_preferred_to_previous_subsidiary_section():
    generator = object.__new__(RAGGenerator)
    generator.llm = Mock()
    generator.llm.invoke.return_value = SimpleNamespace(content='Direktur: Budi [S1].')
    response = generator.generate('siapa direktur jec?', [
        evidence(164, 'Direktur: Nama Anak\nSusunan Direksi Perseroan adalah sebagai berikut:'),
        evidence(165, '**Direktur**: Budi\nPenunjukan Dewan Komisaris dan Direksi Perseroan telah memenuhi persyaratan.'),
    ])
    assert response.source_pages == [165]
    assert 'Nama Anak' not in generator.llm.invoke.call_args.args[0][1].content


@pytest.mark.parametrize('failure', ['build', 'publish'])
def test_failed_replacement_restores_both_indexes(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(config, 'CHROMA_DIR', tmp_path / 'chroma')
    monkeypatch.setattr(config, 'BM25_INDEX_PATH', tmp_path / 'bm25.pkl')
    monkeypatch.setattr(indexer_module, 'Embedder', Mock())
    indexer = DualIndexer()
    old = [Chunk('old', 'aset lama', 'narrative', 1, '', 'TEST.pdf'),
           Chunk('other', 'emiten lain', 'narrative', 1, '', 'OTHER.pdf')]
    indexer.build_index(old, embeddings=np.ones((2, 1024), dtype=np.float32))
    original_build = indexer.build_index
    def failing_build(*args, **kwargs):
        original_build(*args, **kwargs)
        raise RuntimeError('write failed')
    publish = Mock(side_effect=RuntimeError('publish failed')) if failure == 'publish' else Mock()
    if failure == 'build':
        monkeypatch.setattr(indexer, 'build_index', failing_build)
    with pytest.raises(RuntimeError, match='failed'):
        indexer.replace_document('TEST.pdf', [Chunk('new', 'aset baru', 'narrative', 2, '', 'TEST.pdf')],
                                 np.ones((1, 1024), dtype=np.float32), publish)
    assert set(indexer._collection.get()['ids']) == {'old', 'other'}
    assert indexer._load_bm25()
    assert {c.chunk_id for c in indexer._corpus_chunks} == {'old', 'other'}


def test_evaluation_rejects_missing_chunk_annotations(tmp_path):
    questions = tmp_path / 'questions.jsonl'
    truth = tmp_path / 'truth.jsonl'
    questions.write_text(json.dumps(dict(question_id='q1', doc_source='TEST.pdf', question='Aset?', category='numeric', answerable=True)))
    truth.write_text(json.dumps(dict(question_id='q1', answer='100', source_pages=[1])))
    assert any('relevant_chunk_ids' in error for error in validate(questions, truth))
    assert validate(questions, truth, 'page') == []
