import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pymupdf
import pytest
import config
from src.generation.generator import RAGGenerator
from src.ingestion.chunker import Chunk
from src.retrieval.indexer import DualIndexer
from src.retrieval import indexer as indexer_module
from src.services import pipeline as pipeline_module
from src.services.pipeline import RAGPipeline
from src.services.index_lock import IndexWriteBusyError, index_write_lock
from src.services import prospectus_upload as upload_module
from src.services.prospectus_upload import ProspectusUploadError, upload_prospectus
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


def prospectus_pdf(*, pages=1, password=None):
    with pymupdf.open() as pdf:
        for _ in range(pages):
            page = pdf.new_page()
            page.insert_text((72, 72), "Dana penawaran umum perdana saham digunakan untuk modal kerja Perseroan.")
        if password:
            return pdf.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password)
        return pdf.tobytes()


@pytest.fixture
def upload_storage(tmp_path, monkeypatch):
    prospectuses = tmp_path / "prospectuses"
    prospectuses.mkdir()
    monkeypatch.setattr(config, "PROSPEKTUS_DIR", prospectuses)
    monkeypatch.setattr(config, "CACHE_REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "BM25_INDEX_PATH", tmp_path / "bm25.pkl")
    monkeypatch.setattr(config, "PARSED_CACHE_DIR", tmp_path / "parsed")
    monkeypatch.setattr(config, "PARSER_MODE", "pdfplumber")
    monkeypatch.setattr(upload_module, "check_embedding_connection", Mock())
    embedder = Mock()
    embedder.embed_texts.side_effect = lambda texts, **kwargs: np.ones((len(texts), 1024), dtype=np.float32)
    monkeypatch.setattr(indexer_module, "Embedder", Mock(return_value=embedder))
    return prospectuses


@pytest.mark.parametrize("name,content,error", [
    ("anda_2026.pdf", b"%PDF-", "KODE_TAHUN.pdf"),
    ("../ANDA_2026.pdf", b"%PDF-", "KODE_TAHUN.pdf"),
    ("ANDA_2026.pdf", b"", "kosong"),
    ("ANDA_2026.pdf", b"plain text", "dibaca sebagai PDF"),
    ("ANDA_2026.pdf", b"%PDF-1.7\ninvalid", "dibaca sebagai PDF"),
])
def test_upload_rejects_invalid_input(upload_storage, name, content, error):
    with pytest.raises(ProspectusUploadError, match=error):
        upload_prospectus(name, content)
    assert not list(upload_storage.iterdir())


def test_upload_validates_size_pages_and_password(upload_storage, monkeypatch):
    monkeypatch.setattr(upload_module, "MAX_UPLOAD_MB", 1)
    monkeypatch.setattr(upload_module, "MAX_UPLOAD_PAGES", 1)
    with pytest.raises(ProspectusUploadError, match="Ukuran PDF"):
        upload_prospectus("ANDA_2026.pdf", b"x" * (1024 * 1024 + 1))
    with pytest.raises(ProspectusUploadError, match="maksimal 1 halaman"):
        upload_prospectus("ANDA_2026.pdf", prospectus_pdf(pages=2))
    with pytest.raises(ProspectusUploadError, match="kata sandi"):
        upload_prospectus("ANDA_2026.pdf", prospectus_pdf(password="secret"))
    assert not list(upload_storage.iterdir())


@pytest.mark.parametrize("stored_in", ["disk", "legacy registry"])
def test_upload_rejects_existing_names_case_insensitively(upload_storage, stored_in):
    if stored_in == "disk":
        (upload_storage / "anda_2026.PDF").write_bytes(b"existing PDF")
    else:
        config.CACHE_REGISTRY_PATH.write_text(
            json.dumps({"anda_2026.PDF": {"index_schema_version": "legacy"}}), encoding="utf-8",
        )
    with pytest.raises(ProspectusUploadError, match="sudah"):
        upload_prospectus("ANDA_2026.pdf", prospectus_pdf())
    upload_module.check_embedding_connection.assert_not_called()


def test_upload_refuses_a_concurrent_index_write(upload_storage):
    with index_write_lock(), pytest.raises(IndexWriteBusyError):
        upload_prospectus("ANDA_2026.pdf", prospectus_pdf())
    assert not list(upload_storage.iterdir())


def test_upload_preserves_an_unreadable_registry(upload_storage):
    config.CACHE_REGISTRY_PATH.write_bytes(b"{incomplete")
    with pytest.raises(RuntimeError, match="Registry prospektus gagal dimuat"):
        upload_prospectus("ANDA_2026.pdf", prospectus_pdf())
    assert config.CACHE_REGISTRY_PATH.read_bytes() == b"{incomplete"
    assert not list(upload_storage.iterdir())


def test_upload_keeps_pdf_registry_and_indexes_in_sync(upload_storage, monkeypatch):
    original_pdf = prospectus_pdf()
    upload_prospectus("LAMA_2025.pdf", original_pdf)
    original_registry = config.CACHE_REGISTRY_PATH.read_bytes()
    content = prospectus_pdf(pages=2)
    with monkeypatch.context() as broken_storage:
        broken_storage.setattr(RAGPipeline, "_save_registry", Mock(side_effect=OSError("write failed")))
        with pytest.raises(OSError, match="write failed"):
            upload_prospectus("ANDA_2026.pdf", content)

    assert {path.name for path in upload_storage.iterdir()} == {"LAMA_2025.pdf"}
    assert config.CACHE_REGISTRY_PATH.read_bytes() == original_registry
    counts = RAGPipeline().indexer.get_document_chunk_counts()
    assert set(counts["chroma"]) == set(counts["bm25"]) == {"LAMA_2025.pdf"}

    upload_prospectus("ANDA_2026.pdf", content)
    assert (upload_storage / "ANDA_2026.pdf").read_bytes() == content
    pipeline = RAGPipeline()
    assert set(pipeline.get_indexed_docs()) == {"LAMA_2025.pdf", "ANDA_2026.pdf"}
    assert pipeline.get_indexed_docs()["ANDA_2026.pdf"]["total_pages"] == 2
    counts = pipeline.indexer.get_document_chunk_counts()
    assert counts["chroma"] == counts["bm25"]
