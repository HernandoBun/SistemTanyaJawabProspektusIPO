from types import SimpleNamespace
from unittest.mock import Mock
from src.generation.generator import RAGGenerator


def test_letterhead_is_not_evidence_of_parent_company_officers():
    generator = object.__new__(RAGGenerator)
    generator.llm = Mock()
    candidate = SimpleNamespace(chunk=SimpleNamespace(
        content='No. Ref.: 354/2026\nPT RANS ENTERTAINMEN INDONESIA TBK\nDIREKSI\nDirektur Utama: Contoh Nama'
    ))
    result = generator.generate('siapa direktur rans', [candidate])
    generator.llm.invoke.assert_not_called()
    assert result.source_pages == []
    assert result.source_chunks == []
    assert result.retrieval_candidates == [candidate]
    assert 'emiten induk' in result.answer
