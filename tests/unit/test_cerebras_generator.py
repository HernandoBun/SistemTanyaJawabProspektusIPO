from types import SimpleNamespace

import pytest

import config
from src.generation import generator


class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs

    def invoke(self, messages):
        return SimpleNamespace(
            content="Jawaban uji",
            usage_metadata={"total_tokens": 42},
            response_metadata={},
        )


def test_generator_requires_openrouter_api_key(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        generator.RAGGenerator()


def test_generator_initializes_openrouter_client(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(config, "OPENROUTER_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
    monkeypatch.setattr(generator, "ChatOpenAI", FakeChatOpenAI)

    instance = generator.RAGGenerator()

    assert instance.llm.init_kwargs == {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "qwen/qwen-2.5-72b-instruct",
        "temperature": 0.1,
        "max_tokens": 2048,
        "extra_body": {"reasoning": {"effort": "none"}},
        "timeout": config.OPENROUTER_TIMEOUT,
        "max_retries": config.OPENROUTER_MAX_RETRIES,
        "default_headers": {
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "IPO Prospectus Q&A",
        },
    }


def test_empty_retrieval_does_not_call_openrouter(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(config, "OPENROUTER_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
    monkeypatch.setattr(generator, "ChatOpenAI", FakeChatOpenAI)

    response = generator.RAGGenerator().generate(
        question="Pertanyaan uji",
        retrieved_chunks=[],
    )

    assert response.model_used == "qwen/qwen-2.5-72b-instruct"
    assert response.source_chunks == []
    assert response.tokens_used == 0


def test_clean_generated_answer_removes_bracket_citations():
    raw = (
        "Rincian penggunaannya mencakup gaji karyawan dan perizinan [Hal. 17]. "
        "Selain itu, dana dialokasikan untuk pelatihan [Hal. 17; Hal. 243]. "
        "Pertumbuhan tetap stabil [1]. Selesai [..]."
    )
    cleaned = generator.clean_generated_answer(raw)
    assert "[Hal." not in cleaned
    assert "[1]" not in cleaned
    assert "[..]" not in cleaned
    assert "gaji karyawan dan perizinan." in cleaned
    assert "pelatihan." in cleaned
    assert "stabil." in cleaned


def test_clean_generated_answer_removes_unclosed_and_pdf_filenames():
    raw = "Berdasarkan dokumen swap_2026.pdf, modal dialokasikan [Hal. 240"
    cleaned = generator.clean_generated_answer(raw)
    assert "swap_2026.pdf" not in cleaned
    assert "[Hal." not in cleaned
    assert "modal dialokasikan" in cleaned


def test_generator_strips_citations_from_generated_answer(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(config, "OPENROUTER_MODEL_NAME", "qwen/qwen-2.5-72b-instruct")

    class CitationMockLLM:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return SimpleNamespace(
                content="Alokasi untuk modal kerja [Hal. 17].",
                usage_metadata={"total_tokens": 10},
                response_metadata={},
            )

    monkeypatch.setattr(generator, "ChatOpenAI", CitationMockLLM)
    chunk = SimpleNamespace(
        page_number=17,
        chunk_type="narrative",
        section_heading="Penggunaan Dana",
        table_title="",
        financial_unit="",
        content="Penggunaan dana...",
    )
    response = generator.RAGGenerator().generate(
        question="Untuk apa dana digunakan?",
        retrieved_chunks=[SimpleNamespace(chunk=chunk, rrf_score=0.03)],
    )
    assert response.answer == "Alokasi untuk modal kerja."
    assert "[Hal. 17]" not in response.answer


def test_clean_generated_answer_removes_prose_page_citations():
    raw = (
        "Pada bagian struktur direksi di halaman 329, Direktur Utama adalah Johan A.M.M. Hutauruk. "
        "Sementara itu, pada halaman 239 dan 232, Direktur Utama tercatat masing-masing sebagai Budi Djatmiko "
        "dan Iwan Soebijantoro. Selain itu, pada halaman 169 disebutkan bahwa Ngo Adrian menjabat Direktur."
    )
    cleaned = generator.clean_generated_answer(raw)
    assert "halaman" not in cleaned.lower()
    assert "329" not in cleaned
    assert "239" not in cleaned
    assert "232" not in cleaned
    assert "169" not in cleaned
    assert "Johan A.M.M. Hutauruk" in cleaned
    assert "Budi Djatmiko" in cleaned
    assert "Ngo Adrian" in cleaned

