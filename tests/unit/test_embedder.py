from unittest.mock import MagicMock, Mock
import numpy as np
import pytest
import config
from src.retrieval import embedder

def test_embedder_initializes_openrouter(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "openrouter")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(index=0, embedding=[0.1]*1024)]
    mock_client.embeddings.create.return_value = mock_resp

    monkeypatch.setattr(embedder, "_get_openai_client", lambda: mock_client)

    emb = embedder.Embedder(provider="openrouter")
    assert emb.provider == "openrouter"
    assert emb.embedding_dim == 1024

    vecs = emb.embed_texts(["contoh teks"])
    assert vecs.shape == (1, 1024)
    # Check L2 normalization
    norm = np.linalg.norm(vecs[0])
    assert pytest.approx(norm, 0.001) == 1.0


def test_embed_query_sends_raw_bge_m3_query_without_prefix(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(index=0, embedding=[0.1] * 1024)]
    mock_client.embeddings.create.return_value = mock_resp
    monkeypatch.setattr(embedder, "_get_openai_client", lambda: mock_client)

    emb = embedder.Embedder(provider="openrouter")
    emb.embed_query("  Berapa total aset tahun 2025?  ")

    mock_client.embeddings.create.assert_called_once_with(
        model=emb.model_name,
        input=["Berapa total aset tahun 2025?"],
    )


def test_oversized_embedding_text_keeps_head_and_tail(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_MAX_INPUT_CHARS", 100)
    text = "A" * 100 + "Z" * 100

    prepared = embedder._prepare_embedding_text(text)

    assert len(prepared) == 100
    assert prepared.startswith("A")
    assert prepared.endswith("Z")
    assert "dipotong hanya untuk embedding" in prepared

def test_embedder_does_not_change_provider_if_no_key(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "openrouter")

    mock_local = MagicMock()
    mock_local.get_sentence_embedding_dimension.return_value = 1024
    mock_local.encode.return_value = np.zeros((1, 1024), dtype=np.float32)
    monkeypatch.setattr(embedder, "_get_local_model", lambda: mock_local)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        embedder.Embedder(provider="openrouter")
    mock_local.encode.assert_not_called()


def test_cached_workflow_does_not_require_http_client_initialization(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    factory = Mock(side_effect=ImportError("Optional HTTP proxy dependency unavailable"))
    monkeypatch.setattr(embedder, "_get_openai_client", factory)
    emb = embedder.Embedder(provider="openrouter")
    assert emb.embed_texts([]).shape == (0, 1024)
    factory.assert_not_called()
    with pytest.raises(ImportError, match="proxy dependency"):
        emb.embed_query("Berapa total aset?")
    factory.assert_called_once_with()


def test_embedding_requests_share_one_lazily_initialized_client(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "test-key")
    client = MagicMock()
    client.embeddings.create.return_value.data = [MagicMock(index=0, embedding=[.1] * 1024)]
    factory = Mock(return_value=client)
    monkeypatch.setattr(embedder, "_get_openai_client", factory)
    emb = embedder.Embedder(provider="openrouter")
    emb.embed_query("Pertanyaan pertama")
    emb.embed_query("Pertanyaan kedua")
    factory.assert_called_once_with()
    assert client.embeddings.create.call_count == 2
