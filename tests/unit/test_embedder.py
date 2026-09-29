from unittest.mock import MagicMock
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
