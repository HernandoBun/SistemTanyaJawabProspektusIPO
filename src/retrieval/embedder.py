"""
src/retrieval/embedder.py — Embedding Model Wrapper

Mendukung BAAI/bge-m3 via:
1. OpenRouter API (baai/bge-m3) — Cepat, tanpa beban komputasi lokal / CPU.
2. Local SentenceTransformer (BAAI/bge-m3) — Offline fallback.

Fitur:
- Dimensi 1024
- Normalisasi L2 untuk Cosine Similarity
- Query BGE-M3 tanpa instruction prefix
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# Singleton instances
_openai_client = None
_local_model_instance = None


def _prepare_embedding_text(text: str, limit_override: Optional[int] = None) -> str:
    """Bound model input while preserving both ends of oversized tables."""
    clean_text = text if text.strip() else " "
    limit = limit_override or config.EMBEDDING_MAX_INPUT_CHARS
    if len(clean_text) <= limit:
        return clean_text

    marker = "\n\n[... bagian tengah dipotong hanya untuk embedding ...]\n\n"
    available = limit - len(marker)
    if available < 2:
        return clean_text[:limit]
    head_length = (available * 2) // 3
    tail_length = available - head_length
    logger.warning(
        "Embedding input truncated from %d to %d characters; stored chunk remains complete.",
        len(clean_text),
        limit,
    )
    return (
        clean_text[:head_length]
        + marker
        + clean_text[-tail_length:]
    )


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Apply L2 normalization so cosine similarity equals dot product."""
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm > 0 else vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.where(norms > 0, vectors / norms, vectors)


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI

            _openai_client = OpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
                timeout=config.OPENROUTER_TIMEOUT,
                max_retries=config.OPENROUTER_MAX_RETRIES,
                default_headers={
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "IPO Prospectus Q&A",
                },
            )
        except ImportError as exc:
            raise RuntimeError(
                "Paket 'openai' diperlukan untuk embedding OpenRouter. "
                "Jalankan: pip install openai"
            ) from exc
    return _openai_client


def _get_local_model():
    global _local_model_instance
    if _local_model_instance is None:
        logger.info(f"Loading local embedding model: {config.EMBEDDING_MODEL_NAME}")
        from sentence_transformers import SentenceTransformer

        _local_model_instance = SentenceTransformer(
            config.EMBEDDING_MODEL_NAME,
            trust_remote_code=True,
        )
        logger.info("Local embedding model loaded successfully.")
    return _local_model_instance


class Embedder:
    """
    Wrapper around BAAI/bge-m3 for encoding text chunks into dense 1024-d vectors.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or config.EMBEDDING_PROVIDER).lower()
        self.embedding_dim = 1024
        if self.provider not in {"openrouter", "local"}:
            raise ValueError("EMBEDDING_PROVIDER harus openrouter atau local.")

        if self.provider == "openrouter":
            if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY.startswith("your_"):
                raise ValueError("Isi OPENROUTER_API_KEY untuk embedding BGE-M3. Provider tidak diganti otomatis.")
            else:
                self.client = _get_openai_client()
                self.model_name = config.OPENROUTER_EMBEDDING_MODEL
                logger.info(
                    f"Embedder initialized with OpenRouter: {self.model_name} (dim: {self.embedding_dim})"
                )

        if self.provider == "local":
            self.model = _get_local_model()
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(
                f"Embedder initialized locally: {config.EMBEDDING_MODEL_NAME} (dim: {self.embedding_dim})"
            )

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
        progress_callback=None,
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        if batch_size <= 0:
            raise ValueError("Ukuran batch embedding harus positif.")

        logger.info(
            f"Encoding {len(texts)} texts via {self.provider.upper()} ({getattr(self, 'model_name', config.EMBEDDING_MODEL_NAME)})..."
        )

        if self.provider == "openrouter":
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                sanitized_batch = [_prepare_embedding_text(t) for t in batch]
                try:
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=sanitized_batch,
                    )
                except Exception as exc:
                    err_msg = str(exc).lower()
                    if "context length" in err_msg or "input_tokens" in err_msg or "8192" in err_msg or "400" in err_msg:
                        logger.warning("Embedding batch exceeded token limit, retrying with tighter truncation (4000 chars)...")
                        tighter_batch = [_prepare_embedding_text(t, limit_override=4000) for t in batch]
                        response = self.client.embeddings.create(
                            model=self.model_name,
                            input=tighter_batch,
                        )
                    else:
                        raise
                # Sort by index in case API returns out of order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                if [item.index for item in sorted_data] != list(range(len(batch))):
                    raise ValueError("Respons embedding tidak lengkap atau urutannya tidak valid.")
                all_embeddings.extend([item.embedding for item in sorted_data])
                if progress_callback:
                    progress_callback(min(i + batch_size, len(texts)), len(texts))

            raw_vectors = np.array(all_embeddings, dtype=np.float32)
            if raw_vectors.shape != (len(texts), self.embedding_dim) or not np.isfinite(raw_vectors).all():
                raise ValueError("Embedding harus berisi vektor numerik 1024 dimensi untuk setiap chunk.")
            normalized_vectors = _l2_normalize(raw_vectors)
            logger.info(f"Encoding complete. Shape: {normalized_vectors.shape}")
            return normalized_vectors

        # Local SentenceTransformer encoding
        embeddings = self.model.encode(
            [_prepare_embedding_text(text) for text in texts],
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        logger.info(f"Encoding complete. Shape: {embeddings.shape}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string without an instruction prefix.

        BGE-M3 no longer requires adding retrieval instructions to queries.
        Query and passage embeddings therefore use their original text.
        """
        clean_query = _prepare_embedding_text(query.strip())

        if self.provider == "openrouter":
            response = self.client.embeddings.create(
                model=self.model_name,
                input=[clean_query],
            )
            raw_vec = np.array(response.data[0].embedding, dtype=np.float32)
            return _l2_normalize(raw_vec)

        embedding = self.model.encode(
            clean_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding
