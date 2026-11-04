"""RAGAS scoring with OpenRouter Qwen 3.7 Flash and project BGE-M3."""
from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import config

RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def extract_retrieved_contexts(response: Any) -> list[str]:
    """Use all contexts sent to the model, rather than only selected citations."""
    context_texts = getattr(response, "generation_context_texts", None)
    if context_texts is not None:
        return [text.strip() for text in context_texts if isinstance(text, str) and text.strip()]
    contexts: list[str] = []
    chunks = getattr(response, "generation_contexts", None)
    if chunks is None:  # Compatibility with responses created before context capture.
        chunks = getattr(response, "source_chunks", [])
    for retrieved in chunks or []:
        chunk = getattr(retrieved, "chunk", retrieved)
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content.strip():
            contexts.append(content.strip())
    return contexts


def metric_result_value_reason(result: Any) -> tuple[float, str | None]:
    """Normalize RAGAS MetricResult or a numeric test double."""
    value = getattr(result, "value", result)
    reason = getattr(result, "reason", None)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("RAGAS mengembalikan skor non-finite (NaN/Infinity).")
    return value, reason


@dataclass
class RagasScore:
    score: float
    reason: str | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {"score": self.score, "reason": self.reason}


class RagasMetricError(RuntimeError):
    """Keep completed metrics when a later metric fails."""

    def __init__(self, metric: str, partial_scores: dict, error: Exception):
        self.metric = metric
        self.partial_scores = dict(partial_scores)
        details = str(error)
        if any(term in details.lower() for term in ("max_tokens", "finish_reason='length'", "output is incomplete")):
            description = "Keluaran evaluator terpotong oleh batas token."
        else:
            description = (details.splitlines() or [type(error).__name__])[0][:400]
        if config.OPENROUTER_API_KEY:
            description = description.replace(config.OPENROUTER_API_KEY, "[redacted]")
        super().__init__(f"{metric}: {type(error).__name__}: {description}")


class RagasEvaluator:
    """Score one RAG response with four reference-based RAGAS metrics.

    RAGAS imports are intentionally lazy. Unit tests and deterministic
    evaluation can therefore run without loading RAGAS or the embedding model.
    """

    def __init__(self, *, max_tokens: int | None = None, reasoning_enabled: bool = False) -> None:
        self.max_tokens = max_tokens if max_tokens is not None else config.RAGAS_EVALUATOR_MAX_TOKENS
        if self.max_tokens <= 0:
            raise ValueError("Batas token evaluator harus lebih dari nol.")
        self.reasoning_enabled = reasoning_enabled
        if not config.OPENROUTER_API_KEY or config.OPENROUTER_API_KEY.startswith("your_"):
            raise ValueError(
                "OPENROUTER_API_KEY tidak ditemukan; RAGAS memerlukan evaluator LLM."
            )

        # Disable RAGAS analytics for a quiet, reproducible local experiment.
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import BaseRagasEmbedding
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS belum terpasang. Jalankan: python -m pip install -r requirements.txt"
            ) from exc

        from src.retrieval.embedder import Embedder

        class ProjectBgeM3Embeddings(BaseRagasEmbedding):
            """Modern RAGAS adapter for the same project embedding provider."""

            def __init__(self) -> None:
                super().__init__()
                self._embedder = Embedder()

            def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
                return self._embedder.embed_texts(
                    texts, show_progress=False
                ).tolist()

            def embed_text(self, text: str, **kwargs) -> list[float]:
                return self._embedder.embed_query(text).tolist()

            async def aembed_text(self, text: str, **kwargs) -> list[float]:
                return await asyncio.to_thread(self.embed_text, text)

            async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
                return await asyncio.to_thread(self.embed_texts, texts)

        self._client = AsyncOpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            timeout=config.OPENROUTER_TIMEOUT,
            max_retries=config.OPENROUTER_MAX_RETRIES,
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "IPO Prospectus Q&A",
            },
        )
        request_options = {}
        if urlsplit(config.OPENROUTER_BASE_URL).hostname == "openrouter.ai":
            request_options["extra_body"] = {"reasoning": {"enabled": reasoning_enabled}}
        evaluator_llm = llm_factory(
            config.RAGAS_EVALUATOR_MODEL,
            provider="openai",
            client=self._client,
            temperature=config.RAGAS_EVALUATOR_TEMPERATURE,
            max_tokens=self.max_tokens,
            **request_options,
        )
        evaluator_embeddings = ProjectBgeM3Embeddings()
        # Preserve a single event loop and HTTP connection pool across every metric/sample.
        self._runner = asyncio.Runner()

        self.metrics = {
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "answer_relevancy": AnswerRelevancy(
                llm=evaluator_llm, embeddings=evaluator_embeddings
            ),
            "context_precision": ContextPrecision(llm=evaluator_llm),
            "context_recall": ContextRecall(llm=evaluator_llm),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "ragas_api": "collections-v0.4",
            "evaluator_model": config.RAGAS_EVALUATOR_MODEL,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "embedding_provider": config.EMBEDDING_PROVIDER,
            "context_source": "generation_context_texts",
            "max_tokens": self.max_tokens,
            "reasoning_enabled": self.reasoning_enabled,
        }

    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str,
        contexts: list[str],
        existing_scores: dict | None = None,
        on_metric_scored=None,
    ) -> dict[str, dict[str, float | str | None]]:
        """Return faithfulness, relevancy, precision, and recall scores."""
        arguments = dict(question=question, answer=answer, reference=reference, contexts=contexts)
        if existing_scores:
            arguments["existing_scores"] = existing_scores
        if on_metric_scored is not None:
            arguments["on_metric_scored"] = on_metric_scored
        return self._runner.run(self.ascore(**arguments))

    def close(self) -> None:
        try:
            self._runner.run(self._client.close())
        finally:
            self._runner.close()

    async def ascore(
        self, *, question: str, answer: str, reference: str, contexts: list[str],
        existing_scores: dict | None = None, on_metric_scored=None,
    ) -> dict[str, dict[str, float | str | None]]:
        if not contexts:
            raise ValueError("RAGAS tidak dapat dijalankan tanpa retrieved contexts.")

        inputs = {
            "faithfulness": dict(user_input=question, response=answer, retrieved_contexts=contexts),
            "answer_relevancy": dict(user_input=question, response=answer),
            "context_precision": dict(user_input=question, reference=reference, retrieved_contexts=contexts),
            "context_recall": dict(user_input=question, reference=reference, retrieved_contexts=contexts),
        }
        normalized = dict(existing_scores or {})
        if any(name not in RAGAS_METRIC_NAMES for name in normalized):
            raise ValueError("Checkpoint berisi nama metrik RAGAS yang tidak dikenal.")
        for item in normalized.values():
            metric_result_value_reason(item["score"])
        for name in RAGAS_METRIC_NAMES:
            if name in normalized:
                continue
            try:
                result = await self.metrics[name].ascore(**inputs[name])
                value, reason = metric_result_value_reason(result)
                normalized[name] = RagasScore(value, reason).as_dict()
            except Exception as exc:
                raise RagasMetricError(name, normalized, exc) from exc
            if on_metric_scored is not None:
                on_metric_scored(name, normalized[name])
        return normalized
