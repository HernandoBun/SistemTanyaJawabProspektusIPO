"""RAGAS scoring with OpenRouter Qwen 3.7 Flash and project BGE-M3."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import config

RAGAS_METRIC_NAMES = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def extract_retrieved_contexts(response: Any) -> list[str]:
    """Extract clean chunk text from a GeneratorResponse-like object."""
    contexts: list[str] = []
    for retrieved in getattr(response, "source_chunks", []) or []:
        chunk = getattr(retrieved, "chunk", retrieved)
        content = getattr(chunk, "content", "")
        if isinstance(content, str) and content.strip():
            contexts.append(content.strip())
    return contexts


def metric_result_value_reason(result: Any) -> tuple[float, str | None]:
    """Normalize RAGAS MetricResult or a numeric test double."""
    value = getattr(result, "value", result)
    reason = getattr(result, "reason", None)
    return float(value), reason


@dataclass
class RagasScore:
    score: float
    reason: str | None = None

    def as_dict(self) -> dict[str, float | str | None]:
        return {"score": self.score, "reason": self.reason}


class RagasEvaluator:
    """Score one RAG response with four reference-based RAGAS metrics.

    RAGAS imports are intentionally lazy. Unit tests and deterministic
    evaluation can therefore run without loading RAGAS or the embedding model.
    """

    def __init__(self) -> None:
        if not config.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY tidak ditemukan; RAGAS memerlukan evaluator LLM."
            )

        # Disable RAGAS analytics for a quiet, reproducible local experiment.
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

        try:
            from langchain_core.embeddings import Embeddings
            from langchain_openai import ChatOpenAI
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
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

        class ProjectBgeM3Embeddings(Embeddings):
            """LangChain adapter that reuses the exact project embedding model."""

            def __init__(self) -> None:
                self._embedder = Embedder()

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return self._embedder.embed_texts(
                    texts, show_progress=False
                ).tolist()

            def embed_query(self, text: str) -> list[float]:
                return self._embedder.embed_query(text).tolist()

        judge = ChatOpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            model=config.RAGAS_EVALUATOR_MODEL,
            temperature=config.RAGAS_EVALUATOR_TEMPERATURE,
            max_tokens=config.RAGAS_EVALUATOR_MAX_TOKENS,
            extra_body={"reasoning": {"effort": "none"}},
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "IPO Prospectus Q&A",
            },
        )
        evaluator_llm = LangchainLLMWrapper(judge)
        evaluator_embeddings = LangchainEmbeddingsWrapper(ProjectBgeM3Embeddings())

        self.metrics = {
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "answer_relevancy": AnswerRelevancy(
                llm=evaluator_llm, embeddings=evaluator_embeddings
            ),
            "context_precision": ContextPrecision(llm=evaluator_llm),
            "context_recall": ContextRecall(llm=evaluator_llm),
        }

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "ragas_api": "collections-v0.4",
            "evaluator_model": config.RAGAS_EVALUATOR_MODEL,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
        }

    def score(
        self,
        *,
        question: str,
        answer: str,
        reference: str,
        contexts: list[str],
    ) -> dict[str, dict[str, float | str | None]]:
        """Return faithfulness, relevancy, precision, and recall scores."""
        if not contexts:
            raise ValueError("RAGAS tidak dapat dijalankan tanpa retrieved contexts.")

        raw_results = {
            "faithfulness": self.metrics["faithfulness"].score(
                response=answer,
                retrieved_contexts=contexts,
            ),
            "answer_relevancy": self.metrics["answer_relevancy"].score(
                user_input=question,
                response=answer,
            ),
            "context_precision": self.metrics["context_precision"].score(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
            "context_recall": self.metrics["context_recall"].score(
                user_input=question,
                reference=reference,
                retrieved_contexts=contexts,
            ),
        }

        normalized = {}
        for name, result in raw_results.items():
            value, reason = metric_result_value_reason(result)
            normalized[name] = RagasScore(value, reason).as_dict()
        return normalized
