"""Evaluate retrieval independently from answer generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING
import config

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if TYPE_CHECKING:
    from src.services.pipeline import RAGPipeline


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def precision_at_k(retrieved_pages: list[int], relevant_pages: list[int], k: int) -> float:
    if k <= 0 or not retrieved_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    hits = sum(1 for page in top_k if page in relevant_pages)
    return hits / k


def recall_at_k(retrieved_pages: list[int], relevant_pages: list[int], k: int) -> float:
    if not relevant_pages:
        return 0.0
    return len(set(retrieved_pages[:k]) & set(relevant_pages)) / len(set(relevant_pages))


def reciprocal_rank(retrieved_pages: list[int], relevant_pages: list[int]) -> float:
    relevant = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def evaluate(
    questions_path: str | Path,
    ground_truth_path: str | Path,
    *,
    methods: dict[str, dict[str, int]] | None = None,
    output_path: str | Path | None = None,
    pipeline: "RAGPipeline | None" = None,
    relevance_unit: str = "page",
) -> dict:
    if relevance_unit not in {"page", "chunk"}:
        raise ValueError("relevance_unit harus page atau chunk")
    questions = load_jsonl(questions_path)
    ground_truth = {row["question_id"]: row for row in load_jsonl(ground_truth_path)}
    if pipeline is None:
        from src.services.pipeline import RAGPipeline

        pipeline = RAGPipeline()
    methods = methods or {
        "sparse_bm25": {"dense_k": 0, "sparse_k": 10, "final_k": 6},
        "dense_bge_m3": {"dense_k": 10, "sparse_k": 0, "final_k": 6},
        "hybrid_rrf": {"dense_k": 10, "sparse_k": 10, "final_k": 6},
    }

    all_results: dict[str, list[dict]] = {}
    for method_name, parameters in methods.items():
        rows: list[dict] = []
        for question in questions:
            qid = question["question_id"]
            truth = ground_truth.get(qid)
            if truth is None:
                raise ValueError(f"Ground truth tidak ditemukan untuk {qid}")

            answerable = bool(question.get("answerable", truth.get("answerable", True)))
            expected_pages = truth.get("source_pages", [])
            if relevance_unit == "chunk" and answerable and not truth.get("relevant_chunk_ids"):
                raise ValueError(f"{qid}: lengkapi relevant_chunk_ids untuk evaluasi chunk sesuai proposal.")
            pipeline.set_active_document(question["doc_source"])
            chunks = pipeline.retrieve(question["question"], **parameters)
            pages = [item.chunk.page_number for item in chunks]
            retrieved_ids = [item.chunk.chunk_id for item in chunks]
            ranked = retrieved_ids if relevance_unit == "chunk" else pages
            expected = truth.get("relevant_chunk_ids", []) if relevance_unit == "chunk" else expected_pages

            if answerable:
                row = {
                    "question_id": qid,
                    "doc_source": question["doc_source"],
                    "category": question.get("category", ""),
                    "retrieved_pages": pages,
                    "retrieved_chunk_ids": retrieved_ids,
                    "precision_at_1": precision_at_k(ranked, expected, 1),
                    "precision_at_3": precision_at_k(ranked, expected, 3),
                    "precision_at_6": precision_at_k(ranked, expected, 6),
                    "recall_at_1": recall_at_k(ranked, expected, 1),
                    "recall_at_3": recall_at_k(ranked, expected, 3),
                    "recall_at_6": recall_at_k(ranked, expected, 6),
                    "mrr": reciprocal_rank(ranked, expected),
                }
            else:
                row = {
                    "question_id": qid,
                    "doc_source": question["doc_source"],
                    "category": question.get("category", ""),
                    "retrieved_pages": pages,
                }
                # The 0.02 cutoff belongs to hybrid RRF, not a single ranking.
                if parameters.get("dense_k", 0) and parameters.get("sparse_k", 0):
                    row["unanswerable_rejection"] = float(
                        not chunks or max(item.rrf_score for item in chunks) < config.MIN_RRF_SCORE
                    )
            rows.append(row)
        all_results[method_name] = rows

    summary = summarize(all_results)
    payload = {"relevance_unit": relevance_unit, "summary": summary, "results": all_results}
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def summarize(results: dict[str, list[dict]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for method, rows in results.items():
        answerable = [row for row in rows if "mrr" in row]
        rejected = [row for row in rows if "unanswerable_rejection" in row]
        summary[method] = {
            "n": len(rows),
            "precision_at_1": mean(row["precision_at_1"] for row in answerable) if answerable else 0.0,
            "precision_at_3": mean(row["precision_at_3"] for row in answerable) if answerable else 0.0,
            "precision_at_6": mean(row["precision_at_6"] for row in answerable) if answerable else 0.0,
            "recall_at_1": mean(row["recall_at_1"] for row in answerable) if answerable else 0.0,
            "recall_at_3": mean(row["recall_at_3"] for row in answerable) if answerable else 0.0,
            "recall_at_6": mean(row["recall_at_6"] for row in answerable) if answerable else 0.0,
            "mrr": mean(row["mrr"] for row in answerable) if answerable else 0.0,
            "unanswerable_rejection": mean(row["unanswerable_rejection"] for row in rejected) if rejected else None,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/evaluation/questions.jsonl")
    parser.add_argument("--ground-truth", default="data/evaluation/ground_truth.jsonl")
    parser.add_argument("--output", default="evaluation/results/retrieval.json")
    parser.add_argument("--relevance-unit", choices=["chunk", "page"], default="chunk")
    args = parser.parse_args()
    result = evaluate(args.questions, args.ground_truth, output_path=args.output, relevance_unit=args.relevance_unit)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
