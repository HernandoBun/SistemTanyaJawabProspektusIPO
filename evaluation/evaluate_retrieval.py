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
        rows = []
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}, baris {number}: JSON tidak valid ({exc.msg})") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}, baris {number}: setiap baris harus berupa objek JSON")
            rows.append(row)
        return rows


def precision_at_k(retrieved_pages: list, relevant_pages: list, k: int) -> float:
    if k <= 0 or not retrieved_pages:
        return 0.0
    top_k = retrieved_pages[:k]
    hits = sum(1 for page in top_k if page in relevant_pages)
    return hits / k


def recall_at_k(retrieved_pages: list, relevant_pages: list, k: int) -> float:
    if k <= 0 or not relevant_pages:
        return 0.0
    return len(set(retrieved_pages[:k]) & set(relevant_pages)) / len(set(relevant_pages))


def reciprocal_rank(retrieved_pages: list, relevant_pages: list) -> float:
    relevant = set(relevant_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def rank_record(item, rank: int) -> dict:
    return {
        "rank": rank, "chunk_id": item.chunk.chunk_id,
        "page_number": item.chunk.page_number, "rrf_score": item.rrf_score,
        **{name: getattr(item, name, None) for name in
           ("dense_rank", "sparse_rank", "dense_score", "sparse_score")},
    }


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
    if not questions:
        raise ValueError("Dataset pertanyaan kosong")
    for question in questions:
        truth = ground_truth.get(question["question_id"])
        if truth is None:
            raise ValueError(f"Ground truth tidak ditemukan untuk {question['question_id']}")
        if relevance_unit == "chunk" and question.get("answerable", truth.get("answerable", True)) and not truth.get("relevant_chunk_ids"):
            raise ValueError(f"{question['question_id']}: lengkapi relevant_chunk_ids untuk evaluasi chunk sesuai proposal.")
    if pipeline is None:
        from src.services.pipeline import RAGPipeline

        pipeline = RAGPipeline()
    methods = methods or {
        "sparse_bm25": {"dense_k": 0, "sparse_k": 10, "final_k": 6},
        "dense_bge_m3": {"dense_k": 10, "sparse_k": 0, "final_k": 6},
        "hybrid_rrf": {"dense_k": 10, "sparse_k": 10, "final_k": 6},
    }

    all_results: dict[str, list[dict]] = {name: [] for name in methods}
    canonical = {(0, 10, 6): "bm25_only", (10, 0, 6): "dense_only", (10, 10, 6): "hybrid_rrf"}
    method_keys = {name: canonical.get((params.get("dense_k"), params.get("sparse_k"),
                                      params.get("final_k"))) for name, params in methods.items()}
    # Inspect the class to avoid treating dynamically generated Mock attributes as an API.
    shared_pools = callable(getattr(type(pipeline), "retrieve_comparison", None)) and all(method_keys.values())
    for question in questions:
        pipeline.set_active_document(question["doc_source"])
        comparison = pipeline.retrieve_comparison(question["question"]) if shared_pools else None
        for method_name, parameters in methods.items():
            qid = question["question_id"]
            truth = ground_truth.get(qid)
            if truth is None:
                raise ValueError(f"Ground truth tidak ditemukan untuk {qid}")

            answerable = bool(question.get("answerable", truth.get("answerable", True)))
            expected_pages = truth.get("source_pages", [])
            if relevance_unit == "chunk" and answerable and not truth.get("relevant_chunk_ids"):
                raise ValueError(f"{qid}: lengkapi relevant_chunk_ids untuk evaluasi chunk sesuai proposal.")
            chunks = (comparison["rankings"][method_keys[method_name]] if comparison is not None
                      else pipeline.retrieve(question["question"], **parameters))
            pages = [item.chunk.page_number for item in chunks]
            retrieved_ids = [item.chunk.chunk_id for item in chunks]
            ranked = retrieved_ids if relevance_unit == "chunk" else pages
            # A page/chunk is one relevance unit even if it appears multiple times.
            ranked = list(dict.fromkeys(ranked))
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
                    **{f"mrr_at_{k}": reciprocal_rank(ranked[:k], expected) for k in (1, 3, 6)},
                }
            else:
                row = {
                    "question_id": qid,
                    "doc_source": question["doc_source"],
                    "category": question.get("category", ""),
                    "retrieved_pages": pages,
                    "retrieved_chunk_ids": retrieved_ids,
                }
                # The 0.02 cutoff belongs to hybrid RRF, not a single ranking.
                if parameters.get("dense_k", 0) and parameters.get("sparse_k", 0):
                    row["unanswerable_rejection"] = float(
                        not chunks or max(item.rrf_score for item in chunks) < config.MIN_RRF_SCORE
                    )
            row.update({
                "question": question["question"],
                "answerable": answerable,
                "paper_category": question.get("paper_category"),
                "paper_category_status": question.get("paper_category_status"),
                "annotation_status": truth.get("annotation_status"),
                "expected_pages": expected_pages,
                "relevant_chunk_ids": truth.get("relevant_chunk_ids", []),
                "ranked_relevance_units": ranked,
                "retrieved_count": len(chunks),
                "ranking": [rank_record(item, rank) for rank, item in enumerate(chunks, 1)],
            })
            if comparison is not None:
                row["search_query"] = comparison["search_query"]
                row["candidate_rankings"] = {
                    name: [rank_record(item, rank) for rank, item in enumerate(candidates, 1)]
                    for name, candidates in comparison["candidate_rankings"].items()
                }
            all_results[method_name].append(row)

    summary = summarize(all_results)
    payload = {"relevance_unit": relevance_unit, "methods": methods,
               "summary": summary, "results": all_results,
               "protocol": {"cutoffs": [1, 3, 6], "shared_candidate_pools": bool(shared_pools),
                            "rrf_k": config.RRF_K, "answer_generation": False,
                            "llm_relevance_grading": False}}
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
            "n_answerable": len(answerable),
            "n_unanswerable": len(rows) - len(answerable),
            **{name: mean(row[name] for row in answerable) if answerable else None
               for name in ("precision_at_1", "precision_at_3", "precision_at_6",
                            "recall_at_1", "recall_at_3", "recall_at_6",
                            "mrr_at_1", "mrr_at_3", "mrr_at_6", "mrr")},
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
