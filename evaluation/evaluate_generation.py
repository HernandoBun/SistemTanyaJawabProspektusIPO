"""Metrics and runner for generated answers."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from evaluation.evaluate_retrieval import load_jsonl
from evaluation.ragas_evaluator import RAGAS_METRIC_NAMES, extract_retrieved_contexts

if TYPE_CHECKING:
    from src.services.pipeline import RAGPipeline


def normalize_answer(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[^\w%]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def parse_indonesian_number(value: str) -> float:
    cleaned = re.sub(r"[^\d,.-]", "", value)
    if not cleaned:
        raise ValueError(f"Bukan angka: {value!r}")
    if "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif cleaned.count(".") > 1 or re.search(r"\.\d{3}$", cleaned):
        cleaned = cleaned.replace(".", "")
    return float(cleaned)


def extract_numbers(text: str) -> list[float]:
    matches = re.findall(r"(?<!\w)-?\d+(?:\.\d{3})*(?:,\d+)?%?", text)
    return [parse_indonesian_number(match) for match in matches]


def numerical_accuracy(prediction: str, reference: str, tolerance: float = 0.05) -> float:
    predicted = extract_numbers(prediction)
    expected = extract_numbers(reference)
    if not expected:
        return 1.0
    if not predicted:
        return 0.0
    matched = 0
    for expected_number in expected:
        # Years are categorical values, not continuous financial quantities.
        # Applying a 5% tolerance would incorrectly treat 2025 as matching 2024.
        is_year = expected_number.is_integer() and 1900 <= expected_number <= 2100
        if is_year:
            is_match = any(number == expected_number for number in predicted)
        else:
            denominator = max(abs(expected_number), 1.0)
            is_match = any(
                abs(number - expected_number) / denominator <= tolerance
                for number in predicted
            )
        if is_match:
            matched += 1
    return matched / len(expected)


def citation_page_accuracy(predicted_pages: list[int], expected_pages: list[int]) -> float:
    if not expected_pages:
        return float(not predicted_pages)
    return float(bool(set(predicted_pages) & set(expected_pages)))


def evaluate(
    questions_path: str | Path,
    ground_truth_path: str | Path,
    *,
    output_path: str | Path | None = None,
    pipeline: "RAGPipeline | None" = None,
    use_ragas: bool = False,
    continue_on_ragas_error: bool = False,
    ragas_evaluator=None,
) -> dict:
    questions = load_jsonl(questions_path)
    truth = {row["question_id"]: row for row in load_jsonl(ground_truth_path)}
    if not questions:
        raise ValueError("Dataset pertanyaan kosong")
    if pipeline is None:
        from src.services.pipeline import RAGPipeline

        pipeline = RAGPipeline()
    rows = []
    if use_ragas and ragas_evaluator is None:
        from evaluation.ragas_evaluator import RagasEvaluator

        ragas_evaluator = RagasEvaluator()

    failure = None
    metadata = ragas_evaluator.metadata if ragas_evaluator is not None else None
    try:
        for question in questions:
            reference = truth[question["question_id"]]
            row = {
                "question_id": question["question_id"],
                "doc_source": question["doc_source"],
                "question": question["question"],
                "category": question.get("category", ""),
                "answerable": question.get("answerable", reference.get("answerable", True)),
                "reference": reference.get("answer", ""),
                "expected_pages": reference.get("source_pages", []),
            }
            try:
                pipeline.set_active_document(question["doc_source"])
                response = pipeline.query(question["question"])
            except Exception as exc:
                row.update(status="generation_error", error=str(exc))
                rows.append(row)
                failure = exc
                break
            contexts = extract_retrieved_contexts(response)
            row.update({
                "status": "generated",
                "prediction": response.answer,
                "source_pages": response.source_pages,
                "retrieved_contexts": contexts,
                "context_count": len(contexts),
                "model_used": getattr(response, "model_used", ""),
                "tokens_used": getattr(response, "tokens_used", 0),
                "exact_match": exact_match(response.answer, reference.get("answer", "")),
                "token_f1": token_f1(response.answer, reference.get("answer", "")),
                "numerical_accuracy": numerical_accuracy(response.answer, reference.get("answer", "")),
                "citation_page_accuracy": citation_page_accuracy(
                    response.source_pages, reference.get("source_pages", [])
                ),
            })
            if ragas_evaluator is not None:
                if not row["answerable"]:
                    row["ragas_status"] = "skipped_unanswerable"
                elif not contexts:
                    row["ragas_status"] = "skipped_no_context"
                else:
                    try:
                        row["ragas"] = ragas_evaluator.score(
                            question=question["question"],
                            answer=response.answer,
                            reference=reference.get("answer", ""),
                            contexts=contexts,
                        )
                        row["ragas_status"] = "scored"
                    except Exception as exc:
                        row["ragas_status"] = "error"
                        row["ragas_error"] = str(exc)
                        if not continue_on_ragas_error:
                            failure = exc
            rows.append(row)
            if failure is not None:
                break
    finally:
        if ragas_evaluator is not None:
            try:
                ragas_evaluator.close()
            except Exception as exc:
                if failure is None:
                    failure = exc

    metric_names = ("exact_match", "token_f1", "numerical_accuracy", "citation_page_accuracy")
    generated = [row for row in rows if row["status"] == "generated"]
    summary = {name: mean(row[name] for row in generated) if generated else None for name in metric_names}
    summary["n"] = len(rows)
    summary["n_expected"] = len(questions)
    summary["n_generated"] = len(generated)
    summary["n_generation_errors"] = len(rows) - len(generated)
    payload = {"status": "failed" if failure is not None else "completed",
               "method": "hybrid_rrf", "summary": summary, "results": rows}
    if ragas_evaluator is not None:
        scored_rows = [row for row in rows if row.get("ragas_status") == "scored"]
        summary["ragas_n_scored"] = len(scored_rows)
        summary["ragas_n_skipped"] = sum(
            row.get("ragas_status", "").startswith("skipped_") for row in rows
        )
        summary["ragas_n_no_context"] = sum(row.get("ragas_status") == "skipped_no_context" for row in rows)
        summary["ragas_n_unanswerable"] = sum(row.get("ragas_status") == "skipped_unanswerable" for row in rows)
        summary["ragas_n_errors"] = sum(
            row.get("ragas_status") == "error" for row in rows
        )
        for name in RAGAS_METRIC_NAMES:
            summary[f"ragas_{name}"] = (
                mean(row["ragas"][name]["score"] for row in scored_rows)
                if scored_rows
                else None
            )
        payload["ragas_metadata"] = metadata
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if failure is not None:
        raise RuntimeError("Evaluasi generasi/RAGAS gagal; hasil yang sudah selesai disimpan jika output_path tersedia.") from failure
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/evaluation/questions.jsonl")
    parser.add_argument("--ground-truth", default="data/evaluation/ground_truth.jsonl")
    parser.add_argument("--output", default="evaluation/results/generation.json")
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Tambahkan Faithfulness, Answer Relevancy, Context Precision, dan Context Recall",
    )
    parser.add_argument(
        "--continue-on-ragas-error",
        action="store_true",
        help="Catat error per pertanyaan dan lanjutkan evaluasi",
    )
    args = parser.parse_args()
    result = evaluate(
        args.questions,
        args.ground_truth,
        output_path=args.output,
        use_ragas=args.with_ragas,
        continue_on_ragas_error=args.continue_on_ragas_error,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
