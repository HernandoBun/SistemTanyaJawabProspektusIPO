"""Validate the evaluation dataset before running experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_retrieval import load_jsonl


def validate(questions_path: str | Path, ground_truth_path: str | Path, relevance_unit: str = "chunk") -> list[str]:
    questions = load_jsonl(questions_path)
    references = load_jsonl(ground_truth_path)
    errors: list[str] = []
    if not questions:
        errors.append("Dataset pertanyaan kosong")
    question_ids = [item.get("question_id") for item in questions]
    reference_ids = [item.get("question_id") for item in references]
    if len(question_ids) != len(set(question_ids)):
        errors.append("question_id duplikat pada questions.jsonl")
    if len(reference_ids) != len(set(reference_ids)):
        errors.append("question_id duplikat pada ground_truth.jsonl")
    if set(question_ids) != set(reference_ids):
        errors.append("question_id questions dan ground truth tidak sama")

    reference_map = {item.get("question_id"): item for item in references}
    for item in questions:
        qid = item.get("question_id", "<tanpa-id>")
        for field in ("question_id", "doc_source", "question", "category", "answerable"):
            if field not in item:
                errors.append(f"{qid}: field {field!r} tidak tersedia")
        if type(item.get("answerable")) is not bool:
            errors.append(f"{qid}: answerable harus boolean true/false")
        reference = reference_map.get(qid, {})
        if item.get("answerable"):
            if not reference.get("answer"):
                errors.append(f"{qid}: pertanyaan answerable tetapi jawaban referensi kosong")
            if not reference.get("source_pages"):
                errors.append(f"{qid}: pertanyaan answerable tetapi halaman referensi kosong")
            ids = reference.get("relevant_chunk_ids")
            if relevance_unit == "chunk" and (not isinstance(ids, list) or not ids
                    or any(not isinstance(value, str) or not value.strip() for value in ids)):
                errors.append(f"{qid}: lengkapi relevant_chunk_ids dari anotasi manual")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/evaluation/questions.jsonl")
    parser.add_argument("--ground-truth", default="data/evaluation/ground_truth.jsonl")
    parser.add_argument("--relevance-unit", choices=["chunk", "page"], default="chunk")
    args = parser.parse_args()
    errors = validate(args.questions, args.ground_truth, args.relevance_unit)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        raise SystemExit(1)
    print("Dataset evaluasi valid.")


if __name__ == "__main__":
    main()
