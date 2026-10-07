"""Validate the evaluation dataset before running experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.evaluate_retrieval import load_jsonl


def validate(questions_path: str | Path, ground_truth_path: str | Path, relevance_unit: str = "chunk",
             *, require_reference_answers: bool = True) -> list[str]:
    errors: list[str] = []
    if relevance_unit not in {"chunk", "page"}:
        return ["relevance_unit harus chunk atau page"]
    try:
        questions = load_jsonl(questions_path)
        references = load_jsonl(ground_truth_path)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    if not questions:
        errors.append("Dataset pertanyaan kosong")
    if not references:
        errors.append("Dataset ground truth kosong")
    for label, rows in (("questions", questions), ("ground truth", references)):
        for number, row in enumerate(rows, 1):
            if not isinstance(row.get("question_id"), str) or not row["question_id"].strip():
                errors.append(f"{label} baris {number}: question_id harus teks tidak kosong")
    # Avoid hashing malformed IDs before reporting their schema errors.
    question_ids = [item["question_id"] for item in questions if isinstance(item.get("question_id"), str)]
    reference_ids = [item["question_id"] for item in references if isinstance(item.get("question_id"), str)]
    if len(question_ids) != len(set(question_ids)):
        errors.append("question_id duplikat pada questions.jsonl")
    if len(reference_ids) != len(set(reference_ids)):
        errors.append("question_id duplikat pada ground_truth.jsonl")
    if set(question_ids) != set(reference_ids):
        errors.append("question_id questions dan ground truth tidak sama")

    reference_map = {item["question_id"]: item for item in references if isinstance(item.get("question_id"), str)}
    for item in questions:
        qid = item.get("question_id", "<tanpa-id>")
        for field in ("question_id", "doc_source", "question", "category", "answerable"):
            if field not in item:
                errors.append(f"{qid}: field {field!r} tidak tersedia")
        for field in ("doc_source", "question", "category"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"{qid}: {field} harus teks tidak kosong")
        if type(item.get("answerable")) is not bool:
            errors.append(f"{qid}: answerable harus boolean true/false")
        if "paper_category" in item:
            from evaluation.paper_tables import CATEGORIES
            if not isinstance(item["paper_category"], str) or item["paper_category"] not in CATEGORIES:
                errors.append(f"{qid}: paper_category harus explicit_facts, paraphrases, numerical_tables, atau across_sections")
        reference = reference_map.get(qid, {}) if isinstance(qid, str) else {}
        if "answerable" in reference and (type(reference["answerable"]) is not bool
                or reference["answerable"] != item.get("answerable")):
            errors.append(f"{qid}: answerable pada ground truth dan pertanyaan tidak konsisten")
        pages = reference.get("source_pages", [])
        if not isinstance(pages, list) or any(type(page) is not int or page < 1 for page in pages):
            errors.append(f"{qid}: source_pages harus daftar nomor halaman PDF positif")
        elif len(pages) != len(set(pages)):
            errors.append(f"{qid}: source_pages duplikat")
        ids = reference.get("relevant_chunk_ids", [])
        valid_ids = isinstance(ids, list) and all(isinstance(value, str) and value.strip() for value in ids)
        if not valid_ids:
            errors.append(f"{qid}: relevant_chunk_ids harus daftar ID teks tidak kosong")
        elif len(ids) != len(set(ids)):
            errors.append(f"{qid}: relevant_chunk_ids duplikat")
        if item.get("answerable"):
            if require_reference_answers and (not isinstance(reference.get("answer"), str) or not reference["answer"].strip()):
                errors.append(f"{qid}: pertanyaan answerable tetapi jawaban referensi kosong")
            if not reference.get("source_pages"):
                errors.append(f"{qid}: pertanyaan answerable tetapi halaman referensi kosong")
            if relevance_unit == "chunk" and (not valid_ids or not ids):
                errors.append(f"{qid}: lengkapi relevant_chunk_ids dari anotasi manual")
        elif item.get("answerable") is False and (pages or ids):
            errors.append(f"{qid}: pertanyaan unanswerable harus memiliki sumber relevan kosong")
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
