"""Export Chapter 4 templates or recompute tables from saved rankings; no API calls."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import config
from evaluation.evaluate_retrieval import load_jsonl
from evaluation.reporting import export_paper_tables, file_fingerprint, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, help="method_comparison.json; kosong untuk template")
    parser.add_argument("--questions", type=Path, help="Dataset dengan paper_category, jika hasil lama belum memuatnya")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    directory = args.output_dir or config.BASE_DIR / "evaluation/results" / (
        "paper_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        retrieval = json.loads(args.results.read_text(encoding="utf-8-sig")) if args.results else None
        questions = load_jsonl(args.questions) if args.questions else None
        directory.mkdir(parents=True, exist_ok=False)
        tables = export_paper_tables(directory, retrieval=retrieval, questions=questions)
        write_json(directory / "export_manifest.json", {
            "results": file_fingerprint(args.results) if args.results else None,
            "questions": file_fingerprint(args.questions) if args.questions else None,
            "status": tables["status"], "api_calls": False,
        })
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(f"Tabel paper: {directory.resolve()}")
    for warning in tables["warnings"]:
        print(f"[CATATAN] {warning}")


if __name__ == "__main__":
    main()
