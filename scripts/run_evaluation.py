"""Run reproducible retrieval/RAGAS experiments separately from the application."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import config

from evaluation.compare_methods import METHODS
from evaluation.evaluate_generation import evaluate as evaluate_generation
from evaluation.evaluate_retrieval import evaluate as evaluate_retrieval
from evaluation.preflight import check_readiness
from evaluation.reporting import experiment_metadata, export_reports, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["retrieval", "ragas", "all"], default="retrieval")
    parser.add_argument("--questions", type=Path, default=config.BASE_DIR / "data/evaluation/questions.jsonl")
    parser.add_argument("--ground-truth", type=Path, default=config.BASE_DIR / "data/evaluation/ground_truth.jsonl")
    parser.add_argument("--output-dir", type=Path, help="Folder baru untuk satu eksperimen; tidak menimpa hasil lama")
    parser.add_argument("--check-only", action="store_true", help="Periksa kesiapan tanpa memanggil API")
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Jalankan evaluasi generasi sekaligus empat metrik RAGAS",
    )
    parser.add_argument("--continue-on-ragas-error", action="store_true")
    parser.add_argument("--relevance-unit", choices=["chunk", "page"], default="chunk")
    args = parser.parse_args()
    stage = "all" if args.with_ragas else args.stage
    directory = args.output_dir or config.BASE_DIR / "evaluation/results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.exit(1, f"Folder hasil sudah ada; gunakan folder baru: {directory}\n")
    readiness = check_readiness(args.questions, args.ground_truth, stage=stage, relevance_unit=args.relevance_unit)
    write_json(directory / "readiness.json", readiness)
    print(f"Folder hasil: {directory.resolve()}", flush=True)
    for warning in readiness["warnings"]:
        print(f"[CATATAN] {warning}", flush=True)
    if readiness["errors"]:
        parser.exit(1, "Evaluasi belum siap:\n- " + "\n- ".join(readiness["errors"]) + "\n")
    if args.check_only:
        print("Pemeriksaan lokal lulus. Koneksi API dan kualitas jawaban belum diuji.")
        return
    manifest = experiment_metadata(args.questions, args.ground_truth)
    manifest.update(stage=stage, relevance_unit=args.relevance_unit, status="running", methods=METHODS)
    write_json(directory / "manifest.json", manifest)
    generation = None
    try:
        from src.services.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        if not pipeline.is_ready():
            raise RuntimeError("Pipeline belum siap; periksa indeks dokumen terlebih dahulu")
        if stage in {"retrieval", "all"}:
            print("Menjalankan BM25-only, dense-only, dan hybrid RRF...", flush=True)
            retrieval = evaluate_retrieval(args.questions, args.ground_truth, methods=METHODS,
                output_path=directory / "method_comparison.json", relevance_unit=args.relevance_unit, pipeline=pipeline)
            export_reports(directory, retrieval=retrieval)
            manifest["retrieval_protocol"] = retrieval["protocol"]
            if args.relevance_unit == "chunk":
                tables = json.loads((directory / "paper_tables.json").read_text(encoding="utf-8"))
                for warning in tables["warnings"]:
                    print(f"[CATATAN] {warning}", flush=True)
            print(json.dumps(retrieval["summary"], ensure_ascii=False, indent=2), flush=True)
        if stage in {"ragas", "all"} or args.with_generation:
            print("Menjalankan evaluasi jawaban hybrid...", flush=True)
            generation = evaluate_generation(args.questions, args.ground_truth,
                output_path=directory / "generation.json", use_ragas=stage in {"ragas", "all"},
                continue_on_ragas_error=args.continue_on_ragas_error, pipeline=pipeline)
            export_reports(directory, generation=generation)
            print(json.dumps(generation["summary"], ensure_ascii=False, indent=2), flush=True)
        manifest["status"] = "completed"
        if generation and generation["summary"].get("ragas_n_errors", 0):
            manifest["status"] = "completed_with_errors"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        if (directory / "generation.json").is_file():
            generation = json.loads((directory / "generation.json").read_text(encoding="utf-8"))
            export_reports(directory, generation=generation)
        parser.exit(1, f"[GAGAL] {exc}\nLihat hasil dan manifest di {directory.resolve()}\n")
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)
    print("Evaluasi selesai. JSON dan CSV tersimpan pada folder hasil.")


if __name__ == "__main__":
    main()
