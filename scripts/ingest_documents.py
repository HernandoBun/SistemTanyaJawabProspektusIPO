import argparse
import logging
import sys
import warnings
from pathlib import Path

# Suppress harmless Pydantic v2 metadata warnings from third-party libraries (LlamaParse/LlamaIndex)
warnings.filterwarnings("ignore", message=".*validate_default.*")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from src.services.pipeline import RAGPipeline
from scripts.rebuild_indexes import select_pdfs
from src.services.diagnostics import check_embedding_connection, describe_error

def main(argv=None):
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--documents", nargs="+", metavar="PDF")
    arguments.add_argument("--force", action="store_true")
    arguments.add_argument("--refresh-parse-cache", action="store_true")
    arguments.add_argument("--verbose", action="store_true")
    args = arguments.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    last_progress = None

    def progress(stage, fraction, message=""):
        nonlocal last_progress
        percent = int(fraction * 100)
        marker = (stage, percent // 10)
        if marker != last_progress:
            print(f"  {stage}: {percent}% — {message}", flush=True)
            last_progress = marker

    try:
        pdfs = select_pdfs(args.documents)
        if not pdfs:
            raise ValueError(f"Belum ada PDF di {config.PROSPEKTUS_DIR}")
        check_embedding_connection()
        pipeline = RAGPipeline()
        print(f"Ditemukan {len(pdfs)} dokumen.", flush=True)
        for pdf in pdfs:
            last_progress = None
            print(f"Memproses: {pdf.name}", flush=True)
            pipeline.ingest(
                pdf, force=args.force, refresh_parse_cache=args.refresh_parse_cache,
                progress_callback=progress,
            )
            print(f"[OK] Indexed {pdf.name}", flush=True)
        print("Ingestion complete.", flush=True)
        return 0
    except Exception as exc:
        print(f"[GAGAL] {describe_error(exc)}", file=sys.stderr, flush=True)
        if args.verbose:
            logging.exception("Rincian kegagalan pemrosesan")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
