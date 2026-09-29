from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def backup_existing() -> Path:
    backup_dir = config.DATA_DIR / "backups" / datetime.now().strftime("indexes_%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=False)
    if config.CHROMA_DIR.exists():
        shutil.move(str(config.CHROMA_DIR), str(backup_dir / "chroma_db"))
    if config.BM25_INDEX_PATH.exists():
        shutil.move(str(config.BM25_INDEX_PATH), str(backup_dir / "bm25_index.pkl"))
    if config.CACHE_REGISTRY_PATH.exists():
        shutil.copy2(config.CACHE_REGISTRY_PATH, backup_dir / "index_registry.json")
        config.CACHE_REGISTRY_PATH.unlink()
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return backup_dir


def select_pdfs(document_names: list[str] | None = None) -> list[Path]:
    available = {path.name: path for path in config.PROSPEKTUS_DIR.glob("*.pdf")}
    if not document_names:
        return sorted(available.values())
    requested = list(dict.fromkeys(document_names))
    missing = [name for name in requested if name not in available]
    if missing:
        raise ValueError(
            "PDF tidak ditemukan: " + ", ".join(missing)
            + ". Gunakan nama file lengkap sesuai data/raw/prospectuses."
        )
    return [available[name] for name in requested]


def rebuild(
    assume_yes: bool = False,
    document_names: list[str] | None = None,
    refresh_parse_cache: bool = False,
) -> None:
    pdfs = select_pdfs(document_names)
    if not pdfs:
        raise RuntimeError(f"Tidak ada PDF di {config.PROSPEKTUS_DIR}")
    print(f"Parser mode: {config.PARSER_MODE}")
    print("Dokumen yang akan diindeks:")
    for pdf in pdfs:
        print(f"  - {pdf.name}")
    if not assume_yes:
        answer = input(f"Bangun ulang indeks untuk {len(pdfs)} PDF? Ketik 'ya': ").strip().lower()
        if answer != "ya":
            print("Dibatalkan.")
            return

    backup_dir = backup_existing()
    print(f"Indeks lama dicadangkan ke: {backup_dir}")

    from src.services.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    for number, pdf in enumerate(pdfs, start=1):
        print(f"[{number}/{len(pdfs)}] {pdf.name}")
        stats = pipeline.ingest(
            pdf,
            refresh_parse_cache=refresh_parse_cache,
        )
        if stats.get("parse_cache_migrated"):
            cache_status = "cache v2 dimigrasi ke v3"
        else:
            cache_status = "cache v3" if stats.get("parse_cache_hit") else "parser"
        print(
            f"  Parser: {stats.get('parser_used', '?')} "
            f"({cache_status}), chunks: {stats.get('total_chunks', 0)}"
        )
    print(f"Selesai. {len(pdfs)} dokumen telah diindeks ulang.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="Lewati konfirmasi interaktif")
    parser.add_argument(
        "--documents",
        nargs="+",
        metavar="PDF",
        help="Bangun indeks hanya untuk nama PDF yang disebutkan",
    )
    parser.add_argument(
        "--refresh-parse-cache",
        action="store_true",
        help="Abaikan cache dan panggil parser kembali",
    )
    args = parser.parse_args()
    rebuild(args.yes, args.documents, args.refresh_parse_cache)


if __name__ == "__main__":
    main()
