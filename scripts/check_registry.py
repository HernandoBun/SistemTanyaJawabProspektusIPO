"""Validate consistency between PDFs, registry, BM25, and ChromaDB."""
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def check(expected_documents: list[str] | None = None) -> int:
    registry = {}
    if config.CACHE_REGISTRY_PATH.exists():
        registry = json.loads(config.CACHE_REGISTRY_PATH.read_text(encoding="utf-8"))

    all_pdf_names = {path.name for path in config.PROSPEKTUS_DIR.glob("*.pdf")}
    pdf_names = set(expected_documents) if expected_documents else all_pdf_names
    unknown = sorted(pdf_names - all_pdf_names)
    registered_names = set(registry)
    missing = sorted(pdf_names - registered_names)
    stale = sorted(registered_names - pdf_names)

    print(f"PDF tersedia       : {len(pdf_names)}")
    print(f"Dokumen di registry: {len(registered_names)}")
    for name in sorted(registered_names & pdf_names):
        info = registry[name]
        print(f"[OK] {name}: {info.get('chunk_count', '?')} chunk")
    for name in missing:
        print(f"[BELUM DIINDEKS] {name}")
    for name in stale:
        print(f"[NAMA LAMA/TIDAK ADA] {name}")
    for name in unknown:
        print(f"[PDF TIDAK DITEMUKAN] {name}")

    try:
        from src.retrieval.indexer import DualIndexer

        indexer = DualIndexer()
        loaded = indexer.load_index()
        chroma_count = indexer.get_collection_count() if loaded else 0
        bm25_count = len(indexer._corpus_chunks) if loaded else 0
        registry_count = sum(item.get("chunk_count", 0) for item in registry.values())
        per_document = indexer.get_document_chunk_counts() if loaded else {"chroma": {}, "bm25": {}}
        print(f"Chunk ChromaDB     : {chroma_count}")
        print(f"Chunk BM25         : {bm25_count}")
        print(f"Chunk registry     : {registry_count}")
        per_document_match = True
        for name, info in sorted(registry.items()):
            expected = info.get("chunk_count", 0)
            dense = per_document["chroma"].get(name, 0)
            sparse = per_document["bm25"].get(name, 0)
            status = "OK" if expected == dense == sparse else "TIDAK COCOK"
            print(
                f"[{status}] {name}: registry={expected}, "
                f"ChromaDB={dense}, BM25={sparse}"
            )
            per_document_match &= expected == dense == sparse
        counts_match = (
            loaded
            and chroma_count == bm25_count == registry_count
            and per_document_match
        )
    except Exception as exc:
        print(f"[GAGAL MEMERIKSA INDEKS] {exc}")
        counts_match = False

    valid = not missing and not stale and not unknown and counts_match
    print("[VALID] Registry dan indeks konsisten." if valid else "[PERLU REBUILD] Data tidak konsisten.")
    return 0 if valid else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--documents", nargs="+", metavar="PDF",
        help="Validasi hanya indeks pilot untuk PDF yang disebutkan",
    )
    args = parser.parse_args()
    raise SystemExit(check(args.documents))
