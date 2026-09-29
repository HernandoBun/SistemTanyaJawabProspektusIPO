"""
Script to safely remove or reset ChromaDB, BM25 Index, and Registry.
Usage:
    python scripts/clean_indexes.py --all           # Delete all indexes & reset registry
    python scripts/clean_indexes.py --doc BACH_2026.pdf  # Remove only a specific document
"""
import argparse
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def clean_all(delete_parse_cache: bool = False):
    print("Mereset seluruh indeks (ChromaDB + BM25 + Registry)...")
    
    # 1. Hapus ChromaDB
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        print("  [OK] ChromaDB collection dibersihkan.")

    # 2. Hapus BM25 Index
    if config.BM25_INDEX_PATH.exists():
        config.BM25_INDEX_PATH.unlink(missing_ok=True)
        print("  [OK] Berkas BM25 index dibersihkan.")

    # 3. Hapus Registry
    if config.CACHE_REGISTRY_PATH.exists():
        config.CACHE_REGISTRY_PATH.unlink(missing_ok=True)
        print("  [OK] Berkas registry dokumen dibersihkan.")

    # 4. Opsional: Hapus cache parsed JSON
    if delete_parse_cache and config.PARSED_CACHE_DIR.exists():
        shutil.rmtree(config.PARSED_CACHE_DIR, ignore_errors=True)
        config.PARSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("  [OK] Cache parsed PDF dibersihkan.")

    print("\nStatus: Indeks 100% bersih dan kosong.")

def remove_doc(doc_name: str):
    print(f"Menghapus dokumen '{doc_name}' dari ChromaDB dan BM25...")
    from src.retrieval.indexer import DualIndexer
    import json
    
    indexer = DualIndexer()
    if not indexer.load_index():
        print("  [!] Indeks belum ada atau kosong.")
        return
        
    success = indexer.remove_document(doc_name)
    if success:
        print(f"  [OK] '{doc_name}' berhasil dihapus dari ChromaDB dan BM25.")
        # Update registry
        if config.CACHE_REGISTRY_PATH.exists():
            try:
                registry = json.loads(config.CACHE_REGISTRY_PATH.read_text(encoding="utf-8"))
                if doc_name in registry:
                    del registry[doc_name]
                    config.CACHE_REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")
                    print(f"  [OK] '{doc_name}' dihapus dari index_registry.json.")
            except Exception as e:
                print(f"  [!] Gagal update registry: {e}")
    else:
        print(f"  [!] Gagal menghapus '{doc_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Pembersih Indeks ChromaDB & BM25")
    parser.add_argument("--all", action="store_true", help="Reset total semua indeks dan registry")
    parser.add_argument("--doc", type=str, help="Hapus hanya dokumen tertentu (contoh: BACH_2026.pdf)")
    parser.add_argument("--delete-parsed-cache", action="store_true", help="Ikut hapus cache JSON hasil parsing LlamaParse")
    
    args = parser.parse_args()
    if args.all:
        clean_all(delete_parse_cache=args.delete_parsed_cache)
    elif args.doc:
        remove_doc(args.doc)
    else:
        print("Gunakan salah satu argumen:")
        print("  python scripts/clean_indexes.py --all")
        print("  python scripts/clean_indexes.py --doc NAMA_DOKUMEN.pdf")
        print("  python scripts/clean_indexes.py --all --delete-parsed-cache")

if __name__ == "__main__":
    main()
