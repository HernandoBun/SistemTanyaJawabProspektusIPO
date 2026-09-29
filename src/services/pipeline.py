"""
src/pipeline.py — Orchestration Pipeline

INGESTION PIPELINE:
PDF → Parser → Preprocessor → Chunker → Embedder → Indexer (ChromaDB + BM25)

QUERY PIPELINE:
Question → HybridRetriever (filtered by active doc) → RAGGenerator → Answer

Features:
- Document cache registry: indexed docs are NOT re-processed on reload
- Multi-document index: all docs share one ChromaDB collection
- Active document switching: instantly switch query scope per emiten
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from src.ingestion.parser import PDFParser, ParsedPage, get_document_metadata
from src.ingestion.preprocessor import preprocess_query
from src.ingestion.chunker import ProspectusChunker, Chunk
from src.retrieval.retriever import HybridRetriever, RetrievedChunk
import config

if TYPE_CHECKING:
    from src.generation.generator import GeneratorResponse, RAGGenerator
    from src.retrieval.indexer import DualIndexer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main orchestration class for the IPO Prospectus RAG system.

    Usage:
        pipeline = RAGPipeline()

        # Ingest a document (skipped automatically if already cached)
        pipeline.ingest("PROSPEKTUS IPO/07-rans-prospektus-2026.pdf")

        # Switch active document for queries
        pipeline.set_active_document("07-rans-prospektus-2026.pdf")

        # Query
        result = pipeline.query("Berapa total aset perusahaan?")
        print(result.answer)
    """

    def __init__(self):
        from src.retrieval.indexer import DualIndexer

        self.indexer = DualIndexer()
        self._retriever: Optional[HybridRetriever] = None
        self._generator: Optional[RAGGenerator] = None
        self._active_doc: Optional[str] = None   # filename of active document
        self._is_ready = False

        # Load registry (tracks which docs have been indexed)
        self._registry: Dict[str, dict] = self._load_registry()

        # Try to load existing index on startup
        self._try_load_existing_index()

    # ------------------------------------------------------------------ #
    #  Registry Management
    # ------------------------------------------------------------------ #

    def _load_registry(self) -> Dict[str, dict]:
        """Load the JSON registry of indexed documents from disk."""
        if config.CACHE_REGISTRY_PATH.exists():
            try:
                with open(config.CACHE_REGISTRY_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load registry: {e}")
        return {}

    def _save_registry(self) -> None:
        """Publish a complete registry only after the indexes are written."""
        destination = config.CACHE_REGISTRY_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._registry, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)

    def is_doc_indexed(self, pdf_name: str) -> bool:
        """Return True only for an index built with the current schema."""
        metadata = self._registry.get(pdf_name, {})
        return metadata.get("index_schema_version") == config.INDEX_SCHEMA_VERSION

    def get_indexed_docs(self) -> Dict[str, dict]:
        """Return current-schema indexed documents (doc_name → metadata)."""
        return {
            name: dict(metadata)
            for name, metadata in self._registry.items()
            if metadata.get("index_schema_version") == config.INDEX_SCHEMA_VERSION
        }

    def set_active_document(self, doc_name: str) -> None:
        """
        Set the active document for queries.
        Queries will be filtered to only this document's chunks.
        """
        self._active_doc = doc_name
        logger.info(f"Active document set to: {doc_name}")

    def get_active_document(self) -> Optional[str]:
        return self._active_doc

    def remove_document(self, doc_name: str) -> bool:
        """
        Remove a document from the index and registry.
        Returns True if successful.
        """
        success = self.indexer.remove_document(doc_name)
        if success and doc_name in self._registry:
            del self._registry[doc_name]
            self._save_registry()
            # If this was the active doc, clear it
            if self._active_doc == doc_name:
                self._active_doc = None
                # Set active to first remaining doc if any
                if self._registry:
                    self._active_doc = next(iter(self._registry))
            logger.info(f"Document '{doc_name}' removed from index and registry.")
        return success

    # ------------------------------------------------------------------ #
    #  Ingestion
    # ------------------------------------------------------------------ #

    def ingest(
        self,
        pdf_path: str | Path,
        progress_callback: Optional[Callable] = None,
        force: bool = False,
        refresh_parse_cache: bool = False,
    ) -> dict:
        """
        Full ingestion pipeline: PDF → Index.

        Automatically SKIPS processing if the document is already in the cache
        registry, unless force=True is passed.

        Args:
            pdf_path: Path to the PDF prospectus file.
            progress_callback: Optional callable(stage, pct, msg) for UI updates.
            force: If True, re-process even if already cached.
            refresh_parse_cache: If True, call the parser again instead of
                reusing a matching cached parse result.

        Returns:
            dict with ingestion statistics. Includes 'skipped': True if cached.
        """
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.name
        with pdf_path.open("rb") as handle:
            content_hash = hashlib.file_digest(handle, "sha256").hexdigest()

        def _cb(stage: str, pct: float, msg: str = ""):
            if progress_callback:
                progress_callback(stage, pct, msg)
            logger.info(f"[{stage}] {pct:.0%} — {msg}")

        # ── Cache check ──────────────────────────────────────────────────
        if (not force and self.is_doc_indexed(doc_name)
                and self._registry[doc_name].get("content_sha256") == content_hash):
            logger.info(f"'{doc_name}' already indexed — skipping ingestion.")
            self.set_active_document(doc_name)
            _cb("done", 1.0, f"✅ '{doc_name}' sudah diindeks sebelumnya.")
            cached = self._registry[doc_name]
            return {**cached, "skipped": True, "pdf_name": doc_name}

        start_time = time.time()
        stats: dict = {}

        # ── Stage 1: Parse PDF ───────────────────────────────────────────
        _cb("parsing", 0.0, f"Memulai parsing {doc_name}...")
        parser = PDFParser(pdf_path)
        pages: List[ParsedPage] = parser.parse(
            progress_callback=lambda cur, tot: _cb(
                "parsing", cur / tot, f"Halaman {cur}/{tot}"
            ),
            refresh_cache=refresh_parse_cache,
        )
        stats["total_pages"] = len(pages)
        stats["parser_used"] = parser.parser_used
        stats["parse_cache_hit"] = parser.cache_hit
        stats["parse_cache_migrated"] = parser.cache_migrated
        stats["parse_cache_path"] = str(parser.cache_path or "")
        _cb("parsing", 1.0, f"✓ {len(pages)} halaman selesai diparsing")

        # Preprocessing runs centrally inside PDFParser before the parse cache
        # is saved, so both fresh and cached pages follow identical rules.
        preprocessing = dict(parser.preprocessing_report)
        stats["preprocessing"] = preprocessing
        stats["preprocessing_version"] = preprocessing.get(
            "version", config.PREPROCESSING_VERSION
        )
        stats["removed_header_footer_lines"] = (
            preprocessing.get("removed_header_lines", 0)
            + preprocessing.get("removed_footer_lines", 0)
        )
        _cb(
            "preprocessing",
            1.0,
            "✓ Preprocessing selesai: "
            f"{stats['removed_header_footer_lines']} baris header/footer dihapus",
        )

        # ── Stage 2: Chunk ───────────────────────────────────────────────
        _cb("chunking", 0.0, "Memulai proses chunking...")
        chunker = ProspectusChunker(doc_source=doc_name)
        chunks: List[Chunk] = chunker.chunk_pages(pages)
        stats["total_chunks"] = len(chunks)
        stats["narrative_chunks"] = sum(1 for c in chunks if c.chunk_type == "narrative")
        stats["table_chunks"] = sum(1 for c in chunks if c.chunk_type == "table")
        _cb("chunking", 1.0,
            f"✓ {len(chunks)} chunks ({stats['narrative_chunks']} narasi, "
            f"{stats['table_chunks']} tabel)")

        # ── Stage 3: Embed + Index (APPEND mode) ─────────────────────────
        _cb("embedding", 0.0, f"Memuat model {config.EMBEDDING_MODEL_NAME}...")

        def _index_progress(step: str, cur: int, total: int):
            if step == "embedding":
                _cb("embedding", cur / max(total, 1),
                    f"Embedding {cur}/{total} chunks...")
            elif step == "indexing_dense":
                _cb("indexing", 0.3, "Menyimpan ke ChromaDB...")
            elif step == "indexing_sparse":
                _cb("indexing", 0.7, "Membangun indeks BM25...")
            elif step == "done":
                _cb("indexing", 1.0, "✓ Index selesai dibangun")

        if not chunks:
            raise ValueError("Tidak ada chunk yang dapat diindeks dari PDF ini.")
        embeddings = self.indexer.embedder.embed_texts(
            [chunk.content for chunk in chunks],
            progress_callback=lambda current, total: _index_progress("embedding", current, total),
        )
        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = round(elapsed, 2)
        stats["pdf_name"] = doc_name
        stats["indexed_at"] = datetime.now().isoformat(timespec="seconds")
        stats["skipped"] = False

        # ── Save to registry ─────────────────────────────────────────────
        new_metadata = {
            "content_sha256": content_hash,
            "index_schema_version": config.INDEX_SCHEMA_VERSION,
            "chunk_count": stats["total_chunks"],
            "narrative_chunks": stats["narrative_chunks"],
            "table_chunks": stats["table_chunks"],
            "total_pages": stats["total_pages"],
            "elapsed_seconds": stats["elapsed_seconds"],
            "indexed_at": stats["indexed_at"],
            "parser_used": stats["parser_used"],
            "parse_cache_hit": stats["parse_cache_hit"],
            "parse_cache_migrated": stats["parse_cache_migrated"],
            "parse_cache_path": stats["parse_cache_path"],
            "preprocessing_version": stats["preprocessing_version"],
            "removed_header_footer_lines": stats["removed_header_footer_lines"],
            "preprocessing": stats["preprocessing"],
        }
        old_registry = dict(self._registry)
        def publish():
            stats["elapsed_seconds"] = round(time.time() - start_time, 2)
            new_metadata["elapsed_seconds"] = stats["elapsed_seconds"]
            self._registry[doc_name] = new_metadata
            try:
                self._save_registry()
            except Exception:
                self._registry = old_registry
                raise

        self.indexer.replace_document(
            doc_name, chunks, embeddings, publish, progress_callback=_index_progress,
        )
        self._retriever = HybridRetriever(self.indexer)
        self._is_ready = True
        self.set_active_document(doc_name)

        _cb("done", 1.0, f"✓ Selesai dalam {stats['elapsed_seconds']:.1f} detik")
        logger.info(f"Ingestion complete: {stats}")
        return stats

    def query(
        self,
        question: str,
        dense_k: int = config.DENSE_TOP_K,
        sparse_k: int = config.SPARSE_TOP_K,
        final_k: int = config.FINAL_TOP_K,
    ) -> GeneratorResponse:
        """
        End-to-end query: question → retrieved context → LLM answer.
        Retrieval is automatically scoped to the active document.
        """
        if not self._is_ready:
            raise RuntimeError(
                "Pipeline belum siap. Silakan proses dokumen terlebih dahulu."
            )

        if not preprocess_query(question):
            raise ValueError("Pertanyaan kosong setelah preprocessing.")

        logger.info(f"Query (active_doc={self._active_doc}): '{question}'")

        retrieved = self.retrieve(question, dense_k, sparse_k, final_k)

        if not retrieved or max(r.rrf_score for r in retrieved) < config.MIN_RRF_SCORE:
            from src.generation.generator import GeneratorResponse
            return GeneratorResponse(
                answer="Informasi tersebut tidak tersedia di dokumen prospektus ini.",
                source_pages=[], source_chunks=[], citations=[],
                model_used=config.OPENROUTER_MODEL_NAME,
                retrieval_candidates=retrieved,
            )
        if self._generator is None:
            self._init_generator()
        if self._generator is None:
            raise RuntimeError("Layanan jawaban belum siap. Periksa konfigurasi OpenRouter.")

        # Extract readable company name from filename for LLM context
        company_name = self._active_doc or "perusahaan ini"
        if company_name:
            import re
            from pathlib import Path
            stem = Path(company_name).stem
            stem = re.sub(r'\s*\(\d+\)', '', stem)
            stem = re.sub(r'[_\s]+', '-', stem)
            parts = stem.lower().split('-')
            SKIP = {'prospektus','prospectus','2026','2025','2024',
                    '01','02','03','04','05','06','07','08','09','10'}
            tickers = [p.upper() for p in parts if p and p not in SKIP and not p.isdigit()]
            company_name = tickers[0] if tickers else stem.upper()

        response = self._generator.generate(
            question=question,
            retrieved_chunks=retrieved,
            company_name=company_name,
        )

        return response

    def retrieve(
        self,
        question: str,
        dense_k: int = config.DENSE_TOP_K,
        sparse_k: int = config.SPARSE_TOP_K,
        final_k: int = config.FINAL_TOP_K,
    ) -> List[RetrievedChunk]:
        """Return ranked chunks without invoking the LLM."""
        if not self._is_ready or self._retriever is None:
            raise RuntimeError(
                "Pipeline belum siap. Silakan proses dokumen terlebih dahulu."
            )
        if not self._active_doc:
            raise RuntimeError(
                "Dokumen aktif belum dipilih. Panggil set_active_document() terlebih dahulu."
            )

        question = preprocess_query(question)
        if not question:
            raise ValueError("Pertanyaan kosong setelah preprocessing.")

        return self._retriever.retrieve(
            query=question,
            dense_k=dense_k,
            sparse_k=sparse_k,
            final_k=final_k,
            doc_source_filter=self._active_doc,
        )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def is_ready(self) -> bool:
        return self._is_ready

    def get_doc_metadata(self) -> dict:
        return self._registry.get(self._active_doc, {}) if self._active_doc else {}

    def get_index_stats(self) -> dict:
        current_registry = self.get_indexed_docs()
        total_chunks = sum(
            v.get("chunk_count", 0) for v in current_registry.values()
        ) if current_registry else 0
        return {
            "chunk_count": total_chunks,
            "index_ready": bool(current_registry) and self.indexer.is_indexed(),
            "doc_count": len(current_registry),
            "active_doc": self._active_doc,
        }

    def _try_load_existing_index(self) -> None:
        """Try to load an existing index from disk on startup."""
        try:
            if self.indexer.load_index():
                self._retriever = HybridRetriever(self.indexer)
                self._is_ready = True
                logger.info(
                    f"Existing index loaded. No active doc set. "
                    f"Registry: {list(self._registry.keys())}"
                )
        except Exception as e:
            logger.warning(f"Could not load existing index: {e}")
            self._is_ready = False

    def _init_generator(self) -> None:
        """Initialize the LLM generator."""
        try:
            from src.generation.generator import RAGGenerator

            self._generator = RAGGenerator()
        except ValueError as e:
            logger.warning(f"Generator not initialized: {e}")
            self._generator = None
