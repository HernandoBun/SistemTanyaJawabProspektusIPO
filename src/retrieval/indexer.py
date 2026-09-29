"""
src/indexer.py — Dual Indexing: ChromaDB (Dense) + BM25 (Sparse)

Menyimpan:
1. ChromaDB: Vektor embedding + metadata untuk dense retrieval
2. BM25Okapi: Indeks leksikal untuk sparse/keyword retrieval
3. Corpus pickle: Teks asli chunk untuk BM25 scoring
"""

from __future__ import annotations

import logging
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

from src.ingestion.chunker import Chunk
from src.retrieval.embedder import Embedder
from src.retrieval.tokenizer import _tokenize
import config

logger = logging.getLogger(__name__)


class DualIndexer:
    """
    Manages both ChromaDB (dense) and BM25 (sparse) indices.
    """

    def __init__(self):
        self.embedder = Embedder()

        # --- ChromaDB Setup ---
        self._chroma_client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection: Optional[chromadb.Collection] = None

        # --- BM25 State (one independent corpus per prospectus) ---
        self._bm25_by_document: Dict[str, BM25Okapi] = {}
        self._corpus_chunks_by_document: Dict[str, List[Chunk]] = {}
        self._corpus_chunks: List[Chunk] = []

    def _create_cosine_collection(self):
        """Create a Chroma collection with cosine distance explicitly set."""
        collection = self._chroma_client.create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_SPACE},
        )
        self._validate_cosine_collection(collection)
        return collection

    @staticmethod
    def _validate_cosine_collection(collection) -> None:
        """Reject an old/default-L2 collection before similarity is computed."""
        metadata = collection.metadata or {}
        actual_space = metadata.get("hnsw:space")
        if actual_space != config.CHROMA_DISTANCE_SPACE:
            raise RuntimeError(
                "Koleksi ChromaDB tidak menggunakan cosine distance "
                f"(aktual={actual_space!r}). Bangun ulang indeks agar "
                f"hnsw:space={config.CHROMA_DISTANCE_SPACE!r}."
            )

    # ------------------------------------------------------------------ #
    #  Public: Build Index
    # ------------------------------------------------------------------ #

    def build_index(
        self,
        chunks: List[Chunk],
        progress_callback=None,
        reset: bool = False,
        embeddings=None,
    ) -> None:
        """
        Build or append to both dense (ChromaDB) and sparse (BM25) indices.

        Args:
            chunks: List of Chunk objects to index.
            progress_callback: Optional callable(step, current, total)
            reset: If True, wipe everything and start fresh.
                   If False (default), APPEND new chunks to existing index.
        """
        if not chunks:
            logger.warning("No chunks to index!")
            return

        incoming_ids = [chunk.chunk_id for chunk in chunks]
        if len(incoming_ids) != len(set(incoming_ids)):
            duplicates = sorted(
                item for item, count in Counter(incoming_ids).items() if count > 1
            )
            raise ValueError(
                "Duplicate chunk_id pada dokumen yang akan diindeks: "
                + ", ".join(duplicates[:5])
            )

        if embeddings is None:
            embeddings = self.embedder.embed_texts(
                [chunk.content for chunk in chunks], show_progress=True,
            )
        if len(embeddings) != len(chunks):
            raise ValueError("Jumlah embedding tidak sesuai jumlah chunk.")
        logger.info(f"{'Resetting and building' if reset else 'Appending'} index for {len(chunks)} chunks...")

        # ---- ChromaDB ----
        if reset:
            try:
                self._chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
                logger.info(f"Deleted existing collection: {config.CHROMA_COLLECTION_NAME}")
            except Exception:
                pass
            self._collection = self._create_cosine_collection()
            old_corpus_chunks: List[Chunk] = []
        else:
            # Append mode: get or create collection
            try:
                self._collection = self._chroma_client.get_collection(config.CHROMA_COLLECTION_NAME)
                self._validate_cosine_collection(self._collection)
                logger.info(f"Appending to existing collection ({self._collection.count()} chunks)")
            except Exception:
                try:
                    self._collection = self._create_cosine_collection()
                except Exception as exc:
                    raise RuntimeError(
                        "Koleksi ChromaDB yang ada tidak kompatibel. "
                        "Jalankan scripts.rebuild_indexes agar indeks dibuat ulang "
                        "dengan cosine distance."
                    ) from exc
            # Load existing corpus for BM25 rebuild
            old_corpus_chunks = list(self._corpus_chunks)

            # Chroma IDs are global inside a collection. Abort instead of
            # silently dropping chunks when an ID already exists.
            existing_ids: List[str] = []
            for start in range(0, len(incoming_ids), 500):
                found = self._collection.get(
                    ids=incoming_ids[start:start + 500],
                    include=["metadatas"],
                )
                existing_ids.extend(found.get("ids", []))
            if existing_ids:
                raise RuntimeError(
                    "Chunk ID sudah ada di ChromaDB. Hentikan append dan gunakan "
                    "force ingest atau rebuild. Contoh: "
                    + ", ".join(existing_ids[:5])
                )

        if progress_callback:
            progress_callback("indexing_dense", 0, len(chunks))

        # Add new chunks to ChromaDB in batches of 500
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            self._collection.add(
                ids=[c.chunk_id for c in batch_chunks],
                embeddings=batch_embeddings.tolist(),
                documents=[c.content for c in batch_chunks],
                metadatas=[{
                    "chunk_type":     c.chunk_type,
                    "page_number":    c.page_number,
                    "section_heading": c.section_heading,
                    "doc_source":     c.doc_source,
                    "char_count":     c.char_count,
                    "chapter":        c.chapter,
                    "subchapter":     c.subchapter,
                    "table_title":    c.table_title,
                    "financial_unit": c.financial_unit,
                } for c in batch_chunks],
            )
            logger.info(f"  ChromaDB: added {min(i + batch_size, len(chunks))}/{len(chunks)}")

        # ---- BM25: rebuild with old + new chunks ----
        if progress_callback:
            progress_callback("indexing_sparse", 0, len(chunks))

        combined_chunks = old_corpus_chunks + chunks
        self._rebuild_bm25_by_document(combined_chunks)
        self._save_bm25()

        logger.info("Index built/updated successfully.")
        if progress_callback:
            progress_callback("done", len(chunks), len(chunks))

    def remove_document(self, doc_source: str) -> bool:
        """
        Remove all chunks of a specific document from both indexes.
        Returns True if successful.
        """
        if self._collection is None:
            return False
        try:
            # Remove from ChromaDB
            self._collection.delete(where={"doc_source": doc_source})
            logger.info(f"Removed '{doc_source}' from ChromaDB")

            # Rebuild BM25 without this doc
            remaining = [c for c in self._corpus_chunks if c.doc_source != doc_source]
            self._rebuild_bm25_by_document(remaining)
            self._save_bm25()
            logger.info(f"BM25 rebuilt without '{doc_source}' ({len(remaining)} chunks remain)")
            return True
        except Exception as e:
            logger.error(f"Failed to remove document: {e}")
            return False

    def replace_document(self, doc_source, chunks, embeddings, publish, progress_callback=None):
        """Restore the previous document on a failed write or registry publish.

        Offline writers must run sequentially. This recovers Python exceptions;
        it is not a cross-store transaction against power loss or process kills.
        """
        snapshot = self._collection.get(
            where={"doc_source": doc_source},
            include=["documents", "metadatas", "embeddings"],
        ) if self._collection is not None else None
        old_corpus = list(self._corpus_chunks)
        try:
            if self._collection is not None and not self.remove_document(doc_source):
                raise RuntimeError(f"Gagal menyiapkan penggantian indeks {doc_source}")
            self.build_index(chunks, embeddings=embeddings, progress_callback=progress_callback)
            publish()
        except Exception:
            try:
                if self._collection is not None:
                    self._collection.delete(where={"doc_source": doc_source})
                    if snapshot and snapshot["ids"]:
                        for start in range(0, len(snapshot["ids"]), 500):
                            self._collection.add(**{
                                key: snapshot[key][start:start + 500]
                                for key in ("ids", "documents", "metadatas", "embeddings")
                            })
                self._rebuild_bm25_by_document(old_corpus)
                self._save_bm25()
            except Exception as recovery_error:
                raise RuntimeError(
                    f"Pengindeksan {doc_source} gagal dan pemulihan belum berhasil. "
                    "Hentikan pengindeksan dan periksa penyimpanan sebelum mencoba lagi."
                ) from recovery_error
            raise

    # ------------------------------------------------------------------ #
    #  Public: Load Existing Index
    # ------------------------------------------------------------------ #

    def load_index(self) -> bool:
        """
        Load existing indices from disk.

        Returns:
            True if successfully loaded, False if no index exists.
        """
        # Load ChromaDB
        try:
            self._collection = self._chroma_client.get_collection(
                config.CHROMA_COLLECTION_NAME
            )
            self._validate_cosine_collection(self._collection)
            logger.info(
                f"ChromaDB collection loaded: {self._collection.count()} chunks"
            )
        except Exception as e:
            logger.warning(f"ChromaDB collection not found: {e}")
            return False

        # Load BM25
        if not self._load_bm25():
            logger.warning("BM25 index not found.")
            return False

        logger.info("Index loaded successfully.")
        return True

    def is_indexed(self) -> bool:
        """Check if an index has been built/loaded."""
        return self._collection is not None and bool(self._bm25_by_document)

    def get_collection_count(self) -> int:
        """Return number of chunks in ChromaDB collection."""
        if self._collection is None:
            return 0
        return self._collection.count()

    def get_document_chunk_counts(self) -> dict[str, dict[str, int]]:
        """Return per-document counts from both dense and sparse indexes."""
        if self._collection is None:
            return {"chroma": {}, "bm25": {}}
        result = self._collection.get(include=["metadatas"])
        chroma_counts = Counter(
            metadata.get("doc_source", "")
            for metadata in result.get("metadatas", [])
            if metadata.get("doc_source")
        )
        bm25_counts = Counter(
            chunk.doc_source for chunk in self._corpus_chunks if chunk.doc_source
        )
        return {"chroma": dict(chroma_counts), "bm25": dict(bm25_counts)}

    # ------------------------------------------------------------------ #
    #  Public: Query
    # ------------------------------------------------------------------ #

    def dense_search(
        self,
        query_embedding: np.ndarray,
        k: int = config.DENSE_TOP_K,
        doc_source_filter: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search ChromaDB for semantically similar chunks.
        Optionally filter to only return chunks from a specific document.
        """
        if self._collection is None:
            raise RuntimeError("Index not loaded. Call load_index() or build_index() first.")

        count = self._collection.count()
        if count == 0:
            return []

        query_kwargs = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        if doc_source_filter:
            query_kwargs["where"] = {"doc_source": doc_source_filter}

        results = self._collection.query(**query_kwargs)

        chunks_scores: List[Tuple[Chunk, float]] = []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist
            chunk = Chunk(
                chunk_id=chunk_id,
                content=doc,
                chunk_type=meta.get("chunk_type", "narrative"),
                page_number=meta.get("page_number", 0),
                section_heading=meta.get("section_heading", ""),
                doc_source=meta.get("doc_source", ""),
                chapter=meta.get("chapter", ""),
                subchapter=meta.get("subchapter", ""),
                table_title=meta.get("table_title", ""),
                financial_unit=meta.get("financial_unit", ""),
            )
            chunks_scores.append((chunk, similarity))

        return chunks_scores

    def sparse_search(
        self,
        query: str,
        k: int = config.SPARSE_TOP_K,
        doc_source_filter: Optional[str] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search BM25 index for keyword/lexical matches.
        Optionally filter to only return chunks from a specific document.
        """
        if not self._bm25_by_document:
            raise RuntimeError("BM25 index not loaded.")

        if doc_source_filter is None:
            if len(self._bm25_by_document) != 1:
                raise ValueError(
                    "Sparse search memerlukan doc_source_filter karena BM25 "
                    "dibangun terpisah untuk setiap emiten."
                )
            doc_source_filter = next(iter(self._bm25_by_document))

        bm25 = self._bm25_by_document.get(doc_source_filter)
        corpus_chunks = self._corpus_chunks_by_document.get(doc_source_filter, [])
        if bm25 is None or not corpus_chunks:
            return []

        tokenized_query = _tokenize(query)
        scores = bm25.get_scores(tokenized_query)

        top_k_indices = np.argsort(scores)[::-1][:k]

        results: List[Tuple[Chunk, float]] = []
        for idx in top_k_indices:
            if scores[idx] <= 0:
                continue
            chunk = corpus_chunks[idx]
            results.append((chunk, float(scores[idx])))
            if len(results) >= k:
                break

        return results

    # ------------------------------------------------------------------ #
    #  Private: BM25 Persistence
    # ------------------------------------------------------------------ #

    def _rebuild_bm25_by_document(self, chunks: List[Chunk]) -> None:
        """Build independent BM25 statistics for every prospectus."""
        grouped: defaultdict[str, List[Chunk]] = defaultdict(list)
        for chunk in chunks:
            if not chunk.doc_source:
                raise ValueError(f"Chunk tanpa doc_source: {chunk.chunk_id}")
            grouped[chunk.doc_source].append(chunk)

        self._corpus_chunks = list(chunks)
        self._corpus_chunks_by_document = dict(grouped)
        self._bm25_by_document = {
            doc_source: BM25Okapi([_tokenize(chunk.content) for chunk in doc_chunks])
            for doc_source, doc_chunks in self._corpus_chunks_by_document.items()
        }

    def _save_bm25(self) -> None:
        """Persist BM25 index and corpus to disk."""
        data = {
            "schema_version": config.BM25_INDEX_SCHEMA_VERSION,
            "bm25_by_document": self._bm25_by_document,
            "corpus_chunks_by_document": self._corpus_chunks_by_document,
            "corpus_chunks": self._corpus_chunks,
        }
        temporary = config.BM25_INDEX_PATH.with_suffix(".tmp")
        with open(temporary, "wb") as f:
            pickle.dump(data, f)
        temporary.replace(config.BM25_INDEX_PATH)
        logger.info(f"BM25 index saved to {config.BM25_INDEX_PATH}")

    def _load_bm25(self) -> bool:
        """Load BM25 index and corpus from disk."""
        if not config.BM25_INDEX_PATH.exists():
            return False
        try:
            with open(config.BM25_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            if data.get("schema_version") == config.BM25_INDEX_SCHEMA_VERSION:
                self._bm25_by_document = data["bm25_by_document"]
                self._corpus_chunks_by_document = data["corpus_chunks_by_document"]
                self._corpus_chunks = data["corpus_chunks"]
            else:
                legacy_chunks = data.get("corpus_chunks", [])
                if not legacy_chunks:
                    raise ValueError("Corpus BM25 lama kosong atau tidak valid.")
                logger.warning(
                    "Migrating legacy global BM25 index to per-document BM25 in memory."
                )
                self._rebuild_bm25_by_document(legacy_chunks)
                self._save_bm25()
            logger.info(
                "BM25 per-document loaded: %d chunks across %d documents",
                len(self._corpus_chunks),
                len(self._bm25_by_document),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False
