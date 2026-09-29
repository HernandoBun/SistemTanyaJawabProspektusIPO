"""
src/retriever.py — Hybrid Search with Reciprocal Rank Fusion (RRF)

Menggabungkan:
1. Dense Retrieval: Pencarian semantik via ChromaDB + BAAI/bge-m3
2. Sparse Retrieval: Pencarian leksikal via BM25Okapi
3. RRF: Reranking matematis untuk menggabungkan hasil kedua pencarian

Formula RRF:
    score(d) = Σ 1 / (k + rank_i(d))
    
Dimana k=60 (konstanta RRF), rank_i adalah posisi dokumen d di hasil pencarian i.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Dict, Tuple

from src.ingestion.chunker import Chunk
import config

if TYPE_CHECKING:
    from src.retrieval.indexer import DualIndexer

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk with its retrieval metadata and RRF score."""
    chunk: Chunk
    rrf_score: float
    dense_rank: int           # Rank from dense search (0 = not found)
    sparse_rank: int          # Rank from sparse search (0 = not found)
    dense_score: float
    sparse_score: float


# ------------------------------------------------------------------ #
#  Domain-Specific Query Expansion (Indonesian Financial & Prospectus)
# ------------------------------------------------------------------ #

PROSPECTUS_DOMAIN_EXPANSIONS = {
    "laba bersih": ["laba tahun berjalan", "laba neto"],
    "laba kotor": ["laba bruto", "margin laba kotor"],
    "laba operasional": ["laba usaha", "laba operasi"],
    "laba usaha": ["laba usaha", "laba operasional"],
    "laba": ["laba tahun berjalan"],
    "beban": ["beban pokok pendapatan", "beban usaha", "beban operasional"],
    "pendapatan": ["pendapatan usaha", "penjualan neto"],
    "utang": ["liabilitas", "kewajiban"],
    "rasio utang": ["liabilitas terhadap ekuitas"],
    "modal": ["ekuitas", "modal disetor"],
    "belanja modal": ["capital expenditure", "capex"],
    "modal kerja": ["operational expenditure", "opex"],
    "dana hasil ipo": ["rencana penggunaan dana", "penawaran umum perdana saham"],
    "penggunaan dana": ["rencana penggunaan dana", "alokasi dana"],
    "dana ipo": ["rencana penggunaan dana", "penawaran umum"],
    "alokasi dana": ["rencana penggunaan dana", "penawaran umum"],
    "dana": ["rencana penggunaan dana", "alokasi"],
    "ipo": ["penawaran umum perdana saham"],
    "saham publik": ["penawaran umum", "saham baru", "masyarakat"],
    "akuisisi": ["pengambilalihan", "penyertaan saham"],
    "pengendali": ["pemegang saham pengendali", "ultimate beneficial owner"],
    "direktur": ["susunan direksi", "pengurus"],
    "direksi": ["susunan direksi", "pengurus"],
    "komisaris": ["dewan komisaris", "pengurus"],
    "produk": ["segmen usaha", "pendapatan terbesar"],
    "kompetitor": ["pesaing", "persaingan usaha", "pangsa pasar"],
    "kapasitas": ["kapasitas terpasang", "tingkat utilitas"],
    "dividen": ["kebijakan dividen", "pembagian dividen"],
    "risiko": ["faktor risiko", "risiko utama"],
    "pemasok": ["pemasok utama", "ketergantungan pemasok"],
    "pelanggan": ["pelanggan utama", "kontribusi pendapatan"],
}


def expand_query_for_retrieval(query: str) -> str:
    """
    Expand user query with formal Indonesian prospectus and accounting terms.
    Preserves original query words while enriching search representation.
    """
    if not query:
        return ""
    q_lower = query.lower()
    additions = ["perseroan"]
    for term, synonyms in PROSPECTUS_DOMAIN_EXPANSIONS.items():
        if term in q_lower:
            additions.extend(synonyms)

    seen = set(q_lower.split())
    words = []
    for addition in additions:
        for word in addition.split():
            if word not in seen and word not in words:
                words.append(word)

    if not words:
        return query
    return f"{query} {' '.join(words[:8])}"


class HybridRetriever:
    """
    Hybrid Search Retriever combining dense (ChromaDB) and sparse (BM25)
    retrieval with Reciprocal Rank Fusion for final reranking.
    """

    def __init__(self, indexer: "DualIndexer"):
        self.indexer = indexer
        self.embedder = indexer.embedder

    def retrieve(
        self,
        query: str,
        dense_k: int = config.DENSE_TOP_K,
        sparse_k: int = config.SPARSE_TOP_K,
        final_k: int = config.FINAL_TOP_K,
        rrf_k: int = config.RRF_K,
        doc_source_filter: str | None = None,
    ) -> List[RetrievedChunk]:
        """
        Perform hybrid search and return top-k reranked chunks.

        Args:
            query: User's question string.
            dense_k: How many candidates to fetch from ChromaDB.
            sparse_k: How many candidates to fetch from BM25.
            final_k: Final number of chunks to return after RRF.
            rrf_k: RRF constant (higher = more uniform merging).

        Returns:
            List of RetrievedChunk objects sorted by RRF score descending.
        """
        # Expand query for retrieval using prospectus domain terms
        search_query = expand_query_for_retrieval(query)

        # ---- Step 1: Embed the query ----
        query_embedding = self.embedder.embed_query(search_query) if dense_k > 0 else None

        # ---- Step 2: Dense search ----
        dense_results: List[Tuple[Chunk, float]] = (
            self.indexer.dense_search(
                query_embedding, k=dense_k, doc_source_filter=doc_source_filter
            )
            if dense_k > 0 else []
        )
        logger.debug(f"Dense search returned {len(dense_results)} results")

        # ---- Step 3: Sparse search ----
        sparse_results: List[Tuple[Chunk, float]] = (
            self.indexer.sparse_search(
                search_query, k=sparse_k, doc_source_filter=doc_source_filter
            )
            if sparse_k > 0 else []
        )
        logger.debug(f"Sparse search returned {len(sparse_results)} results")

        # ---- Step 4: Reciprocal Rank Fusion ----
        fused = self._rrf_merge(
            dense_results,
            sparse_results,
            rrf_k=rrf_k,
        )

        # Return top final_k
        top_results = fused[:final_k]

        logger.info(
            f"Hybrid retrieval: dense={len(dense_results)}, "
            f"sparse={len(sparse_results)}, "
            f"fused={len(fused)}, "
            f"returned={len(top_results)}"
        )
        return top_results

    def _rrf_merge(
        self,
        dense_results: List[Tuple[Chunk, float]],
        sparse_results: List[Tuple[Chunk, float]],
        rrf_k: int = 60,
    ) -> List[RetrievedChunk]:
        """
        Merge and rerank results using Reciprocal Rank Fusion.

        RRF score for document d:
            rrf_score(d) = Σ_i 1 / (k + rank_i(d))

        Documents not present in a ranking list get no contribution from that list.
        """
        # Build lookup dicts: content_key → (chunk, score, rank)
        dense_map: Dict[tuple[str, str], Tuple[Chunk, float, int]] = {}
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            key = self._chunk_key(chunk)
            dense_map[key] = (chunk, score, rank)

        sparse_map: Dict[tuple[str, str], Tuple[Chunk, float, int]] = {}
        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            key = self._chunk_key(chunk)
            sparse_map[key] = (chunk, score, rank)

        # Collect all unique chunk keys
        all_keys = set(dense_map.keys()) | set(sparse_map.keys())

        fused_chunks: List[RetrievedChunk] = []

        for key in all_keys:
            # Dense contribution
            if key in dense_map:
                chunk, d_score, d_rank = dense_map[key]
                d_rrf = 1.0 / (rrf_k + d_rank)
            else:
                chunk, d_score, d_rank = sparse_map[key][0], 0.0, 0
                d_rrf = 0.0

            # Sparse contribution
            if key in sparse_map:
                s_chunk, s_score, s_rank = sparse_map[key]
                s_rrf = 1.0 / (rrf_k + s_rank)
                # Prefer the chunk from whichever source found it
                if key not in dense_map:
                    chunk = s_chunk
            else:
                s_score, s_rank = 0.0, 0
                s_rrf = 0.0

            rrf_score = d_rrf + s_rrf

            fused_chunks.append(RetrievedChunk(
                chunk=chunk,
                rrf_score=rrf_score,
                dense_rank=d_rank,
                sparse_rank=s_rank,
                dense_score=d_score,
                sparse_score=s_score,
            ))

        # Sort by RRF score descending
        fused_chunks.sort(key=lambda x: (-x.rrf_score, x.chunk.doc_source, x.chunk.chunk_id))
        return fused_chunks

    @staticmethod
    def _chunk_key(chunk: Chunk) -> tuple[str, str]:
        """
        Identity of a chunk, independent of repeated headings or table prefixes.
        """
        return (chunk.doc_source, chunk.chunk_id)
