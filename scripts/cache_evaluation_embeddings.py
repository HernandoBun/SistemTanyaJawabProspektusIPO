"""Cache actual offline BGE-M3 query vectors, using an installed local model.

This command can run in a different Python environment from RAGAS. Cache identity
includes the dataset and exact expanded queries. No reference answers are embedded
as queries, and no passage vectors are substituted for query vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import config
from evaluation.evaluate_retrieval import load_jsonl
from evaluation.reporting import file_fingerprint, write_json
from src.ingestion.preprocessor import preprocess_query
from src.retrieval.embedder import _prepare_embedding_text
from src.retrieval.retriever import expand_query_for_retrieval


def build_cache(questions_path: Path, destination: Path) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    questions = load_jsonl(questions_path)
    queries = list(dict.fromkeys(_prepare_embedding_text(expand_query_for_retrieval(
        preprocess_query(q["question"]))) for q in questions))
    with config.BM25_INDEX_PATH.open("rb") as handle:
        snapshot = pickle.load(handle)
    # Probe normal narrative passages, independent of relevance labels and queries.
    probes = [next(c for c in snapshot["corpus_chunks_by_document"][doc]
                   if c.chunk_type == "narrative" and 100 < len(c.content) < 1000)
              for doc in sorted({q["doc_source"] for q in questions})]
    print(f"Loading cached {config.EMBEDDING_MODEL_NAME} on CPU; {len(queries)} queries", flush=True)
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, local_files_only=True, device="cpu")
    texts = queries + [_prepare_embedding_text(c.content) for c in probes]
    vectors = model.encode(texts, batch_size=8, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    if vectors.shape != (len(texts), 1024) or not np.isfinite(vectors).all():
        raise ValueError("Invalid BGE-M3 embedding shape or values")
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / "query_vectors.npy", vectors[:len(queries)], allow_pickle=False)
    np.save(destination / "probe_vectors.npy", vectors[len(queries):], allow_pickle=False)
    write_json(destination / "manifest.json", dict(
        model=config.EMBEDDING_MODEL_NAME, provider="local_cached_sentence_transformers_cpu",
        python=sys.version, packages={p:version(p) for p in ("torch", "transformers", "sentence-transformers", "numpy")},
        questions=file_fingerprint(questions_path), queries=queries,
        query_sha256=[hashlib.sha256(q.encode()).hexdigest() for q in queries],
        probe_chunk_ids=[c.chunk_id for c in probes],
        query_vectors=file_fingerprint(destination / "query_vectors.npy"),
        probe_vectors=file_fingerprint(destination / "probe_vectors.npy"),
        pooling="model's saved SentenceTransformer modules; normalized dense vectors",
        local_model_config=getattr(model[0].auto_model.config, "_commit_hash", None)))
    print(f"Saved {len(queries)} query vectors and {len(probes)} probe vectors: {destination}", flush=True)


class CachedQueryEmbedder:
    def __init__(self, directory: Path, questions_path: Path):
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["questions"]["sha256"] != file_fingerprint(questions_path)["sha256"]:
            raise ValueError("Embedding cache belongs to a different question set")
        path = directory / "query_vectors.npy"
        if self.manifest["query_vectors"]["sha256"] != file_fingerprint(path)["sha256"]:
            raise ValueError("Embedding cache checksum mismatch")
        vectors = np.load(path, allow_pickle=False)
        if vectors.shape != (len(self.manifest["queries"]), 1024) or not np.isfinite(vectors).all():
            raise ValueError("Invalid cached query vectors")
        self._vectors = dict(zip(self.manifest["queries"], vectors))

    def embed_query(self, query):
        key = _prepare_embedding_text(query.strip())
        if key not in self._vectors:
            raise ValueError("Query absent from frozen embedding cache; no approximate substitute allowed")
        return self._vectors[key].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=config.DATA_DIR / "evaluation/thesis_benchmark/questions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=config.BASE_DIR / "evaluation/cache/thesis_bge_m3")
    args = parser.parse_args()
    build_cache(args.questions, args.output_dir)


if __name__ == "__main__":
    main()
