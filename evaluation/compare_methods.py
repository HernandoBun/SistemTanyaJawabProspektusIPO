"""Ablation study: BM25-only, dense-only, and hybrid RRF."""
from __future__ import annotations

import argparse
import json

from evaluation.evaluate_retrieval import evaluate


METHODS = {
    "bm25_only": {"dense_k": 0, "sparse_k": 10, "final_k": 6},
    "dense_only": {"dense_k": 10, "sparse_k": 0, "final_k": 6},
    "hybrid_rrf": {"dense_k": 10, "sparse_k": 10, "final_k": 6},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="data/evaluation/questions.jsonl")
    parser.add_argument("--ground-truth", default="data/evaluation/ground_truth.jsonl")
    parser.add_argument("--output", default="evaluation/results/method_comparison.json")
    parser.add_argument("--relevance-unit", choices=["chunk", "page"], default="chunk")
    args = parser.parse_args()
    payload = evaluate(
        args.questions,
        args.ground_truth,
        methods=METHODS,
        output_path=args.output,
        relevance_unit=args.relevance_unit,
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
