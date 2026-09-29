"""Run retrieval ablation and optional deterministic/RAGAS evaluation."""
from __future__ import annotations

import argparse

from evaluation.compare_methods import METHODS
from evaluation.evaluate_generation import evaluate as evaluate_generation
from evaluation.evaluate_retrieval import evaluate as evaluate_retrieval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument(
        "--with-ragas",
        action="store_true",
        help="Jalankan evaluasi generasi sekaligus empat metrik RAGAS",
    )
    parser.add_argument("--continue-on-ragas-error", action="store_true")
    parser.add_argument("--relevance-unit", choices=["chunk", "page"], default="chunk")
    args = parser.parse_args()
    from evaluation.validate_dataset import validate
    errors = validate("data/evaluation/questions.jsonl", "data/evaluation/ground_truth.jsonl", args.relevance_unit)
    if errors:
        parser.exit(1, "Evaluasi belum dapat dijalankan:\n- " + "\n- ".join(errors) + "\n")
    evaluate_retrieval(
        "data/evaluation/questions.jsonl",
        "data/evaluation/ground_truth.jsonl",
        methods=METHODS,
        output_path="evaluation/results/method_comparison.json",
        relevance_unit=args.relevance_unit,
    )
    if args.with_generation or args.with_ragas:
        evaluate_generation(
            "data/evaluation/questions.jsonl",
            "data/evaluation/ground_truth.jsonl",
            output_path="evaluation/results/generation.json",
            use_ragas=args.with_ragas,
            continue_on_ragas_error=args.continue_on_ragas_error,
        )


if __name__ == "__main__":
    main()
