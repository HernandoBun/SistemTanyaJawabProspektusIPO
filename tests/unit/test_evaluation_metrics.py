from evaluation.evaluate_generation import (
    citation_page_accuracy,
    numerical_accuracy,
    parse_indonesian_number,
    token_f1,
)
from evaluation.evaluate_retrieval import precision_at_k, recall_at_k, reciprocal_rank


def test_indonesian_number_parser():
    assert parse_indonesian_number("1.250.000.000") == 1_250_000_000
    assert parse_indonesian_number("18,7") == 18.7


def test_numerical_accuracy_requires_all_reference_numbers():
    assert numerical_accuracy("Rp1.250.000.000", "Rp1.250.000.000") == 1.0
    assert numerical_accuracy("2025 sebesar Rp1.250", "2024 Rp1.250") == 0.5


def test_token_f1_counts_duplicate_tokens():
    assert token_f1("aset aset naik", "aset naik") < 1.0


def test_retrieval_metrics():
    assert precision_at_k([9, 5, 3], [5], 1) == 0.0
    assert precision_at_k([5, 9, 3], [5], 1) == 1.0
    assert precision_at_k([9, 5, 3], [5], 2) == 0.5
    assert recall_at_k([9, 5, 3], [5], 2) == 1.0
    assert reciprocal_rank([9, 5, 3], [5]) == 0.5
    assert citation_page_accuracy([5, 7], [7]) == 1.0

