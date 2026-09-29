from types import SimpleNamespace

from evaluation.ragas_evaluator import (
    RagasScore,
    extract_retrieved_contexts,
    metric_result_value_reason,
)


def test_extract_retrieved_contexts_ignores_empty_content():
    response = SimpleNamespace(
        source_chunks=[
            SimpleNamespace(chunk=SimpleNamespace(content="  konteks pertama  ")),
            SimpleNamespace(chunk=SimpleNamespace(content="")),
            SimpleNamespace(chunk=SimpleNamespace(content="konteks kedua")),
        ]
    )
    assert extract_retrieved_contexts(response) == ["konteks pertama", "konteks kedua"]


def test_metric_result_value_reason_supports_ragas_result():
    result = SimpleNamespace(value=0.75, reason="cukup didukung")
    assert metric_result_value_reason(result) == (0.75, "cukup didukung")
    assert metric_result_value_reason(1) == (1.0, None)


def test_ragas_score_serialization():
    assert RagasScore(0.9, "didukung").as_dict() == {
        "score": 0.9,
        "reason": "didukung",
    }
