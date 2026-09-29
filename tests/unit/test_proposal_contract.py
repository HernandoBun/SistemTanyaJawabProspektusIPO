from types import SimpleNamespace
from unittest.mock import Mock
import json

import pytest
import config
from src.services.pipeline import RAGPipeline
from src.services import diagnostics
from src.generation import generator
from evaluation.evaluate_retrieval import evaluate


@pytest.mark.parametrize('score,expected_calls', [(0.0199, 0), (0.02, 1), (0.03, 1)])
def test_best_rrf_score_gates_generation(monkeypatch, score, expected_calls):
    monkeypatch.setattr(config, 'MIN_RRF_SCORE', 0.02)
    pipeline = object.__new__(RAGPipeline)
    pipeline._is_ready = True
    pipeline._active_doc = 'TEST.pdf'
    pipeline._generator = Mock()
    pipeline.retrieve = Mock(return_value=[SimpleNamespace(rrf_score=score)])
    question = '  Berapa total aset?  '
    response = pipeline.query(question)
    assert pipeline._generator.generate.call_count == expected_calls
    if expected_calls:
        assert pipeline._generator.generate.call_args.kwargs['question'] == question
    else:
        assert response.source_chunks == []
        assert response.source_pages == []
        assert len(response.retrieval_candidates) == 1


def test_no_retrieval_does_not_initialize_llm():
    pipeline = object.__new__(RAGPipeline)
    pipeline._is_ready = True
    pipeline._active_doc = 'TEST.pdf'
    pipeline._generator = None
    pipeline._init_generator = Mock(side_effect=AssertionError('LLM should not load'))
    pipeline.retrieve = Mock(return_value=[])
    assert pipeline.query('Berapa aset?').source_pages == []


def test_dns_failure_is_actionable(monkeypatch):
    monkeypatch.setattr(config, 'EMBEDDING_PROVIDER', 'openrouter')
    monkeypatch.setattr(config, 'OPENROUTER_API_KEY', 'test')
    monkeypatch.setattr(diagnostics.socket, 'getaddrinfo', Mock(side_effect=OSError('DNS')))
    with pytest.raises(ConnectionError, match='DNS'):
        diagnostics.check_embedding_connection()


def test_chunk_metrics_use_ids_not_coincident_page_numbers(tmp_path):
    questions = tmp_path/'questions.jsonl'
    truth = tmp_path/'truth.jsonl'
    questions.write_text(json.dumps({'question_id':'q1','doc_source':'TEST.pdf','question':'Aset?','answerable':True}))
    truth.write_text(json.dumps({'question_id':'q1','source_pages':[3],'relevant_chunk_ids':['correct']}))
    chunks = [SimpleNamespace(chunk=SimpleNamespace(chunk_id='wrong',page_number=3),rrf_score=.03)]
    pipeline=Mock()
    pipeline.retrieve.return_value=chunks
    result=evaluate(questions,truth,pipeline=pipeline,relevance_unit='chunk')
    assert result['summary']['hybrid_rrf']['precision_at_1']==0
    assert result['summary']['hybrid_rrf']['recall_at_1']==0


def test_empty_llm_answer_is_failure_not_reasoning_output(monkeypatch):
    monkeypatch.setattr(config, 'OPENROUTER_API_KEY', 'test')
    client=Mock()
    client.invoke.return_value=SimpleNamespace(content='',additional_kwargs={'reasoning_content':'internal'})
    monkeypatch.setattr(generator, 'ChatOpenAI', Mock(return_value=client))
    chunk=SimpleNamespace(page_number=1,chunk_type='table',section_heading='Aset',table_title='',financial_unit='Rupiah',content='100')
    with pytest.raises(RuntimeError,match='kosong'):
        generator.RAGGenerator().generate('Aset?', [SimpleNamespace(chunk=chunk,rrf_score=.03)])
