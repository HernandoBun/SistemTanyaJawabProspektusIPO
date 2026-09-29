"""Export candidate chunks for human review; never manufacture ground truth."""
import argparse
import json
import pickle
from pathlib import Path

import config
from evaluation.evaluate_retrieval import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='output/anotasi_kandidat.json')
    args = parser.parse_args()
    # Only read the locally generated index, never an uploaded pickle.
    with config.BM25_INDEX_PATH.open('rb') as handle:
        corpus = pickle.load(handle)['corpus_chunks']
    truths = {row['question_id']: row for row in load_jsonl('data/evaluation/ground_truth.jsonl')}
    rows = []
    for question in load_jsonl('data/evaluation/questions.jsonl'):
        truth = truths.get(question['question_id'], {})
        candidates = [c for c in corpus if c.doc_source == question['doc_source']
                      and c.page_number in truth.get('source_pages', [])]
        rows.append({**question, 'reference_answer': truth.get('answer', ''),
                     'instruction': 'Periksa terhadap PDF asli. Pilih hanya chunk yang mendukung jawaban; kandidat bukan label relevansi.',
                     'candidates': [{'chunk_id': c.chunk_id, 'page': c.page_number, 'content': c.content}
                                    for c in candidates]})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Kandidat anotasi: {destination}. Ground truth tidak diubah.')


if __name__ == '__main__':
    main()
