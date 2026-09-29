"""Summarize real user ratings; never synthesize participant responses."""
import argparse
import json
from pathlib import Path
from statistics import mean

DIMENSIONS = ('relevance', 'usefulness', 'satisfaction')


def summarize_ratings(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('Belum ada penilaian pengguna.')
    seen = set()
    for row in rows:
        identity = (row.get('participant_id'), row.get('question_id'))
        if not all(identity) or identity in seen:
            raise ValueError('Identitas peserta/pertanyaan kosong atau penilaian duplikat.')
        seen.add(identity)
        for dimension in DIMENSIONS:
            score = row.get(dimension)
            if type(score) is not int or not 1 <= score <= 5:
                raise ValueError(f'{dimension} harus bilangan bulat 1 sampai 5.')
    return {
        'responses': len(rows),
        'participants': len({r['participant_id'] for r in rows}),
        'scale': {'min': 1, 'max': 5},
        'mean': {d: mean(r[d] for r in rows) for d in DIMENSIONS},
        'distribution': {d: {str(s): sum(r[d] == s for r in rows) for s in range(1, 6)} for d in DIMENSIONS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('evaluation/results/human.json'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    summary = summarize_ratings(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
