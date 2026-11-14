"""The small test of the generative laboratory, from the command line.

Three tiny transformers identical except for attention (ordinary, last tokens only, concepts
plus last tokens) are trained from scratch on memory tasks (near, far, more facts than concepts)
and on the characters of a text; the cost per token is measured at growing context lengths.
See acan/lm/small_test.py.

Example: python scripts/probe_small_lm.py --text .data/wikitext2-test.txt --budget full --output reports/small-lm.json
"""
import argparse
import json
from pathlib import Path

from acan.lm.small_test import BUDGETS, read_corpus, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text', type=Path, required=True, help='a .txt file or a .jsonl file with a text field')
    parser.add_argument('--budget', choices=sorted(BUDGETS), default='quick')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    text = read_corpus(args.text, args.text.suffix.lower() == '.jsonl')
    result = run(text, args.budget, report=lambda phase: print(phase, flush=True), seed=args.seed)
    for kind, item in result['memory'].items():
        print(f"memoria · {kind}: {100*item['accuracy']:.1f}% risposte giuste")
    for name, case in result['memory_cases'].items():
        print(f"memoria {name} ({case['pairs']} coppie, risposta a {case['nearest_answer_distance']} token): "
              f"normale {100*case['dense']['accuracy']:.1f}%, concetti {100*case['concepts']['accuracy']:.1f}%")
    for row in result['cost']['rows']:
        print(f"costo a {row['length']} token: normale {row['dense']['milliseconds_per_token']:.3f} ms/token, "
              f"concetti {row['concepts']['milliseconds_per_token']:.3f} ms/token")
    for kind, item in result['text'].items():
        print(f"testo · {kind}: {item['bits_per_character']:.3f} bit per carattere")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
