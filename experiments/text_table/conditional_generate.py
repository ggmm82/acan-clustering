"""Completa un prompt con un checkpoint locale prodotto da conditional_run."""

import argparse
import pickle
from pathlib import Path

from tokenizers import Tokenizer
from .conditional_cursor import ConditionalText


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("prompt")
    args = parser.parse_args()
    if args.tokens < 0:
        parser.error("Il numero di token deve essere non negativo.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    with args.checkpoint.open("rb") as stream:
        model = ConditionalText(*pickle.load(stream))
    if tokenizer.get_vocab_size() != len(model.geometry.vectors):
        parser.error("Il vocabolario del tokenizer è incompatibile con il checkpoint.")
    prefix = tokenizer.encode(args.prompt).ids
    print(tokenizer.decode(prefix + model.generate(prefix, args.tokens)))


if __name__ == "__main__":
    main()
