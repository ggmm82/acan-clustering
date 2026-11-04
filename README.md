# ACAN clustering

ACAN (Algebraic Concept Attention Network) provides concept discovery and
tabular autocompletion with a NumPy backend and an optional PyTorch backend.
Requires Python 3.9 or newer.

## Development setup

From the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

Install the optional PyTorch backend with `python -m pip install -e '.[torch]'`.

## Contents

- `acan/`: library source
- `tests/`: unit tests
- `benchmarks/`: experimental benchmark script
- `experiments/`: archived results and figures
- `paper/`: paper source

The benchmark script currently contains absolute paths from its original
environment and requires path configuration before use elsewhere.

## License

MIT; see [LICENSE](LICENSE).
