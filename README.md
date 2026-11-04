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

## Interactive application

```sh
python -m pip install -e '.[ui]'
python -m acan.vis.render
```

Open `http://localhost:4000` in your browser.

1. Upload a dataset in **Data**, select targets in **Model**, and click **Learn Model**.
2. Inspect the learned graph and held-out results in **Evaluation**.
3. Use **Download Model** to save a JSON model. Later, use **Load Model JSON**
   in **Model** to restore it, including its target labels.
4. In **Inference**, upload a new UTF-8 CSV and choose **Predici Labels** or
   **Completa campi vuoti**. Click **Esegui inferenza**, inspect the preview,
   and download the complete CSV. No training occurs in this tab.

Prediction ignores any supplied target values and predicts the saved model's
labels. Completion fills empty cells in columns present in the trained schema,
preserving observed values; include the columns you want completed in the CSV.
Each incomplete row needs at least one known attribute. A model without target
labels can still be used in completion mode.

Training jobs are isolated by signed browser session and a second concurrent
training in the same session is rejected. Use one server process: jobs are kept
in memory, are lost on restart, and completed jobs expire after one hour when
new jobs start. Separate browser profiles have separate sessions; tabs sharing
cookies share a session. This is a local research application, without accounts
or durable job storage. Inference uses the model held in the current page.

The default development command enables Dash debugging. For local runs without
the debugger/reloader, use:

```sh
python -c "from acan.vis.render import app; app.run(debug=False, port=4000)"
```

## Contents

For Docker deployment and the PR → CI → prerelease → stable release process,
see [Controlled Docker releases](docs/RELEASING.md) and [Contributing](CONTRIBUTING.md).

- `acan/`: library source
- `tests/`: unit tests
- `benchmarks/`: experimental benchmark script
- `experiments/`: archived results and figures
- `paper/`: paper source

The benchmark script currently contains absolute paths from its original
environment and requires path configuration before use elsewhere.

## License

MIT; see [LICENSE](LICENSE).
