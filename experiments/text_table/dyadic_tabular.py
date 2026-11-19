"""Verifica comparativa sui file originali dell'utente, split e opzioni fissati."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from acan.core.concepts import ConceptModel
from acan.vis.datasets import clean_frame, read_arff, target_and_filters


def measure(path):
    raw = (
        read_arff(path.read_bytes())
        if path.suffix == ".arff"
        else pd.read_csv(path)
        if path.suffix == ".csv"
        else pd.read_excel(path)
    )
    data = clean_frame(raw)
    if path.stem == "GiocoSet":
        outputs = []
        for frame in (data, data.drop(columns="Gioco")):
            m = ConceptModel().fit(frame)
            outputs.append(
                dict(
                    concepts=m.concepts,
                    ari=float(
                        adjusted_rand_score(data.Tempo + " " + data.Luogo, m.assignments(frame).argmax(1))
                    ),
                )
            )
        return dict(all_columns=outputs[0], without_outcome=outputs[1])
    y = target_and_filters(data)[0][0]
    data = data.dropna(subset=[y]).copy()
    numeric = path.stem == "autoPrice"
    if numeric:
        data[y] = pd.to_numeric(data[y])
    else:
        data[y] = data[y].astype(str)
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, stratify=None if numeric else data[y]
    )
    model = ConceptModel().fit(train, labels=[y])
    pred = model.predict(test.drop(columns=y), y)
    score = np.abs(pred - test[y]).mean() if numeric else (pred == test[y]).mean()
    return dict(
        train=len(train),
        test=len(test),
        metric="MAE" if numeric else "accuracy",
        value=float(score),
        levels=[level["concepts"] for level in model.levels],
        prediction_sha256=hashlib.sha256(json.dumps(pred.tolist()).encode()).hexdigest(),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    results = {}
    for source in sorted(args.dir.rglob("*"), key=lambda p: p.name):
        if source.suffix not in (".arff", ".csv", ".xlsx"):
            continue
        start = time.monotonic()
        results[source.name] = dict(
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            **measure(source),
            seconds=time.monotonic() - start,
        )
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(source.name, flush=True)
