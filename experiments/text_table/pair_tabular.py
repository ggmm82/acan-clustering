"""Held-out tests of the same parent-local categorical readout on ordinary tables."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from acan.core.concepts import ConceptModel
from acan.vis.datasets import clean_frame, read_arff, target_and_filters


def run(directory):
    results = {}
    for filename in ("vote.arff", "breast-cancer.arff", "mushroom.arff"):
        frame = clean_frame(read_arff(next(directory.rglob(filename)).read_bytes()))
        label = target_and_filters(frame)[0][0]
        frame = frame.dropna(subset=[label]).copy()
        frame[label] = frame[label].astype(str)
        train, test = train_test_split(frame, test_size=0.2, random_state=42, stratify=frame[label])
        build, judge = train_test_split(train, test_size=0.2, random_state=42, stratify=train[label])
        model = ConceptModel().fit(build, labels=[label])

        def score():
            p, classes = model.predict_proba(test.drop(columns=label), label)
            y = pd.Index(classes).get_indexer(test[label])
            return dict(
                accuracy=float((p.argmax(1) == y).mean()),
                nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
            )

        before = score()
        categorical = [x for x in model.encoder.categorical if x != label]
        report = {}
        if len(categorical) >= 2:
            parent, child = categorical[:2]
            report = model.fit_category_pair(label, parent, child, build, judge)
            report["columns"] = [parent, child]
        results[filename] = dict(
            build=len(build),
            judgment=len(judge),
            test=len(test),
            before=before,
            after=score(),
            local=report,
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.data), indent=2) + "\n")
