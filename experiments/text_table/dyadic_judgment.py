"""Misura il giudizio tabulare su una colonna categorica larga. Scrive i punteggi per verificare la parità."""

import numpy as np, pandas as pd, time, json, resource
from acan.core.concepts import ConceptModel, Encoder


def main():
    rng = np.random.default_rng(71)
    d = pd.DataFrame(rng.normal(size=(800, 6)))
    d.columns = ["x" + str(i) for i in range(6)]
    d["label"] = ["v" + str(i) for i in rng.integers(0, 8000, len(d))]
    m = ConceptModel(covariance=False)
    m.labels = ["label"]
    m.encoder = Encoder().fit(d)
    m.encoder.categories["label"] = ["v" + str(i) for i in range(8000)]
    m.encoder.width = 8000
    data = m.encoder.transform(d)
    m._roots(data)
    t = rng.dirichlet(np.ones(64), len(d))
    s = m._statistics(t, data)
    m.statistics = s
    rows = np.arange(256)
    w = m._weights(6.0)
    start = time.perf_counter()
    out = []
    for distinct in [False, True]:
        m.distinct = distinct
        for structure in [False, True]:
            out.extend(m._score(s, data, w, rows, structure))
    print(
        json.dumps(
            {
                "seconds": time.perf_counter() - start,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "scores": out,
            }
        )
    )


if __name__ == "__main__":
    main()
