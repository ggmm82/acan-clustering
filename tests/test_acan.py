"""
Comprehensive test suite for ACAN (Algebraic Concept Attention Network v8).
"""

import unittest
import numpy as np
import pandas as pd
from acan import (
    ModularClusterNet,
    AlgebraicClusterModel,
    GraphClusterModel,
    AlgebraicClusterLearner,
    ACANPredictor,
)


class TestACANFramework(unittest.TestCase):
    def test_01_modular_cluster_net_forward_and_shapes(self):
        n, m = 40, 8
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(n, m))
        net = ModularClusterNet(n, m, k=n, tau=0.2, alpha=0.5)
        net.initialize_G_from_data(X)
        net.set_root_statistics(X.mean(axis=0), X.min(axis=0), X.max(axis=0))
        res = net(X)

        self.assertEqual(res["T"].shape, (n, n))
        self.assertEqual(res["G"].shape, (n, m))
        self.assertEqual(res["R"].shape, (n, m))
        # Verify competitive slot attention: columns of T sum to 1.0 along dim=0
        col_sums = np.sum(np.asarray(res["T"]), axis=0)
        np.testing.assert_allclose(col_sums, np.ones(n), atol=1e-5)
        self.assertGreater(float(res["cp"]), 0.0)

    def test_02_concept_compression_and_training(self):
        # Create dataset with 2 distinct behavioral clusters and 4 irrelevant noise columns
        rng = np.random.default_rng(123)
        rows = []
        for i in range(60):
            if i < 30:
                rows.append({"f1": 0.0, "f2": 1.0, "noise1": rng.uniform(0, 1), "noise2": rng.uniform(0, 1), "label": "A"})
            else:
                rows.append({"f1": 1.0, "f2": 0.0, "noise1": rng.uniform(0, 1), "noise2": rng.uniform(0, 1), "label": "B"})
        df = pd.DataFrame(rows)
        gcm = AlgebraicClusterLearner.learn(df, labels=["label"], max_epochs=80, lr=0.08, tau=0.18, alpha=0.6)
        root = gcm.clusterModel[gcm.rootId]

        # Active concepts k_active must be strictly compressed compared to n=60
        self.assertLess(root.k(), 60)
        self.assertGreaterEqual(root.k(), 2)

        # Evaluate accuracy on the dataset
        err, Y_test, Y_pred, Y_conf, _ = gcm.evaluateError(df, ["label"])
        acc = (Y_test.idxmax(axis=1) == Y_pred.idxmax(axis=1)).mean()
        self.assertGreaterEqual(acc, 0.95)
        self.assertGreater(float(Y_conf.values.mean()), 0.5)

    def test_03_universal_masked_tabular_autocompletion(self):
        # Dataset with strong bidirectional functional dependencies: A <-> B <-> C
        rows = []
        for _ in range(25):
            rows.append({"Tempo": "Sole", "Luogo": "Spiazzale", "Gioco": "SI"})
            rows.append({"Tempo": "Sole", "Luogo": "Palazzetto", "Gioco": "NO"})
            rows.append({"Tempo": "Piove", "Luogo": "Spiazzale", "Gioco": "NO"})
            rows.append({"Tempo": "Piove", "Luogo": "Palazzetto", "Gioco": "SI"})
        df = pd.DataFrame(rows)
        gcm = AlgebraicClusterLearner.learn(df, labels=[], max_epochs=90, lr=0.08, tau=0.15, alpha=0.6)

        # Mask different columns on different rows (dynamic target completion)
        masked_df = pd.DataFrame([
            {"Tempo": "Sole", "Luogo": "Spiazzale", "Gioco": np.nan},
            {"Tempo": "Piove", "Luogo": np.nan, "Gioco": "SI"},
            {"Tempo": np.nan, "Luogo": "Spiazzale", "Gioco": "SI"},
        ])
        completed = gcm.predict_masked(masked_df)
        self.assertEqual(completed.loc[0, "Gioco"], "SI")
        self.assertEqual(completed.loc[1, "Luogo"], "Palazzetto")
        self.assertEqual(completed.loc[2, "Tempo"], "Sole")

    def test_04_json_serialization_and_rule_extraction(self):
        df = pd.DataFrame({
            "x": [0.1, 0.12, 0.9, 0.88, 0.11, 0.92],
            "y": [0.2, 0.18, 0.8, 0.82, 0.21, 0.79],
            "cls": ["C1", "C1", "C2", "C2", "C1", "C2"],
        })
        gcm = AlgebraicClusterLearner.learn(df, labels=["cls"], max_epochs=40)
        json_str = gcm.to_json()
        loaded_gcm = GraphClusterModel.loadModel(json_str)
        self.assertEqual(loaded_gcm.rootId, gcm.rootId)
        rules = loaded_gcm.clusterModel[loaded_gcm.rootId].extractRules(minMax=loaded_gcm.minMax)
        self.assertGreaterEqual(len(rules), 1)


if __name__ == "__main__":
    unittest.main()
