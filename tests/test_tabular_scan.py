"""The scan of a table: the blocks of the language layer with the tabular cost, one pass, any size."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from acan.core.concepts import ConceptModel, Encoder
from acan.core.scan import KEYS, Radius, Scan, aggregate, rows_as_concepts, scan


def table(n, seed, missing=True):
    """Groups of unequal size with numbers, categories, missing values and a class."""
    rng, world = np.random.default_rng(seed), np.random.default_rng(0)
    centres, kinds, colours = world.normal(0, 3, (12, 4)), world.integers(0, 4, 12), world.integers(0, 3, 12)
    group = rng.choice(12, n, p=world.dirichlet(np.full(12, 1.)))
    frame = pd.DataFrame(centres[group]+rng.normal(0, 1, (n, 4)), columns=list('abcd'))
    frame['colour'] = pd.Series(np.where(rng.random(n) < .9, colours[group], rng.integers(0, 3, n))).map(lambda v: 'rgb'[v])
    if missing:
        frame.loc[rng.random(n) < .1, 'b'] = np.nan
        frame.loc[rng.random(n) < .05, 'colour'] = None
    frame['kind'] = pd.Series(np.where(rng.random(n) < .9, kinds[group], rng.integers(0, 4, n))).map(lambda v: f'k{v}')
    return frame


def prepared(frame, labels=('kind',)):
    model = ConceptModel(seed=0)
    model.labels, model.encoder = list(labels), Encoder().fit(frame)
    base = model.encoder.transform(frame)
    model._roots(base)
    return model, base


class TabularScanTests(unittest.TestCase):
    def test_radius_written_on_the_sums_is_the_radius_of_the_model(self):
        frame = table(400, 1)
        for labels in (('kind',), ()):
            model, base = prepared(frame, labels)
            rows = rows_as_concepts(model, base)
            member = np.eye(20)[np.random.default_rng(0).integers(0, 20, len(frame))]
            groups = {key: member.T@value for key, value in rows.items()}
            radius = Radius(model)
            np.testing.assert_allclose(radius(rows), model._radius(rows), atol=1e-12)
            np.testing.assert_allclose(radius(groups), model._radius(groups), atol=1e-12)
            one = {key: value[3:4] for key, value in groups.items()}
            np.testing.assert_allclose(radius(groups, one), model._radius({key: value+one[key] for key, value in groups.items()}), atol=1e-12)
            seen = rows_as_concepts(model, model._context(groups, base, model._weights(0.)))      # the scale of a level above
            np.testing.assert_allclose(radius(seen), model._radius(seen), atol=1e-12)

    def test_copies_are_added_together_first_and_every_sum_is_kept(self):
        frame = pd.concat([table(8, 2, missing=False)]*4, ignore_index=True)     # every row four times
        model, base = prepared(frame, ())
        rows = {key: value[None] for key, value in rows_as_concepts(model, base).items()}
        left = aggregate(model, rows, 8)
        self.assertEqual(left['C'].shape, (1, 8))
        np.testing.assert_allclose(left['C'][0], 4.)                             # the four copies of each row, and nothing else
        for key in KEYS:
            np.testing.assert_allclose(left[key].sum(1), rows[key].sum(1), atol=1e-9)
        np.testing.assert_allclose(Radius(model)(left), 0., atol=1e-6)

    def test_one_pass_in_chunks_or_in_parts_keeps_every_row_and_few_concepts(self):
        frame = table(3000, 3)
        model, base = prepared(frame)
        whole = scan(model, base, 16)
        chunks = Scan(model, 16)
        parts = Scan(model, 16)
        for start in range(0, 3000, 700):
            piece = tuple(item[start:start+700] for item in base)
            chunks.push(piece)
            parts.absorb(Scan(model, 16).push(piece))
        rows = rows_as_concepts(model, base)
        for found in (whole, chunks.result(), parts.result()):
            self.assertLessEqual(len(found['C']), 16*(int(np.log2(3000/16))+1)+16)
            for key in KEYS:
                np.testing.assert_allclose(found[key].sum(0), rows[key].sum(0), rtol=1e-9, atol=1e-6)

    def test_encoder_from_chunks_is_the_encoder_of_the_whole_table(self):
        frame = table(1000, 4)
        whole = Encoder().fit(frame)
        parts = Encoder().fit_chunks(frame.iloc[start:start+130] for start in range(0, 1000, 130))
        self.assertEqual((whole.numeric, whole.categorical, whole.categories), (parts.numeric, parts.categorical, parts.categories))
        np.testing.assert_allclose(whole.mean, parts.mean, atol=1e-10)
        np.testing.assert_allclose(whole.std, parts.std, atol=1e-10)

    def test_a_table_read_a_chunk_at_a_time_predicts_as_one_held_in_memory(self):
        train, test = table(6000, 5), table(2000, 6)
        held = ConceptModel(seed=0).fit(train, ['kind'])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'table.csv'
            train.to_csv(path, index=False)
            read = lambda: pd.read_csv(path, chunksize=900)
            for workers in (1, 2):
                streamed = ConceptModel(seed=0, refine_rows=2000, workers=workers).fit_chunks(read, ['kind'])
                self.assertEqual(streamed.rows, 6000)
                self.assertAlmostEqual(float(streamed.statistics['C'].sum()), 6000., places=3)     # sums over every row
                accuracy = (streamed.predict(test, 'kind') == test['kind']).mean()
                self.assertGreater(accuracy, (held.predict(test, 'kind') == test['kind']).mean()-.03)
        large = ConceptModel(seed=0, refine_rows=2000, chunk_rows=1500).fit(train, ['kind'])      # the same path from a frame
        self.assertAlmostEqual(float(large.statistics['C'].sum()), 6000., places=3)

    def test_same_seed_same_model(self):
        frame = table(1500, 7)
        first, second = (ConceptModel(seed=3).fit(frame, ['kind']) for _ in range(2))
        np.testing.assert_array_equal(first.statistics['U'], second.statistics['U'])


if __name__ == '__main__':
    unittest.main()
