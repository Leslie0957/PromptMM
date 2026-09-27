"""Synthetic CPU protocol checks. No real assets or held-out splits."""

import sys
from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from cross_interaction_p1a_coverage import coverage


class P1aCoverageTests(unittest.TestCase):
    def setUp(self):
        # Eight users, 12 items. Items 2/4/6 each have four train edges;
        # item 8 is low support, and item 11 has no train edge.
        edges = [(u, item) for item in (2, 4, 6) for u in range(4)]
        edges += [(0, 8), (1, 8), (6, 9), (7, 10)]
        rows, cols = zip(*edges)
        self.matrix = sp.csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(8, 12))

    def test_deterministic_train_only_and_valid_negatives(self):
        first = coverage(self.matrix, seed=19, batch_size=6, per_stratum=2, exclude=())
        self.assertEqual(first, coverage(self.matrix, seed=19, batch_size=6,
                                         per_stratum=2, exclude=()))
        self.assertEqual(first['kind'], 'train_only_feasibility_not_launch_plan')
        self.assertEqual(sum(first['pool_edges'].values()), first['edges'])
        train = {(u, p) for u, p in zip(*self.matrix.nonzero())}
        for batch in first['contexts']:
            for u, p, n in batch:
                self.assertIn((u, p), train)
                self.assertNotIn((u, n), train)
        for item in first['items']:
            seen = set()
            for pool in item['pools'].values():
                pairs = {tuple(pair) for pair in pool['positive_pairs']}
                self.assertFalse(seen & pairs)
                seen |= pairs

    def test_exclusion_and_sparse_shortfall(self):
        out = coverage(self.matrix, seed=19, batch_size=6, per_stratum=3,
                       exclude=(2, 4, 6))
        self.assertFalse({2, 4, 6} & {x['item'] for x in out['items']})
        self.assertEqual(out['degree_0'], 6)
        self.assertEqual(out['degree_1_to_3'], 3)
        self.assertTrue(any(s['shortfall'] > 0 for s in out['strata']))
        self.assertEqual(out['strata'][0]['nonzero_items'] +
                         out['strata'][1]['nonzero_items'] +
                         out['strata'][2]['nonzero_items'], 6)

    def test_reject_user_with_no_negative(self):
        full = sp.csr_matrix(np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, 'train-unobserved'):
            coverage(full, seed=1, batch_size=2, exclude=())


if __name__ == '__main__':
    unittest.main()
