"""Synthetic-only checks for the frozen P1a train probe protocol."""
from pathlib import Path
import sys
import unittest

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from cross_interaction_p1a_plan import make_plan, digest, validate_plan


class PlanTests(unittest.TestCase):
    def setUp(self):
        edges = [(u, i) for i in range(1, 7) for u in range(6)]
        rows, cols = zip(*edges)
        self.matrix = sp.csr_matrix((np.ones(len(edges)), (rows, cols)), shape=(8, 12))
        self.positives = {u: set(self.matrix.indices[self.matrix.indptr[u]:self.matrix.indptr[u+1]])
                          for u in range(6)}
        self.old = {'batch': [[0, 1, 11]], 'probes': {'1': {
            'C': {'positive': {'triples': [[1, 1, 10]]},
                  'negative': {'triples': [[1, 2, 11]]}}}}}

    def test_deterministic_roles_and_exposure(self):
        a = make_plan(self.matrix, self.old, seed=17)
        b = make_plan(self.matrix, self.old, seed=17)
        self.assertEqual(digest(a), digest(b))
        self.assertEqual(a, b)
        self.assertEqual(a['max_nonzero_pairs'], len(a['items']) * 16)
        self.assertEqual(a['states'], ['cold2022', 'warm2022'])
        self.assertTrue(a['items'])
        for row in a['items']:
            self.assertTrue(all(m > 0 for m in row['multiplicity']))
            for probes in row['probes']:
                old_used = {(0, 1), (1, 1), (1, 2)}
                self.assertFalse(old_used & {
                    (u, p) for role in probes['C'].values()
                    for u, p, _ in role['triples']})
                for pool in probes.values():
                    pos = pool['positive']
                    neg = pool['negative']
                    self.assertEqual(len(pos['triples']), pos['base_draws'] * 3)
                    self.assertEqual(len(neg['triples']), neg['base_draws'])
                    self.assertEqual(neg['negative_repeats'], 1 if neg['base_draws'] else 0)
                    self.assertTrue(0 <= pos['mass'] <= 1)
                    self.assertTrue(0 <= neg['mass'] <= 1)

    def test_invalid_probe_rejected(self):
        plan = make_plan(self.matrix, self.old, seed=17)
        target = next(p for row in plan['items'] for p in row['probes']
                      if p['A']['positive']['triples'])
        target['A']['positive']['triples'][0][2] = target['A']['positive']['triples'][0][1]
        with self.assertRaisesRegex(ValueError, 'Invalid probe'):
            validate_plan(plan, self.positives, self.matrix.shape)


if __name__ == '__main__':
    unittest.main()
