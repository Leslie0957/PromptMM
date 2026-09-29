import sys
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'tools'))
from neighbor_shared_residual import effective_items, features, group_metrics, triplet_loss
from run_neighbor_shared_residual import check_profile, run_serial_arms, summarize


class NeighborResidualTests(unittest.TestCase):
    def setUp(self):
        self.q = np.arange(30 * 64, dtype=np.float32).reshape(30, 64) / 1000
        self.ids = np.array([0, 1, 2])
        self.image = np.tile(np.arange(3, 23), (3, 1))
        self.text = np.tile(np.arange(5, 25), (3, 1))
        self.x, _ = features(self.q, self.ids, self.image, self.text)

    def test_zero_scores_frozen_and_export(self):
        q = torch.tensor(self.q, requires_grad=True)
        x = torch.tensor(self.x)
        u = torch.randn(4, 64, requires_grad=True)
        w = torch.nn.Parameter(torch.zeros(64, 128))
        for arm, extra in [('B', None), ('N', w), ('R', w), ('F', torch.nn.Parameter(torch.zeros(2)))]:
            folded = effective_items(q, x, arm, extra)
            self.assertTrue(torch.equal(folded, q))
            self.assertTrue(torch.equal(u @ folded.T, u @ q.T))
        self.assertTrue(torch.equal(x[3:], torch.zeros_like(x[3:])))
        with torch.no_grad():
            w.normal_(0, .01)
        folded = effective_items(q, x, 'N', w)
        self.assertTrue(torch.allclose(u @ folded.T, u @ (q + x @ w.T).T, atol=1e-6))
        self.assertFalse(torch.equal(folded[0], q[0]))
        self.assertTrue(torch.equal(folded[4], q[4]))
        self.assertFalse(x.requires_grad)

    def test_positive_negative_gradient_and_shared_effect(self):
        q = torch.tensor(self.q, requires_grad=True)
        x = torch.tensor(self.x)
        u = torch.ones(1, 64, requires_grad=True)
        w = torch.nn.Parameter(torch.zeros(64, 128))
        ids = torch.tensor([[0], [0], [1]])
        loss = triplet_loss(u, q, x, 'N', w, ids)
        loss.backward()
        self.assertIsNotNone(w.grad)
        self.assertGreater(float(w.grad.abs().sum()), 0)
        self.assertGreater(float(q.grad[0].abs().sum()), 0)
        self.assertGreater(float(q.grad[1].abs().sum()), 0)
        self.assertTrue(torch.allclose(q.grad[0], -q.grad[1], atol=1e-7))
        self.assertIsNone(x.grad)
        before = effective_items(q, x, 'N', w).detach().clone()
        with torch.no_grad():
            w -= .01 * w.grad
        after = effective_items(q, x, 'N', w).detach()
        self.assertGreater(float((after[2] - before[2]).abs().sum()), 0)

    def test_group_denominators(self):
        val = SimpleNamespace(indptr=np.array([0, 2, 3]), indices=np.array([0, 2, 1]))
        metrics = group_metrics(np.array([[0, 1], [1, 2]]), np.array([0, 1]), val,
                                np.array([0, 1, 2]))
        self.assertEqual(metrics['denominator'], [1, 1, 1])
        self.assertEqual(metrics['hits'], [1, 1, 0])
        self.assertAlmostEqual(metrics['recall20'], .75)
        self.assertEqual(sum(metrics['exposure']), 4)

    def test_profile_and_incomplete_summary_stop(self):
        import json
        import tempfile
        p = json.loads((ROOT / 'docs/research/innovation2/NEIGHBOR_SHARED_RESIDUAL_PROFILE_V1.json').read_text())
        check_profile(p)
        p['arms'].append('C')
        with self.assertRaises(ValueError):
            check_profile(p)
        p['arms'].pop()
        with tempfile.TemporaryDirectory() as d, self.assertRaises((FileNotFoundError, RuntimeError)):
            summarize(Path(d), p)

    def test_serial_hard_failure_does_not_skip(self):
        called = []
        def call(arm):
            called.append(arm)
            if arm == 'N':
                raise RuntimeError('hard failure')
        with self.assertRaisesRegex(RuntimeError, 'hard failure'):
            run_serial_arms(['B', 'N', 'F', 'R'], call)
        self.assertEqual(called, ['B', 'N'])


if __name__ == '__main__':
    unittest.main()
