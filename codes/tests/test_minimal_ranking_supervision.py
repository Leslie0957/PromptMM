import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'tools'))
from minimal_ranking_supervision import (anchor_loss, candidate_rows, initial_scales,
                                         mixed_tables, rank_kl)
from run_minimal_ranking_supervision import test_denial, train_step


class ObjectiveTests(unittest.TestCase):
    def test_kl_zero_gradient_and_teacher_detach(self):
        user = torch.tensor([[1., 2.]], requires_grad=True)
        item = torch.tensor([[[.3, .1], [.2, .4], [.9, -.2]]], requires_grad=True)
        logits = (user.detach()[:, None, :] * item.detach()).sum(-1)
        teacher = logits.clone().requires_grad_(True)
        zero = rank_kl(user, item, teacher, torch.zeros(1), torch.ones(1))
        self.assertLess(abs(float(zero.detach())), 1e-6)
        gradients = torch.autograd.grad(zero, (user, item), retain_graph=True)
        self.assertLess(max(float(g.abs().max()) for g in gradients), 1e-6)
        self.assertIsNone(torch.autograd.grad(zero, teacher, allow_unused=True)[0])
        changed = rank_kl(user, item, teacher + torch.tensor([[2., 0., -2.]]),
                          torch.zeros(1), torch.ones(1))
        self.assertGreater(float(changed.detach()), 0.01)

    def test_anchor_coordinate_squares_and_scale(self):
        initial_u = torch.tensor([[1., 2.]])
        initial_i = torch.tensor([[2., 0.], [0., 2.]])
        su, si = initial_scales(initial_u, initial_i)
        self.assertAlmostEqual(su, 2.5)
        self.assertAlmostEqual(si, 2.)
        zero = anchor_loss(initial_u, initial_i[:1], initial_i[1:],
                           initial_u, initial_i[:1], initial_i[1:], su, si)
        self.assertEqual(float(zero), 0.)
        shifted = anchor_loss(initial_u + torch.tensor([[1., -1.]]), initial_i[:1],
                              initial_i[1:], initial_u, initial_i[:1], initial_i[1:], su, si)
        self.assertAlmostEqual(float(shifted), 1 / 2.5 / 3, places=6)

    def test_mixed_dot(self):
        u = torch.tensor([[1., 2.]])
        i = torch.tensor([[3., 4.], [5., 6.]])
        tu = torch.tensor([[2., -1.]])
        ti = torch.tensor([[7., 8.], [9., 10.]])
        mu, mi = mixed_tables(u, i, tu, ti)
        self.assertTrue(torch.allclose(mu @ mi.T, .5 * (u @ i.T) + .5 * (tu @ ti.T), atol=1e-6))

    def test_candidate_filter_tie_rng_isolation(self):
        train = csr_matrix((np.ones(2), ([0, 1], [0, 1])), shape=(2, 80))
        user = torch.ones((2, 2))
        item = torch.zeros((80, 2))
        item[2:4] = 1.
        np.random.seed(17)
        before = np.random.get_state()
        a = candidate_rows(train, user, item, seed=20260930)
        b = candidate_rows(train, user, item, seed=20260930)
        after = np.random.get_state()
        self.assertTrue(np.array_equal(a[1], b[1]))
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertEqual(a[1][0, :2].tolist(), [2, 3])
        for row, seen in enumerate((0, 1)):
            self.assertEqual(len(set(a[1][row])), 64)
            self.assertNotIn(seen, a[1][row])

    def test_step_and_abort(self):
        model = torch.nn.Module()
        model.user_id_embedding = torch.nn.Embedding(2, 2)
        model.item_id_embedding = torch.nn.Embedding(4, 2)
        initial = {'user': model.user_id_embedding.weight.detach().clone(),
                   'item': model.item_id_embedding.weight.detach().clone()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids = torch.tensor([[0, 1], [0, 1], [2, 3]])
        candidates = {'row_for_user': torch.tensor([0, 1]),
                      'ids': torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
                      'logits': torch.zeros((2, 4)), 'mu': torch.zeros(2),
                      'scale': torch.ones(2)}
        values = train_step(model, optimizer, 'B', ids, candidates, initial, (1., 1.))
        self.assertTrue(np.isfinite(values['total']))
        self.assertEqual(len(optimizer.state), 2)
        with torch.no_grad():
            model.user_id_embedding.weight[0, 0] = float('nan')
        with self.assertRaises(RuntimeError):
            train_step(model, optimizer, 'A', ids, candidates, initial, (1., 1.))

    def test_test_path_denial_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = test_denial(root)
            with self.assertRaises(PermissionError):
                (root / 'test_mat').open('rb')
            self.assertEqual(count['denied_test_open_attempts'], 1)


if __name__ == '__main__':
    unittest.main()
