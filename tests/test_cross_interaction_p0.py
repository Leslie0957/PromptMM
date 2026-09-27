"""Synthetic only: no real assets, model checkpoints, GPU or split files."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
import cross_interaction_p0 as p0
from td_distill_model_no_projection import TDDistillNoProjectionModel, bpr_loss, directional_distillation_loss


class P0Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = TDDistillNoProjectionModel(6, 9, 4, 4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=6e-5, weight_decay=0.01)
        self.batch = [[0, 2, 7], [1, 2, 8], [2, 3, 2], [3, 4, 6]]
        self.tensor = torch.tensor(self.batch)
        # Synthetic history creates nonzero moments and step (not fresh AdamW).
        for _ in range(3):
            self.optimizer.zero_grad()
            p0.per_loss(self.model, self.tensor).mean().backward()
            self.optimizer.step()
        self.checkpoint = {'model_state_dict': deepcopy(self.model.state_dict()),
                           'optimizer_state_dict': deepcopy(self.optimizer.state_dict())}
        self.target = torch.randn(9, 4)
        self.probes = {'C': {'positive': {'triples': [[4, 2, 5]], 'mass': 0.2},
                             'negative': {'triples': [[5, 1, 2]], 'mass': 0.1}}}

    def test_zero_and_source_unchanged(self):
        before = deepcopy(self.checkpoint)
        with patch('torch.load', side_effect=AssertionError('No assets allowed')):
            out = p0.paired_u1(self.checkpoint, self.target, self.batch, self.probes, 2, 0)
        self.assertEqual(out['delta_norm'], 0)
        for k, v in before['model_state_dict'].items():
            self.assertTrue(torch.equal(v, self.checkpoint['model_state_dict'][k]))
        for i, state in before['optimizer_state_dict']['state'].items():
            for k, v in state.items():
                self.assertTrue(torch.equal(v, self.checkpoint['optimizer_state_dict']['state'][i][k]))

    def test_multiplicity_reduction_matches_formal_loss(self):
        selected = self.tensor[:, 1] == 2
        pos = self.model.item_id_embedding(self.tensor[:, 1][selected])
        expected = directional_distillation_loss(pos, self.target[self.tensor[:, 1][selected]]) * selected.sum() / len(selected)
        torch.testing.assert_close(p0.item_kd(self.model, self.tensor, self.target, 2), expected)
        self.assertEqual(float(p0.item_kd(self.model, self.tensor, self.target, 0).detach()), 0)

    def test_reference_adamw_and_utility_sign(self):
        weight = p0.COEFFICIENTS['image']
        result = p0.paired_u1(self.checkpoint, self.target, self.batch, self.probes, 2, weight)
        models = []
        for coefficient in (0, weight):
            model = deepcopy(self.model)
            opt = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
            opt.load_state_dict(deepcopy(self.checkpoint['optimizer_state_dict']))
            u, p, n = self.tensor.T
            loss = bpr_loss(model.user_id_embedding(u), model.item_id_embedding(p), model.item_id_embedding(n))
            # Independent masked full-batch reduction reference.
            terms = (torch.nn.functional.normalize(model.item_id_embedding(p), dim=-1) -
                     torch.nn.functional.normalize(self.target[p], dim=-1)).square().mean(-1)
            loss = loss + coefficient * (terms * (p == 2)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            models.append(model)
        probe = torch.tensor(self.probes['C']['positive']['triples'])
        expected = float((p0.per_loss(models[0], probe).double() - p0.per_loss(models[1], probe).double()).mean().detach())
        self.assertAlmostEqual(result['pools']['C']['positive']['u1'], expected, places=12)
        self.assertEqual(result['multiplicity'], 2)

    def test_optimizer_branch_independence(self):
        a, oa = p0.branch(self.checkpoint, 'cpu')
        b, ob = p0.branch(self.checkpoint, 'cpu')
        sa, sb = next(iter(oa.state.values())), next(iter(ob.state.values()))
        sa['exp_avg'].add_(9)
        self.assertFalse(torch.equal(sa['exp_avg'], sb['exp_avg']))
        self.assertTrue(torch.equal(sb['exp_avg'], self.checkpoint['optimizer_state_dict']['state'][0]['exp_avg']))

    def test_missing_optimizer_rejected(self):
        with self.assertRaises(KeyError):
            p0.validate_checkpoint({'model_state_dict': self.checkpoint['model_state_dict']})

    def test_disjoint_edges_and_low_degree(self):
        matrix = sp.csr_matrix(([1] * 10, ([0, 1, 2, 3, 4, 5, 0, 1, 2, 0], [0, 0, 0, 0, 0, 0, 1, 1, 1, 2])), shape=(6, 9))
        pools, positives, degree, frequency = p0.pools_from_train(matrix)
        sets = [set(map(tuple, p)) for p in pools.values()]
        self.assertEqual(sum(map(len, sets)), 10)
        for i, a in enumerate(sets):
            for b in sets[i + 1:]:
                self.assertFalse(a & b)
        absent = p0.role_probe(pools['C'], positives, degree, 9, 2, 'positive', 8)
        self.assertEqual(absent['triples'], [])
        batch = p0.update_batch(pools['update'], positives, 9, size=12)
        for u, p, n in batch:
            self.assertIn((u, p), sets[0]); self.assertNotIn(n, positives[u])

    def test_role_mass_is_sampler_probability_not_half(self):
        positives = {0: {0, 1}, 1: {1}}
        degree = np.array([2, 1])
        edges = np.array([[0, 0], [0, 1], [1, 1]])
        pos = p0.role_probe(edges, positives, degree, 4, 0, 'positive', 1)
        neg = p0.role_probe(edges, positives, degree, 4, 0, 'negative', 2)
        self.assertAlmostEqual(pos['mass'], 0.25)
        self.assertAlmostEqual(neg['mass'], 1 / 6)
        self.assertEqual(neg['triples'][0], (1, 1, 0))

    def test_launcher_describe_does_not_load_assets(self):
        spec = importlib.util.spec_from_file_location('p0_launcher', ROOT / 'tools/run_cross_interaction_p0.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with patch.object(sys, 'argv', ['p0', '--describe']), patch.object(module, 'worker', side_effect=AssertionError), patch.object(module, 'source', side_effect=AssertionError):
            module.main()

    def test_source_exempts_only_user_reviews(self):
        spec = importlib.util.spec_from_file_location('p0_launcher', ROOT / 'tools/run_cross_interaction_p0.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with patch.object(module.subprocess, 'check_output', side_effect=['?? check/a.txt\n', 'abc\n']):
            self.assertEqual(module.source(), 'abc')
        with patch.object(module.subprocess, 'check_output', return_value=' M codes/main_mmlight.py\n'):
            with self.assertRaises(RuntimeError): module.source()

    def test_heldout_guard_and_timeout_preservation(self):
        spec = importlib.util.spec_from_file_location('p0_launcher', ROOT / 'tools/run_cross_interaction_p0.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for name in ('data/sports/val_mat', 'D:\\data\\test_mat', 'val.json', 'test.json'):
            with self.assertRaises(RuntimeError): module.deny_heldout('open', (name, 'rb', 0))
        module.deny_heldout('open', ('data/sports/train_mat', 'rb', 0))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(module, 'OUT', Path(tmp) / 'output'), patch.object(module, 'source', return_value='synthetic'), patch.object(sys, 'argv', ['p0']), patch.object(module.subprocess, 'run', side_effect=module.subprocess.TimeoutExpired('synthetic', 600)):
                with self.assertRaises(SystemExit): module.main()
                self.assertTrue((module.OUT / 'launch.json').exists())
                self.assertIn('hard 600s timeout', (module.OUT / 'supervisor.json').read_text())
                with self.assertRaises(FileExistsError): module.main()


if __name__ == '__main__':
    unittest.main()
