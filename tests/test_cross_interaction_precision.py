"""Synthetic CPU-only precision and launch protocol checks."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
import cross_interaction_p0 as p0
import cross_interaction_precision as precision
from td_distill_model_no_projection import TDDistillNoProjectionModel


def launcher():
    spec = importlib.util.spec_from_file_location('precision_launcher', ROOT / 'tools/run_cross_interaction_precision.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def packet(self, margin, role='positive', zero=False):
        a = TDDistillNoProjectionModel(2, 3, 64, 64)
        with torch.no_grad():
            a.user_id_embedding.weight.zero_()
            a.user_id_embedding.weight[:, 0] = 1
            a.item_id_embedding.weight.zero_()
            a.item_id_embedding.weight[0, 0] = margin
        b = deepcopy(a)
        if not zero:
            with torch.no_grad():
                x = b.item_id_embedding.weight[0, 0]
                x.copy_(torch.nextafter(x, torch.tensor(float('inf'))))
        triple = [0, 0, 1] if role == 'positive' else [0, 1, 0]
        probes = {'C': {role: {'triples': [triple], 'mass': 0.2}}}
        return precision.capture(a, b, probes, 0), probes

    def test_reference_sweep_roles_and_saturation(self):
        for margin in (-1000, -80, -20, -1, 0, 1e-5, 1, 20, 80, 1000):
            for role in ('positive', 'negative'):
                with self.subTest(margin=margin, role=role):
                    rows, probes = self.packet(margin, role)
                    result = precision.evaluate_rows(rows, probes)
                    self.assertTrue(result['consistent'])
                    self.assertTrue(result['rows_unchanged'])
                    value = result['pools']['C'][role]['u1']
                    self.assertGreaterEqual(value, 0) if role == 'positive' else self.assertLessEqual(value, 0)

    def test_zero_and_empty_probes(self):
        rows, probes = self.packet(2, zero=True)
        probes['A'] = {'positive': {'triples': [], 'mass': 0}}
        result = precision.evaluate_rows(rows, probes)
        self.assertEqual(result['pools']['C']['positive']['u1'], 0)
        self.assertIsNone(result['pools']['A']['positive']['u1'])
        self.assertEqual(result['focus_changed_components'], 0)

    def test_cast_before_dot_recovers_cancellation_case(self):
        rows, probes = self.packet(1)
        rows['a_users'][0, :3] = 1
        rows['b_users'][0, :3] = 1
        rows['a_items'][0, :3] = torch.tensor([1e8, 1, -1e8])
        rows['b_items'][0, :3] = torch.tensor([1e8, 1, -1e8])
        rows['b_items'][0, 1] = torch.nextafter(torch.tensor(1.), torch.tensor(float('inf')))
        result = precision.evaluate_rows(rows, probes)
        sample = result['pools']['C']['positive']['samples'][0]
        self.assertTrue(result['consistent'])
        self.assertGreater(sample['stable64'], 0)
        self.assertEqual(sample['cpu32'], 0)
        self.assertGreater(abs(sample['cpu32']-sample['reference']), sample['stable_tolerance'])

    def test_large_delta_fallback(self):
        rows, probes = self.packet(-3)
        rows['b_items'][0, 0] = 9
        result = precision.evaluate_rows(rows, probes)
        self.assertTrue(result['consistent'])
        self.assertGreater(result['pools']['C']['positive']['u1'], 0)

    def test_reference_disagreement_rejected(self):
        rows, probes = self.packet(1)
        with patch.object(precision, 'stable_difference', return_value=1):
            self.assertFalse(precision.evaluate_rows(rows, probes)['consistent'])

    def test_nonlocal_capture_rejected(self):
        a = TDDistillNoProjectionModel(2, 3, 4, 4); b = deepcopy(a)
        with torch.no_grad(): b.item_id_embedding.weight[1].add_(1)
        probes = {'C': {'positive': {'triples': [[0, 0, 1]], 'mass': 1}}}
        with self.assertRaises(ValueError): precision.capture(a, b, probes, 0)

    def test_observer_preserves_float32_update_and_source(self):
        model = TDDistillNoProjectionModel(6, 9, 4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
        batch = [[0, 2, 7], [1, 2, 8], [2, 3, 2], [3, 4, 6]]
        for _ in range(3):
            optimizer.zero_grad(); p0.per_loss(model, torch.tensor(batch)).mean().backward(); optimizer.step()
        checkpoint = dict(model_state_dict=deepcopy(model.state_dict()), optimizer_state_dict=deepcopy(optimizer.state_dict()))
        before = deepcopy(checkpoint)
        target = torch.randn(9, 4)
        probes = {'C': {'positive': {'triples': [[4, 2, 5]], 'mass': 0.2},
                         'negative': {'triples': [[5, 1, 2]], 'mass': 0.1}}}
        legacy = p0.paired_u1(checkpoint, target, batch, probes, 2, p0.COEFFICIENTS['image'])
        observed = p0.paired_u1(checkpoint, target, batch, probes, 2, p0.COEFFICIENTS['image'],
                               observer=lambda a,b,p,i: precision.evaluate_rows(precision.capture(a,b,p,i),p))
        self.assertEqual(legacy, {k:v for k,v in observed.items() if k != 'precision'})
        self.assertTrue(observed['precision']['consistent'])
        for k,v in before['model_state_dict'].items():self.assertTrue(torch.equal(v,checkpoint['model_state_dict'][k]))
        for i,state in before['optimizer_state_dict']['state'].items():
            for k,v in state.items():self.assertTrue(torch.equal(v,checkpoint['optimizer_state_dict']['state'][i][k]))

    def test_rows_roundtrip_and_tolerance_aggregation(self):
        rows, probes = self.packet(1)
        probes['C']['positive']['triples'] *= 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic.pt'; torch.save(rows,path)
            restored = torch.load(path, weights_only=True)
        result = precision.evaluate_rows(restored,probes)
        role = result['pools']['C']['positive']
        self.assertEqual(len(role['samples']),2)
        self.assertEqual(role['tolerance'],role['samples'][0]['stable_tolerance'])
        self.assertEqual(result['pools']['C']['contribution_tolerance'],0.2*role['tolerance'])

    def test_describe_and_heldout_guard(self):
        module=launcher()
        with patch.object(sys,'argv',['precision','--describe']),patch.object(module,'worker',side_effect=AssertionError),patch.object(module,'source',side_effect=AssertionError):
            module.main()
        for name in ('data/test_mat','data/val_mat','val.json','test.json'):
            with self.assertRaises(RuntimeError):module.deny_heldout('open',(name,'rb',0))

    def test_timeout_keeps_artifacts_and_prevents_retry(self):
        module=launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(module,'OUT',Path(tmp)/'out'),patch.object(module,'source',return_value='synthetic'),patch.object(sys,'argv',['precision']),patch.object(module.subprocess,'run',side_effect=module.subprocess.TimeoutExpired('synthetic',600)):
                with self.assertRaises(SystemExit):module.main()
                self.assertIn('hard 600s timeout',(module.OUT/'supervisor.json').read_text())
                with self.assertRaises(FileExistsError):module.main()


if __name__ == '__main__':
    unittest.main()
