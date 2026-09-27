"""Synthetic analysis, signal and launcher guards; no real assets/GPU."""
from pathlib import Path
import importlib.util
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
from copy import deepcopy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from cross_interaction_p1a_analysis import analyze, _gain, PRACTICAL_GAIN
from cross_interaction_p1a_signal import features
from cross_interaction_p0 import paired_u1, per_loss
from td_distill_model_no_projection import TDDistillNoProjectionModel


class P1aRuntimeTests(unittest.TestCase):
    def test_full_and_half_are_separate_real_adamw_steps(self):
        torch.manual_seed(11)
        model = TDDistillNoProjectionModel(5, 8, 4, 4)
        opt = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=.01)
        batch = [[0, 2, 6], [1, 2, 7], [2, 3, 6], [3, 4, 7]]
        for _ in range(3):
            opt.zero_grad()
            per_loss(model, torch.tensor(batch)).mean().backward()
            opt.step()
        checkpoint = {'model_state_dict': deepcopy(model.state_dict()),
                      'optimizer_state_dict': deepcopy(opt.state_dict())}
        targets = torch.randn(8, 4)
        probe = {'A': {'positive': {'triples': [[4, 2, 6]], 'mass': .2},
                       'negative': {'triples': [[4, 3, 2]], 'mass': .1}}}
        full = paired_u1(checkpoint, targets, batch, probe, 2, .3 / 1.3)
        half = paired_u1(checkpoint, targets, batch, probe, 2, .15 / 1.3)
        self.assertEqual(full['multiplicity'], 2)
        self.assertEqual(half['multiplicity'], 2)
        self.assertGreater(full['delta_norm'], 0)
        self.assertGreater(half['delta_norm'], 0)
        self.assertTrue(torch.equal(checkpoint['model_state_dict']['item_id_embedding.weight'],
                                    model.state_dict()['item_id_embedding.weight']))

    def test_a_features_only_and_finite(self):
        torch.manual_seed(8)
        checkpoint = {'model_state_dict': {
            'user_id_embedding.weight': torch.randn(4, 5),
            'item_id_embedding.weight': torch.randn(7, 5)}}
        targets = torch.randn(7, 5)
        probes = {'positive': {'triples': [[0, 2, 5], [1, 2, 6]], 'mass': .12},
                  'negative': {'triples': [[2, 4, 2]], 'mass': .03}}
        before = {k: v.clone() for k, v in checkpoint['model_state_dict'].items()}
        f = features(checkpoint, targets, 2, probes, 1, 7)
        self.assertEqual(f['a_positive_triples'], 2)
        self.assertEqual(f['a_negative_triples'], 1)
        self.assertIsNotNone(f['both_dot'])
        for k, v in before.items():
            self.assertTrue(torch.equal(v, checkpoint['model_state_dict'][k]))

    def test_additive_surrogate_and_complete_set_gate(self):
        rows = [{'item': i, 'layer': 1 if i < 10 else 2,
                 'score': float(i), 'c_full': i * 1e-8, 'c_half': i * .5e-8}
                for i in range(20)]
        result = _gain(rows, 'score')
        self.assertGreater(result['gain_vs_random'], 0)
        self.assertGreater(result['gain_vs_best_control'], PRACTICAL_GAIN)
        self.assertIsNone(_gain(rows[:-1], 'score'))

    def test_full_synthetic_analysis(self):
        items = list(range(23))
        pairs = []
        for state in ('cold2022', 'warm2022'):
            for context in (0, 1):
                pairs.append({'state': state, 'context': context, 'item': 0,
                              'modality': 'image', 'arm': 'zero'})
                for item in items:
                    layer = 0 if item < 3 else 1 if item < 13 else 2
                    for modality in ('image', 'text'):
                        for arm in ('full', 'half'):
                            v = item * 1e-8 * (1 if arm == 'full' else .5)
                            if item == 0:
                                v = None
                            pair = {'state': state, 'context': context, 'item': item,
                                    'layer': layer, 'modality': modality, 'arm': arm,
                                    'pools': {name: {'positive': {'u1': v}, 'contribution': v}
                                              for name in ('A', 'B', 'C')}}
                            if arm == 'full':
                                pair['a_features'] = {key: float(item) for key in
                                    ('local_dot', 'local_cos', 'positive_dot', 'positive_cos',
                                     'both_dot', 'both_cos', 'hard_margin', 'log_frequency',
                                     'representation_gap', 'constant')}
                            pairs.append(pair)
        self.assertEqual(len(pairs), 372)
        report = {'status': 'running_analysis', 'pairs': pairs}
        result = analyze(report)
        self.assertEqual(len(result['groups']), 4)
        self.assertEqual(result['groups'][0]['mid_high_items'], 20)
        self.assertEqual(result['groups'][0]['low_layer'][0]['reason'], 'C positive N/A')
        self.assertEqual(result['groups'][0]['selectors']['b_oracle']['decision'], 'screen_pass')
        report['pairs'].pop()
        with self.assertRaisesRegex(ValueError, 'Complete'):
            analyze(report)

    def test_launcher_describe_and_heldout_denial(self):
        script = ROOT / 'tools/run_cross_interaction_p1a.py'
        result = subprocess.run([sys.executable, '-B', str(script), '--describe'],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertIn('372 pairs', result.stdout)
        spec = importlib.util.spec_from_file_location('p1a_launcher', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaisesRegex(RuntimeError, 'held-out'):
            module.deny_heldout('open', ('D:/fake/test_mat', 'rb'))
        with patch.object(module.subprocess, 'check_output', return_value='?? check/chatgpt.txt\n'):
            self.assertEqual(module.source(), '?? check/chatgpt.txt')
        with patch.object(module.subprocess, 'check_output', return_value='?? unrelated.txt\n'):
            with self.assertRaisesRegex(RuntimeError, 'Committed source'):
                module.source()


if __name__ == '__main__':
    unittest.main()
