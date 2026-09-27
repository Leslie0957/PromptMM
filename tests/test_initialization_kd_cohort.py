"""Full worker lifecycle on tiny synthetic CPU inputs, with all CUDA APIs mocked."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / p) for p in ('tools', 'codes')]
import run_initialization_kd_interaction as launcher
from run_initialization_kd_cohort import load_spec
from initialization_kd_interaction import Protocol
from initialization_kd_runtime import resolve_protocol


class CohortTests(unittest.TestCase):
    def test_frozen_contract(self):
        spec = load_spec()
        self.assertEqual(resolve_protocol(spec), Protocol())
        self.assertIs(launcher._evaluator(spec), launcher.evaluate_validation_fast)
        self.assertIs(launcher._evaluator({}), launcher.evaluate_validation)
        for section, key, value in [('numerics', 'tf32', True),
                                     ('hard_caps', 'parent_wall_seconds', 86401),
                                     ('candidate_screening', 'primary_min_positive_interaction_absolute_recall20', .003)]:
            bad = copy.deepcopy(spec)
            bad[section][key] = value
            with self.assertRaises(RuntimeError):
                resolve_protocol(bad)
        with self.assertRaises(RuntimeError):
            resolve_protocol(dict(spec, evaluator='reference_heapq'))
        with self.assertRaises(RuntimeError):
            launcher._evaluator({'evaluator': 'unknown'})

    def test_full_worker_four_arms_and_acceptance(self):
        p = Protocol(epochs=300, batches_per_epoch=1, batch_size=2, n_users=3, n_items=60, dim=2)
        random = {'user': torch.ones(3, 2) * .1, 'item': torch.arange(120).reshape(60, 2).float() / 100}
        teacher = {k: v.clone() * .8 for k, v in random.items()}
        originals = {k: v.clone() for k, v in random.items()}
        semantics = {'item_image': teacher['item'], 'item_text': teacher['item'],
                     'user_image': teacher['user'], 'user_text': teacher['user']}
        tape = np.tile(np.array([[[[0, 1], [1, 2], [3, 4]]]], dtype=np.int32), (300, 1, 1, 1))
        tape.setflags(write=False)
        train = sp.csr_matrix((np.ones(3), ([0, 1, 2], [1, 2, 3])), shape=(3, 60))
        val = sp.csr_matrix((np.ones(3), ([0, 1, 2], [5, 6, 7])), shape=(3, 60))
        spec = load_spec()
        original_device = torch.device
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                out = Path(tmp)
                for name, result in [('_asset_gate', None), ('_source_gate', 'synthetic'),
                                     ('resolve_protocol', p), ('load_train_val', (train, val)),
                                     ('load_pinned_tensors', (random, teacher, semantics, tape))]:
                    stack.enter_context(patch.object(launcher, name, return_value=result))
                fast = stack.enter_context(patch.object(launcher, 'evaluate_validation_fast', wraps=launcher.evaluate_validation_fast))
                reference = stack.enter_context(patch.object(launcher, 'evaluate_validation', side_effect=AssertionError('Unexpected reference route')))
                stack.enter_context(patch.object(torch, 'device', return_value=original_device('cpu')))
                for name, result in {'is_available': True, 'get_device_properties': SimpleNamespace(total_memory=8*1024**3),
                                     'set_per_process_memory_fraction': None, 'memory_reserved': 0,
                                     'get_device_name': 'synthetic_cpu', 'synchronize': None,
                                     'max_memory_reserved': 0, 'empty_cache': None}.items():
                    stack.enter_context(patch.object(torch.cuda, name, return_value=result))
                launcher._json(out / 'launch_manifest.json', {'source_commit': 'synthetic'})
                launcher._worker(spec, out, {'source_commit': 'synthetic'})
                launcher._accept_completion(spec, out)
                self.assertEqual(fast.call_count, 1204)
                reference.assert_not_called()
                report = json.loads((out / 'report.json').read_text())
                self.assertEqual(list(report['arms']), ['R0', 'R1', 'T0', 'T1'])
                self.assertTrue(all(a['optimizer_steps'] == 300 for a in report['arms'].values()))
                for key in random:
                    self.assertTrue(torch.equal(random[key], originals[key]))
                report['interaction']['I'] += 1
                launcher._json(out / 'report.json', report)
                with self.assertRaisesRegex(RuntimeError, 'interaction'):
                    launcher._accept_completion(spec, out)
        finally:
            torch.use_deterministic_algorithms(prior)


if __name__ == '__main__':
    unittest.main()
