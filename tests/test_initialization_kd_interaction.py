"""Synthetic CPU protocol checks; no repository data/model assets are opened."""

import os
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from initialization_kd_interaction import (  # noqa: E402
    ARMS, Protocol, arm_config, assert_train_val_path, install_test_read_denial,
    interaction, reserve_attempt, run_cohort, select_epochs, validate_inputs)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.p = Protocol(epochs=2, batches_per_epoch=1, batch_size=2,
                          n_users=3, n_items=4, dim=2)
        self.random = {'user': torch.tensor([[.1, .2], [.3, -.4], [-.2, .2]]),
                       'item': torch.tensor([[.1, -.1], [.2, .2], [-.3, .1], [.4, .3]])}
        self.teacher = {'user': torch.tensor([[.2, .1], [.1, .4], [-.1, .3]]),
                        'item': torch.tensor([[.3, -.1], [.2, .3], [-.2, .2], [.1, .4]])}
        self.semantics = {'item_image': self.teacher['item'].clone(),
                          'item_text': self.teacher['item'].flip(0).clone(),
                          'user_image': self.teacher['user'].clone(),
                          'user_text': self.teacher['user'].clone()}
        self.tape = np.array([[[[0, 1], [1, 2], [2, 3]]],
                              [[[1, 2], [2, 3], [0, 1]]]], dtype=np.int32)

    def test_four_arms_identical_inputs_fresh_optimizers_and_original_coefficients(self):
        old_r = {k: v.clone() for k, v in self.random.items()}
        old_t = {k: v.clone() for k, v in self.teacher.items()}
        old_tape = self.tape.copy()
        calls = []

        def val(model, epoch):
            calls.append(epoch)
            return {'recall20': float(epoch), 'ndcg20': float(epoch) / 2}

        result = run_cohort(self.random, self.teacher, self.semantics,
                            self.tape, val, self.p)
        self.assertEqual(tuple(result['arms']), ARMS)
        self.assertEqual(calls, [0, 1, 2] * 4)
        self.assertEqual(result['interaction']['I'], 0.0)
        for arm, run in result['arms'].items():
            self.assertEqual(run['optimizer_steps'], 2)
            self.assertEqual(run['selection'], {'fixed_final_index': 1, 'best_index': 1})
            source = old_r if arm.startswith('R') else old_t
            self.assertTrue(torch.equal(self.random['user'], old_r['user']))
            self.assertTrue(torch.equal(self.teacher['item'], old_t['item']))
            self.assertEqual(len(run['optimizer'].state), 2)
            self.assertEqual({int(v['step']) for v in run['optimizer'].state.values()}, {2})
            self.assertEqual(run['config']['initial'], arm_config(arm, self.p)['initial'])
            self.assertEqual(run['config']['alpha'], 0.0 if arm.endswith('0') else .3)
            self.assertEqual(run['config']['rates']['item_text'], .3)
            self.assertEqual(run['config']['rates']['user_image'], 0.0)
            self.assertEqual(run['initial_validation']['recall20'], 0.0)
            self.assertEqual(source['user'].shape, run['model'].user_id_embedding.weight.shape)
        self.assertFalse(torch.equal(result['arms']['R0']['model'].item_id_embedding.weight,
                                     result['arms']['R1']['model'].item_id_embedding.weight))
        self.assertFalse(torch.equal(result['arms']['T0']['model'].item_id_embedding.weight,
                                     result['arms']['T1']['model'].item_id_embedding.weight))
        self.assertTrue(np.array_equal(self.tape, old_tape))

    def test_cache_shape_tape_and_selection_fail_closed(self):
        validate_inputs(self.random, self.teacher, self.semantics, self.tape, self.p)
        bad = dict(self.semantics)
        bad['item_image'] = bad['item_image'][:2]
        with self.assertRaises(ValueError):
            validate_inputs(self.random, self.teacher, bad, self.tape, self.p)
        changed = self.tape.copy()
        changed[0, 0, 0, 0] = 999
        with self.assertRaises(ValueError):
            validate_inputs(self.random, self.teacher, self.semantics, changed, self.p)
        self.assertEqual(select_epochs([0.2, 0.4, 0.4]),
                         {'fixed_final_index': 2, 'best_index': 1})
        with self.assertRaises(ValueError):
            select_epochs([0.2, float('nan')])
        with self.assertRaises(ValueError):
            interaction({'R0': 0.1})

    def test_split_denial_and_one_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'sports'
            root.mkdir()
            (root / 'train_mat').write_bytes(b'train')
            (root / 'val_mat').write_bytes(b'val')
            (root / 'test_mat').write_bytes(b'test')
            install_test_read_denial(root)
            self.assertEqual((root / 'train_mat').read_bytes(), b'train')
            self.assertEqual((root / 'val_mat').read_bytes(), b'val')
            with self.assertRaises(PermissionError):
                (root / 'test_mat').read_bytes()
            with self.assertRaises(PermissionError):
                os.open(root / 'test_mat', os.O_RDONLY)
            with self.assertRaises(PermissionError):
                assert_train_val_path(root / 'test.json', root)
            out = Path(tmp) / 'new_attempt'
            reserve_attempt(out)
            with self.assertRaises(FileExistsError):
                reserve_attempt(out)
            self.assertEqual((root / 'train_mat').read_bytes(), b'train')

    def test_committed_plan_matches_core_and_has_no_launch_command(self):
        path = Path(__file__).resolve().parents[1] / 'docs/research/INITIALIZATION_KD_INTERACTION_CONFIG_2026-09-27.json'
        spec = json.loads(path.read_text(encoding='utf-8'))
        self.assertIsNone(spec['launch_command'])
        p = Protocol()
        common = spec['common']
        self.assertEqual((p.epochs, p.batches_per_epoch, p.batch_size),
                         (common['training']['epochs'], common['training']['batches_per_epoch'], common['training']['batch_size']))
        self.assertEqual((p.lr, p.weight_decay),
                         (common['optimizer']['learning_rate'], common['optimizer']['weight_decay']))
        for arm in ARMS:
            resolved = arm_config(arm, p)
            self.assertEqual(resolved['initial'], spec['arms'][arm]['initial'])
            self.assertEqual(resolved['alpha'], spec['arms'][arm]['alpha'])
            self.assertEqual(resolved['objective'], spec['arms'][arm]['objective'])


if __name__ == '__main__':
    unittest.main()
