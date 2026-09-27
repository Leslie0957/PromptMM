"""Synthetic-only adapter checks: no repository model or dataset is opened."""

import hashlib
import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import scipy.sparse as sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from initialization_kd_adapter import (  # noqa: E402
    check_budget, evaluate_validation, load_train_val, require_hash)
from initialization_kd_interaction import install_test_read_denial  # noqa: E402


class TinyModel:
    def __init__(self):
        self.user_id_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1., 0.], [0., 1.]]))
        self.item_id_embedding = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1., 0.], [.8, .2], [0., 1.], [-1., 0.]]))


class AdapterTests(unittest.TestCase):
    def test_train_val_only_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = sparse.csr_matrix(([1, 1], ([0, 1], [0, 2])), shape=(2, 4))
            val = sparse.csr_matrix(([1, 1], ([0, 1], [1, 3])), shape=(2, 4))
            for name, matrix in (('train_mat', train), ('val_mat', val)):
                (root / name).write_bytes(pickle.dumps(matrix))
            (root / 'test_mat').write_bytes(b'forbidden')
            train_hash = hashlib.sha256((root / 'train_mat').read_bytes()).hexdigest()
            val_hash = hashlib.sha256((root / 'val_mat').read_bytes()).hexdigest()
            install_test_read_denial(root)
            loaded_train, loaded_val = load_train_val(root, train_hash, val_hash, 2, 4)
            self.assertEqual(loaded_train.nnz, 2)
            self.assertEqual(loaded_val.nnz, 2)
            with self.assertRaises(PermissionError):
                (root / 'test_mat').read_bytes()
            with self.assertRaises(RuntimeError):
                require_hash(root / 'val_mat', '0' * 64)
            with self.assertRaises(RuntimeError):
                load_train_val(root, train_hash, '0' * 64, 2, 4)

    def test_validation_excludes_train_and_uses_val_users_only(self):
        train = sparse.csr_matrix(([1, 1], ([0, 1], [0, 2])), shape=(2, 4))
        val = sparse.csr_matrix(([1, 1], ([0, 1], [1, 3])), shape=(2, 4))
        result = evaluate_validation(TinyModel(), train, val, ks=(1, 2), user_batch=1)
        # User 0's top score item 0 is excluded; item 1 becomes rank 1.
        # User 1's item 2 is excluded; item 3 is ranked below item 1.
        self.assertEqual(result['validation_users'], 2)
        self.assertEqual(result['recall'][0], .5)
        self.assertEqual(result['recall'][1], .5)
        self.assertEqual(result['ndcg'][0], .5)

    def test_resource_boundaries_fail_closed(self):
        limits = {'parent_wall_seconds': 10, 'process_rss_bytes': 100,
                  'cuda_allocator_bytes': 50, 'output_bytes': 10,
                  'minimum_free_disk_bytes': 20}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(check_budget(0, root, limits, 99, 49, 20, now=10)
                             ['parent_wall_seconds'], 10)
            for rss, cuda, disk, now in ((101, 0, 20, 1), (0, 51, 20, 1),
                                         (0, 0, 19, 1), (0, 0, 20, 11)):
                with self.assertRaises(RuntimeError):
                    check_budget(0, root, limits, rss, cuda, disk, now=now)
            (root / 'partial').write_bytes(b'x' * 11)
            with self.assertRaises(RuntimeError):
                check_budget(0, root, limits, 0, 0, 20, now=1)

    def test_current_plan_cannot_launch_or_open_assets(self):
        config = Path(__file__).resolve().parents[1] / 'docs/research/INITIALIZATION_KD_INTERACTION_CONFIG_2026-09-27.json'
        spec = json.loads(config.read_text(encoding='utf-8'))
        self.assertEqual(spec['status'], 'preparation_only_not_launchable')
        self.assertIsNone(spec['launch_command'])
        from importlib.util import module_from_spec, spec_from_file_location
        launcher = Path(__file__).resolve().parents[1] / 'tools/run_initialization_kd_interaction.py'
        module_spec = spec_from_file_location('init_kd_launcher', launcher)
        module = module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        with self.assertRaisesRegex(RuntimeError, 'Preparation only'):
            module._source_gate(spec)

    def test_synthetic_parent_kills_overrun_and_preserves_partial(self):
        from importlib.util import module_from_spec, spec_from_file_location
        launcher = Path(__file__).resolve().parents[1] / 'tools/run_initialization_kd_interaction.py'
        module_spec = spec_from_file_location('init_kd_supervisor', launcher)
        module = module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        caps = {'parent_wall_seconds': .3, 'process_rss_bytes': 10**9,
                'cuda_allocator_bytes': 10**9, 'output_bytes': 10**6,
                'minimum_free_disk_bytes': 0}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            worker = "from pathlib import Path; import time; Path(%r).write_text('kept'); time.sleep(5)" % str(output / 'partial.txt')
            with self.assertRaisesRegex(RuntimeError, 'parent_wall_seconds'):
                module._supervise({'hard_caps': caps}, output,
                                  [sys.executable, '-B', '-c', worker], poll_seconds=.05)
            self.assertEqual((output / 'partial.txt').read_text(), 'kept')
            self.assertEqual(json.loads((output / 'supervisor.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
