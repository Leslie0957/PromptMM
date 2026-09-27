"""Synthetic CPU launch/acceptance tests. No dataset, model asset or GPU read."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'tools')]
from initialization_kd_runtime import (bind_worker, digest_json, resolve_protocol,
                                       validate_artifacts, validate_environment, environment)
from initialization_kd_interaction import Protocol, run_arm
from run_initialization_kd_interaction import _json, _accept_completion, _supervise
import run_initialization_kd_interaction as launcher
from run_initialization_kd_resource_smoke import load_spec


class RuntimeTests(unittest.TestCase):
    def test_smoke_contract_and_environment(self):
        spec = load_spec()  # committed config text only
        p = resolve_protocol(spec)
        self.assertEqual((p.epochs, p.batches_per_epoch, p.batch_size), (1, 8, 1024))
        self.assertEqual(validate_environment(spec['environment']), environment())
        for field in ('training', 'kd', 'optimizer', 'evaluation', 'dimensions'):
            bad = copy.deepcopy(spec)
            bad['common'][field]['unexpected'] = True
            with self.assertRaises(RuntimeError):
                resolve_protocol(bad)
        for field, key in (('numerics', 'tf32'), ('hard_caps', 'attempts'), ('smoke', 'epochs')):
            bad = copy.deepcopy(spec)
            bad[field][key] = 999
            with self.assertRaises(RuntimeError):
                resolve_protocol(bad)
        bad = dict(spec['environment'], torch='wrong')
        with self.assertRaises(RuntimeError):
            validate_environment(bad)

    def test_parent_binding_source_and_single_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            spec = {'fixture': 1}
            _json(out / 'launch_manifest.json', {
                'token_sha256': hashlib.sha256(b'secret').hexdigest(), 'parent_pid': os.getpid(),
                'source_commit': 'source', 'config_digest': digest_json(spec)})
            for token, pid, head, config in (('', os.getpid(), 'source', spec),
                                           ('secret', 0, 'source', spec),
                                           ('secret', os.getpid(), 'changed', spec),
                                           ('secret', os.getpid(), 'source', {})):
                with self.assertRaises(RuntimeError):
                    bind_worker(out, out, config, head, token, pid)
            with self.assertRaises(RuntimeError):
                bind_worker(out, out / 'other', spec, 'source', 'secret', os.getpid())
            bind_worker(out, out, spec, 'source', 'secret', os.getpid())
            with self.assertRaises(FileExistsError):
                bind_worker(out, out, spec, 'source', 'secret', os.getpid())

    def make_artifacts(self, out):
        p = Protocol(epochs=2, batches_per_epoch=1, batch_size=2, n_users=3, n_items=4, dim=2)
        random = {'user': torch.ones(3, 2) * .1, 'item': torch.ones(4, 2) * .2}
        teacher = {key: value.clone() for key, value in random.items()}
        semantics = {'item_image': teacher['item'], 'item_text': teacher['item'],
                     'user_image': teacher['user'], 'user_text': teacher['user']}
        tape = np.array([[[[0, 1], [1, 2], [2, 3]]], [[[1, 2], [2, 3], [0, 1]]]], dtype=np.int32)
        spec = {'mode': 'resource_smoke', 'hard_caps': {'parent_wall_seconds': 10,
                'cuda_allocator_bytes': 10**9, 'process_rss_bytes': 10**9,
                'output_bytes': 10**7, 'minimum_free_disk_bytes': 0}}
        digest = digest_json(spec)
        (out / 'T1').mkdir()
        curve = []
        def callback(arm, epoch, metric, model, optimizer, best, final):
            curve.append({'epoch': epoch, 'validation': metric})
            for label, save in (('best', best), ('final', final)):
                if save:
                    torch.save({'arm': arm, 'epoch': epoch, 'metric': metric,
                                'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'source_commit': 'synthetic', 'config_digest': digest},
                               out / arm / (label + '.pt'))
        def val(model, epoch):
            return {'recall20': .5, 'ndcg20': .25}
        result = run_arm('T1', random, teacher, semantics, tape, val, p, epoch_callback=callback)
        _json(out / 'T1/curve.json', curve)
        _json(out / 'report.json', {'status': 'completed', 'source_commit': 'synthetic',
              'config_digest': digest, 'test_file_reads': 0, 'selection_split': 'Validation',
              'validation_calls': 1, 'peak_cuda_reserved_bytes': 0, 'peak_sampled_rss_bytes': 0,
              'arms': {'T1': {
                  'selection': result['selection'], 'final_validation': result['curve'][-1],
                  'optimizer_steps': 2, 'best_checkpoint': 'T1/best.pt', 'final_checkpoint': 'T1/final.pt'}}})
        _json(out / 'launch_manifest.json', {'source_commit': 'synthetic'})
        hashes = validate_artifacts(out, ('T1',), p, 'synthetic', digest)
        _json(out / 'acceptance.json', {'status': 'passed', 'source_commit': 'synthetic',
                                      'config_digest': digest, 'hashes': hashes})
        return p, spec

    def test_complete_output_and_parent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p, spec = self.make_artifacts(out)
            hashes = validate_artifacts(out, ('T1',), p, 'synthetic', digest_json(spec))
            self.assertEqual(len(hashes), 4)
            _accept_completion(spec, out)
            _supervise(spec, out, command=[sys.executable, '-B', '-c', 'pass'], poll_seconds=.05)
            self.assertEqual(json.loads((out / 'supervisor.json').read_text())['status'], 'completed')

    def test_corrupt_checkpoint_and_incomplete_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p, spec = self.make_artifacts(out)
            checkpoint = torch.load(out / 'T1/final.pt', weights_only=True)
            checkpoint['optimizer']['state'][0]['step'].fill_(9)
            torch.save(checkpoint, out / 'T1/final.pt')
            with self.assertRaisesRegex(RuntimeError, 'step mismatch'):
                validate_artifacts(out, ('T1',), p, 'synthetic', digest_json(spec))
            with self.assertRaises(RuntimeError):
                _accept_completion(spec, out)
            _json(out / 'T1/curve.json', [])
            with self.assertRaisesRegex(RuntimeError, 'Incomplete epoch'):
                validate_artifacts(out, ('T1',), p, 'synthetic', digest_json(spec))

    def test_exit_zero_without_output_is_failure(self):
        caps = {'parent_wall_seconds': 10, 'cuda_allocator_bytes': 10**9,
                'process_rss_bytes': 10**9, 'output_bytes': 10**7, 'minimum_free_disk_bytes': 0}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'acceptance'):
                _supervise({'hard_caps': caps}, out,
                           command=[sys.executable, '-B', '-c', 'pass'], poll_seconds=.05)
            self.assertEqual(json.loads((out / 'supervisor.json').read_text())['status'], 'failed')

    def test_actual_worker_flow_with_only_synthetic_cpu_inputs(self):
        import scipy.sparse as sparse
        p = Protocol(epochs=1, batches_per_epoch=8, batch_size=2, n_users=3, n_items=64, dim=2)
        random = {'user': torch.ones(3, 2) * .1,
                  'item': torch.arange(128, dtype=torch.float32).reshape(64, 2) / 128}
        teacher = {key: value.clone() for key, value in random.items()}
        semantics = {'item_image': teacher['item'], 'item_text': teacher['item'],
                     'user_image': teacher['user'], 'user_text': teacher['user']}
        tape = np.tile(np.array([[[[0, 1], [1, 2], [2, 3]]]], dtype=np.int32), (1, 8, 1, 1))
        train = sparse.csr_matrix(([1, 1], ([0, 1], [1, 2])), shape=(3, 64))
        val = sparse.csr_matrix(([1, 1], ([0, 1], [60, 61])), shape=(3, 64))
        spec = load_spec()
        original_device = torch.device
        prior_determinism = torch.are_deterministic_algorithms_enabled()
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            out = Path(tmp)
            stack.enter_context(patch.object(launcher, '_asset_gate'))
            stack.enter_context(patch.object(launcher, '_source_gate', return_value='synthetic'))
            stack.enter_context(patch.object(launcher, 'resolve_protocol', return_value=p))
            stack.enter_context(patch.object(launcher, 'load_pinned_tensors', return_value=(random, teacher, semantics, tape)))
            stack.enter_context(patch.object(launcher, 'load_train_val', return_value=(train, val)))
            stack.enter_context(patch.object(torch, 'device', return_value=original_device('cpu')))
            for name, result in {'is_available': True, 'get_device_properties': SimpleNamespace(total_memory=8*1024**3),
                                 'set_per_process_memory_fraction': None, 'memory_reserved': 0,
                                 'get_device_name': 'synthetic_cpu', 'synchronize': None,
                                 'max_memory_reserved': 0, 'empty_cache': None}.items():
                stack.enter_context(patch.object(torch.cuda, name, return_value=result))
            _json(out / 'launch_manifest.json', {'source_commit': 'synthetic'})
            launcher._worker(spec, out, {'source_commit': 'synthetic'})
            _accept_completion(spec, out)
            report = json.loads((out / 'report.json').read_text())
            self.assertEqual(report['validation_calls'], 1)
            self.assertEqual(report['arms']['T1']['optimizer_steps'], 8)
            self.assertEqual(report['arms']['T1']['initial_validation'],
                             {'skipped': 'resource_smoke_no_epoch_zero_ranking'})
        torch.use_deterministic_algorithms(prior_determinism)


if __name__ == '__main__':
    unittest.main()
