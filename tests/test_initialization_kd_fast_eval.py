"""Synthetic exact selection and metric parity, including boundary ties."""
from pathlib import Path
import sys
import unittest
import tempfile
import hashlib
import json
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'codes'))
from initialization_kd_fast_eval import exact_topk, reference_topk, evaluate_validation_fast
from initialization_kd_adapter import evaluate_validation
from initialization_kd_eval_parity import compare_evaluators
from unittest.mock import patch


class FastEvalTests(unittest.TestCase):
    def assert_parity(self, scores, seen, k):
        before = scores.copy()
        actual = exact_topk(scores, seen, k)
        self.assertEqual(actual, reference_topk(scores, seen, k))
        self.assertTrue(np.array_equal(before, scores))
        self.assertEqual(len(actual), k)
        self.assertFalse(set(actual) & seen)

    def test_ties_signed_zero_negative_near_float_limits(self):
        for dtype in (np.float32, np.float64):
            values = [np.zeros(100, dtype=dtype), np.full(100, -3, dtype=dtype),
                      np.array([0., -0.] * 50, dtype=dtype),
                      np.array([1.] * 20 + [0.] * 60 + [-1.] * 20, dtype=dtype),
                      np.array([np.finfo(dtype).max, -np.finfo(dtype).max,
                                1., np.nextafter(dtype(1), dtype(2)), 0., -0.], dtype=dtype)]
            for scores in values:
                for seen in (set(), {0}, set(range(len(scores) // 2))):
                    available = len(scores) - len(seen)
                    for k in {1, min(20, available), min(50, available), available}:
                        self.assert_parity(scores, seen, k)

    def test_randomized_exact_ids(self):
        rng = np.random.default_rng(20220927)
        for n in (51, 64, 257, 18357):
            for _ in range(16):
                scores = rng.normal(size=n).astype(np.float32)
                if _ % 2:
                    scores = np.round(scores, 1)  # dense ties around top-K boundary
                seen = set(rng.choice(n, size=n // 4, replace=False).tolist())
                self.assert_parity(scores, seen, min(50, n - len(seen)))

    def test_invalid_or_insufficient_candidates(self):
        for scores, seen, k in ((np.array([1., np.nan]), set(), 1),
                                (np.array([1., np.inf]), set(), 1),
                                (np.zeros(4), {0, 1, 2}, 2),
                                (np.zeros(4), set(), 0), (np.zeros(4), {9}, 1)):
            with self.assertRaises(ValueError):
                exact_topk(scores, seen, k)

    def test_manual_parity_config_and_saved_rank_acceptance(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
        from run_initialization_kd_eval_parity import load_spec, accept
        from initialization_kd_runtime import digest_json
        from initialization_kd_adapter import sha256
        from run_initialization_kd_interaction import _json
        spec = load_spec()
        self.assertEqual(spec['comparison']['training_steps'], 0)
        self.assertEqual(spec['mode'], 'evaluation_parity')
        self.assertEqual(spec['hard_caps']['attempts'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ranks = np.tile(np.arange(50, dtype='<i4'), (35598, 1))
            names = ('reference_top50.npy', 'fast_top50.npy')
            for name in names:
                np.save(out / name, ranks, allow_pickle=False)
            _json(out / 'launch_manifest.json', {'source_commit': 'synthetic'})
            report = {'status': 'completed', 'source_commit': 'synthetic', 'config_digest': digest_json(spec),
                      'training_steps': 0, 'test_reads': 0, 'peak_cuda_reserved_bytes': 0,
                      'peak_sampled_rss_bytes': 0, 'parity': {
                          'status': 'passed', 'users_checked': 35598, 'k': 50,
                          'score_rows_exact': True, 'ordered_top50_exact': True, 'metrics_exact': True,
                          'reference_metrics': {'recall20': .5}, 'fast_metrics': {'recall20': .5},
                          'rank_sha256': hashlib.sha256(ranks.tobytes()).hexdigest(),
                          'reference_seconds': 2, 'fast_seconds': 1, 'speedup_observed': 2}}
            _json(out / 'report.json', report)
            _json(out / 'acceptance.json', {'report_sha256': sha256(out / 'report.json'),
                    'config_digest': digest_json(spec), 'ranks': {name: sha256(out / name) for name in names}})
            accept(spec, out)
            ranks[0, 0] = 51
            np.save(out / names[1], ranks, allow_pickle=False)
            with self.assertRaises(RuntimeError):
                accept(spec, out)

    def test_full_metrics_users_order_batches_and_no_input_mutation(self):
        rng = np.random.default_rng(91)
        users = torch.tensor(rng.normal(size=(7, 4)), dtype=torch.float32)
        items = torch.tensor(rng.normal(size=(80, 4)), dtype=torch.float32)
        items[30:60] = items[0]  # equal-score items straddling multiple K boundaries
        model = SimpleNamespace(user_id_embedding=SimpleNamespace(weight=users),
                                item_id_embedding=SimpleNamespace(weight=items))
        train = sparse.csr_matrix((np.ones(7), (np.arange(7), np.arange(7))), shape=(7, 80))
        val = sparse.csr_matrix((np.ones(4), ([0, 2, 2, 6], [31, 32, 70, 50])), shape=(7, 80))
        before_u, before_i = users.clone(), items.clone()
        before_t, before_v = train.copy(), val.copy()
        for batch in (1, 2, 256):
            old = evaluate_validation(model, train, val, user_batch=batch)
            new = evaluate_validation_fast(model, train, val, user_batch=batch)
            self.assertEqual(old, new)
            self.assertEqual(new['validation_users'], 3)
        self.assertTrue(torch.equal(users, before_u) and torch.equal(items, before_i))
        self.assertEqual((train != before_t).nnz, 0)
        self.assertEqual((val != before_v).nnz, 0)

        comparison = compare_evaluators(model, train, val)
        self.assertTrue(comparison['score_rows_exact'])
        self.assertTrue(comparison['ordered_top50_exact'])
        self.assertEqual(comparison['reference_metrics'], comparison['fast_metrics'])
        self.assertEqual(comparison['users_checked'], 3)
        with patch('initialization_kd_eval_parity.exact_topk',
                   side_effect=lambda scores, seen, k: list(reversed(reference_topk(scores, seen, k)))):
            with self.assertRaisesRegex(RuntimeError, 'parity failed'):
                compare_evaluators(model, train, val)


if __name__ == '__main__':
    unittest.main()
