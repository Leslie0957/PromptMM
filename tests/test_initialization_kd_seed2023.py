"""Synthetic seed2023 declaration and four-arm routing checks."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ('tools', 'codes')]
import run_initialization_kd_cohort_seed2023 as next_run
from run_initialization_kd_cohort import load_spec as prior_run
from initialization_kd_runtime import resolve_protocol
import tests.test_initialization_kd_cohort as old_tests


class Seed2023Tests(unittest.TestCase):
    def test_only_declared_seed_and_asset_delta(self):
        old, new = prior_run(), next_run.load_spec()
        self.assertEqual(resolve_protocol(new), resolve_protocol(old))
        changed = copy.deepcopy(new)
        for key in ('seed', 'run_id', 'output_namespace', 'launch_command'):
            changed[key] = old[key]
        for key in ('random_initial', 'triplet_tape'):
            changed['common']['assets'][key] = old['common']['assets'][key]
        self.assertEqual(changed, old)
        self.assertNotEqual(new['common']['assets']['random_initial']['sha256'], old['common']['assets']['random_initial']['sha256'])
        self.assertNotEqual(new['common']['assets']['triplet_tape']['sha256'], old['common']['assets']['triplet_tape']['sha256'])
        self.assertEqual(new['environment'], old['environment'])
        self.assertEqual(new['hard_caps'], old['hard_caps'])
        self.assertEqual(new['candidate_screening'], old['candidate_screening'])

    def test_closed_delta_rejects_identity_drift(self):
        original = json.loads(next_run.DELTA.read_text(encoding='utf-8'))
        for change in ({'seed': 2022}, {'output_namespace': 'exp/other'},
                       {'random_initial': {'path': 'wrong', 'sha256': '0'*64}},
                       {'extra': True}):
            bad = dict(original, **change)
            with patch.object(next_run, 'DELTA', SimpleNamespace(read_text=lambda **_: json.dumps(bad))):
                with self.assertRaises(RuntimeError):
                    next_run.load_spec()
        for seed in (2024, -1):
            with self.assertRaises(RuntimeError):
                resolve_protocol(dict(next_run.load_spec(), seed=seed))

    def test_full_synthetic_worker_with_seed2023_route(self):
        with patch.object(old_tests, 'load_spec', next_run.load_spec):
            old_tests.CohortTests().test_full_worker_four_arms_and_acceptance()


if __name__ == '__main__':
    unittest.main()
