"""Synthetic-only tests; never loads project data or checkpoints."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from modal_neighbor_core import (binary_matrix, normalize, neighbors, matched_random,
                                 associations, measures, bootstrap, screen)
from run_modal_neighbor_diagnostic import DataGuard, limit_reason, supervise, sha


class ModalNeighborTests(unittest.TestCase):
    def test_binary_and_duplicate_labels(self):
        m = sparse.coo_matrix(([1, 1, 1], ([0, 0, 1], [1, 1, 0])), shape=(2, 3))
        a, duplicates = binary_matrix(m, (2, 3))
        self.assertEqual(duplicates, 1)
        self.assertEqual(a.nnz, 2)
        with self.assertRaises(ValueError):
            binary_matrix(m, (2, 3), strict=True)
        with self.assertRaises(ValueError):
            binary_matrix(sparse.csr_matrix([[np.nan]]), (1, 1))

    def test_graph_self_ties_blocks_zero(self):
        x, valid = normalize([[1, 0], [1, 0], [1, 0], [0, 1], [0, 0]])
        self.assertFalse(valid[4])
        q, c = np.array([0, 1, 2]), np.arange(4)
        a, s = neighbors(x, q, c, 2, 1)
        b, t = neighbors(x, q, c[::-1], 2, 3)
        np.testing.assert_array_equal(a, [[1, 2], [0, 2], [0, 1]])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(s, t, rtol=0, atol=1e-15)
        with self.assertRaises(ValueError):
            neighbors(x, q, np.array([0, 1]), 2)
        with self.assertRaises(ValueError):
            normalize([[float('inf'), 0]])

    def test_exact_matching_saturation_and_order(self):
        q = np.array([0, 4])
        degree = np.array([1, 1, 1, 2, 2, 2])
        real = np.array([[1, 2, 3], [0, 1, 3]])
        c = np.arange(6)
        r, v, pools = matched_random(q, real, c, degree, 8, 0, 20260929)
        r2, v2, _ = matched_random(q[::-1], real[::-1], c, degree, 8, 0, 20260929)
        np.testing.assert_array_equal(r, r2[:, ::-1])
        np.testing.assert_array_equal(v, v2[::-1])
        self.assertEqual(v[0], 1/3)
        for repeat in r:
            for j, row in enumerate(repeat):
                self.assertNotIn(q[j], row)
                self.assertEqual(len(set(row)), 3)
                np.testing.assert_array_equal(np.sort(degree[row]), np.sort(degree[real[j]]))
            self.assertTrue({1, 2}.issubset(set(repeat[0])))

    def test_association_denominators_and_empty(self):
        train = sparse.csr_matrix([[1, 0, 1, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        counts = associations(train, [0, 1, 2], [0, 0, 1], np.array([[0, 2], [1, 3]]))
        np.testing.assert_array_equal(counts, [2, 0, 1])
        row = measures(counts, np.array([[0, 0, 1], [0, 0, 0]]), [5, 5, 6], 2)
        self.assertAlmostEqual(row['real'], 2/3)
        self.assertAlmostEqual(row['null'], 1/6)
        self.assertAlmostEqual(row['delta'], .5)
        self.assertAlmostEqual(row['target_macro_real'], .75)
        self.assertAlmostEqual(row['target_macro_null'], .25)
        self.assertAlmostEqual(row['overlap_fraction_real'], .5)
        self.assertIsNone(measures(np.zeros(3), np.zeros((2, 3)), [0, 0, 1], 2)['ratio'])
        self.assertIsNone(measures(np.zeros(0), np.zeros((2, 0)), [], 2)['delta'])

    def test_bootstrap_and_screens(self):
        np.testing.assert_allclose(bootstrap([0, 0, 1], np.ones(3)*.01, 30, 7), [.01, .01])
        self.assertEqual(screen(.01, [.001, .02], [.001, .02], 1, 1), 'candidate_signal')
        self.assertEqual(screen(0, [-.001, .001], [-.001, .001], 1, 1), 'screen_stop')
        self.assertEqual(screen(.01, [.001, .02], [.001, .02], .9, 1), 'inconclusive_coverage')
        self.assertEqual(screen(.01, [.001, .02], [.001, .02], 1, .7), 'inconclusive_matching_support')

    def test_data_guard_in_isolated_process(self):
        with tempfile.TemporaryDirectory() as temp:
            # Actual audit hook, actual attempted reads, synthetic files only.
            code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from run_modal_neighbor_diagnostic import DataGuard
r=Path(sys.argv[2]); (r/'data').mkdir()
v=r/'data/val_mat'; t=r/'data/test_mat'
v.write_text('synthetic');t.write_text('synthetic')
g=DataGuard(r,v);sys.addaudithook(g)
for p in [v,t]:
 try:p.read_text();raise AssertionError('guard failed')
 except PermissionError:pass
g.graph_sealed=True
assert v.read_text()=='synthetic'
assert g.test_denied==1 and g.validation_denied==1
'''
            subprocess.run([sys.executable, '-B', '-c', code, str(ROOT/'tools'), temp], check=True)

    def test_supervisor_timeout_and_success(self):
        caps = dict(seconds=.2, rss=2**30, output_bytes=2**20, minimum_free_disk=0, sample_seconds=.03)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            result = supervise([sys.executable, '-B', '-c', 'import time;time.sleep(10)'], out, caps, os.environ.copy())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['limit'], 'seconds')
        with tempfile.TemporaryDirectory() as temp:
            caps['seconds'] = 10
            result = supervise([sys.executable, '-B', '-c', 'print("synthetic")'], Path(temp), caps, os.environ.copy())
            self.assertEqual(result['status'], 'completed')
        sample = dict(seconds=0, rss=2**31, output_bytes=0, free_disk=1)
        self.assertEqual(limit_reason(sample, caps), 'rss')

    def test_frozen_profile_and_launch_refusal(self):
        profile_path = ROOT/'docs/research/innovation2/MODAL_NEIGHBOR_PROFILE_V1.json'
        p = json.loads(profile_path.read_text())
        a = json.loads((ROOT/'docs/research/innovation2/RANKING_GAP_PROFILE_V1.json').read_text())
        for key in ('train', 'validation', 'teacher'):
            self.assertEqual(p['inputs'][key]['sha256'], a[key]['sha256'])
        self.assertEqual((p['k'], p['random_repeats'], p['bootstrap_repeats']), (20, 100, 1000))
        self.assertFalse((ROOT/p['output']).exists())
        r = subprocess.run([sys.executable, '-B', str(ROOT/'tools/run_modal_neighbor_diagnostic.py'), '--formal'], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('Preparation only', r.stderr)
        self.assertFalse((ROOT/p['output']).exists())

    def test_synthetic_worker_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            code = '''
import os,sys,json,pickle,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,sys.argv[1])
import numpy as np
import torch
from scipy import sparse
import run_modal_neighbor_diagnostic as runner
r=Path(sys.argv[2]);runner.ROOT=r
(r/'data').mkdir();out=r/'out';out.mkdir()
train=sparse.lil_matrix((4,9))
for u,items in enumerate([[0,3,4],[1,3,5],[2,4,5],[3,4,5,6]]):train[u,items]=1
val=sparse.lil_matrix((4,9))
for u,i in enumerate([7,0,8,7]):val[u,i]=1
for name,m in [('train',train),('validation',val)]:
 with (r/'data'/name).open('wb') as f:pickle.dump(m.tocsr(),f)
rng=np.random.default_rng(8)
torch.save({k:torch.tensor(rng.normal(size=(9,64))) for k in ['image_items','text_items']},r/'teacher.pt')
p={'inputs':{k:{'path':path,'sha256':runner.sha(r/path)} for k,path in [('train','data/train'),('validation','data/validation'),('teacher','teacher.pt')]},
'shape':[4,9],'expected_low_items':3,'expected_low_max_degree':1,'expected_low_validation_pairs':4,
'k':2,'block':2,'random_repeats':4,'random_seed':20260929,'bootstrap_repeats':20,'user_bootstrap_seed':20260930,'item_bootstrap_seed':20260931,'threshold':.005}
runner.worker(p,out)
report=json.loads((out/'report.json').read_text());seal=json.loads((out/'graph_seal.json').read_text());acc=json.loads((out/'acceptance.json').read_text())
assert acc['status']=='passed' and acc['test_denied_attempts']==0 and acc['validation_early_denied_attempts']==0
assert seal['time']<=report['validation_started']
assert report['support']['pairs']==4
for name in ['image','text']:
 with np.load(out/(name+'_graphs.npz')) as z:
  known=[set(x) for x in [[0,3,4],[1,3,5],[2,4,5],[3,4,5,6]]]
  target=[7,0,8,7];q=z['queries'].tolist()
  real=sum(bool(set(z['real'][q.index(i)]) & known[u]) for u,i in enumerate(target))/4
  null=np.mean([bool(set(g[q.index(i)]) & known[u]) for g in z['random'] for u,i in enumerate(target)])
 assert abs(report['modalities'][name]['delta']-(real-null))<1e-12
for name,h in acc['hashes'].items():assert runner.sha(out/name)==h
'''
            subprocess.run([sys.executable, '-B', '-c', code, str(ROOT/'tools'), temp], check=True)


if __name__ == '__main__':
    unittest.main()
