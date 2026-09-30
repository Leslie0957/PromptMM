"""Synthetic only: no repository dataset/checkpoint loads."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_own_anchor_control import ROOT, affine_residual, control_features, data_guard, paired
from initialization_kd_fast_eval import exact_topk


class OwnAnchorTests(unittest.TestCase):
    def test_affine_and_mask_and_original_scales(self):
        rng = np.random.default_rng(7)
        q = rng.normal(size=(60,64)).astype(np.float32)
        ids = np.array([1,5,9,20])
        graphs = [rng.integers(0,60,size=(4,20)) for _ in range(2)]
        xr, xc, scales, centers = control_features(q, ids, graphs)
        w = rng.normal(size=(64,128)).astype(np.float32)
        self.assertTrue(np.allclose(xc @ w.T, affine_residual(q,ids,w,scales,centers),rtol=1e-5,atol=1e-5))
        other = np.setdiff1d(np.arange(60),ids)
        self.assertTrue((xc[other] == 0).all())
        expected = (centers[0]-q[ids])/scales[0]/np.sqrt(2.)
        np.testing.assert_array_equal(xc[ids,:64], expected.astype(np.float32))
        self.assertFalse(np.array_equal(xr,xc))

    def test_constant_centers_no_intervention(self):
        q = np.random.default_rng(2).normal(size=(40,64)).astype(np.float32)
        ids = np.array([25,26,27])
        graph = np.tile(np.arange(20),(3,1))
        xr,xc,_,_ = control_features(q,ids,[graph,graph])
        np.testing.assert_allclose(xr,xc,rtol=1e-6,atol=1e-7)

    def test_paired_accounting(self):
        val = csr_matrix(([1,1,1],([0,0,1],[1,2,3])),shape=(2,6))
        r = np.array([[0,1],[2,3]])
        c = np.array([[1,2],[0,2]])
        u,i = paired(r,c,np.array([0,1]),val,6)
        np.testing.assert_array_equal(u,[[1,0,1,2],[0,1,1,1]])
        self.assertEqual(int((i[:,0]-i[:,1]).sum()),0)
        self.assertEqual(float(np.mean((u[:,0]-u[:,1])/u[:,3])),-.25)

    def test_ranking_ties_exclusion_and_nonfinite(self):
        scores = np.ones(30,dtype=np.float32)
        self.assertEqual(exact_topk(scores,{0,1},20),list(range(2,22)))
        scores[4] = np.nan
        with self.assertRaises(ValueError):
            exact_topk(scores,set(),20)

    def test_data_guard_without_open(self):
        counts = {'denied_data_opens':0,'train_mat_opens':0,'val_mat_opens':0}
        for name in ['train_mat','val_mat']:
            data_guard('open',(str(ROOT/'data/sports'/name),),counts)
        for name in ['test_mat','../baby/train_mat','unknown']:
            with self.assertRaises(PermissionError):
                data_guard('open',(str(ROOT/'data/sports'/name),),counts)
        self.assertEqual(counts['denied_data_opens'],3)


if __name__ == '__main__':
    unittest.main()
