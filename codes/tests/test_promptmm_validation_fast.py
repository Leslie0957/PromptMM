import sys
from pathlib import Path
import unittest
import numpy as np
import scipy.sparse as sp
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from promptmm_validation_fast import rank_fast,rank_legacy,evaluate_fast
from promptmm_release_validation import evaluate_validation, evaluate_training_validation

class FastParity(unittest.TestCase):
    def test_exact_ranks_random_ties_exclusions_and_boundaries(self):
        rng=np.random.default_rng(71)
        for dtype in (np.float32,np.float64):
            for n in (50,51,100,1000):
                for mode in range(4):
                    scores=(rng.normal(size=n) if mode==0 else rng.integers(-2,3,size=n)).astype(dtype)
                    if mode==2:scores[:]=0
                    if mode==3:scores[:]=1;scores[::3]=-0.
                    candidates=rng.permutation(n).tolist()
                    for k in (1,10,50):
                        self.assertEqual(rank_fast(scores,candidates,k),rank_legacy(scores,candidates,k))
                    if n>50:
                        candidates=candidates[1:]
                        self.assertEqual(rank_fast(scores,candidates,50),rank_legacy(scores,candidates,50))
        for bad in (np.nan,np.inf,-np.inf):
            with self.assertRaises(FloatingPointError):rank_fast(np.array([bad]),[0],1)
        with self.assertRaises(ValueError):rank_fast(np.ones(2),[0],2)

    def test_exact_end_to_end_metrics(self):
        train=sp.csr_matrix(([1.]*3,([0,1,2],[0,1,2])),shape=(3,100))
        val=sp.csr_matrix(([1.]*5,([0,0,1,2,2],[3,99,4,5,80])),shape=(3,100))
        torch.manual_seed(12);ue=torch.randn(3,8);ie=torch.randn(100,8);ue[2].zero_()
        class Model:
            def eval(self):pass
            def __call__(self,adj):return ue,ie
        for batch in (1,2,256):
            self.assertEqual(evaluate_validation(Model(),None,train,val,batch),evaluate_training_validation(Model(),None,train,val,batch))
        with self.assertRaises(ValueError):evaluate_fast(Model(),None,train,sp.csr_matrix(val.shape))

if __name__=='__main__':unittest.main()
