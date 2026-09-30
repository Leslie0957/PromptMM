import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from scalar_calibration_core import sample_pairs,objective,derivative,solve,merge_signed,decision
from run_scalar_calibration import access_guard,ROOT


class ScalarTests(unittest.TestCase):
    def test_derivative_finite_difference(self):
        m=np.array([[.2,-.8],[1.,2.]])
        d=np.array([[1,-1],[0,1]])
        a=np.array([.3,-.2]);eps=1e-6
        for u in range(2):
            hi=a.copy();lo=a.copy();hi[u]+=eps;lo[u]-=eps
            self.assertAlmostEqual((objective(m,d,hi,.01)-objective(m,d,lo,.01))/(2*eps),derivative(m,d,a,.01)[u]/2,places=8)

    def test_convex_solution_and_signs(self):
        m=np.zeros((3,4));d=np.array([[1]*4,[-1]*4,[0]*4])
        u,us=solve(m,d);g,gs=solve(m,d,per_user=False)
        self.assertGreater(u[0],0);self.assertLess(u[1],0);self.assertEqual(u[2],0)
        self.assertLess(us['max_stationarity_error'],1e-8)
        self.assertLess(us['objective'],gs['objective']);self.assertEqual(g[0],0)
        grid=np.linspace(u[0]-.1,u[0]+.1,101)
        self.assertLessEqual(objective(m[:1],d[:1],u[:1],.01),min(objective(m[:1],d[:1],x,.01) for x in grid)+1e-12)

    def test_extreme_margins_and_invalid(self):
        a,r=solve(np.array([[10000.,-10000.]]),np.array([[1,-1]]))
        self.assertTrue(np.isfinite(a).all());self.assertTrue(np.isfinite(r['objective']))
        for ridge in [0,-1,np.nan]:
            with self.assertRaises(ValueError):solve(np.zeros((2,2)),np.ones((2,2)),ridge)

    def test_sampling_repeatable_no_seen(self):
        t=csr_matrix(np.array([[1,0,1,0],[0,1,0,0]]))
        a,b=sample_pairs(t,64,8721);c,d=sample_pairs(t,64,8721)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
        for u in range(2):
            seen=set(t[u].indices);self.assertTrue(set(a[u])<=seen);self.assertFalse(set(b[u])&seen)
        with self.assertRaises(ValueError):sample_pairs(csr_matrix(np.ones((1,2))),3,1)

    def test_merge_matches_full_sort_and_monotone(self):
        rng=np.random.default_rng(19);ni=60;k=20
        scores=rng.normal(size=(4,ni)).astype(np.float32);scores[0]=1
        ids=np.empty((4,2,k),int);cache=np.empty_like(ids,dtype=np.float32)
        for u in range(4):
            for g in range(2):
                choices=np.arange(g*30,(g+1)*30);chosen=choices[np.lexsort((choices,-scores[u,choices]))[:k]]
                ids[u,g]=chosen;cache[u,g]=scores[u,chosen]
        for offset in [-100.,-1.,0.,1.,100.]:
            out=merge_signed(ids,cache,offset)
            for u in range(4):
                full=scores[u].astype(float)+offset*(np.arange(ni)<30)
                expect=np.lexsort((np.arange(ni),-full))[:k]
                np.testing.assert_array_equal(out[u],expect)
        n=[(merge_signed(ids,cache,x)<30).sum(1) for x in [-100,-1,0,1,100]]
        self.assertTrue((np.diff(n,axis=0)>=0).all())
        # Signed individual offsets and exhausted first group padding.
        ids[0,0,2:]=-1
        out=merge_signed(ids,cache,np.array([100.,0.,-1.,1.]))
        self.assertEqual((out[0]<30).sum(),2)

    def test_export_identity(self):
        rng=np.random.default_rng(8);u=rng.normal(size=(4,64));i=rng.normal(size=(7,64));a=rng.normal(size=4);h=np.arange(7)%2
        np.testing.assert_allclose(u@i.T+a[:,None]*h,np.column_stack((u,a))@np.column_stack((i,h)).T,atol=1e-12)

    def test_guard_and_screen(self):
        state={'sealed':False,'denied':0,'opens':[]}
        access_guard('open',(ROOT/'data/sports/train_mat',),state)
        with self.assertRaises(PermissionError):access_guard('open',(ROOT/'data/sports/val_mat',),state)
        state['sealed']=True;access_guard('open',(ROOT/'data/sports/val_mat',),state)
        with self.assertRaises(PermissionError):access_guard('open',(ROOT/'data/sports/test_mat',),state)
        r={'low_U_minus_B':1,'low_U_minus_G':1,'recall_U_minus_B':0.,'recall_U_minus_G':0.}
        self.assertEqual(decision([r]*3),'candidate_signal')
        self.assertEqual(decision([dict(r,recall_U_minus_G=-1e-12)]*3),'screen_stop')

if __name__=='__main__':unittest.main()
