import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_per_user_slot_control import ROOT,construct,guard,metrics,decide


class SlotTests(unittest.TestCase):
    def fixture(self):
        mask=np.arange(50)<25
        ids=np.tile(np.array([np.arange(20),np.arange(25,45)]),(3,1,1))
        scores=np.tile(np.arange(20,0,-1),(3,2,1)).astype(np.float32)
        r=np.array([np.arange(25,45),np.arange(20),np.ravel(np.column_stack((np.arange(10),np.arange(25,35))))])
        return ids,scores,r,mask

    def test_zero_full_mixed_prefixes(self):
        ids,scores,r,mask=self.fixture();h,k=construct(ids,scores,r,mask)
        np.testing.assert_array_equal(k,[0,20,10]);np.testing.assert_array_equal(h,r)
        np.testing.assert_array_equal(mask[h],mask[r])

    def test_changes_identity_preserves_slots(self):
        ids,scores,r,mask=self.fixture();r[2,::2]=np.arange(10,20)
        h,k=construct(ids,scores,r,mask)
        np.testing.assert_array_equal(h[2,::2],np.arange(10))
        np.testing.assert_array_equal(mask[h],mask[r])

    def test_invalid_padding_order_and_ties(self):
        ids,scores,r,mask=self.fixture();scores[:]=1
        construct(ids,scores,r,mask)
        ids[1,0,0]=-1
        with self.assertRaises(RuntimeError):construct(ids,scores,r,mask)
        ids,scores,r,mask=self.fixture();scores[0,0,1]=99
        with self.assertRaises(RuntimeError):construct(ids,scores,r,mask)

    def test_guard_blocks_both_before_seal(self):
        state={'sealed':False,'denied':0,'opens':[]}
        for name in ['train_mat','val_mat','test_mat']:
            with self.assertRaises(PermissionError):guard('open',(str(ROOT/'data/sports'/name),),state)
        state['sealed']=True
        guard('open',(str(ROOT/'data/sports/train_mat'),),state)
        with self.assertRaises(PermissionError):guard('open',(str(ROOT/'data/sports/test_mat'),),state)

    def test_metrics_subgroups(self):
        val=csr_matrix(([1,1,1,1],([0,0,0,0],[0,1,2,3])),shape=(1,4))
        m=metrics(np.array([[0,1,2]]),np.array([0]),val,np.array([0,0,1,2]),np.array([True,False,False,False]))
        self.assertEqual(m['hits'],[2,1,0]);self.assertEqual(m['recall20'],.75)
        self.assertEqual(m['subgroup_hits'],[1,2,1]);self.assertEqual(m['denominator'],[2,1,1])

    def test_decision(self):
        a={'low_H_minus_R':-3,'recall_H_minus_R':-.0002}
        b={'low_H_minus_R':-4,'recall_H_minus_R':.0002}
        self.assertEqual(decide([a]*3),'B_within_group_sufficient')
        self.assertEqual(decide([b]*3),'B_within_group_insufficient')
        self.assertEqual(decide([a,b,a]),'unresolved')


if __name__=='__main__':unittest.main()
