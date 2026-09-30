import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from exposure_bonus_core import merge,calibrate,decision
from run_exposure_matched_bonus import access_guard,ROOT


def reference(scores,mask,beta,k=20):
    # Full candidate scalar scan, separate from vectorized cache merge.
    remaining=set(range(len(scores)));result=[]
    for _ in range(k):
        winner=None
        for i in sorted(remaining):
            if winner is None:
                winner=i;continue
            j=winner
            if mask[i]==mask[j]:
                better=scores[i]>scores[j] or (scores[i]==scores[j] and i<j)
            elif mask[i]:
                d=float(scores[j])-float(scores[i]);better=beta>d or (beta==d and i<j)
            else:
                d=float(scores[i])-float(scores[j]);better=beta<d or (beta==d and i<j)
            if better:winner=i
        result.append(winner);remaining.remove(winner)
    return result


def cache(scores,mask):
    ids=np.full((1,2,20),-1,dtype=np.int32);values=np.zeros_like(ids,dtype=np.float32)
    for g,sel in enumerate([mask,~mask]):
        group=np.flatnonzero(sel);chosen=group[np.lexsort((group,-scores[group]))][:20]
        ids[0,g,:len(chosen)]=chosen;values[0,g,:len(chosen)]=scores[chosen]
    return ids,values


class BonusTests(unittest.TestCase):
    def test_cached_full_reference(self):
        rng=np.random.default_rng(43)
        for _ in range(12):
            s=rng.integers(-10,10,size=63).astype(np.float32)/7
            mask=rng.random(63)<.3
            ids,sc=cache(s,mask)
            counts=[]
            for b in [0.,1e-15,.1,1.,10000000000000000.]:
                top,e=merge(ids,sc,b)
                self.assertEqual(top[0].tolist(),reference(s,mask,b));counts.append(int(e[0]))
            self.assertEqual(counts,sorted(counts))

    def test_tiny_difference_and_padding(self):
        s=np.ones(35,dtype=np.float32);s[1]=np.nextafter(s[1],np.float32(2))
        mask=np.zeros(35,dtype=bool);mask[:2]=True
        ids,sc=cache(s,mask)
        for beta in [0.,float(s[1]-s[0]),1e30]:
            self.assertEqual(merge(ids,sc,beta)[0][0].tolist(),reference(s,mask,beta))
        ids[:]=-1
        with self.assertRaises(ValueError):merge(ids,sc,0)

    def test_search_exact_and_unmatched(self):
        s=np.arange(60,dtype=np.float32);mask=np.arange(60)<30;ids,sc=cache(s,mask)
        c=calibrate(ids,sc,10,59)
        self.assertTrue(c['matched']);self.assertLessEqual(len(c['trace']),50)
        self.assertEqual(c['exposure'],10)
        c=calibrate(ids,sc,21,59);self.assertFalse(c['matched']);self.assertEqual(c['reason'],'above_range')
        c=calibrate(*cache(-s,mask),0,59);self.assertFalse(c['matched']);self.assertEqual(c['beta'],0)

    def test_step_jump_nearest_lower(self):
        s=np.zeros(40,dtype=np.float32);mask=np.arange(40)>=20;ids,sc=cache(s,mask)
        c=calibrate(ids,sc,10,0)
        self.assertFalse(c['matched']);self.assertEqual(c['beta'],0.);self.assertEqual(len(c['trace']),50)

    def test_phase_guard(self):
        state={'sealed':False,'denied':0,'opens':[]}
        with self.assertRaises(PermissionError):access_guard('open',(str(ROOT/'data/sports/val_mat'),),state)
        state['sealed']=True
        access_guard('open',(str(ROOT/'data/sports/val_mat'),),state)
        with self.assertRaises(PermissionError):access_guard('open',(str(ROOT/'data/sports/test_mat'),),state)
        self.assertEqual(len(state['opens']),1)

    def test_decision_boundaries(self):
        a={'matched':True,'low_Q_minus_R':-3,'recall_Q_minus_R':-.0002}
        self.assertEqual(decision([a]*3),'simple_control_sufficient')
        b={'matched':True,'low_Q_minus_R':-4,'recall_Q_minus_R':.0002}
        self.assertEqual(decision([b]*3),'simple_control_insufficient')
        self.assertEqual(decision([a,b,a]),'unresolved')
        self.assertEqual(decision([dict(a,matched=False)]),'unresolved')


if __name__=='__main__':unittest.main()
