import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import torch
from gradient_diagnostic import inspect_batch,relation,COEFFICIENTS
from td_distill_model_no_projection import TDDistillNoProjectionModel

class GradientDiagnosticTests(unittest.TestCase):
    def test_numpy_batch_serialization(self):
        import json
        import numpy as np
        import tempfile
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
        from diagnose_sports_gradients import native_batch_ids, OUT
        from promptmm_release_validation import atomic_json
        raw=([0,np.int64(1)],[np.int32(2),2],[np.int32(3),np.int64(4)])
        before=np.random.get_state()
        batch=native_batch_ids(raw)
        self.assertEqual(batch,[[0,1],[2,2],[3,4]])
        self.assertTrue(all(type(v) is int for side in batch for v in side))
        self.assertTrue(np.array_equal(before[1],np.random.get_state()[1]))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'batches.json';atomic_json(path,[batch])
            self.assertEqual(json.loads(path.read_text()),[batch])
        self.assertTrue(str(OUT).endswith('sports_sharedinit_seed2022_v2'))

    def test_actual_sampler_serialization(self):
        import json
        import random
        import numpy as np
        from unittest.mock import patch
        from diagnose_sports_gradients import native_batch_ids
        with patch.object(sys,'argv',['test']):
            from utility.load_data import Data
        data=Data.__new__(Data);data.n_users=2;data.n_items=5;data.batch_size=2
        data.exist_users=[0,1];data.train_items={0:[0,1],1:[2]}
        random.seed(2022);np.random.seed(2022)
        raw=data.sample();batch=native_batch_ids(raw)
        self.assertEqual(json.loads(json.dumps(batch)),batch)
        for a,b in zip(raw,batch):self.assertEqual(list(a),b)
        for u,p,n in zip(*batch):
            self.assertIn(p,data.train_items[u]);self.assertNotIn(n,data.train_items[u])

    def test_relations(self):
        a=torch.tensor([3.,4.]);r=relation(a,a*2)
        self.assertAlmostEqual(r['norm'],5);self.assertAlmostEqual(r['relative_to_bpr'],.5)
        self.assertAlmostEqual(r['cosine_to_bpr'],1)
        self.assertIsNone(relation(a*0,a)['cosine_to_bpr'])
        self.assertAlmostEqual(relation(a,-a)['cosine_to_bpr'],-1)

    def test_actual_losses_weights_and_no_mutation(self):
        torch.manual_seed(7)
        m=TDDistillNoProjectionModel(3,5,4,4,user_head_names=())
        before={k:v.clone() for k,v in m.state_dict().items()}
        targets={'image_items':torch.randn(5,4),'text_items':torch.randn(5,4)}
        batch=tuple(torch.tensor(x) for x in ([0,1,0],[1,1,2],[3,4,3]))
        r=inspect_batch(m,targets,batch)
        self.assertAlmostEqual(r['losses']['total'],r['losses']['bpr']+r['losses']['image']*COEFFICIENTS['image']+r['losses']['text']*COEFFICIENTS['text'],places=6)
        self.assertEqual(r['gradients']['users']['weighted_distill']['norm'],0)
        self.assertIsNone(r['gradients']['users']['weighted_distill']['cosine_to_bpr'])
        self.assertGreater(r['gradients']['items']['weighted_distill']['norm'],0)
        for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        self.assertAlmostEqual(r['gradients']['items']['weighted_image']['norm']/r['gradients']['items']['image']['norm'],COEFFICIENTS['image'],places=6)

    def test_nonfinite_rejected(self):
        m=TDDistillNoProjectionModel(2,3,4,4,user_head_names=())
        with self.assertRaises(RuntimeError):
            inspect_batch(m,{'image_items':torch.full((3,4),float('nan')),'text_items':torch.ones(3,4)},tuple(torch.tensor(x) for x in ([0],[0],[1])))

    def test_runner_no_update_or_ranking(self):
        import ast
        p=Path(__file__).resolve().parents[2]/'tools/diagnose_sports_gradients.py'
        tree=ast.parse(p.read_text())
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('step',attrs);self.assertNotIn('test',attrs)
        self.assertNotIn('AdamW',attrs)
        self.assertIn('exist_ok=False',p.read_text())
        self.assertNotIn('main_mmlight as',p.read_text())

if __name__=='__main__':unittest.main()
