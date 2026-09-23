import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'codes')]
import diagnose_sports_initialization as diag

class DiagnosticTests(unittest.TestCase):
    def test_numeric_differences(self):
        a={k:torch.ones(2,3) for k in diag.NAMES};b={k:v.clone() for k,v in a.items()}
        self.assertTrue(all(v['exact'] for v in diag.compare(a,b).values()))
        b['users'][0,0]+=0.25
        row=diag.compare(a,b)['users'];self.assertEqual(row['different_elements'],1);self.assertEqual(row['max_abs'],.25)
        b['items'][0,0]=float('nan')
        with self.assertRaises(ValueError):diag.compare(a,b)

    def test_hook_returns_before_student_construction(self):
        tree=ast.parse((ROOT/'codes/main_mmlight.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Trainer')
        train=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='train')
        gate=next(n for n in train.body if isinstance(n,ast.If) and 'initialization_diagnostic' in ast.unparse(n.test))
        self.assertIsInstance(gate.body[0],ast.Return)
        constructor=next(n for n in ast.walk(train) if isinstance(n,ast.Call) and ast.unparse(n.func)=='TDDistillNoProjectionModel')
        self.assertLess(gate.lineno,constructor.lineno)

    def test_worker_zero_update_and_exclusive_outputs(self):
        class Teacher(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer('v',torch.ones(2,3));self.eval()
            def forward(self,*args):return tuple(self.v.clone() for _ in range(6))
        class Trainer:
            def __init__(self,**kw):
                self.teacher_model=Teacher();self.prompt_module=Teacher();self.ui_graph=self.iu_graph=None
                self.run_manifest_path='synthetic';self.run_manifest={'code_fingerprints':{},'dataset_identity':{},'teacher_checkpoint_fingerprint':{}}
            def _update_run_manifest(self,**kw):self.run_manifest.update(kw)
            def train(self,initialization_diagnostic=None):
                self.initialization_diagnostic=initialization_diagnostic
                return initialization_diagnostic(self,self.teacher_model())
        entry=SimpleNamespace(Trainer=Trainer,select_dataset=lambda:None,set_seed=lambda s:None,args=SimpleNamespace(seed=2022),
                              np=SimpleNamespace(__version__='fixture'),dgl=SimpleNamespace(__version__='fixture'),
                              data_generator=SimpleNamespace(n_users=2,n_items=2))
        with tempfile.TemporaryDirectory() as d,patch.object(diag,'OUT',Path(d)),patch.object(diag,'source_state',return_value='fixture'),patch.dict(sys.modules,{'main_mmlight':entry}),patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'get_device_name',return_value='fixture'),patch.object(torch.optim.AdamW,'step'),patch.object(sys,'argv',list(sys.argv)),contextlib.redirect_stdout(io.StringIO()):
            diag.worker(0)
            r=json.loads((Path(d)/'process0/report.json').read_text())
            self.assertEqual(r['status'],'completed');self.assertEqual(r['teacher_forwards'],3)
            self.assertEqual(r['optimizer_steps'],0);self.assertTrue(r['teacher_prompt_unchanged'])
            self.assertTrue((Path(d)/'process0/initial_tensors.pt').exists())
            with self.assertRaises(FileExistsError):diag.worker(0)

if __name__=='__main__':unittest.main()
