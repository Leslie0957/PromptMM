"""Synthetic-only Validation isolation, metric parity and checkpoint contracts."""
import ast
import contextlib
import heapq
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'codes'))
import promptmm_release_validation as runner
from promptmm_release import ReleaseStudent
from utility import metrics


class ValidationContracts(unittest.TestCase):
    def test_metric_parity_including_ties_and_legacy_ndcg(self):
        # Extract exact pure functions without executing the Test-loading module.
        source = (ROOT/'codes/utility/batch_test.py').read_text(encoding='utf-8')
        nodes = ast.parse(source).body
        namespace = dict(heapq=heapq, metrics=metrics, np=np)
        for node in nodes:
            if isinstance(node, ast.FunctionDef) and node.name in ('ranklist_by_heapq', 'get_performance'):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<reference>', 'exec'), namespace)
        train = sp.csr_matrix(([1.,1.], ([0,1],[0,1])), shape=(3,70))
        val = sp.csr_matrix(([1.]*6, ([0,0,1,1,2,2],[2,66,3,67,4,68])), shape=(3,70))
        torch.manual_seed(8)
        ue, ie = torch.randn(3,4), torch.randn(70,4)
        # One complete tie row tests the exact existing set-order/heapq policy.
        ue[2].zero_()
        class Model:
            def eval(self): pass
            def __call__(self, adj): return ue, ie
        actual = runner.evaluate_validation(Model(), None, train, val, batch_size=2)
        expected = {k: np.zeros(4) for k in actual}
        scores = (ue @ ie.T).numpy()
        for u in range(3):
            truth = val[u].indices.tolist()
            candidates = list(set(range(70)) - set(train[u].indices))
            relevance, auc = namespace['ranklist_by_heapq'](truth, candidates, scores[u], runner.KS)
            result = namespace['get_performance'](truth, relevance, auc, runner.KS)
            for k in expected: expected[k] += result[k] / 3
        for k in expected: np.testing.assert_allclose(actual[k], expected[k], atol=1e-15, rtol=0)
        with patch.object(Model, '__call__', return_value=(ue*float('nan'), ie)):
            with self.assertRaises(FloatingPointError): runner.evaluate_validation(Model(),None,train,val)

    def test_loader_never_needs_test_and_checks_identity_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            train = sp.csr_matrix(([1.],([0],[0])),shape=(2,3))
            val = sp.csr_matrix(([1.],([0],[1])),shape=(2,3))
            with (p/'val_mat').open('wb') as f: pickle.dump(val,f)
            loaded = runner.load_validation(p, train, runner.sha256(p/'val_mat'))
            self.assertEqual(loaded.nnz,1)
            with self.assertRaises(RuntimeError): runner.load_validation(p,train,'wrong')
            with self.assertRaises(ValueError): runner.load_validation(p,val,runner.sha256(p/'val_mat'))
            self.assertEqual([x.name for x in p.iterdir()], ['val_mat'])

    def test_best_selection_earliest_tie_atomic_restore_and_alias(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'best.pt'
            student = ReleaseStudent(2,3,4,1)
            student.init_user_item_embed(torch.randn(2,4),torch.randn(3,4))
            result = {'recall':[.1,.2,.3,.4]}
            best, yes = runner.save_best(path,student,1,result,{},-1.)
            self.assertTrue(yes)
            digest = runner.sha256(path)
            with torch.no_grad(): student.user_id_embedding.weight.add_(1.)
            for value in (.2,.1):
                score, yes = runner.save_best(path,student,2,{'recall':[0,value,0,0]}, {},best)
                self.assertFalse(yes); self.assertEqual(score,.2)
                self.assertEqual(runner.sha256(path),digest)
            saved = runner.restore_best(path,student)
            self.assertEqual(saved['best_epoch'],1)
            student.assert_aliases()
            best, yes = runner.save_best(path,student,3,{'recall':[0,.3,0,0]}, {},best)
            self.assertTrue(yes)
            self.assertEqual(runner.restore_best(path,student)['best_epoch'],3)
            with self.assertRaises(FloatingPointError):
                runner.save_best(path,student,4,{'recall':[0,float('nan')]},{},best)
            self.assertFalse((Path(d)/'best.pt.tmp').exists())
            bad=torch.load(path,weights_only=False)
            bad['model_state_dict']['user_id_embedding_pre.weight'].add_(1)
            torch.save(bad,path)
            with self.assertRaises(RuntimeError): runner.restore_best(path,student)

    def test_cli_budget_and_exclusive_directory(self):
        self.assertEqual(runner.parse_args(['--promptmm_release_validation']).epochs,30)
        for extra in (['--epochs','300'],['--seed','2023'],['--run_final_test','true'],['--student_lr','.1'],['--resume']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(['--promptmm_release_validation']+extra)
        with tempfile.TemporaryDirectory() as d:
            runner.claim_run_directory(d)
            with self.assertRaises(FileExistsError): runner.claim_run_directory(d)

    def test_synthetic_end_to_end_budget_and_failure_preservation(self):
        from contextlib import ExitStack
        from promptmm_release import ResourceConfig
        class Teacher(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.ones(64)*.01)
            def forward(self, ui, iu, prompt):
                u, i = prompt()
                u, i = u + self.bias, i + self.bias
                return u, i, i, i, u, u
        for fail in (False, True):
            with self.subTest(failure=fail), tempfile.TemporaryDirectory() as d, ExitStack() as stack:
                root=Path(d)
                data=root/'data/sports'; data.mkdir(parents=True)
                train=sp.csr_matrix(([1.,1.,1.],([0,1,2],[0,1,2])),shape=(3,70))
                val=sp.csr_matrix(([1.,1.,1.],([0,1,2],[3,4,5])),shape=(3,70))
                for name,value in [('train_mat',train),('val_mat',val)]:
                    with (data/name).open('wb') as f: pickle.dump(value,f)
                for name in ('image_feat.npy','text_feat.npy'): np.save(data/name,np.ones((70,2)))
                prompt={'item_hard_token':torch.randn(70,64)*.01,'user_hard_token':torch.randn(3,64)*.01}
                for name in ('trans_user','trans_item'):
                    for k,v in torch.nn.Linear(64,64).state_dict().items(): prompt[name+'.'+k]=v
                checkpoint=dict(dataset='sports',best_epoch=37,paper_ready_eligible=False,
                    dataset_identity=dict(conversion_manifest={},
                        matrices={k:{'sha256':runner.sha256(data/n)} for k,n in [('train','train_mat'),('validation','val_mat')]},
                        features={k:{'sha256':runner.sha256(data/n)} for k,n in [('image','image_feat.npy'),('text','text_feat.npy')]}),
                    teacher_inference_config={'weight_size':[64,64]},teacher_model=Teacher().state_dict(),prompt_module=prompt)
                path=root/'teacher.pt'; torch.save(checkpoint,path)
                # Source hash allowlist is exercised against isolated fixture files.
                for name in ('main_mmlight.py','promptmm_release.py','promptmm_release_resource.py','Models_mmlight.py',
                             'utility/metrics.py','utility/dataset_profiles.py','utility/sports_validation_reuse.py'):
                    f=root/'codes'/name; f.parent.mkdir(parents=True,exist_ok=True); f.write_text('# synthetic source')
                source=root/'codes/promptmm_release_validation.py'; source.write_text('# synthetic runner identity')
                stack.enter_context(patch.object(runner,'__file__',str(source)))
                stack.enter_context(patch.object(runner,'ROOT',root))
                stack.enter_context(patch.object(runner,'EPOCHS',2))
                stack.enter_context(patch.object(runner,'load_teacher_class',return_value=Teacher))
                stack.enter_context(patch.object(runner,'load_dgl',return_value=SimpleNamespace(seed=lambda s:None,__version__='synthetic')))
                stack.enter_context(patch('promptmm_release.ResourceConfig',return_value=ResourceConfig(batch_size=2)))
                stack.enter_context(patch('utility.dataset_profiles.SPORTS_TEACHER_CHECKPOINT','teacher.pt'))
                stack.enter_context(patch('utility.sports_validation_reuse.SPORTS_TEACHER_SHA256',runner.sha256(path)))
                stack.enter_context(patch.object(runner.subprocess,'check_output',side_effect=lambda args,**kw: 'fixturehead' if args[1]=='rev-parse' else ''))
                real_device=torch.device
                stack.enter_context(patch.object(torch,'device',side_effect=lambda name:real_device('cpu' if name=='cuda:0' else name)))
                for name,value in dict(is_available=True,is_current_stream_capturing=False,get_device_name='synthetic CPU',
                                       get_device_properties=SimpleNamespace(total_memory=0),max_memory_allocated=0,max_memory_reserved=0).items():
                    stack.enter_context(patch.object(torch.cuda,name,return_value=value))
                for name in ('manual_seed_all','synchronize','reset_peak_memory_stats'):
                    stack.enter_context(patch.object(torch.cuda,name,return_value=None))
                def candidates(train,users,pos,*args):
                    if fail: raise RuntimeError('synthetic sampler failure')
                    return torch.tensor([[p,69] for p in pos])
                stack.enter_context(patch('promptmm_release.release_candidates',side_effect=candidates))
                with contextlib.redirect_stdout(io.StringIO()):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError,'synthetic sampler failure'):
                            runner.main(['--promptmm_release_validation'])
                    else: self.assertEqual(runner.main(['--promptmm_release_validation']),0)
                report=json.loads((root/'exp/promptmm_release'/runner.run_identity(2022,2)/'report.json').read_text())
                self.assertEqual(report['status'],'failed' if fail else 'completed')
                self.assertEqual(report['test_evaluations'],0)
                self.assertFalse(report['test_split_loaded'])
                if not fail:
                    self.assertEqual(report['optimizer_steps_completed'],4)
                    self.assertEqual(report['validation_evaluations'],3)
                    self.assertEqual(len(report['curve']),2)
                    self.assertTrue(report['best_checkpoint_roundtrip'])
                    self.assertTrue(report['teacher_unchanged'])
                with self.assertRaises(FileExistsError):runner.claim_run_directory(root,runner.run_identity(2022,2))

    def test_three_seed120_describe_resolves_seed_and_identity(self):
        for seed in (2022,2023,2024):
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                runner.main(['--promptmm_release_validation','--seed',str(seed),'--epochs','120','--describe'])
            spec=json.loads(output.getvalue())
            self.assertEqual(spec['config']['seed'],seed)
            self.assertEqual(spec['epochs'],120)
            self.assertEqual(spec['run_id'],runner.run_identity(seed,120))

    def test_describe_dispatch_isolated(self):
        script = """import sys,runpy
sys.path.insert(0,'codes')
sys.argv=['codes/main_mmlight.py','--promptmm_release_validation','--describe']
try: runpy.run_path('codes/main_mmlight.py',run_name='__main__')
except SystemExit as e: assert e.code==0
assert all(x not in sys.modules for x in ['utility.batch_test','utility.load_data','utility.parser','dgl'])
"""
        result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        spec=json.loads(result.stdout)
        self.assertEqual(spec['epochs'],30)
        self.assertFalse(spec['early_stopping'])
        self.assertFalse(spec['test_split_loaded'])
        self.assertNotIn('steps',spec['config'])


if __name__ == '__main__': unittest.main()
