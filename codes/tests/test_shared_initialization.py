import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'codes'),str(ROOT/'tools')]
import shared_initialization as shared
import run_sports_sharedinit_pair as launcher
from utility.dataset_profiles import SPORTS_STUDENT_PROFILES,SPORTS_SHARED_INIT_PROFILES
from td_distill_model_no_projection import TDDistillNoProjectionModel
from initialization_audit import verify_td_teacher_copy
from promptmm_release_resource import sha256

class SharedInit(unittest.TestCase):
    def test_only_alpha_and_real_parser(self):
        a,b=[SPORTS_STUDENT_PROFILES[p]['defaults'] for p in SPORTS_SHARED_INIT_PROFILES]
        self.assertEqual({k for k in a if a[k]!=b[k]},{'td_distill_alpha'})
        for name in SPORTS_SHARED_INIT_PROFILES:
            code="import sys,json;sys.path.insert(0,'codes');from utility.parser import args;print(json.dumps(vars(args)))"
            p=subprocess.run([sys.executable,'-B','-c',code,'--dataset','sports','--student_profile',name],cwd=ROOT,capture_output=True,text=True,check=True)
            a=json.loads(p.stdout);self.assertEqual(a['student_config_overrides'],{})
            self.assertTrue(a['td_init_from_teacher']);self.assertFalse(a['run_final_test']);self.assertEqual(a['epoch'],300)

    def test_loader_pinning_rng_copy_and_targets(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path=root/shared.ASSET;path.parent.mkdir(parents=True)
            t={k:torch.ones(3 if k in ('users','image_users','text_users') else 7,64) for k in shared.NAMES}
            torch.save(t,path)
            with patch.object(shared,'ASSET_SHA',sha256(path)):
                rng=torch.get_rng_state().clone();a,identity=shared.load_shared(root,'cpu',3,7)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                s=TDDistillNoProjectionModel(3,7,64,64);s.init_user_item_embed(a['users'],a['items'])
                init=verify_td_teacher_copy(s,a['users'],a['items']);self.assertEqual(init['users'],identity['tensors']['users'])
                self.assertIs(shared.semantic_targets(a)['item_text'],a['text_items'])
                with self.assertRaises(ValueError):shared.load_shared(root,'cpu',4,7)
                path.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):shared.load_shared(root,'cpu',3,7)

    def test_wiring_and_failure_no_next_arm(self):
        tree=ast.parse((ROOT/'codes/main_mmlight.py').read_text(encoding='utf-8'))
        code=ast.unparse(tree);self.assertIn('td_teacher_semantics = semantic_targets(self.shared_td_tensors)',code)
        self.assertIn("self.u_final_embed = self.shared_td_tensors['users']",code)
        with tempfile.TemporaryDirectory() as d,patch.object(launcher,'load_shared',return_value=({},{})),patch.object(launcher,'digest',return_value=launcher.ASSET_SHA),patch.object(launcher.subprocess,'check_output',side_effect=lambda args,**kw:'head' if args[1]=='rev-parse' else ''),patch.object(launcher.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'fixture')) as call:
            with self.assertRaises(subprocess.CalledProcessError):launcher.run(Path(d),'python')
            self.assertEqual(call.call_count,1)
            r=json.loads((Path(d)/'exp/initialization_checks'/launcher.RUN_ID/'batch.json').read_text());self.assertEqual(r['status'],'failed')
            with self.assertRaises(FileExistsError):launcher.run(Path(d),'python')
            self.assertEqual(call.call_count,1)

if __name__=='__main__':unittest.main()
