import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'codes'));sys.path.insert(0,str(ROOT/'tools'))
import torch
from initialization_audit import initialize_release,verify_td_teacher_copy
from promptmm_release import ReleaseStudent
from td_distill_model_no_projection import TDDistillNoProjectionModel
import promptmm_release_validation as release
from utility.dataset_profiles import SPORTS_STUDENT_PROFILES,SPORTS_TD_TEACHER_INIT_PROFILE as PROFILE
import run_sports_initialization_pair as batch

class InitializationPair(unittest.TestCase):
    def test_profile_delta_and_parser(self):
        old=SPORTS_STUDENT_PROFILES['sports_student_full_seed2022_val300_v1']['defaults']
        new=SPORTS_STUDENT_PROFILES[PROFILE]['defaults']
        self.assertEqual({k for k in new if new[k]!=old[k]},{'td_init_from_teacher'})
        code="import sys,json;sys.path.insert(0,'codes');from utility.parser import args;print(json.dumps(vars(args)))"
        p=subprocess.run([sys.executable,'-B','-c',code,'--dataset','sports','--student_profile',PROFILE],cwd=ROOT,capture_output=True,text=True,check=True)
        args=json.loads(p.stdout)
        self.assertEqual(args['student_config_overrides'],{})
        self.assertTrue(args['td_init_from_teacher']);self.assertFalse(args['run_final_test'])

    def test_release_initialization_alias_and_rng(self):
        for mode in ('teacher','random'):
            torch.manual_seed(2022);s=ReleaseStudent(3,70,64,1)
            users=torch.ones(3,64);items=torch.ones(70,64)*2
            previous=s.user_id_embedding.weight.detach().clone()
            rng=torch.get_rng_state().clone()
            record=initialize_release(s,users,items,mode)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(s.user_id_embedding.weight,users if mode=='teacher' else previous))
            s.assert_aliases();self.assertEqual(record['mode'],mode)
            (s.user_id_embedding.weight.sum()+s.user_id_embedding_pre.weight.sum()).backward()
            opt=torch.optim.AdamW(s.parameters(),lr=6e-5,foreach=False);opt.step();s.assert_aliases()

    def test_td_copy_and_bad_shape(self):
        s=TDDistillNoProjectionModel(3,70,64,64)
        u=torch.ones(3,64);i=torch.ones(70,64)
        s.init_user_item_embed(u,i)
        self.assertTrue(verify_td_teacher_copy(s,u,i)['independent_storage'])
        with self.assertRaises(ValueError):initialize_release(ReleaseStudent(3,70,64,1),u[:2],i,'teacher')

    def test_release_cli_identity(self):
        cmd=['--promptmm_release_validation','--seed','2022','--epochs','300','--student_lr','6e-5','--student_initialization','random']
        output=io.StringIO()
        with contextlib.redirect_stdout(output):release.main(cmd+['--describe'])
        spec=json.loads(output.getvalue());self.assertEqual(spec['run_id'],batch.RELEASE_ID)
        self.assertEqual(spec['student_initialization'],'random')
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):release.parse_args(cmd+['--seed','2023'])
        self.assertNotEqual(release.run_identity(2022,300,6e-5),batch.RELEASE_ID)

    def test_serial_failure_stops_and_blocks_retry(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(batch.subprocess,'check_output',side_effect=lambda args,**kw:'head' if args[1]=='rev-parse' else ''),patch.object(batch.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'fixture')) as run:
                with self.assertRaises(subprocess.CalledProcessError):batch.run_batch(Path(d),'python')
                self.assertEqual(run.call_count,1)
                r=json.loads((Path(d)/'exp/initialization_checks'/batch.BATCH_ID/'batch.json').read_text())
                self.assertEqual(r['status'],'failed');self.assertEqual(r['completed'],[])
                with self.assertRaises(FileExistsError):batch.run_batch(Path(d),'python')
                self.assertEqual(run.call_count,1)
        cmds=batch.commands('python');self.assertEqual(len(cmds),2)
        self.assertIn(PROFILE,cmds[0]);self.assertIn('random',cmds[1])

if __name__=='__main__':unittest.main()
