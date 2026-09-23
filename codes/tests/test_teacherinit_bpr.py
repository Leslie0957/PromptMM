import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'codes'),str(ROOT/'tools')]
from utility.dataset_profiles import SPORTS_STUDENT_PROFILES,SPORTS_TD_TEACHER_INIT_PROFILE,SPORTS_BPR_TEACHER_INIT_PROFILE
from initialization_audit import EXPECTED_TD_TEACHER_INITIAL, require_same_td_initial
import run_sports_teacherinit_bpr as runner

class TeacherBPR(unittest.TestCase):
    def test_only_alpha_changes(self):
        a=SPORTS_STUDENT_PROFILES[SPORTS_TD_TEACHER_INIT_PROFILE]['defaults']
        b=SPORTS_STUDENT_PROFILES[SPORTS_BPR_TEACHER_INIT_PROFILE]['defaults']
        self.assertEqual({k for k in b if a[k]!=b[k]},{'td_distill_alpha'})
        code="import sys,json;sys.path.insert(0,'codes');from utility.parser import args;print(json.dumps(vars(args)))"
        p=subprocess.run([sys.executable,'-B','-c',code,'--dataset','sports','--student_profile',SPORTS_BPR_TEACHER_INIT_PROFILE],cwd=ROOT,text=True,capture_output=True,check=True)
        a=json.loads(p.stdout);self.assertEqual(a['student_config_overrides'],{})
        self.assertEqual(a['td_distill_alpha'],0);self.assertTrue(a['td_init_from_teacher']);self.assertFalse(a['run_final_test'])

    def test_initial_hash_guard(self):
        a=copy.deepcopy(EXPECTED_TD_TEACHER_INITIAL);require_same_td_initial(a)
        for key in ('sha256','shape','dtype'):
            bad=copy.deepcopy(a);bad['items'][key]='wrong'
            with self.assertRaises(RuntimeError):require_same_td_initial(bad)

    def test_failure_preserved_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ref=root/runner.REFERENCE;ref.parent.mkdir(parents=True);ref.write_text(json.dumps({'initialization':EXPECTED_TD_TEACHER_INITIAL}))
            with patch.object(runner,'REFERENCE_SHA',runner.digest(ref)),patch.object(runner.subprocess,'check_output',side_effect=lambda args,**kw:'head' if args[1]=='rev-parse' else ''),patch.object(runner.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'fixture')) as call:
                with self.assertRaises(subprocess.CalledProcessError):runner.run(root,'python')
                report=json.loads((root/'exp/initialization_checks'/runner.RUN_ID/'run.json').read_text());self.assertEqual(report['status'],'failed')
                with self.assertRaises(FileExistsError):runner.run(root,'python')
                self.assertEqual(call.call_count,1)

if __name__=='__main__':unittest.main()
