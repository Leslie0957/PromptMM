import unittest
import sys,json,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'codes'),str(ROOT/'tools')]
import run_sports_sharedinit_alpha3 as runner
from utility.dataset_profiles import SPORTS_ALPHA3_PROFILE,SPORTS_SHARED_INIT_PROFILES,SPORTS_SHARED_TENSOR_PROFILES,SPORTS_TEACHER_INIT_PROFILES,SPORTS_STUDENT_PROFILES
class Alpha3Tests(unittest.TestCase):
    def test_delta_and_parser(self):
        a=SPORTS_STUDENT_PROFILES[SPORTS_SHARED_INIT_PROFILES[0]]['defaults'];b=SPORTS_STUDENT_PROFILES[SPORTS_ALPHA3_PROFILE]['defaults']
        self.assertEqual({k for k in a if a[k]!=b[k]},{'td_distill_alpha'});self.assertEqual(b['td_distill_alpha'],3.0)
        self.assertEqual(len(SPORTS_SHARED_INIT_PROFILES),2)
        self.assertIn(SPORTS_ALPHA3_PROFILE,SPORTS_SHARED_TENSOR_PROFILES);self.assertIn(SPORTS_ALPHA3_PROFILE,SPORTS_TEACHER_INIT_PROFILES)
        code="import sys,json;sys.path.insert(0,'codes');from utility.parser import args;print(json.dumps(vars(args)))"
        p=subprocess.run([sys.executable,'-B','-c',code,'--dataset','sports','--student_profile',SPORTS_ALPHA3_PROFILE],cwd=ROOT,capture_output=True,text=True,check=True)
        args=json.loads(p.stdout)
        for k,v in b.items():self.assertEqual(args[k],v)
        self.assertEqual(args['student_config_overrides'],{})
        self.assertIn('in SPORTS_SHARED_TENSOR_PROFILES:',(ROOT/'codes/main_mmlight.py').read_text())

    def test_single_failure_and_no_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/runner.ANCHOR;p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'status':'completed','shared_initialization_asset':{}}))
            with patch.object(runner,'digest',return_value=runner.ANCHOR_SHA),patch.object(runner,'load_shared',return_value=({},{})),patch.object(runner,'check_source'),patch.object(runner.subprocess,'check_output',return_value='head'),patch.object(runner.subprocess,'run',side_effect=subprocess.CalledProcessError(1,'fixture')) as call:
                with self.assertRaises(subprocess.CalledProcessError):runner.run(root,'python')
                self.assertEqual(call.call_count,1)
                report=root/'exp/initialization_checks'/runner.RUN_ID/'run.json'
                self.assertEqual(json.loads(report.read_text())['status'],'failed')
                with self.assertRaises(FileExistsError):runner.run(root,'python')
                self.assertEqual(call.call_count,1)
if __name__=='__main__':unittest.main()
