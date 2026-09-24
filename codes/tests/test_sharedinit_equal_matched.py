import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'codes'), str(ROOT / 'tools')]
import run_sports_sharedinit_equal_matched as runner
from utility.dataset_profiles import (
    SPORTS_EQUAL_MATCHED_PROFILE, SPORTS_SHARED_INIT_PROFILES,
    SPORTS_SHARED_TENSOR_PROFILES, SPORTS_TEACHER_INIT_PROFILES,
    SPORTS_STUDENT_PROFILES,
)


class EqualMatchedTests(unittest.TestCase):
    def test_profile_delta_and_parser(self):
        baseline = SPORTS_STUDENT_PROFILES[SPORTS_SHARED_INIT_PROFILES[0]]['defaults']
        arm = SPORTS_STUDENT_PROFILES[SPORTS_EQUAL_MATCHED_PROFILE]['defaults']
        self.assertEqual({key for key in baseline if baseline[key] != arm[key]},
                         {'td_distill_alpha', 'td_item_text_rate'})
        self.assertEqual(arm['td_distill_alpha'], 0.4111064309069839)
        self.assertEqual(arm['td_item_image_rate'], 1.0)
        self.assertEqual(arm['td_item_text_rate'], 1.0)
        self.assertEqual(len(SPORTS_SHARED_INIT_PROFILES), 2)
        self.assertIn(SPORTS_EQUAL_MATCHED_PROFILE, SPORTS_SHARED_TENSOR_PROFILES)
        self.assertIn(SPORTS_EQUAL_MATCHED_PROFILE, SPORTS_TEACHER_INIT_PROFILES)
        code = "import sys,json;sys.path.insert(0,'codes');from utility.parser import args;print(json.dumps(vars(args)))"
        result = subprocess.run(
            [sys.executable, '-B', '-c', code, '--dataset', 'sports',
             '--student_profile', SPORTS_EQUAL_MATCHED_PROFILE],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        resolved = json.loads(result.stdout)
        for key, value in arm.items():
            self.assertEqual(resolved[key], value)
        self.assertEqual(resolved['student_config_overrides'], {})

    def test_single_failure_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            anchor = root / runner.ANCHOR
            anchor.parent.mkdir(parents=True)
            anchor.write_text(json.dumps({'status': 'completed',
                                          'shared_initialization_asset': {}}))
            with (patch.object(runner, 'digest', side_effect=lambda path:
                               runner.CALIBRATION_SHA if path == root / runner.CALIBRATION
                               else runner.ANCHOR_SHA),
                  patch.object(runner, 'load_shared', return_value=({}, {})),
                  patch.object(runner, 'check_source'),
                  patch.object(runner.subprocess, 'check_output', return_value='head'),
                  patch.object(runner.subprocess, 'run',
                               side_effect=subprocess.CalledProcessError(1, 'fixture')) as child):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run(root, 'python')
                self.assertEqual(child.call_count, 1)
                report = root / 'exp/initialization_checks' / runner.RUN_ID / 'run.json'
                self.assertEqual(json.loads(report.read_text())['status'], 'failed')
                with self.assertRaises(FileExistsError):
                    runner.run(root, 'python')
                self.assertEqual(child.call_count, 1)


if __name__ == '__main__':
    unittest.main()
