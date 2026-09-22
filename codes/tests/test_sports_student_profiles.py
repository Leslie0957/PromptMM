import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'codes'))
from utility.dataset_profiles import SPORTS_STUDENT_PROFILE_NAMES
from utility.sports_validation_reuse import is_pinned_sports_validation_reuse, SPORTS_TEACHER_SHA256
from utility.experiment_protocol import validate_teacher_checkpoint_metadata


class SportsStudentTest(unittest.TestCase):
    def resolve(self, name):
        code = "import sys,json;sys.path.insert(0,'codes');from utility.parser import args;print(json.dumps(vars(args)))"
        out = subprocess.run([sys.executable, '-B', '-c', code, '--dataset', 'sports',
                              '--student_profile', name], cwd=ROOT, check=True,
                             capture_output=True, text=True)
        return SimpleNamespace(**json.loads(out.stdout))

    def test_profiles_pair_and_gate(self):
        args = [self.resolve(n) for n in SPORTS_STUDENT_PROFILE_NAMES if 'teacherinit' not in n]
        for a in args:
            self.assertEqual(a.dataset_config_overrides, {})
            self.assertEqual(a.student_config_overrides, {})
            self.assertFalse(a.if_train_teacher or a.teacher_only or a.run_final_test or a.td_init_from_teacher)
            expected_seed = int(a.student_profile.split('_seed')[1].split('_')[0])
            budget = 300 if '_val300_' in a.student_profile else 120
            self.assertEqual((a.seed, a.epoch, a.early_stopping_patience), (expected_seed,budget,budget))
            self.assertEqual(a.hard_token_seed, 2022)
            self.assertEqual(a.student_embed_size, 64)
            self.assertEqual(a.student_lr, 0.00006)
            self.assertEqual(a.student_model_type, 'td_distill_no_projection')
            self.assertEqual(a.td_user_image_rate + a.td_user_text_rate, 0)
            with patch('utility.sports_validation_reuse.file_fingerprint', return_value={'sha256':SPORTS_TEACHER_SHA256}):
                self.assertTrue(is_pinned_sports_validation_reuse(a, 'teacher.pt'))
                for field,value in [('run_final_test',True),('if_train_teacher',True),
                                    ('student_lr',0.001),('seed',a.seed+1),('epoch',121),
                                    ('hard_token_seed',2023),
                                    ('eval_protocol','legacy_test_selected'),('teacher_only',True)]:
                    bad=copy.copy(a); setattr(bad,field,value)
                    with self.assertRaises(ValueError):is_pinned_sports_validation_reuse(bad,'teacher.pt')
            with patch('utility.sports_validation_reuse.file_fingerprint', return_value={'sha256':'wrong'}):
                with self.assertRaises(ValueError):is_pinned_sports_validation_reuse(a,'teacher.pt')
        self.assertEqual(len(args), 18)
        for offset in range(0,18,3):
            bpr, full, image = args[offset:offset+3]
            self.assertEqual(bpr.td_distill_alpha, 0)
            self.assertAlmostEqual(full.td_distill_alpha / (1 + full.td_item_text_rate), image.td_distill_alpha)
            self.assertEqual(image.td_item_text_rate, 0)
            self.assertEqual(full.td_item_text_rate, 0.3)
        allowed = {'seed', 'student_profile', 'student_config_profile', 'student_config_source'}
        for index in range(3,9):
            baseline = vars(args[index % 3])
            actual = vars(args[index])
            self.assertEqual({k for k in actual if actual[k] != baseline[k]}, allowed)
        for index in range(9,18):
            baseline = vars(args[index-9])
            actual = vars(args[index])
            self.assertEqual({k for k in actual if actual[k] != baseline[k]},
                             {'epoch', 'early_stopping_patience', 'student_profile',
                              'student_config_profile', 'student_config_source'})

    def test_relaxation_preserves_identity_and_blocker_checks(self):
        c = dict(evaluation_protocol='val_test_once_v1', selection_split='validation',
                 primary_k=20, dataset='sports', dataset_identity={'hash':'a'},
                 teacher_inference_config={'embed_size':64}, candidate_exclusion_policy='train_only',
                 paper_ready_eligible=False, paper_ready_blockers=['final test evaluation is disabled'],
                 hard_token_cache={'image':{'cache_identity':'x'}})
        def check(candidate, require=False):
            validate_teacher_checkpoint_metadata(candidate,'val_test_once_v1',20,'sports','teacher.pt',
                c['dataset_identity'],c['teacher_inference_config'],False,c['paper_ready_blockers'],
                'train_only',require_paper_ready_checkpoint=require)
        check(c)
        with self.assertRaises(ValueError):check(c,True)
        for field,value in [('dataset_identity',{'hash':'wrong'}),('teacher_inference_config',{}),
                            ('paper_ready_blockers',['other']),('selection_split','test'),
                            ('hard_token_cache',{}),('dataset','baby')]:
            bad=dict(c);bad[field]=value
            with self.assertRaises(ValueError):check(bad)


if __name__ == '__main__':unittest.main()
