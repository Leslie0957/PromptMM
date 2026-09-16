import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]


class SportsTeacherProfileTest(unittest.TestCase):
    def resolve(self, extra=()):
        script = (
            "import sys,json; sys.path.insert(0,'codes'); "
            "from utility.parser import args; print(json.dumps(vars(args)))"
        )
        result = subprocess.run(
            [sys.executable, '-B', '-c', script, '--dataset', 'sports',
             '--gpu_id', '0', '--if_train_teacher', 'true', '--teacher_only',
             'true', '--run_final_test', 'false', *extra],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def test_real_parser_manual_command_is_validation_only(self):
        args = self.resolve()
        self.assertEqual(args['dataset_config_profile'], 'sports_teacher_validation120_v1')
        self.assertEqual(args['dataset_config_overrides'], {})
        self.assertEqual(args['student_config_profile'], 'none')
        self.assertEqual(args['eval_protocol'], 'val_test_once_v1')
        self.assertTrue(args['teacher_only'] and args['if_train_teacher'])
        self.assertFalse(args['run_final_test'])
        self.assertFalse(args['run_efficiency_benchmark'])
        self.assertFalse(args['allow_teacher_alias_overwrite'])
        self.assertEqual(args['teacher_checkpoint'], '')
        self.assertEqual((args['seed'], args['epoch'], args['early_stopping_patience']), (2022, 120, 120))
        self.assertEqual(args['smoke_train_batches'], 0)
        self.assertEqual(args['embed_size'], 64)
        self.assertEqual(args['hard_token_seed'], 2022)
        self.assertEqual(args['hard_token_type'], 'pca')
        self.assertEqual(args['duplicate_modalities_policy'], 'error')

    def test_cli_delta_is_visible(self):
        args = self.resolve(['--lr', '0.001'])
        self.assertEqual(args['dataset_config_overrides']['lr']['resolved'], 0.001)


if __name__ == '__main__':
    unittest.main()
