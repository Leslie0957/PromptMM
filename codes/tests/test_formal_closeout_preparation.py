"""No-Test checks for Baby budget, ranking, fail-fast planning and access guard."""
import contextlib
import io
import json
from pathlib import Path
import pickle
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
sys.path.insert(0, str(ROOT/'tools'))
import formal_closeout_eval as evaluator
import promptmm_release_baby_formal as baby
import run_innovation1_formal_closeout as cohort


class PreparationContracts(unittest.TestCase):
    def test_selected_td_and_release_states_without_test(self):
        from promptmm_release import ReleaseStudent
        from promptmm_release_validation import save_best
        train = sp.csr_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, 70))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            users, items = torch.randn(2, 64), torch.randn(70, 64)
            state = {'user_id_embedding.weight': users.clone(),
                     'item_id_embedding.weight': items.clone()}
            td_full = root/'td_full.pth'
            td_infer = root/'td_infer.pth'
            torch.save({'model_state_dict': state, 'student_model_type': 'td_distill_no_projection',
                        'best_epoch': 7, 'best_selection_recall': .1}, td_full)
            torch.save(state, td_infer)
            source = {'method': 'bpr', 'checkpoint_path': td_full, 'infer_path': td_infer}
            report = {'best_selection_epoch': 7, 'best_selection_recall': .1}
            ue, ie, selected = evaluator.selected_embeddings(source, report, train, torch.device('cpu'))
            self.assertTrue(torch.equal(ue, users))
            self.assertTrue(torch.equal(ie, items))
            self.assertEqual(selected['selected_epoch'], 7)
            torch.save(dict(state, **{'item_id_embedding.weight': items + 1}), td_infer)
            with self.assertRaisesRegex(RuntimeError, 'full/infer'):
                evaluator.selected_embeddings(source, report, train, torch.device('cpu'))

            release_best = root/'best.pt'
            student = ReleaseStudent(2, 70, 64, 1)
            student.init_user_item_embed(users.clone(), items.clone())
            metric = {'recall': [0., .2, 0., 0.]}
            save_best(release_best, student, 3, metric, {}, -1.)
            source = {'method': 'promptmm_release', 'checkpoint_path': release_best}
            report = {'config': {'layers': 1}, 'best_epoch': 3, 'best_validation': metric}
            ue, ie, selected = evaluator.selected_embeddings(source, report, train, torch.device('cpu'))
            self.assertEqual(ue.shape, (2, 64))
            self.assertEqual(ie.shape, (70, 64))
            self.assertTrue(selected['alias_preserved'])
            self.assertTrue(torch.isfinite(ue).all())

    def test_baby_trainer_synthetic_complete_without_test(self):
        from contextlib import ExitStack
        from promptmm_release import ResourceConfig
        class Teacher(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.ones(64) * .01)
            def forward(self, ui, iu, prompt):
                u, i = prompt()
                u, i = u + self.bias, i + self.bias
                return u, i, i, i, u, u
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            data = root/'data/baby'
            data.mkdir(parents=True)
            train = sp.csr_matrix(([1., 1., 1.], ([0, 1, 2], [0, 1, 2])), shape=(3, 70))
            val = sp.csr_matrix(([1., 1., 1.], ([0, 1, 2], [3, 4, 5])), shape=(3, 70))
            for name, matrix in (('train_mat', train), ('val_mat', val)):
                with (data/name).open('wb') as handle:
                    pickle.dump(matrix, handle)
            for name in ('image_feat.npy', 'text_feat.npy'):
                np.save(data/name, np.ones((70, 2)))
            prompt = {'item_hard_token': torch.randn(70, 64)*.01,
                      'user_hard_token': torch.randn(3, 64)*.01}
            for prefix in ('trans_user', 'trans_item'):
                for name, value in torch.nn.Linear(64, 64).state_dict().items():
                    prompt[prefix+'.'+name] = value
            checkpoint = dict(dataset='baby', best_epoch=22, paper_ready_eligible=True,
                              evaluation_protocol='val_test_once_v1', selection_split='validation',
                              dataset_identity=dict(conversion_manifest={},
                                  matrices={key: {'sha256': baby.sha256(data/filename)}
                                            for key, filename in [('train', 'train_mat'),
                                                                  ('validation', 'val_mat')]},
                                  features={key: {'sha256': baby.sha256(data/filename)}
                                            for key, filename in [('image', 'image_feat.npy'),
                                                                  ('text', 'text_feat.npy')]}),
                              teacher_inference_config={'weight_size': [64, 64]},
                              teacher_model=Teacher().state_dict(), prompt_module=prompt)
            teacher_path = root/'Model/baby/teacher_model_val_test_once_v1.pt'
            teacher_path.parent.mkdir(parents=True)
            torch.save(checkpoint, teacher_path)
            for filename in ('main_mmlight.py', 'promptmm_release.py',
                             'promptmm_release_resource.py', 'promptmm_validation_fast.py',
                             'initialization_audit.py', 'Models_mmlight.py',
                             'utility/metrics.py', 'utility/dataset_profiles.py'):
                path = root/'codes'/filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# fixture source')
            trainer_file = root/'codes/promptmm_release_baby_formal.py'
            trainer_file.write_text('# fixture runner')
            stack.enter_context(patch.object(baby, 'ROOT', root))
            stack.enter_context(patch.object(baby, '__file__', str(trainer_file)))
            stack.enter_context(patch.object(baby, 'EPOCHS', 2))
            stack.enter_context(patch.object(baby, 'PATIENCE', 1))
            stack.enter_context(patch.object(baby, 'BABY_TEACHER_SHA256', baby.sha256(teacher_path)))
            stack.enter_context(patch.object(baby, 'load_teacher_class', return_value=Teacher))
            stack.enter_context(patch.object(baby, 'load_dgl',
                                       return_value=SimpleNamespace(seed=lambda s: None, __version__='fixture')))
            stack.enter_context(patch('promptmm_release.ResourceConfig',
                                      return_value=ResourceConfig(batch_size=2, learning_rate=6e-5)))
            stack.enter_context(patch.object(baby.subprocess, 'check_output',
                                       side_effect=lambda cmd, **kw: 'fixture-head' if cmd[1] == 'rev-parse' else ''))
            real_device = torch.device
            stack.enter_context(patch.object(torch, 'device',
                                       side_effect=lambda name: real_device('cpu' if name == 'cuda:0' else name)))
            for name, value in dict(is_available=True, get_device_name='fixture CPU',
                                    get_device_properties=SimpleNamespace(total_memory=0),
                                    max_memory_allocated=0, max_memory_reserved=0).items():
                stack.enter_context(patch.object(torch.cuda, name, return_value=value))
            for name in ('manual_seed_all', 'synchronize', 'reset_peak_memory_stats'):
                stack.enter_context(patch.object(torch.cuda, name, return_value=None))
            stack.enter_context(patch('promptmm_release.release_candidates',
                                      side_effect=lambda train, users, pos, *args:
                                      torch.tensor([[p, 69] for p in pos])))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(baby.main(['--baby_release_formal']), 0)
            report_path = root/'exp/promptmm_release_baby'/baby.run_identity(2022)/'report.json'
            report = json.loads(report_path.read_text())
            self.assertEqual(report['status'], 'validation_completed')
            self.assertIn(report['stop_reason'], ('patience7', 'cap1000'))
            self.assertEqual(report['validation_evaluations'], report['completed_epochs'] + 1)
            self.assertEqual(report['optimizer_steps_completed'], report['completed_epochs'] * 2)
            self.assertEqual(report['test_evaluations'], 0)
            self.assertFalse(report['test_split_loaded'])
            self.assertTrue(report['best_checkpoint_roundtrip'])
            self.assertFalse((data/'test_mat').exists())

    def test_baby_budget_and_earliest_tie(self):
        args = baby.parse_args(['--baby_release_formal', '--seed', '2024'])
        self.assertEqual(args.seed, 2024)
        self.assertEqual(baby.EPOCHS, 1000)
        self.assertEqual(baby.PATIENCE, 7)
        self.assertIn('cap1000_patience7_seed2024_lr6e5', baby.run_identity(2024))
        for extra in ('--epochs', '--student_lr', '--run_final_test'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                baby.parse_args(['--baby_release_formal', extra, '1'])
        misses = 0
        for i in range(6):
            misses, stop = baby.advance_patience(False, misses)
            self.assertEqual(misses, i + 1)
            self.assertFalse(stop)
        misses, stop = baby.advance_patience(False, misses)
        self.assertEqual((misses, stop), (7, True))
        self.assertEqual(baby.advance_patience(True, misses), (0, False))

    def test_evaluation_matches_legacy_ranking_with_ties(self):
        train = sp.csr_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, 70))
        truth = sp.csr_matrix(([1., 1.], ([0, 1], [2, 3])), shape=(2, 70))
        ue = torch.zeros(2, 64)
        ie = torch.zeros(70, 64)
        actual, users = evaluator.evaluate_embeddings(ue, ie, train, truth)
        self.assertEqual(users, 2)
        from promptmm_validation_fast import rank_legacy, accumulate
        expected = {k: np.zeros(4) for k in actual}
        for user in (0, 1):
            candidates = list(set(range(70)) - set(train[user].indices))
            positive = set(truth[user].indices)
            accumulate(expected, rank_legacy(np.zeros(70), candidates, 50),
                       positive, evaluator.KS, 2)
        for key in expected:
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=0)

    def test_serial_plan_is_frozen_and_preflights_all_before_test(self):
        steps = cohort.plan()
        self.assertEqual(len(steps), 27)
        self.assertEqual([s['phase'] for s in steps],
                         ['baby_training'] * 3 + ['no_test_validation'] * 12 +
                         ['final_test_once'] * 12)
        self.assertEqual(len({s['name'] for s in steps}), 27)
        self.assertEqual(set(evaluator.SLOTS), set(cohort.SLOTS))

    def test_serial_runner_stops_before_next_step_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [dict(name='fixture_failure', phase='no_test_validation',
                          command=[sys.executable, '-c', 'raise RuntimeError("fixture stop")']),
                     dict(name='forbidden_next', phase='final_test_once',
                          command=[sys.executable, '-c', 'open("should_not_exist", "w")'])]
            with patch.object(cohort, 'ROOT', root), \
                 patch.object(cohort, 'COHORT', root/'exp/formal_closeout_cohort/innovation1_fixed_v1'), \
                 patch.object(cohort, 'clean_head', return_value='fixture-head'), \
                 patch.object(cohort, 'plan', return_value=steps), \
                 patch.object(cohort, 'check_expected_absent', return_value=None):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'fixture_failure'):
                    cohort.run()
                result = json.loads((cohort.COHORT/'batch.json').read_text())
                self.assertEqual(result['status'], 'failed')
                self.assertEqual([x['name'] for x in result['steps']], ['fixture_failure'])
                self.assertFalse((root/'should_not_exist').exists())

    def test_failed_test_access_marks_attempt_and_refuses_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'docs/research').mkdir(parents=True)
            (root/'data/baby').mkdir(parents=True)
            (root/'codes/utility').mkdir(parents=True)
            (root/'codes/utility/metrics.py').write_text('# fixture metric identity')
            (root/'codes/promptmm_validation_fast.py').write_text('# fixture ranking identity')
            for name in ('formal_closeout_eval.py', 'promptmm_release.py',
                         'promptmm_release_validation.py'):
                (root/'codes'/name).write_text('# fixture evaluator source')
            source_dir = root/'exp/promptmm_release_baby/baby_promptmm_release_cap1000_patience7_seed2022_lr6e5_v1'
            source_dir.mkdir(parents=True)
            train = sp.csr_matrix(([1.], ([0], [0])), shape=(2, 70))
            val = sp.csr_matrix(([1.], ([0], [1])), shape=(2, 70))
            for name, value in (('train_mat', train), ('val_mat', val)):
                with (root/'data/baby'/name).open('wb') as handle:
                    pickle.dump(value, handle)
            # Deliberately create no test_mat: the marker must survive its first access failure.
            (source_dir/'best.pt').write_bytes(b'fixture selected checkpoint')
            ue, ie = torch.zeros(2, 64), torch.zeros(70, 64)
            expected_val, _ = evaluator.evaluate_embeddings(ue, ie, train, val)
            source = dict(status='validation_completed', dataset='baby',
                          identity='PromptMM-release-Baby-sharedTeacher-v1',
                          config={'learning_rate': 6e-5}, early_stopping_patience=7,
                          epochs_cap=1000, test_evaluations=0, test_split_loaded=False,
                          best_checkpoint_sha256=evaluator.sha256(source_dir/'best.pt'),
                          input_sha256={'train_mat': evaluator.sha256(root/'data/baby/train_mat'),
                                        'val_mat': evaluator.sha256(root/'data/baby/val_mat')},
                          shared_teacher={'sha256': evaluator.identity_teacher_sha('baby')},
                          best_epoch=1,
                          best_validation=expected_val)
            (source_dir/'report.json').write_text(json.dumps(source), encoding='utf-8')
            identity = {'matrices': {
                'train': {'sha256': source['input_sha256']['train_mat'], 'shape': [2, 70]},
                'validation': {'sha256': source['input_sha256']['val_mat']},
                'test': {'sha256': 'not-a-real-test-hash'}}}
            ledger = {'baby_existing_formal': [{'data_identity': identity}]}
            ledger_path = root/'docs/research/INNOVATION1_REUSE_ASSETS_2026-09-26.json'
            ledger_path.write_text(json.dumps(ledger), encoding='utf-8')
            preflight_dir = root/'exp/formal_closeout_preflight/b3_2022'
            preflight_dir.mkdir(parents=True)
            preflight = {'status': 'passed', 'launch_commit': 'fixture-head',
                         'ledger_sha256': evaluator.sha256(ledger_path),
                         'source_sha256': evaluator.sha256(source_dir/'report.json'),
                         'selected_checkpoint_sha256': source['best_checkpoint_sha256'],
                         'source_selected_epoch': 1, 'validation_recall_delta': 0.}
            (preflight_dir/'report.json').write_text(json.dumps(preflight), encoding='utf-8')
            with patch.object(evaluator, 'ROOT', root), patch.object(evaluator, 'LEDGER', ledger_path), \
                 patch.object(evaluator, '__file__', str(root/'codes/formal_closeout_eval.py')), \
                 patch.object(evaluator, 'clean_head', return_value='fixture-head'), \
                 patch.object(evaluator, 'selected_embeddings',
                              return_value=(ue, ie, {'selected_epoch': 1})), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'get_device_name', return_value='fixture'), \
                 patch.object(torch, 'device', return_value=torch.device('cpu')):
                with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                    evaluator.run('b3_2022', 'final-test-once')
                result = json.loads((root/'exp/formal_closeout_eval/b3_2022/report.json').read_text())
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(result['test_access_started'], result['error'])
                self.assertEqual(result['student_test_attempts'], 1)
                self.assertFalse(result['test_split_loaded'])
                with self.assertRaises(FileExistsError):
                    evaluator.run('b3_2022', 'final-test-once')


if __name__ == '__main__':
    unittest.main()
