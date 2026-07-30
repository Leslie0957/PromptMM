import contextlib
import io
import os
import pickle
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from utility.experiment_protocol import (  # noqa: E402
    LEGACY_PROTOCOL,
    PAPER_READY_PROTOCOL,
    dataset_identity_from_preflight,
    early_stopping_non_improvement_limit,
    protocol_uses_validation,
    resolve_primary_k_index,
    restore_checkpoint_then_evaluate,
    run_dataset_preflight,
    training_batch_count,
    validate_teacher_checkpoint_metadata,
    write_json,
)
from utility.hard_token_cache import (  # noqa: E402
    load_legacy_hard_token_cache,
    load_or_create_hard_token_cache,
)


class ExperimentProtocolTest(unittest.TestCase):
    def _write_dataset(
        self, root, overlap=False, identical_modalities=False, cold_start=False
    ):
        dataset_dir = Path(root) / 'tiny'
        dataset_dir.mkdir(parents=True)

        if cold_start:
            train = sp.csr_matrix(
                ([1.0, 1.0], ([0, 1], [0, 0])), shape=(2, 3)
            )
            validation_col = 1
        else:
            train = sp.csr_matrix(
                (
                    [1.0, 1.0, 1.0, 1.0],
                    ([0, 0, 1, 1], [0, 1, 1, 2]),
                ),
                shape=(2, 3),
            )
            validation_col = 0 if overlap else 2
        validation = sp.csr_matrix(([1.0], ([0], [validation_col])), shape=(2, 3))
        test_col = 2 if cold_start else 0
        test = sp.csr_matrix(([1.0], ([1], [test_col])), shape=(2, 3))
        for file_name, matrix in (
            ('train_mat', train),
            ('val_mat', validation),
            ('test_mat', test),
        ):
            with (dataset_dir / file_name).open('wb') as file_obj:
                pickle.dump(matrix, file_obj)

        image = np.arange(12, dtype=np.float32).reshape(3, 4)
        text = image.copy() if identical_modalities else image + 1.0
        np.save(dataset_dir / 'image_feat.npy', image)
        np.save(dataset_dir / 'text_feat.npy', text)
        return dataset_dir

    def test_protocol_selection_and_primary_k(self):
        self.assertTrue(protocol_uses_validation(PAPER_READY_PROTOCOL))
        self.assertFalse(protocol_uses_validation(LEGACY_PROTOCOL))
        index, ks = resolve_primary_k_index('[10, 20, 50]')
        self.assertEqual(index, 1)
        self.assertEqual(ks, [10, 20, 50])
        self.assertEqual(
            early_stopping_non_improvement_limit(PAPER_READY_PROTOCOL, 8), 8
        )
        self.assertEqual(
            early_stopping_non_improvement_limit(LEGACY_PROTOCOL, 8), 9
        )
        self.assertEqual(training_batch_count(100, 32), 4)
        self.assertEqual(training_batch_count(100, 32, 1), 1)
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            training_batch_count(100, 32, -1)

    def test_restore_happens_before_final_evaluation(self):
        events = []
        model_state = {'weight': 99}

        def load_checkpoint(path):
            events.append(('restore', path))
            model_state['weight'] = 3
            return {'best_epoch': 3}

        def evaluate_test():
            events.append(('test', None))
            return {'recall': np.array([model_state['weight'] / 12.0])}

        checkpoint, result = restore_checkpoint_then_evaluate(
            'best.pt', load_checkpoint, evaluate_test
        )

        self.assertEqual(events, [('restore', 'best.pt'), ('test', None)])
        self.assertEqual(checkpoint['best_epoch'], 3)
        self.assertAlmostEqual(float(result['recall'][0]), 0.25)

    def test_paper_ready_teacher_requires_matching_metadata(self):
        dataset_identity = {'dataset': 'tiny', 'fingerprint': 'abc'}
        teacher_config = {'embed_size': 2}
        valid_checkpoint = {
            'evaluation_protocol': PAPER_READY_PROTOCOL,
            'selection_split': 'validation',
            'primary_k': 20,
            'dataset': 'tiny',
            'dataset_identity': dataset_identity,
            'teacher_inference_config': teacher_config,
            'paper_ready_eligible': True,
            'paper_ready_blockers': [],
            'hard_token_cache': {'image': {'cache_identity': 'image-cache'}},
        }
        validate_teacher_checkpoint_metadata(
            valid_checkpoint,
            PAPER_READY_PROTOCOL,
            20,
            'tiny',
            'teacher.pt',
            dataset_identity,
            teacher_config,
            True,
            [],
        )
        with self.assertRaisesRegex(ValueError, 'Paper-ready teacher reuse requires'):
            validate_teacher_checkpoint_metadata(
                {},
                PAPER_READY_PROTOCOL,
                20,
                'tiny',
                'legacy_teacher.pt',
                dataset_identity,
                teacher_config,
                True,
                [],
            )
        missing_dataset = dict(valid_checkpoint)
        missing_dataset.pop('dataset')
        with self.assertRaisesRegex(ValueError, 'dataset must equal'):
            validate_teacher_checkpoint_metadata(
                missing_dataset,
                PAPER_READY_PROTOCOL,
                20,
                'tiny',
                'teacher_without_dataset.pt',
                dataset_identity,
                teacher_config,
                True,
                [],
            )
        mismatched_config = dict(valid_checkpoint)
        mismatched_config['teacher_inference_config'] = {'embed_size': 4}
        with self.assertRaisesRegex(ValueError, 'teacher_inference_config'):
            validate_teacher_checkpoint_metadata(
                mismatched_config,
                PAPER_READY_PROTOCOL,
                20,
                'tiny',
                'teacher_wrong_config.pt',
                dataset_identity,
                teacher_config,
                True,
                [],
            )

    def test_legacy_teacher_allows_metadata_free_historical_checkpoint(self):
        validate_teacher_checkpoint_metadata(
            {}, LEGACY_PROTOCOL, 20, 'tiny', 'legacy_teacher.pt'
        )

    def test_preflight_fingerprints_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_dataset(temp_dir)
            report = run_dataset_preflight(temp_dir, 'tiny', 'warn')
            identity = dataset_identity_from_preflight(report)

        self.assertEqual(report['status'], 'passed')
        self.assertEqual(
            report['split_overlaps'],
            {'train_validation': 0, 'train_test': 0, 'validation_test': 0},
        )
        self.assertEqual(len(report['matrices']['train']['sha256']), 64)
        self.assertFalse(report['duplicate_modalities'])
        self.assertEqual(identity['matrices']['train']['sha256'], report['matrices']['train']['sha256'])

    def test_preflight_rejects_split_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_dataset(temp_dir, overlap=True)
            with self.assertRaisesRegex(ValueError, 'overlapping user-item interactions'):
                run_dataset_preflight(temp_dir, 'tiny', 'warn')

    def test_preflight_reports_cold_start_without_filtering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_dataset(temp_dir, cold_start=True)
            report = run_dataset_preflight(temp_dir, 'tiny', 'warn')
            identity = dataset_identity_from_preflight(report)

        self.assertEqual(report['status'], 'warning')
        self.assertTrue(report['cold_start']['has_cold_start'])
        self.assertEqual(
            report['cold_start']['validation']['items_absent_from_train'], [1]
        )
        self.assertEqual(
            report['cold_start']['test']['items_absent_from_train'], [2]
        )
        self.assertEqual(
            report['cold_start']['validation'][
                'interactions_with_items_absent_from_train'
            ],
            1,
        )
        self.assertIn('cold_start', identity)

    def test_preflight_duplicate_modality_policies(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_dataset(temp_dir, identical_modalities=True)
            report = run_dataset_preflight(temp_dir, 'tiny', 'warn')
            self.assertEqual(report['status'], 'warning')
            self.assertTrue(report['duplicate_modalities'])
            with self.assertRaisesRegex(ValueError, 'numerically identical'):
                run_dataset_preflight(temp_dir, 'tiny', 'error')

    def test_preflight_detects_numeric_duplicates_with_different_file_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = self._write_dataset(temp_dir)
            image = np.load(dataset_dir / 'image_feat.npy')
            np.save(dataset_dir / 'text_feat.npy', image.astype(np.float64))
            report = run_dataset_preflight(temp_dir, 'tiny', 'warn')

        self.assertFalse(report['byte_identical_modalities'])
        self.assertTrue(report['numerically_identical_modalities'])
        self.assertTrue(report['duplicate_modalities'])

    def test_preflight_rejects_invalid_sparse_values(self):
        for invalid_value, expected_message in (
            (0.0, 'explicitly stored zero'),
            (float('nan'), 'NaN or infinite'),
            (-1.0, 'negative interaction'),
        ):
            with self.subTest(invalid_value=invalid_value):
                with tempfile.TemporaryDirectory() as temp_dir:
                    dataset_dir = self._write_dataset(temp_dir)
                    invalid_train = sp.csr_matrix(
                        ([invalid_value], ([0], [0])), shape=(2, 3)
                    )
                    with (dataset_dir / 'train_mat').open('wb') as file_obj:
                        pickle.dump(invalid_train, file_obj)
                    with self.assertRaisesRegex(ValueError, expected_message):
                        run_dataset_preflight(temp_dir, 'tiny', 'warn')

    def test_write_json_supports_current_directory_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                write_json('report.json', {'status': 'ok'})
                self.assertTrue(Path('report.json').is_file())
            finally:
                os.chdir(original_cwd)

    def test_hard_token_cache_is_versioned_deterministic_and_preserves_legacy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            features = np.arange(24, dtype=np.float32).reshape(6, 4)
            image_path = dataset_dir / 'image_feat.npy'
            text_path = dataset_dir / 'text_feat.npy'
            np.save(image_path, features)
            np.save(text_path, features)
            legacy_path = dataset_dir / 'hard_token_image_pca'
            legacy_path.write_bytes(b'legacy-cache-must-remain')

            image_values, image_record = load_or_create_hard_token_cache(
                features, image_path, dataset_dir, 'image', 'pca', 2, 2022
            )
            text_values, text_record = load_or_create_hard_token_cache(
                features, text_path, dataset_dir, 'text', 'pca', 2, 2022
            )
            image_values_again, image_record_again = load_or_create_hard_token_cache(
                features, image_path, dataset_dir, 'image', 'pca', 2, 2022
            )

            np.testing.assert_allclose(image_values, text_values, rtol=0, atol=0)
            np.testing.assert_allclose(image_values, image_values_again, rtol=0, atol=0)
            self.assertFalse(image_record['cache_hit'])
            self.assertFalse(text_record['cache_hit'])
            self.assertTrue(image_record_again['cache_hit'])
            self.assertEqual(legacy_path.read_bytes(), b'legacy-cache-must-remain')
            self.assertNotEqual(
                image_record['cache_file']['path'], text_record['cache_file']['path']
            )

    def test_legacy_hard_token_cache_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            legacy_values = np.arange(12, dtype=np.float32).reshape(6, 2)
            legacy_path = dataset_dir / 'hard_token_image_pca'
            with legacy_path.open('wb') as file_obj:
                pickle.dump(legacy_values, file_obj)
            original_bytes = legacy_path.read_bytes()

            loaded_values, record = load_legacy_hard_token_cache(
                dataset_dir, 'image', 'pca', (6, 2)
            )

            np.testing.assert_array_equal(loaded_values, legacy_values)
            self.assertTrue(record['legacy_metadata_free'])
            self.assertEqual(legacy_path.read_bytes(), original_bytes)

    def test_prompt_hard_tokens_are_checkpointed_and_restored_across_seeds(self):
        original_argv = list(sys.argv)
        original_tensor_cuda = torch.Tensor.cuda
        original_module_cuda = torch.nn.Module.cuda

        def tensor_to_cpu(tensor, device=None, non_blocking=False, memory_format=None):
            return tensor.to('cpu')

        def module_to_cpu(module, device=None):
            return module.to('cpu')

        try:
            sys.argv = ['prompt-buffer-smoke']
            torch.Tensor.cuda = tensor_to_cpu
            torch.nn.Module.cuda = module_to_cpu
            import Models_mmlight

            with tempfile.TemporaryDirectory() as temp_dir:
                dataset_dir = Path(temp_dir) / 'tiny'
                dataset_dir.mkdir()
                features = np.arange(24, dtype=np.float32).reshape(6, 4)
                np.save(dataset_dir / 'image_feat.npy', features)
                np.save(dataset_dir / 'text_feat.npy', features + 1.0)

                model_args = Models_mmlight.args
                model_args.data_path = str(Path(temp_dir)) + os.sep
                model_args.dataset = 'tiny'
                model_args.embed_size = 2
                model_args.hard_token_type = 'pca'
                model_args.eval_protocol = PAPER_READY_PROTOCOL
                model_args.seed = 2022
                model_args.hard_token_seed = 2022
                model_args.prompt_dropout = 0.0

                indices = torch.tensor([[0, 1], [0, 5]], dtype=torch.long)
                values = torch.ones(2)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Sparse invariant checks.*')
                    ui_graph = torch.sparse_coo_tensor(
                        indices, values, (2, 6), check_invariants=False
                    ).coalesce()

                with contextlib.redirect_stdout(io.StringIO()):
                    source_prompt = Models_mmlight.PromptLearner(
                        features, features + 1.0, ui_graph
                    )
                source_state = source_prompt.state_dict()
                self.assertIn('item_hard_token', source_state)
                self.assertIn('user_hard_token', source_state)

                model_args.seed = 2023
                with contextlib.redirect_stdout(io.StringIO()):
                    restored_prompt = Models_mmlight.PromptLearner(
                        features, features + 1.0, ui_graph
                    )
                restored_prompt.load_state_dict(source_state, strict=True)

                torch.testing.assert_close(
                    restored_prompt.item_hard_token, source_prompt.item_hard_token
                )
                torch.testing.assert_close(
                    restored_prompt.user_hard_token, source_prompt.user_hard_token
                )
        finally:
            sys.argv = original_argv
            torch.Tensor.cuda = original_tensor_cuda
            torch.nn.Module.cuda = original_module_cuda


if __name__ == '__main__':
    unittest.main()
