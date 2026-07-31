import argparse
import sys
import unittest
from pathlib import Path


CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from utility.dataset_profiles import (  # noqa: E402
    BABY_STUDENT_PROFILE_NAMES,
    BABY_STUDENT_PROFILE_DEFAULTS,
    BABY_STUDENT_PROFILE_NAME,
    BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_DEFAULTS,
    BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_NAME,
    BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_SCOPE,
    BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_SOURCE,
    BABY_TEACHER_PROFILE_DEFAULTS,
    BABY_TEACHER_PROFILE_NAME,
    apply_dataset_profile_defaults,
    apply_student_profile_defaults,
    baby_student_paper_ready_blockers,
    resolved_profile_metadata,
    resolved_student_profile_metadata,
)
from td_distill_model_no_projection import (  # noqa: E402
    TDDistillNoProjectionModel,
    export_infer_state_dict,
)


EXPECTED_BABY_STUDENT_REFERENCE_DEFAULTS = {
    'seed': 2022,
    'eval_protocol': 'val_test_once_v1',
    'Ks': '[10, 20, 40, 50]',
    'test_flag': 'part',
    'dataset_preflight': True,
    'duplicate_modalities_policy': 'error',
    'batch_size': 1024,
    'epoch': 1000,
    'smoke_train_batches': 0,
    'early_stopping_patience': 7,
    'if_train_teacher': False,
    'teacher_only': False,
    'teacher_checkpoint': 'Model/baby/teacher_model_val_test_once_v1.pt',
    'allow_teacher_alias_overwrite': False,
    'student_model_type': 'td_distill_no_projection',
    'student_embed_size': 64,
    'student_lr': 0.00006,
    'student_weight_decay': 0.01,
    'td_init_from_teacher': False,
    'td_distill_alpha': 0.0,
    'td_item_image_rate': 0.0,
    'td_item_text_rate': 0.0,
    'td_user_image_rate': 0.0,
    'td_user_text_rate': 0.0,
    'run_final_test': True,
    'run_efficiency_benchmark': False,
}

DECLARED_CANDIDATE_SEMANTIC_DEFAULTS = {
    'td_distill_alpha': 0.3,
    'td_item_image_rate': 1.0,
    'td_item_text_rate': 0.3,
    'td_user_image_rate': 0.0,
    'td_user_text_rate': 0.0,
}


class DatasetProfilesTest(unittest.TestCase):
    @staticmethod
    def _argument_type(default):
        if isinstance(default, bool):
            return lambda value: value.strip().lower() in {'true', '1', 'yes'}
        return type(default) if not isinstance(default, str) else str

    def _parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='amazon')
        for field_name, default in BABY_TEACHER_PROFILE_DEFAULTS.items():
            parser.add_argument(
                '--' + field_name,
                default=None,
                type=self._argument_type(default),
            )
        return parser

    def _student_parser(self):
        parser = self._parser()
        parser.add_argument('--student_profile', default='')
        for field_name, default in BABY_STUDENT_PROFILE_DEFAULTS.items():
            option = '--' + field_name
            if option in parser._option_string_actions:
                continue
            parser.add_argument(
                option,
                default=None,
                type=self._argument_type(default),
            )
        return parser

    def test_baby_profile_defaults_are_applied_and_pinned(self):
        parser = self._parser()
        apply_dataset_profile_defaults(parser, 'baby')
        args = parser.parse_args(['--dataset', 'baby'])
        metadata = resolved_profile_metadata('baby', args)

        self.assertEqual(metadata['name'], BABY_TEACHER_PROFILE_NAME)
        self.assertEqual(metadata['scope'], 'teacher_only')
        self.assertEqual(metadata['overrides'], {})
        self.assertEqual(args.seed, 2022)
        self.assertEqual(args.sparse, 1)
        self.assertEqual(args.Ks, '[10, 20, 40, 50]')
        self.assertEqual(args.test_flag, 'part')
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.embed_size, 64)
        self.assertAlmostEqual(args.lr, 0.00055)

    def test_cli_value_overrides_profile_and_is_recorded(self):
        parser = self._parser()
        apply_dataset_profile_defaults(parser, 'baby')
        args = parser.parse_args(['--dataset', 'baby', '--batch_size', '256'])
        metadata = resolved_profile_metadata('baby', args)

        self.assertEqual(args.batch_size, 256)
        self.assertEqual(
            metadata['overrides']['batch_size'],
            {'expected': 1024, 'resolved': 256},
        )

    def test_seed_override_is_recorded(self):
        parser = self._parser()
        apply_dataset_profile_defaults(parser, 'baby')
        args = parser.parse_args(['--dataset', 'baby', '--seed', '7'])
        metadata = resolved_profile_metadata('baby', args)

        self.assertEqual(
            metadata['overrides']['seed'],
            {'expected': 2022, 'resolved': 7},
        )

    def test_literal_formatting_does_not_create_false_override(self):
        parser = self._parser()
        apply_dataset_profile_defaults(parser, 'baby')
        args = parser.parse_args(
            ['--dataset', 'baby', '--weight_size', '[64,64]']
        )
        metadata = resolved_profile_metadata('baby', args)

        self.assertNotIn('weight_size', metadata['overrides'])

        args = parser.parse_args(['--dataset', 'baby', '--Ks', '[10,20,40,50]'])
        metadata = resolved_profile_metadata('baby', args)
        self.assertNotIn('Ks', metadata['overrides'])

    def test_baby_student_reference_profile_is_bpr_only_and_pinned(self):
        parser = self._student_parser()
        apply_dataset_profile_defaults(parser, 'baby')
        apply_student_profile_defaults(
            parser, 'baby', BABY_STUDENT_PROFILE_NAME
        )
        args = parser.parse_args(
            [
                '--dataset',
                'baby',
                '--student_profile',
                BABY_STUDENT_PROFILE_NAME,
            ]
        )
        metadata = resolved_student_profile_metadata(
            'baby', args.student_profile, args
        )

        self.assertEqual(metadata['name'], BABY_STUDENT_PROFILE_NAME)
        self.assertEqual(metadata['scope'], 'student_reference')
        self.assertEqual(metadata['overrides'], {})
        self.assertFalse(args.if_train_teacher)
        self.assertFalse(args.teacher_only)
        self.assertEqual(args.student_model_type, 'td_distill_no_projection')
        self.assertEqual(args.student_embed_size, 64)
        self.assertAlmostEqual(args.student_lr, 0.00006)
        self.assertAlmostEqual(args.student_weight_decay, 0.01)
        self.assertFalse(args.td_init_from_teacher)
        self.assertEqual(args.td_distill_alpha, 0.0)
        self.assertEqual(
            (
                args.td_item_image_rate,
                args.td_item_text_rate,
                args.td_user_image_rate,
                args.td_user_text_rate,
            ),
            (0.0, 0.0, 0.0, 0.0),
        )
        self.assertTrue(args.run_final_test)

    def test_baby_student_reference_profile_defaults_are_unchanged(self):
        self.assertEqual(
            BABY_STUDENT_PROFILE_DEFAULTS,
            EXPECTED_BABY_STUDENT_REFERENCE_DEFAULTS,
        )

    def test_candidate_differs_only_in_declared_semantic_defaults(self):
        changed_fields = {
            field_name
            for field_name in BABY_STUDENT_PROFILE_DEFAULTS
            if BABY_STUDENT_PROFILE_DEFAULTS[field_name]
            != BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_DEFAULTS[field_name]
        }

        self.assertEqual(
            set(BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_DEFAULTS),
            set(BABY_STUDENT_PROFILE_DEFAULTS),
        )
        self.assertEqual(
            changed_fields,
            {
                'td_distill_alpha',
                'td_item_image_rate',
                'td_item_text_rate',
            },
        )
        for field_name in (
            set(BABY_STUDENT_PROFILE_DEFAULTS)
            - set(DECLARED_CANDIDATE_SEMANTIC_DEFAULTS)
        ):
            self.assertEqual(
                BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_DEFAULTS[field_name],
                BABY_STUDENT_PROFILE_DEFAULTS[field_name],
            )
        for field_name, expected in DECLARED_CANDIDATE_SEMANTIC_DEFAULTS.items():
            self.assertEqual(
                BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_DEFAULTS[field_name],
                expected,
            )

    def test_candidate_formal_profile_resolves_without_overrides(self):
        parser = self._student_parser()
        apply_dataset_profile_defaults(parser, 'baby')
        apply_student_profile_defaults(
            parser,
            'baby',
            BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_NAME,
        )
        args = parser.parse_args(
            [
                '--dataset',
                'baby',
                '--student_profile',
                BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_NAME,
            ]
        )
        dataset_metadata = resolved_profile_metadata('baby', args)
        student_metadata = resolved_student_profile_metadata(
            'baby', args.student_profile, args
        )

        self.assertEqual(dataset_metadata['overrides'], {})
        self.assertEqual(student_metadata['overrides'], {})
        self.assertEqual(
            (
                student_metadata['name'],
                student_metadata['scope'],
                student_metadata['source'],
            ),
            (
                BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_NAME,
                BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_SCOPE,
                BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_SOURCE,
            ),
        )
        self.assertEqual(
            baby_student_paper_ready_blockers(
                student_metadata['name'],
                student_metadata['scope'],
                student_metadata['source'],
                student_metadata['overrides'],
            ),
            [],
        )
        self.assertEqual(args.seed, 2022)
        self.assertEqual(args.student_model_type, 'td_distill_no_projection')
        self.assertEqual(args.student_embed_size, 64)
        self.assertFalse(args.td_init_from_teacher)
        self.assertAlmostEqual(args.student_lr, 0.00006)
        self.assertAlmostEqual(args.student_weight_decay, 0.01)
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.epoch, 1000)
        self.assertEqual(args.early_stopping_patience, 7)
        self.assertFalse(args.if_train_teacher)
        self.assertFalse(args.allow_teacher_alias_overwrite)
        self.assertEqual(
            args.teacher_checkpoint,
            'Model/baby/teacher_model_val_test_once_v1.pt',
        )

    def test_candidate_cli_override_remains_a_paper_ready_blocker(self):
        parser = self._student_parser()
        apply_dataset_profile_defaults(parser, 'baby')
        apply_student_profile_defaults(
            parser,
            'baby',
            BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_NAME,
        )
        args = parser.parse_args(
            [
                '--dataset',
                'baby',
                '--student_profile',
                BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_NAME,
                '--td_item_text_rate',
                '0.5',
            ]
        )
        metadata = resolved_student_profile_metadata(
            'baby', args.student_profile, args
        )
        blockers = baby_student_paper_ready_blockers(
            metadata['name'],
            metadata['scope'],
            metadata['source'],
            metadata['overrides'],
        )

        self.assertEqual(
            metadata['overrides']['td_item_text_rate'],
            {'expected': 0.3, 'resolved': 0.5},
        )
        self.assertIn(
            'Baby student reference profile has resolved overrides',
            blockers,
        )

    def test_unknown_student_identity_remains_a_paper_ready_blocker(self):
        blockers = baby_student_paper_ready_blockers(
            'baby_unknown_student_v1',
            'student_candidate',
            'undeclared',
            {},
        )
        self.assertEqual(
            blockers,
            ['Baby student profile metadata is missing or mismatched'],
        )

    def test_candidate_preserves_id_only_no_projection_deployment_contract(self):
        model = TDDistillNoProjectionModel(
            n_users=3,
            n_items=4,
            embedding_dim=64,
            item_teacher_dim=64,
            user_teacher_dim=64,
            item_head_names=('image', 'text'),
            user_head_names=(),
        )
        named_parameters = {name for name, _ in model.named_parameters()}
        infer_state = export_infer_state_dict(model)

        self.assertEqual(
            named_parameters,
            {'user_id_embedding.weight', 'item_id_embedding.weight'},
        )
        self.assertIs(
            model.project_items(model.item_id_embedding.weight)['image'],
            model.item_id_embedding.weight,
        )
        self.assertEqual(model.project_users(model.user_id_embedding.weight), {})
        self.assertEqual(
            set(infer_state),
            {
                'user_id_embedding.weight',
                'item_id_embedding.weight',
                'embedding_dim',
                'n_users',
                'n_items',
                'variant',
            },
        )
        self.assertEqual(infer_state['variant'], 'td_distill_no_projection')

    def test_baby_student_profile_cli_override_is_recorded(self):
        parser = self._student_parser()
        apply_dataset_profile_defaults(parser, 'baby')
        apply_student_profile_defaults(
            parser, 'baby', BABY_STUDENT_PROFILE_NAME
        )
        args = parser.parse_args(
            [
                '--dataset',
                'baby',
                '--student_profile',
                BABY_STUDENT_PROFILE_NAME,
                '--student_lr',
                '0.001',
                '--run_final_test',
                'false',
            ]
        )
        metadata = resolved_student_profile_metadata(
            'baby', args.student_profile, args
        )

        self.assertEqual(
            metadata['overrides']['student_lr'],
            {'expected': 0.00006, 'resolved': 0.001},
        )
        self.assertEqual(
            metadata['overrides']['run_final_test'],
            {'expected': True, 'resolved': False},
        )

    def test_student_profile_rejects_wrong_dataset(self):
        parser = self._student_parser()
        with self.assertRaisesRegex(ValueError, 'not valid for dataset'):
            apply_student_profile_defaults(
                parser, 'amazon', BABY_STUDENT_PROFILE_NAME
            )

    def test_student_profile_rejects_unknown_name(self):
        parser = self._student_parser()
        with self.assertRaisesRegex(ValueError, 'not valid for dataset'):
            apply_student_profile_defaults(
                parser, 'baby', 'baby_unknown_student_v1'
            )

    def test_supported_student_profile_names_are_explicit(self):
        self.assertEqual(
            BABY_STUDENT_PROFILE_NAMES,
            (
                BABY_STUDENT_PROFILE_NAME,
                BABY_TD_ASYMMETRIC_NO_PROJECTION_PROFILE_NAME,
            ),
        )


if __name__ == '__main__':
    unittest.main()
