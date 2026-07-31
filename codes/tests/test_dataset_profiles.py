import argparse
import sys
import unittest
from pathlib import Path


CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from utility.dataset_profiles import (  # noqa: E402
    BABY_STUDENT_PROFILE_DEFAULTS,
    BABY_STUDENT_PROFILE_NAME,
    BABY_TEACHER_PROFILE_DEFAULTS,
    BABY_TEACHER_PROFILE_NAME,
    apply_dataset_profile_defaults,
    apply_student_profile_defaults,
    resolved_profile_metadata,
    resolved_student_profile_metadata,
)


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


if __name__ == '__main__':
    unittest.main()
