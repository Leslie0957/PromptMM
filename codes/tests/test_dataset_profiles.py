import argparse
import sys
import unittest
from pathlib import Path


CODES_DIR = Path(__file__).resolve().parents[1]
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from utility.dataset_profiles import (  # noqa: E402
    BABY_TEACHER_PROFILE_DEFAULTS,
    BABY_TEACHER_PROFILE_NAME,
    apply_dataset_profile_defaults,
    resolved_profile_metadata,
)


class DatasetProfilesTest(unittest.TestCase):
    def _parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='amazon')
        for field_name, default in BABY_TEACHER_PROFILE_DEFAULTS.items():
            value_type = type(default) if not isinstance(default, str) else str
            parser.add_argument('--' + field_name, default=None, type=value_type)
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


if __name__ == '__main__':
    unittest.main()
