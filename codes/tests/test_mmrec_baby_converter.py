import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CODES_DIR = REPO_ROOT / 'codes'
for import_path in (REPO_ROOT, CODES_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from tools.convert_mmrec_baby import (  # noqa: E402
    CORE_SOURCE_FILES,
    ConversionError,
    convert_baby,
    convert_mmrec,
    file_fingerprint,
    parse_args,
)
from utility.experiment_protocol import (  # noqa: E402
    dataset_identity_from_preflight,
    run_dataset_preflight,
)


class MMRecBabyConverterTest(unittest.TestCase):
    def test_dataset_defaults_and_explicit_paths(self):
        for dataset in ('baby', 'sports'):
            args = parse_args([] if dataset == 'baby' else ['--dataset', dataset])
            self.assertEqual(args.source, REPO_ROOT / 'data' / '_incoming' / ('mmrec_' + dataset))
            self.assertEqual(args.output, REPO_ROOT / 'data' / dataset)
        args = parse_args(['--dataset', 'sports', '--source', 'custom_raw', '--output', 'custom_out'])
        self.assertEqual(args.source, Path('custom_raw'))
        self.assertEqual(args.output, Path('custom_out'))

    def test_sports_conversion_and_wrong_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(temp_dir)
            (source / 'baby.inter').rename(source / 'sports.inter')
            names = ('sports.inter',) + CORE_SOURCE_FILES[1:]
            (source / 'download_manifest.json').write_text(json.dumps({
                'files': [{'name': name, **file_fingerprint(source / name)} for name in names]
            }), encoding='utf-8')
            output = Path(temp_dir) / 'sports'
            with self.assertRaises(FileNotFoundError):
                convert_baby(source, output)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ConversionError, 'Unsupported'):
                convert_mmrec(source, output, dataset='../sports')
            before = {name: file_fingerprint(source / name) for name in names}
            manifest = convert_mmrec(source, output, dataset='sports')
            self.assertEqual(manifest['dataset']['name'], 'sports')
            self.assertEqual(before, manifest['source']['artifacts'])
            self.assertEqual(before, {name: file_fingerprint(source / name) for name in names})
            for name, pairs in [('train_mat', {(0, 0), (1, 1)}),
                                ('val_mat', {(0, 2)}), ('test_mat', {(1, 2)})]:
                with (output / name).open('rb') as f:
                    matrix = pickle.load(f)
                self.assertEqual(matrix.shape, (2, 3))
                self.assertEqual(set(zip(*matrix.nonzero())), pairs)
                self.assertTrue(np.all(matrix.data == 1))
            for name in CORE_SOURCE_FILES[1:]:
                self.assertEqual(file_fingerprint(output / name), before[name])
            preflight = run_dataset_preflight(str(Path(temp_dir)), 'sports', 'error')
            self.assertEqual(preflight['status'], 'warning')
            self.assertEqual(preflight['conversion_manifest']['cold_item_policy'], 'retain_official')

    def _write_source(self, root, interactions=None):
        source = Path(root) / 'source'
        source.mkdir()
        (source / 'i_id_mapping.csv').write_text(
            'asin\titemID\nA\t0\nB\t1\nC\t2\n', encoding='utf-8'
        )
        (source / 'u_id_mapping.csv').write_text(
            'user_id\tuserID\nU0\t0\nU1\t1\n', encoding='utf-8'
        )
        np.save(
            source / 'image_feat.npy',
            np.arange(12, dtype=np.float64).reshape(3, 4),
        )
        np.save(
            source / 'text_feat.npy',
            np.arange(6, dtype=np.float32).reshape(3, 2),
        )
        if interactions is None:
            interactions = [
                (0, 0, 5.0, 100, 0),
                (1, 1, 4.0, 101, 0),
                (0, 2, 3.0, 102, 1),
                (1, 2, 2.0, 103, 2),
            ]
        lines = ['userID\titemID\trating\ttimestamp\tx_label']
        lines.extend('\t'.join(map(str, row)) for row in interactions)
        (source / 'baby.inter').write_text(
            '\n'.join(lines) + '\n', encoding='utf-8'
        )
        self._refresh_download_manifest(source)
        return source

    def _refresh_download_manifest(self, source):
        manifest = {
            'schema_version': 1,
            'status': 'downloaded_and_audited',
            'files': [
                {
                    'name': file_name,
                    **file_fingerprint(source / file_name),
                }
                for file_name in CORE_SOURCE_FILES
            ],
        }
        (source / 'download_manifest.json').write_text(
            json.dumps(manifest, indent=2) + '\n', encoding='utf-8'
        )

    def test_converts_official_split_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(temp_dir)
            output = Path(temp_dir) / 'baby'
            source_before = {
                name: file_fingerprint(source / name)
                for name in CORE_SOURCE_FILES
            }

            manifest = convert_baby(source, output)

            matrices = {}
            for file_name in ('train_mat', 'val_mat', 'test_mat'):
                with (output / file_name).open('rb') as file_obj:
                    matrices[file_name] = pickle.load(file_obj).tocsr()
            self.assertEqual(matrices['train_mat'].shape, (2, 3))
            self.assertEqual(
                [
                    matrices['train_mat'].nnz,
                    matrices['val_mat'].nnz,
                    matrices['test_mat'].nnz,
                ],
                [2, 1, 1],
            )
            np.testing.assert_array_equal(
                matrices['train_mat'].data, np.ones(2, dtype=np.float32)
            )
            self.assertEqual(
                manifest['dataset']['cold_start']['validation'][
                    'items_absent_from_train'
                ],
                [2],
            )
            self.assertEqual(
                manifest['protocol']['interaction_semantics'],
                'implicit_positive_1.0',
            )
            self.assertEqual(
                file_fingerprint(output / 'image_feat.npy'),
                file_fingerprint(source / 'image_feat.npy'),
            )
            self.assertEqual(
                source_before,
                {
                    name: file_fingerprint(source / name)
                    for name in CORE_SOURCE_FILES
                },
            )

            preflight = run_dataset_preflight(
                str(Path(temp_dir)) + os.sep, 'baby', 'error'
            )
            identity = dataset_identity_from_preflight(preflight)
            self.assertEqual(preflight['status'], 'warning')
            self.assertEqual(
                preflight['conversion_manifest']['cold_item_policy'],
                'retain_official',
            )
            self.assertIn('cold_start', identity)
            self.assertIn('conversion_manifest', identity)

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(temp_dir)
            output = Path(temp_dir) / 'baby'
            output.mkdir()
            marker = output / 'user-owned.txt'
            marker.write_text('preserve', encoding='utf-8')

            with self.assertRaisesRegex(FileExistsError, 'Refusing to overwrite'):
                convert_baby(source, output)

            self.assertEqual(marker.read_text(encoding='utf-8'), 'preserve')

    def test_rejects_source_hash_mismatch_before_writing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(temp_dir)
            output = Path(temp_dir) / 'baby'
            interaction_path = source / 'baby.inter'
            interaction_text = interaction_path.read_text(encoding='utf-8')
            interaction_path.write_text(
                interaction_text.replace('\t5.0\t', '\t4.0\t', 1),
                encoding='utf-8',
            )

            with self.assertRaisesRegex(ConversionError, 'SHA256'):
                convert_baby(source, output)

            self.assertFalse(output.exists())

    def test_rejects_duplicate_pair_and_feature_row_mismatch(self):
        invalid_cases = (
            (
                'duplicate',
                [
                    (0, 0, 5.0, 100, 0),
                    (1, 1, 4.0, 101, 0),
                    (0, 2, 3.0, 102, 1),
                    (0, 2, 2.0, 103, 2),
                ],
                'Duplicate user-item pair',
            ),
            (
                'invalid_label',
                [
                    (0, 0, 5.0, 100, 0),
                    (1, 1, 4.0, 101, 0),
                    (0, 2, 3.0, 102, 1),
                    (1, 2, 2.0, 103, 9),
                ],
                'x_label 9 is invalid',
            ),
        )
        for case_name, interactions, expected_error in invalid_cases:
            with self.subTest(case_name=case_name):
                with tempfile.TemporaryDirectory() as temp_dir:
                    source = self._write_source(temp_dir, interactions)
                    output = Path(temp_dir) / 'baby'
                    with self.assertRaisesRegex(ConversionError, expected_error):
                        convert_baby(source, output)
                    self.assertFalse(output.exists())

        with tempfile.TemporaryDirectory() as temp_dir:
            source = self._write_source(temp_dir)
            np.save(
                source / 'image_feat.npy',
                np.arange(8, dtype=np.float64).reshape(2, 4),
            )
            self._refresh_download_manifest(source)
            with self.assertRaisesRegex(ConversionError, 'rows'):
                convert_baby(source, Path(temp_dir) / 'baby')


if __name__ == '__main__':
    unittest.main()
