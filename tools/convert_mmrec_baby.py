import argparse
import csv
import hashlib
import json
import math
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / 'data' / '_incoming' / 'mmrec_baby'
DEFAULT_OUTPUT = REPO_ROOT / 'data' / 'baby'
EXPECTED_INTERACTION_COLUMNS = [
    'userID',
    'itemID',
    'rating',
    'timestamp',
    'x_label',
]
CORE_SOURCE_FILES = (
    'baby.inter',
    'i_id_mapping.csv',
    'image_feat.npy',
    'text_feat.npy',
    'u_id_mapping.csv',
)
LABEL_TO_SPLIT = {0: 'train', 1: 'validation', 2: 'test'}


class ConversionError(ValueError):
    pass


def file_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return {
        'bytes': path.stat().st_size,
        'sha256': digest.hexdigest(),
    }


def _load_and_validate_source_manifest(source_dir):
    manifest_path = source_dir / 'download_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            'Source download manifest is required for hash validation: {}'.format(
                manifest_path
            )
        )
    with manifest_path.open('r', encoding='utf-8') as file_obj:
        manifest = json.load(file_obj)
    recorded_files = {
        item.get('name'): item
        for item in manifest.get('files', [])
        if isinstance(item, dict)
    }
    fingerprints = {}
    for file_name in CORE_SOURCE_FILES:
        file_path = source_dir / file_name
        if not file_path.is_file():
            raise FileNotFoundError('Missing Baby source file: {}'.format(file_path))
        recorded = recorded_files.get(file_name)
        if not recorded:
            raise ConversionError(
                'Source manifest does not identify required file: {}'.format(file_name)
            )
        actual = file_fingerprint(file_path)
        if actual['bytes'] != int(recorded.get('bytes', -1)):
            raise ConversionError(
                '{} size does not match download_manifest.json'.format(file_name)
            )
        if actual['sha256'] != str(recorded.get('sha256', '')).lower():
            raise ConversionError(
                '{} SHA256 does not match download_manifest.json'.format(file_name)
            )
        fingerprints[file_name] = actual
    return manifest_path, manifest, fingerprints


def _read_mapping(path, raw_column, numeric_column):
    with path.open('r', encoding='utf-8-sig', newline='') as file_obj:
        reader = csv.DictReader(file_obj, delimiter='\t')
        expected_header = [raw_column, numeric_column]
        if reader.fieldnames != expected_header:
            raise ConversionError(
                '{} header must be {}, got {}'.format(
                    path.name, expected_header, reader.fieldnames
                )
            )
        raw_ids = []
        numeric_ids = []
        for line_number, row in enumerate(reader, start=2):
            raw_id = row[raw_column]
            if not raw_id:
                raise ConversionError(
                    '{} has an empty {} at line {}'.format(
                        path.name, raw_column, line_number
                    )
                )
            try:
                numeric_id = int(row[numeric_column])
            except (TypeError, ValueError) as error:
                raise ConversionError(
                    '{} has an invalid {} at line {}'.format(
                        path.name, numeric_column, line_number
                    )
                ) from error
            if numeric_id < 0:
                raise ConversionError(
                    '{} has a negative {} at line {}'.format(
                        path.name, numeric_column, line_number
                    )
                )
            raw_ids.append(raw_id)
            numeric_ids.append(numeric_id)

    if len(raw_ids) != len(set(raw_ids)):
        raise ConversionError('{} contains duplicate source IDs'.format(path.name))
    if len(numeric_ids) != len(set(numeric_ids)):
        raise ConversionError('{} contains duplicate numeric IDs'.format(path.name))
    if set(numeric_ids) != set(range(len(numeric_ids))):
        raise ConversionError(
            '{} numeric IDs must be contiguous from 0'.format(path.name)
        )
    return len(numeric_ids)


def _validate_feature(path, expected_rows):
    feature = np.load(path, mmap_mode='r', allow_pickle=False)
    if feature.ndim != 2:
        raise ConversionError(
            '{} must contain a 2D feature matrix, got {}'.format(
                path.name, feature.shape
            )
        )
    if feature.shape[0] != expected_rows:
        raise ConversionError(
            '{} rows ({}) do not match item count ({})'.format(
                path.name, feature.shape[0], expected_rows
            )
        )
    if not np.issubdtype(feature.dtype, np.number):
        raise ConversionError('{} must use a numeric dtype'.format(path.name))
    for start in range(0, feature.shape[0], 1024):
        if not np.isfinite(feature[start:start + 1024]).all():
            raise ConversionError('{} contains NaN or infinite values'.format(path.name))
    return {
        'shape': list(feature.shape),
        'dtype': str(feature.dtype),
    }


def _read_interactions(path, n_users, n_items):
    split_records = {split_name: [] for split_name in LABEL_TO_SPLIT.values()}
    seen_pairs = {}
    observed_users = set()
    observed_items = set()

    with path.open('r', encoding='utf-8', newline='') as file_obj:
        reader = csv.DictReader(file_obj, delimiter='\t')
        if reader.fieldnames != EXPECTED_INTERACTION_COLUMNS:
            raise ConversionError(
                '{} header must be {}, got {}'.format(
                    path.name, EXPECTED_INTERACTION_COLUMNS, reader.fieldnames
                )
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                user_id = int(row['userID'])
                item_id = int(row['itemID'])
                rating = float(row['rating'])
                int(row['timestamp'])
                label = int(row['x_label'])
            except (TypeError, ValueError) as error:
                raise ConversionError(
                    '{} contains an invalid value at line {}'.format(
                        path.name, line_number
                    )
                ) from error
            if not 0 <= user_id < n_users:
                raise ConversionError(
                    'userID {} is out of range at line {}'.format(
                        user_id, line_number
                    )
                )
            if not 0 <= item_id < n_items:
                raise ConversionError(
                    'itemID {} is out of range at line {}'.format(
                        item_id, line_number
                    )
                )
            if not math.isfinite(rating):
                raise ConversionError('rating is not finite at line {}'.format(line_number))
            if label not in LABEL_TO_SPLIT:
                raise ConversionError(
                    'x_label {} is invalid at line {}'.format(label, line_number)
                )
            pair = (user_id, item_id)
            if pair in seen_pairs:
                raise ConversionError(
                    'Duplicate user-item pair {} at lines {} and {}'.format(
                        pair, seen_pairs[pair], line_number
                    )
                )
            seen_pairs[pair] = line_number
            split_records[LABEL_TO_SPLIT[label]].append(pair)
            observed_users.add(user_id)
            observed_items.add(item_id)

    if observed_users != set(range(n_users)):
        raise ConversionError('Interaction users do not match the user mapping universe')
    if observed_items != set(range(n_items)):
        raise ConversionError('Interaction items do not match the item mapping universe')
    if any(not records for records in split_records.values()):
        raise ConversionError('Train, validation, and test must all be non-empty')
    return split_records


def _cold_start_report(split_records):
    train_users = {user_id for user_id, _ in split_records['train']}
    train_items = {item_id for _, item_id in split_records['train']}
    report = {
        'policy': 'retain_official',
        'train_users': len(train_users),
        'train_items': len(train_items),
    }
    for split_name in ('validation', 'test'):
        records = split_records[split_name]
        cold_users = sorted({user_id for user_id, _ in records} - train_users)
        cold_items = sorted({item_id for _, item_id in records} - train_items)
        cold_user_set = set(cold_users)
        cold_item_set = set(cold_items)
        report[split_name] = {
            'users_absent_from_train': cold_users,
            'items_absent_from_train': cold_items,
            'interactions_with_users_absent_from_train': sum(
                user_id in cold_user_set for user_id, _ in records
            ),
            'interactions_with_items_absent_from_train': sum(
                item_id in cold_item_set for _, item_id in records
            ),
        }
    report['has_cold_start'] = any(
        report[split_name]['users_absent_from_train']
        or report[split_name]['items_absent_from_train']
        for split_name in ('validation', 'test')
    )
    return report


def _build_matrix(records, shape):
    rows = np.fromiter((user_id for user_id, _ in records), dtype=np.int64)
    cols = np.fromiter((item_id for _, item_id in records), dtype=np.int64)
    values = np.ones(len(records), dtype=np.float32)
    return sp.coo_matrix((values, (rows, cols)), shape=shape).tocsr()


def _write_pickle(path, value):
    with path.open('wb') as file_obj:
        pickle.dump(value, file_obj, protocol=pickle.HIGHEST_PROTOCOL)


def _resolved_is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def convert_baby(source_dir, output_dir, invocation=None):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError('Baby source directory does not exist: {}'.format(source_dir))
    if output_dir.exists():
        raise FileExistsError(
            'Refusing to overwrite existing output directory: {}'.format(output_dir)
        )
    if output_dir == source_dir or _resolved_is_within(output_dir, source_dir):
        raise ConversionError('Derived output must not be inside the raw source directory')

    staging_dir = output_dir.with_name('.{}.building'.format(output_dir.name))
    if staging_dir.exists():
        raise FileExistsError(
            'Refusing to overwrite an existing conversion staging directory: {}'.format(
                staging_dir
            )
        )

    source_manifest_path, source_manifest, source_fingerprints = (
        _load_and_validate_source_manifest(source_dir)
    )
    n_items = _read_mapping(
        source_dir / 'i_id_mapping.csv', 'asin', 'itemID'
    )
    n_users = _read_mapping(
        source_dir / 'u_id_mapping.csv', 'user_id', 'userID'
    )
    image_report = _validate_feature(source_dir / 'image_feat.npy', n_items)
    text_report = _validate_feature(source_dir / 'text_feat.npy', n_items)
    split_records = _read_interactions(
        source_dir / 'baby.inter', n_users, n_items
    )
    cold_start = _cold_start_report(split_records)
    shape = (n_users, n_items)
    matrices = {
        'train_mat': _build_matrix(split_records['train'], shape),
        'val_mat': _build_matrix(split_records['validation'], shape),
        'test_mat': _build_matrix(split_records['test'], shape),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    for file_name, matrix in matrices.items():
        _write_pickle(staging_dir / file_name, matrix)
    for file_name in (
        'image_feat.npy',
        'text_feat.npy',
        'i_id_mapping.csv',
        'u_id_mapping.csv',
    ):
        shutil.copy2(source_dir / file_name, staging_dir / file_name)

    source_fingerprints_after = {
        file_name: file_fingerprint(source_dir / file_name)
        for file_name in CORE_SOURCE_FILES
    }
    if source_fingerprints_after != source_fingerprints:
        raise ConversionError('Raw source fingerprints changed during conversion')

    output_artifacts = {
        file_name: file_fingerprint(staging_dir / file_name)
        for file_name in (
            'train_mat',
            'val_mat',
            'test_mat',
            'image_feat.npy',
            'text_feat.npy',
            'i_id_mapping.csv',
            'u_id_mapping.csv',
        )
    }
    converter_path = Path(__file__).resolve()
    manifest = {
        'schema_version': 1,
        'status': 'completed',
        'converted_at': datetime.now().astimezone().isoformat(),
        'converter': {
            'path': str(converter_path),
            'sha256': file_fingerprint(converter_path)['sha256'],
            'invocation': invocation or 'library_call',
        },
        'source': {
            'directory': str(source_dir),
            'download_manifest': str(source_manifest_path),
            'download_manifest_sha256': file_fingerprint(source_manifest_path)['sha256'],
            'download_status': source_manifest.get('status'),
            'artifacts': source_fingerprints,
            'verified_unchanged_after_conversion': True,
        },
        'output': {
            'directory': str(output_dir),
            'artifacts': output_artifacts,
        },
        'protocol': {
            'interaction_semantics': 'implicit_positive_1.0',
            'source_rating_retained_as_weight': False,
            'label_mapping': {
                '0': 'train_mat',
                '1': 'val_mat',
                '2': 'test_mat',
            },
            'cold_item_policy': 'retain_official',
        },
        'dataset': {
            'users': n_users,
            'items': n_items,
            'matrix_shape': [n_users, n_items],
            'split_interactions': {
                split_name: len(records)
                for split_name, records in split_records.items()
            },
            'image_features': image_report,
            'text_features': text_report,
            'cold_start': cold_start,
        },
    }
    with (staging_dir / 'conversion_manifest.json').open(
        'w', encoding='utf-8', newline='\n'
    ) as file_obj:
        json.dump(manifest, file_obj, indent=2, sort_keys=True)
        file_obj.write('\n')

    staging_dir.replace(output_dir)
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Convert the audited MMRec Baby dataset into PromptMM matrix format.'
    )
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    invocation = ' '.join([str(Path(sys.executable).resolve()), str(Path(__file__).resolve())] + (argv if argv is not None else sys.argv[1:]))
    manifest = convert_baby(args.source, args.output, invocation=invocation)
    summary = {
        'status': manifest['status'],
        'output': manifest['output']['directory'],
        'users': manifest['dataset']['users'],
        'items': manifest['dataset']['items'],
        'split_interactions': manifest['dataset']['split_interactions'],
        'cold_start': manifest['dataset']['cold_start'],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
