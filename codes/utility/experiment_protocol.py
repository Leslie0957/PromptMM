import ast
import hashlib
import json
import os
import pickle
import shutil
import uuid
from datetime import datetime

import numpy as np
import scipy.sparse as sp


PAPER_READY_PROTOCOL = 'val_test_once_v1'
LEGACY_PROTOCOL = 'legacy_test_best'
SUPPORTED_EVAL_PROTOCOLS = (PAPER_READY_PROTOCOL, LEGACY_PROTOCOL)
PRIMARY_SELECTION_K = 20
TRAIN_ONLY_CANDIDATE_EXCLUSION = 'train_only'
TEACHER_INFERENCE_CONFIG_FIELDS = (
    'embed_size',
    'weight_size',
    'layers',
    'sparse',
    'model_cat_rate',
    'feat_soft_token_rate',
    'soft_token_rate',
    'hard_token_type',
    'hard_token_seed',
)


def protocol_uses_validation(eval_protocol):
    if eval_protocol not in SUPPORTED_EVAL_PROTOCOLS:
        raise ValueError(
            'Unsupported evaluation protocol {!r}; expected one of {}.'.format(
                eval_protocol, SUPPORTED_EVAL_PROTOCOLS
            )
        )
    return eval_protocol == PAPER_READY_PROTOCOL


def candidate_exclusion_policy(eval_protocol):
    protocol_uses_validation(eval_protocol)
    return TRAIN_ONLY_CANDIDATE_EXCLUSION


def early_stopping_non_improvement_limit(eval_protocol, patience):
    protocol_uses_validation(eval_protocol)
    if patience < 0:
        raise ValueError('early_stopping_patience must be non-negative.')
    legacy_extra_epoch = 1 if eval_protocol == LEGACY_PROTOCOL else 0
    return max(1, int(patience) + legacy_extra_epoch)


def resolve_primary_k_index(ks_value, primary_k=PRIMARY_SELECTION_K):
    ks = ast.literal_eval(ks_value) if isinstance(ks_value, str) else list(ks_value)
    if not isinstance(ks, (list, tuple)) or not all(isinstance(k, int) for k in ks):
        raise ValueError('Ks must be a list or tuple of integers, got {!r}.'.format(ks_value))
    if primary_k not in ks:
        raise ValueError('Primary selection K={} is missing from Ks={}.'.format(primary_k, ks))
    return list(ks).index(primary_k), list(ks)


def _sha256(file_path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(file_path, 'rb') as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(file_path):
    resolved_path = os.path.abspath(file_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError('Cannot fingerprint missing file: {}'.format(resolved_path))
    return {
        'path': resolved_path,
        'bytes': int(os.path.getsize(resolved_path)),
        'sha256': _sha256(resolved_path),
    }


def restore_checkpoint_then_evaluate(checkpoint_path, load_checkpoint, evaluate):
    checkpoint = load_checkpoint(checkpoint_path)
    result = evaluate()
    return checkpoint, result


def teacher_inference_config_from_namespace(namespace):
    config = {}
    for field_name in TEACHER_INFERENCE_CONFIG_FIELDS:
        value = getattr(namespace, field_name)
        if field_name == 'weight_size':
            value = ast.literal_eval(value) if isinstance(value, str) else list(value)
        config[field_name] = _jsonable(value)
    return config


def dataset_identity_from_preflight(preflight_report):
    matrices = preflight_report.get('matrices')
    features = preflight_report.get('features')
    if not matrices or not features:
        raise ValueError(
            'A complete dataset preflight report is required to build dataset identity.'
        )
    identity = {
        'dataset': preflight_report.get('dataset'),
        'matrices': {
            split_name: {
                key: matrices[split_name][key]
                for key in ('sha256', 'bytes', 'shape', 'interactions', 'dtype')
            }
            for split_name in ('train', 'validation', 'test')
        },
        'features': {
            modality: {
                key: features[modality][key]
                for key in ('sha256', 'bytes', 'shape', 'dtype')
            }
            for modality in ('image', 'text')
        },
        'split_overlaps': preflight_report.get('split_overlaps'),
        'duplicate_modalities': preflight_report.get('duplicate_modalities'),
    }
    cold_start = preflight_report.get('cold_start')
    if cold_start and cold_start.get('has_cold_start'):
        identity['cold_start'] = cold_start
    conversion_manifest = preflight_report.get('conversion_manifest')
    if conversion_manifest:
        identity['conversion_manifest'] = {
            key: conversion_manifest.get(key)
            for key in (
                'sha256',
                'bytes',
                'status',
                'cold_item_policy',
                'source_download_manifest_sha256',
                'converter_sha256',
            )
        }
    return identity


def training_batch_count(n_train, batch_size, smoke_train_batches=0):
    if n_train <= 0:
        raise ValueError('n_train must be positive.')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')
    if smoke_train_batches < 0:
        raise ValueError('smoke_train_batches must be non-negative.')
    full_batch_count = n_train // batch_size + 1
    if smoke_train_batches:
        return min(full_batch_count, int(smoke_train_batches))
    return full_batch_count


def validate_teacher_checkpoint_metadata(
    checkpoint,
    active_protocol,
    primary_k,
    active_dataset,
    checkpoint_path,
    active_dataset_identity=None,
    active_teacher_inference_config=None,
    active_paper_ready_eligible=None,
    active_paper_ready_blockers=None,
    active_candidate_exclusion_policy=None,
):
    checkpoint_protocol = checkpoint.get('evaluation_protocol')
    checkpoint_selection_split = checkpoint.get('selection_split')
    checkpoint_primary_k = checkpoint.get('primary_k')
    checkpoint_dataset = checkpoint.get('dataset')
    checkpoint_candidate_policy = checkpoint.get('candidate_exclusion_policy')
    if checkpoint_candidate_policy is None and checkpoint_protocol in SUPPORTED_EVAL_PROTOCOLS:
        checkpoint_candidate_policy = candidate_exclusion_policy(checkpoint_protocol)

    if active_protocol == PAPER_READY_PROTOCOL:
        if (
            checkpoint_protocol != PAPER_READY_PROTOCOL
            or checkpoint_selection_split != 'validation'
            or checkpoint_primary_k != primary_k
        ):
            raise ValueError(
                'Paper-ready teacher reuse requires checkpoint metadata '
                'evaluation_protocol=%s, selection_split=validation, primary_k=%d. '
                'Checkpoint %s contains protocol=%r, split=%r, primary_k=%r.' % (
                    PAPER_READY_PROTOCOL,
                    primary_k,
                    checkpoint_path,
                    checkpoint_protocol,
                    checkpoint_selection_split,
                    checkpoint_primary_k,
                )
            )
        if checkpoint_dataset != active_dataset:
            raise ValueError(
                'Paper-ready teacher checkpoint dataset must equal active dataset %r; '
                'checkpoint %s contains %r.' % (
                    active_dataset, checkpoint_path, checkpoint_dataset
                )
            )
        required_frozen_metadata = {
            'dataset_identity': active_dataset_identity,
            'teacher_inference_config': active_teacher_inference_config,
            'paper_ready_eligible': active_paper_ready_eligible,
            'paper_ready_blockers': list(active_paper_ready_blockers or []),
            'candidate_exclusion_policy': active_candidate_exclusion_policy,
        }
        for metadata_name, expected_value in required_frozen_metadata.items():
            if expected_value is None:
                raise ValueError(
                    'Active run is missing required frozen-teacher metadata: {}.'.format(
                        metadata_name
                    )
                )
            checkpoint_value = (
                checkpoint_candidate_policy
                if metadata_name == 'candidate_exclusion_policy'
                else checkpoint.get(metadata_name)
            )
            if checkpoint_value != expected_value:
                raise ValueError(
                    'Teacher checkpoint frozen metadata mismatch for {} at {}.'.format(
                        metadata_name, checkpoint_path
                    )
                )
        if not checkpoint.get('hard_token_cache'):
            raise ValueError(
                'Paper-ready teacher checkpoint is missing hard-token provenance: {}'.format(
                    checkpoint_path
                )
            )
    elif checkpoint_protocol not in (None, LEGACY_PROTOCOL):
        raise ValueError(
            'Legacy protocol run cannot silently reuse a %r teacher checkpoint: %s' % (
                checkpoint_protocol, checkpoint_path
            )
        )

    if (
        active_protocol == LEGACY_PROTOCOL
        and checkpoint_dataset is not None
        and checkpoint_dataset != active_dataset
    ):
        raise ValueError(
            'Teacher checkpoint dataset %r does not match active dataset %r: %s' % (
                checkpoint_dataset, active_dataset, checkpoint_path
            )
        )


def _all_finite(array, row_chunk_size=4096):
    if array.ndim == 0:
        return bool(np.isfinite(array))
    for start in range(0, array.shape[0], row_chunk_size):
        if not np.isfinite(array[start:start + row_chunk_size]).all():
            return False
    return True


def _evaluation_cold_start(train_matrix, evaluation_matrix):
    train_user_seen = np.asarray(train_matrix.getnnz(axis=1)).reshape(-1) > 0
    train_item_seen = np.asarray(train_matrix.getnnz(axis=0)).reshape(-1) > 0
    evaluation_user_seen = np.asarray(evaluation_matrix.getnnz(axis=1)).reshape(-1) > 0
    evaluation_item_seen = np.asarray(evaluation_matrix.getnnz(axis=0)).reshape(-1) > 0
    cold_user_mask = evaluation_user_seen & ~train_user_seen
    cold_item_mask = evaluation_item_seen & ~train_item_seen
    evaluation_coo = evaluation_matrix.tocoo(copy=False)
    return {
        'users_absent_from_train': np.flatnonzero(cold_user_mask).astype(int).tolist(),
        'items_absent_from_train': np.flatnonzero(cold_item_mask).astype(int).tolist(),
        'interactions_with_users_absent_from_train': int(
            np.count_nonzero(~train_user_seen[evaluation_coo.row])
        ),
        'interactions_with_items_absent_from_train': int(
            np.count_nonzero(~train_item_seen[evaluation_coo.col])
        ),
    }


def _arrays_equal(left, right, row_chunk_size=4096):
    if left.shape != right.shape:
        return False
    for start in range(0, left.shape[0], row_chunk_size):
        if not np.array_equal(
            left[start:start + row_chunk_size],
            right[start:start + row_chunk_size],
        ):
            return False
    return True


def run_dataset_preflight(data_path, dataset, duplicate_modalities_policy='warn'):
    if duplicate_modalities_policy not in ('allow', 'warn', 'error'):
        raise ValueError(
            'duplicate_modalities_policy must be allow, warn, or error; got {!r}.'.format(
                duplicate_modalities_policy
            )
        )

    dataset_dir = os.path.abspath(os.path.join(data_path, dataset))
    required_files = {
        'image_features': 'image_feat.npy',
        'text_features': 'text_feat.npy',
        'train_matrix': 'train_mat',
        'validation_matrix': 'val_mat',
        'test_matrix': 'test_mat',
    }
    resolved_files = {
        name: os.path.join(dataset_dir, file_name)
        for name, file_name in required_files.items()
    }
    missing = [path for path in resolved_files.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            'Dataset preflight failed for {}. Missing required files: {}'.format(
                dataset_dir, ', '.join(missing)
            )
        )

    conversion_manifest_path = os.path.join(dataset_dir, 'conversion_manifest.json')
    conversion_manifest_data = None
    if os.path.isfile(conversion_manifest_path):
        with open(conversion_manifest_path, 'r', encoding='utf-8') as file_obj:
            conversion_manifest_data = json.load(file_obj)
        if conversion_manifest_data.get('status') != 'completed':
            raise ValueError(
                'Conversion manifest is not completed: {}'.format(
                    conversion_manifest_path
                )
            )

    matrices = {}
    matrix_report = {}
    for split_name, file_key in (
        ('train', 'train_matrix'),
        ('validation', 'validation_matrix'),
        ('test', 'test_matrix'),
    ):
        with open(resolved_files[file_key], 'rb') as file_obj:
            matrix = pickle.load(file_obj)
        if not sp.issparse(matrix):
            raise TypeError(
                '{} must contain a scipy sparse matrix, got {}.'.format(
                    resolved_files[file_key], type(matrix).__name__
                )
            )
        matrix = matrix.tocsr()
        if matrix.data.size and not np.isfinite(matrix.data).all():
            raise ValueError('{} contains NaN or infinite values.'.format(resolved_files[file_key]))
        if matrix.data.size and np.any(matrix.data == 0):
            raise ValueError('{} contains explicitly stored zero values.'.format(resolved_files[file_key]))
        if matrix.data.size and np.any(matrix.data < 0):
            raise ValueError('{} contains negative interaction values.'.format(resolved_files[file_key]))
        matrices[split_name] = matrix
        matrix_report[split_name] = {
            'shape': list(matrix.shape),
            'interactions': int(matrix.nnz),
            'dtype': str(matrix.dtype),
            **file_fingerprint(resolved_files[file_key]),
        }

    train_shape = matrices['train'].shape
    for split_name in ('validation', 'test'):
        if matrices[split_name].shape != train_shape:
            raise ValueError(
                'Dataset matrix shapes differ: train={} but {}={}.'.format(
                    train_shape, split_name, matrices[split_name].shape
                )
            )
    if any(matrix.nnz == 0 for matrix in matrices.values()):
        raise ValueError('Train, validation, and test matrices must all contain interactions.')

    split_overlaps = {
        'train_validation': int(matrices['train'].multiply(matrices['validation']).nnz),
        'train_test': int(matrices['train'].multiply(matrices['test']).nnz),
        'validation_test': int(matrices['validation'].multiply(matrices['test']).nnz),
    }
    nonzero_overlaps = {
        split_pair: overlap_count
        for split_pair, overlap_count in split_overlaps.items()
        if overlap_count > 0
    }
    if nonzero_overlaps:
        raise ValueError(
            'Dataset splits contain overlapping user-item interactions: {}.'.format(
                nonzero_overlaps
            )
        )

    conversion_policy = (
        conversion_manifest_data.get('protocol', {}).get('cold_item_policy')
        if conversion_manifest_data
        else 'unspecified'
    )
    cold_start = {
        'policy': conversion_policy,
        'validation': _evaluation_cold_start(
            matrices['train'], matrices['validation']
        ),
        'test': _evaluation_cold_start(matrices['train'], matrices['test']),
    }
    cold_start['has_cold_start'] = any(
        cold_start[split_name]['users_absent_from_train']
        or cold_start[split_name]['items_absent_from_train']
        for split_name in ('validation', 'test')
    )

    feature_report = {}
    feature_arrays = {}
    for modality_name, file_key in (
        ('image', 'image_features'),
        ('text', 'text_features'),
    ):
        feature_path = resolved_files[file_key]
        feature_array = np.load(feature_path, mmap_mode='r', allow_pickle=False)
        if feature_array.ndim != 2:
            raise ValueError(
                '{} features must be a 2D array, got shape {}.'.format(
                    modality_name, feature_array.shape
                )
            )
        if feature_array.shape[0] != train_shape[1]:
            raise ValueError(
                '{} feature rows ({}) do not match matrix item count ({}).'.format(
                    modality_name, feature_array.shape[0], train_shape[1]
                )
            )
        finite = _all_finite(feature_array)
        if not finite:
            raise ValueError('{} features contain NaN or infinite values.'.format(modality_name))
        feature_arrays[modality_name] = feature_array
        feature_report[modality_name] = {
            'path': feature_path,
            'shape': list(feature_array.shape),
            'dtype': str(feature_array.dtype),
            'bytes': int(os.path.getsize(feature_path)),
            'finite': finite,
            'sha256': _sha256(feature_path),
        }

    byte_identical_modalities = (
        feature_report['image']['bytes'] == feature_report['text']['bytes']
        and feature_report['image']['sha256'] == feature_report['text']['sha256']
    )
    numerically_identical_modalities = _arrays_equal(
        feature_arrays['image'], feature_arrays['text']
    )
    duplicate_modalities = byte_identical_modalities or numerically_identical_modalities
    warnings = []
    if duplicate_modalities:
        duplicate_message = (
            'image_feat.npy and text_feat.npy are numerically identical; this dataset cannot '
            'independently support an image-text complementarity claim.'
        )
        if duplicate_modalities_policy == 'error':
            raise ValueError(duplicate_message)
        if duplicate_modalities_policy == 'warn':
            warnings.append(duplicate_message)

    conversion_manifest_report = None
    if conversion_manifest_data:
        recorded_artifacts = conversion_manifest_data.get('output', {}).get(
            'artifacts', {}
        )
        actual_artifacts = {
            'train_mat': matrix_report['train'],
            'val_mat': matrix_report['validation'],
            'test_mat': matrix_report['test'],
            'image_feat.npy': feature_report['image'],
            'text_feat.npy': feature_report['text'],
        }
        for file_name, actual in actual_artifacts.items():
            recorded = recorded_artifacts.get(file_name)
            if not recorded:
                raise ValueError(
                    'Conversion manifest is missing output artifact {}'.format(
                        file_name
                    )
                )
            if (
                int(recorded.get('bytes', -1)) != actual['bytes']
                or str(recorded.get('sha256', '')).lower() != actual['sha256']
            ):
                raise ValueError(
                    'Converted artifact does not match conversion manifest: {}'.format(
                        file_name
                    )
                )
        manifest_fingerprint = file_fingerprint(conversion_manifest_path)
        conversion_manifest_report = {
            **manifest_fingerprint,
            'status': conversion_manifest_data.get('status'),
            'cold_item_policy': conversion_policy,
            'source_download_manifest_sha256': conversion_manifest_data.get(
                'source', {}
            ).get('download_manifest_sha256'),
            'converter_sha256': conversion_manifest_data.get('converter', {}).get(
                'sha256'
            ),
        }

    if cold_start['has_cold_start']:
        warnings.append(
            'Evaluation contains entities absent from training: '
            'validation users={}, items={}, interactions(user/item)={}/{}; '
            'test users={}, items={}, interactions(user/item)={}/{}. '
            'Cold-start policy={!r}; document this condition for formal comparison.'.format(
                len(cold_start['validation']['users_absent_from_train']),
                len(cold_start['validation']['items_absent_from_train']),
                cold_start['validation']['interactions_with_users_absent_from_train'],
                cold_start['validation']['interactions_with_items_absent_from_train'],
                len(cold_start['test']['users_absent_from_train']),
                len(cold_start['test']['items_absent_from_train']),
                cold_start['test']['interactions_with_users_absent_from_train'],
                cold_start['test']['interactions_with_items_absent_from_train'],
                conversion_policy,
            )
        )

    return {
        'checked_at': datetime.now().astimezone().isoformat(),
        'dataset': dataset,
        'dataset_dir': dataset_dir,
        'status': 'warning' if warnings else 'passed',
        'duplicate_modalities_policy': duplicate_modalities_policy,
        'duplicate_modalities': duplicate_modalities,
        'byte_identical_modalities': byte_identical_modalities,
        'numerically_identical_modalities': numerically_identical_modalities,
        'matrices': matrix_report,
        'split_overlaps': split_overlaps,
        'cold_start': cold_start,
        'features': feature_report,
        'conversion_manifest': conversion_manifest_report,
        'warnings': warnings,
    }


def namespace_to_dict(namespace):
    return {key: _jsonable(value) for key, value in sorted(vars(namespace).items())}


def result_to_dict(result):
    if result is None:
        return None
    return {key: _jsonable(value) for key, value in result.items()}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(file_path, payload):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    temporary_path = '{}.tmp.{}'.format(file_path, os.getpid())
    try:
        with open(temporary_path, 'w', encoding='utf-8') as file_obj:
            json.dump(_jsonable(payload), file_obj, ensure_ascii=True, indent=2, sort_keys=True)
            file_obj.write('\n')
        os.replace(temporary_path, file_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def publish_file_atomically(source_path, destination_path, allow_overwrite=False):
    source_path = os.path.abspath(source_path)
    destination_path = os.path.abspath(destination_path)
    if source_path == destination_path:
        raise ValueError('Source and destination must be different paths.')
    if not os.path.isfile(source_path):
        raise FileNotFoundError('Publication source does not exist: {}'.format(source_path))
    if os.path.exists(destination_path) and not allow_overwrite:
        raise FileExistsError(
            'Refusing to overwrite existing published artifact: {}'.format(destination_path)
        )

    destination_dir = os.path.dirname(destination_path)
    os.makedirs(destination_dir, exist_ok=True)
    temporary_path = '{}.tmp.{}.{}'.format(
        destination_path, os.getpid(), uuid.uuid4().hex
    )
    try:
        shutil.copy2(source_path, temporary_path)
        if allow_overwrite:
            os.replace(temporary_path, destination_path)
        else:
            # Linking a fully written temporary file gives no-clobber publication
            # on both NTFS and common Linux filesystems.
            os.link(temporary_path, destination_path)
            os.unlink(temporary_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return file_fingerprint(destination_path)
