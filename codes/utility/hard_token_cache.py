import hashlib
import json
import os
import pickle
from datetime import datetime

import numpy as np
import sklearn
from sklearn.decomposition import FastICA, PCA

from utility.experiment_protocol import file_fingerprint


HARD_TOKEN_CACHE_VERSION = 2
SUPPORTED_METHODS = ('pca', 'ica')


def _cache_identity(metadata):
    identity_payload = {
        'format_version': metadata['format_version'],
        'method': metadata['method'],
        'modality': metadata['modality'],
        'n_components': metadata['n_components'],
        'random_state': metadata['random_state'],
        'source_sha256': metadata['source_feature']['sha256'],
        'source_bytes': metadata['source_feature']['bytes'],
        'source_shape': metadata['source_shape'],
        'source_dtype': metadata['source_dtype'],
        'sklearn_version': metadata['sklearn_version'],
    }
    serialized = json.dumps(identity_payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('ascii')).hexdigest()


def _validate_cached_payload(payload, expected_identity, expected_rows, n_components, cache_path):
    if not isinstance(payload, dict) or 'metadata' not in payload or 'values' not in payload:
        raise ValueError(
            'Versioned hard-token cache has an invalid payload and will not be reused: {}'.format(
                cache_path
            )
        )
    if payload['metadata'].get('cache_identity') != expected_identity:
        raise ValueError('Hard-token cache identity mismatch: {}'.format(cache_path))
    values = np.asarray(payload['values'])
    expected_shape = (expected_rows, n_components)
    if values.shape != expected_shape:
        raise ValueError(
            'Hard-token cache {} has shape {}, expected {}.'.format(
                cache_path, values.shape, expected_shape
            )
        )
    if not np.isfinite(values).all():
        raise ValueError('Hard-token cache contains NaN or infinite values: {}'.format(cache_path))
    return values


def load_or_create_hard_token_cache(
    feature_array,
    source_feature_path,
    dataset_dir,
    modality,
    method,
    n_components,
    random_state,
):
    if method not in SUPPORTED_METHODS:
        raise ValueError('Unsupported hard-token cache method {!r}.'.format(method))
    if modality not in ('image', 'text'):
        raise ValueError('Unsupported hard-token modality {!r}.'.format(modality))

    features = np.asarray(feature_array)
    if features.ndim != 2:
        raise ValueError('{} features must be two-dimensional.'.format(modality))
    if n_components <= 0 or n_components > min(features.shape):
        raise ValueError(
            'n_components={} is invalid for {} feature shape {}.'.format(
                n_components, modality, features.shape
            )
        )
    if not np.isfinite(features).all():
        raise ValueError('{} features contain NaN or infinite values.'.format(modality))

    metadata = {
        'format_version': HARD_TOKEN_CACHE_VERSION,
        'method': method,
        'modality': modality,
        'n_components': int(n_components),
        'random_state': int(random_state),
        'source_feature': file_fingerprint(source_feature_path),
        'source_shape': list(features.shape),
        'source_dtype': str(features.dtype),
        'sklearn_version': sklearn.__version__,
    }
    identity = _cache_identity(metadata)
    metadata['cache_identity'] = identity
    cache_name = 'hard_token_{}_{}_v{}_{}.pkl'.format(
        modality, method, HARD_TOKEN_CACHE_VERSION, identity[:16]
    )
    cache_path = os.path.abspath(os.path.join(dataset_dir, cache_name))

    if os.path.isfile(cache_path):
        with open(cache_path, 'rb') as file_obj:
            payload = pickle.load(file_obj)
        values = _validate_cached_payload(
            payload, identity, features.shape[0], n_components, cache_path
        )
        cache_hit = True
    else:
        if method == 'pca':
            transformer = PCA(
                n_components=n_components,
                svd_solver='randomized',
                random_state=random_state,
            )
        else:
            transformer = FastICA(
                n_components=n_components,
                random_state=random_state,
            )
        values = transformer.fit_transform(features)
        if not np.isfinite(values).all():
            raise ValueError(
                '{} hard-token transform produced NaN or infinite values.'.format(method)
            )
        payload = {
            'metadata': {
                **metadata,
                'created_at': datetime.now().astimezone().isoformat(),
            },
            'values': values,
        }
        os.makedirs(dataset_dir, exist_ok=True)
        temporary_path = '{}.tmp.{}'.format(cache_path, os.getpid())
        with open(temporary_path, 'wb') as file_obj:
            pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, cache_path)
        cache_hit = False

    record = {
        **metadata,
        'cache_hit': cache_hit,
        'cache_file': file_fingerprint(cache_path),
    }
    return values, record


def load_legacy_hard_token_cache(dataset_dir, modality, method, expected_shape):
    if method not in SUPPORTED_METHODS:
        raise ValueError('Unsupported legacy hard-token method {!r}.'.format(method))
    if modality not in ('image', 'text'):
        raise ValueError('Unsupported legacy hard-token modality {!r}.'.format(modality))
    cache_path = os.path.abspath(
        os.path.join(dataset_dir, 'hard_token_{}_{}'.format(modality, method))
    )
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            'Legacy protocol requires the preserved historical hard-token cache: {}'.format(
                cache_path
            )
        )
    with open(cache_path, 'rb') as file_obj:
        values = np.asarray(pickle.load(file_obj))
    if tuple(values.shape) != tuple(expected_shape):
        raise ValueError(
            'Legacy hard-token cache {} has shape {}, expected {}.'.format(
                cache_path, values.shape, expected_shape
            )
        )
    if not np.isfinite(values).all():
        raise ValueError('Legacy hard-token cache contains invalid values: {}'.format(cache_path))
    return values, {
        'format_version': 1,
        'legacy_metadata_free': True,
        'method': method,
        'modality': modality,
        'cache_hit': True,
        'cache_file': file_fingerprint(cache_path),
    }
