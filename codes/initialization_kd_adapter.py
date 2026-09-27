"""Isolated preparation adapter. Importing this module does no asset I/O.

The old Data/batch_test modules must never be imported: they read Test on init.
No CLI is exposed while the four-arm declaration is non-launchable.
"""

import hashlib
import heapq
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np

from initialization_kd_interaction import assert_train_val_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
        raise RuntimeError('Asset SHA256 mismatch: ' + str(path))


def load_train_val(dataset_root, expected_train, expected_val, n_users, n_items):
    """Read exactly two hash-pinned trusted sparse pickles after Test hook install."""
    import scipy.sparse as sparse

    root = Path(dataset_root)
    matrices = []
    for name, expected in (('train_mat', expected_train), ('val_mat', expected_val)):
        path = root / name
        assert_train_val_path(path, root)
        require_hash(path, expected)
        with path.open('rb') as stream:
            matrix = pickle.load(stream)
        if not sparse.issparse(matrix) or matrix.shape != (n_users, n_items):
            raise ValueError(name + ' sparse shape/type mismatch')
        matrix = matrix.tocsr(copy=True)
        matrix.sum_duplicates()
        matrix.sort_indices()
        if matrix.nnz == 0 or not np.isfinite(matrix.data).all() or (matrix.data <= 0).any():
            raise ValueError(name + ' values invalid')
        matrices.append(matrix)
    if not np.any(np.diff(matrices[1].indptr)):
        raise ValueError('No Validation users')
    return tuple(matrices)


def _dcg(bits):
    return sum(float(bit) / math.log2(index + 2) for index, bit in enumerate(bits))


def evaluate_validation(model, train, val, ks=(10, 20, 40, 50), user_batch=256,
                        ranker=None):
    """Mirror part/heapq evaluator: Val users, Train exclusion, dot-product rank."""
    import torch

    if train.shape != val.shape or any(k < 1 or k > train.shape[1] for k in ks):
        raise ValueError('Evaluation shape/K mismatch')
    users = np.flatnonzero(np.diff(val.indptr))
    if not len(users):
        raise ValueError('No Validation users')
    max_k = max(ks)
    recall = np.zeros(len(ks), dtype=np.float64)
    ndcg = np.zeros(len(ks), dtype=np.float64)
    item = model.item_id_embedding.weight.detach()
    for offset in range(0, len(users), user_batch):
        chunk = users[offset:offset + user_batch]
        user = model.user_id_embedding.weight[torch.as_tensor(chunk, device=item.device)]
        scores = (user @ item.T).detach().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite Validation scores')
        for row, uid in enumerate(chunk):
            seen = set(train.indices[train.indptr[uid]:train.indptr[uid + 1]].tolist())
            positives = set(val.indices[val.indptr[uid]:val.indptr[uid + 1]].tolist())
            if ranker is None:
                candidates = (index for index in range(train.shape[1]) if index not in seen)
                # Preserve the completed smoke's reference implementation.
                ranked = heapq.nlargest(max_k, candidates, key=scores[row].__getitem__)
            else:
                ranked = ranker(scores[row], seen, max_k)
            if len(ranked) < max_k:
                raise ValueError('Insufficient Validation candidates')
            bits = [int(index in positives) for index in ranked]
            for i, k in enumerate(ks):
                recall[i] += sum(bits[:k]) / len(positives)
                ideal = _dcg(sorted(bits, reverse=True)[:k])
                ndcg[i] += _dcg(bits[:k]) / ideal if ideal else 0.0
    recall /= len(users)
    ndcg /= len(users)
    result = {'recall': recall.tolist(), 'ndcg': ndcg.tolist(),
              'validation_users': int(len(users))}
    if 20 in ks:
        result['recall20'] = float(recall[ks.index(20)])
        result['ndcg20'] = float(ndcg[ks.index(20)])
    return result


def load_pinned_tensors(root, spec, protocol):
    """Future read-only preflight: load only shared/initial/tape, never checkpoint."""
    import torch
    from initialization_kd_interaction import validate_inputs

    assets = spec['common']['assets']
    paths = {name: Path(root) / assets[name]['path'] for name in
             ('shared_cache', 'random_initial', 'triplet_tape')}
    for name, path in paths.items():
        require_hash(path, assets[name]['sha256'])
    cache = torch.load(paths['shared_cache'], map_location='cpu', weights_only=True)
    if set(cache) != {'users', 'items', 'image_items', 'text_items', 'image_users', 'text_users'}:
        raise ValueError('Shared cache keys mismatch')
    random = torch.load(paths['random_initial'], map_location='cpu', weights_only=True)
    teacher = {'user': cache['users'], 'item': cache['items']}
    semantics = {'item_image': cache['image_items'], 'item_text': cache['text_items'],
                 'user_image': cache['image_users'], 'user_text': cache['text_users']}
    tape = np.load(paths['triplet_tape'], mmap_mode='r', allow_pickle=False)
    validate_inputs(random, teacher, semantics, tape, protocol)
    return random, teacher, semantics, tape


def output_bytes(root):
    total = 0
    for base, _, names in os.walk(root):
        for name in names:
            try:
                total += (Path(base) / name).stat().st_size
            except FileNotFoundError:
                # Atomic report/checkpoint rename can remove a listed temp file.
                pass
    return total


def check_budget(started, output_root, limits, rss_bytes, cuda_bytes, free_disk_bytes,
                 now=None):
    """Pure supervisor predicate, suitable for parent-process polling."""
    elapsed = (time.monotonic() if now is None else now) - started
    measured = {'parent_wall_seconds': elapsed, 'process_rss_bytes': rss_bytes,
                'cuda_allocator_bytes': cuda_bytes, 'output_bytes': output_bytes(output_root),
                'free_disk_bytes': free_disk_bytes}
    for key in ('parent_wall_seconds', 'process_rss_bytes', 'cuda_allocator_bytes', 'output_bytes'):
        if key == 'parent_wall_seconds' and limits[key] is None:
            continue  # Explicitly disabled wall cap; keep elapsed telemetry.
        if measured[key] > limits[key]:
            raise RuntimeError('Resource cap exceeded: ' + key)
    if measured['free_disk_bytes'] < limits['minimum_free_disk_bytes']:
        raise RuntimeError('Free disk floor breached')
    return measured
