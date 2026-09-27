"""Compare two exact evaluators on one supplied table state; no file loaders."""
import hashlib
import time
import numpy as np

from initialization_kd_adapter import evaluate_validation
from initialization_kd_fast_eval import exact_topk, reference_topk


def compare_evaluators(model, train, val, synchronize=lambda: None, check=lambda: None,
                       capture=None):
    import torch
    saved_scores, saved_ranks, fast_ranks = [], [], []
    rank_digest = hashlib.sha256()
    position = [0]

    def reference(scores, seen, k):
        check()
        result = reference_topk(scores, seen, k)
        saved_scores.append(hashlib.sha256(scores.tobytes()).digest())
        saved_ranks.append(np.asarray(result, dtype='<i4'))
        return result

    def fast(scores, seen, k):
        check()
        i = position[0]
        result = exact_topk(scores, seen, k)
        if (i >= len(saved_ranks) or hashlib.sha256(scores.tobytes()).digest() != saved_scores[i]
                or not np.array_equal(result, saved_ranks[i])):
            raise RuntimeError('Exact score/ranking parity failed at Val user index ' + str(i))
        rank_digest.update(np.asarray(result, dtype='<i4').tobytes())
        fast_ranks.append(np.asarray(result, dtype='<i4'))
        position[0] += 1
        return result

    with torch.no_grad():
        synchronize()
        start = time.perf_counter()
        old = evaluate_validation(model, train, val, ranker=reference)
        synchronize()
        old_seconds = time.perf_counter() - start
        start = time.perf_counter()
        new = evaluate_validation(model, train, val, ranker=fast)
        synchronize()
        fast_seconds = time.perf_counter() - start
    if old != new or position[0] != len(saved_ranks) or position[0] != old['validation_users']:
        raise RuntimeError('Exact metric/user-count parity failed')
    if capture is not None:
        capture(np.stack(saved_ranks), np.stack(fast_ranks))
    return {'status': 'passed', 'users_checked': position[0], 'k': 50,
            'score_rows_exact': True, 'ordered_top50_exact': True, 'metrics_exact': True,
            'rank_sha256': rank_digest.hexdigest(), 'reference_metrics': old,
            'fast_metrics': new, 'reference_seconds': old_seconds,
            'fast_seconds': fast_seconds, 'speedup_observed': old_seconds / fast_seconds,
            'timing_order': ['reference', 'fast'],
            'timing_includes': 'GEMM, transfers, ranking, metrics, parity hashing/checks'}
