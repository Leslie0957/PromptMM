"""Exact CPU top-K selection; score GEMM and metric arithmetic are unchanged."""
import heapq
import numpy as np


def reference_topk(scores, seen, k):
    candidates = (index for index in range(len(scores)) if index not in seen)
    result = heapq.nlargest(k, candidates, key=scores.__getitem__)
    if len(result) != k:
        raise ValueError('Insufficient Validation candidates')
    return result


def exact_topk(scores, seen, k):
    """Descending finite score, ascending item ID on exact ties (heapq order).

    Partition only determines the threshold. Include ALL scores above it, then
    fill from ascending IDs equal to it. Never trust argpartition's tie order,
    add score epsilons, downcast scores, or sort a truncated arbitrary tie set.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.dtype not in (np.float32, np.float64):
        raise ValueError('Expected one float32/64 score row')
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite scores')
    if k < 1 or len(scores) - len(seen) < k:
        raise ValueError('Insufficient Validation candidates')
    if any(index < 0 or index >= len(scores) for index in seen):
        raise ValueError('Invalid Train item index')
    masked = scores.copy()
    if seen:
        masked[np.fromiter(seen, dtype=np.int64)] = -np.inf
    threshold = np.partition(masked, len(masked) - k)[len(masked) - k]
    above = np.flatnonzero(masked > threshold)
    tied = np.flatnonzero(masked == threshold)[:k - len(above)]
    chosen = np.concatenate((above, tied))
    # IDs resolve ties, including +0/-0, without modifying the scores.
    return chosen[np.lexsort((chosen, -scores[chosen]))].tolist()


def evaluate_validation_fast(model, train, val, ks=(10, 20, 40, 50), user_batch=256):
    from initialization_kd_adapter import evaluate_validation
    return evaluate_validation(model, train, val, ks, user_batch, ranker=exact_topk)
