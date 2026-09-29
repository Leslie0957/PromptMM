"""Pure CPU estimands for the fixed modal-neighbor diagnostic (no data I/O)."""
import numpy as np


def binary_matrix(matrix, shape, strict=False):
    coo = matrix.tocoo(copy=True)
    if coo.shape != tuple(shape) or not np.isfinite(coo.data).all():
        raise ValueError('Invalid matrix shape or values')
    if np.any(coo.data <= 0):
        raise ValueError('Explicit nonpositive interaction')
    pairs = coo.row.astype(np.int64) * shape[1] + coo.col
    duplicates = len(pairs) - len(np.unique(pairs))
    if strict and (duplicates or np.any(coo.data != 1)):
        raise ValueError('Duplicate/nonbinary Validation labels')
    csr = coo.tocsr()
    csr.sum_duplicates()
    csr.data[:] = 1
    csr.sort_indices()
    return csr, int(duplicates)


def normalize(vectors):
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Nonfinite/malformed modality')
    norm = np.linalg.norm(x, axis=1)
    valid = norm > 0
    out = np.zeros_like(x)
    out[valid] = x[valid] / norm[valid, None]
    return out, valid


def neighbors(x, queries, candidates, k, block=128):
    candidates = np.sort(candidates)
    ids = np.empty((len(queries), k), dtype=np.int32)
    scores = np.empty((len(queries), k), dtype=np.float64)
    for start in range(0, len(queries), block):
        sims = x[queries[start:start + block]] @ x[candidates].T
        for row, query in enumerate(queries[start:start + block]):
            available = candidates != query
            if available.sum() < k:
                raise ValueError('Insufficient candidates')
            c, s = candidates[available], sims[row, available]
            order = np.lexsort((c, -s))[:k]
            ids[start + row], scores[start + row] = c[order], s[order]
    return ids, scores


def matched_random(queries, real, candidates, degree, repeats, modality, seed):
    """Exact-degree sampling; true neighbors remain in the null candidate pool."""
    k = real.shape[1]
    pools = {int(d): candidates[degree[candidates] == d]
             for d in np.unique(degree[candidates])}
    result = np.empty((repeats, len(queries), k), dtype=np.int32)
    variable = np.zeros(len(queries))
    sizes = np.empty_like(real)
    for qi, query in enumerate(queries):
        target = real[qi]
        if len(np.unique(target)) != k or query in target or not np.isin(target, candidates).all():
            raise ValueError('Invalid real neighbors')
        ds, counts = np.unique(degree[target], return_counts=True)
        offset = 0
        for d, count in zip(ds, counts):
            pool = pools[int(d)]
            pool = pool[pool != query]
            if len(pool) < count:
                raise ValueError('Insufficient exact-degree pool')
            sizes[qi, offset:offset + count] = len(pool)
            variable[qi] += count * (len(pool) > count)
            for rep in range(repeats):
                rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
                    [seed, rep, modality, int(query), int(d)])))
                result[rep, qi, offset:offset + count] = rng.choice(pool, count, replace=False)
            offset += count
        expected = np.sort(degree[target])
        if not np.all(np.sort(degree[result[:, qi]], axis=1) == expected):
            raise ValueError('Degree matching failed')
        if any(len(np.unique(r)) != k or query in r for r in result[:, qi]):
            raise ValueError('Invalid random neighbors')
    return result, variable / k, sizes


def associations(train, users, query_rows, graph):
    """Number of neighbors in each user's binary Train history."""
    counts = np.empty(len(users), dtype=np.int16)
    for n, (u, qi) in enumerate(zip(users, query_rows)):
        history = train.indices[train.indptr[u]:train.indptr[u + 1]]
        counts[n] = np.isin(graph[qi], history, assume_unique=True).sum()
    return counts


def cluster_summary(labels, delta):
    ids, inverse = np.unique(labels, return_inverse=True)
    return ids, np.bincount(inverse, weights=delta), np.bincount(inverse)


def bootstrap(labels, delta, repeats, seed):
    ids, sums, den = cluster_summary(labels, delta)
    if not len(ids):
        return None
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(repeats)
    for r in range(repeats):
        sample = rng.integers(len(ids), size=len(ids))
        values[r] = sums[sample].sum() / den[sample].sum()
    return np.quantile(values, [.025, .975]).tolist()


def measures(real_counts, random_counts, targets, k):
    n = len(real_counts)
    if not n:
        return {'pairs': 0, 'real': None, 'null': None, 'delta': None, 'ratio': None}
    real = real_counts > 0
    null = (random_counts > 0).mean(axis=0)
    delta = real - null
    ids, sums, den = cluster_summary(targets, delta)
    _, real_sums, _ = cluster_summary(targets, real)
    _, null_sums, _ = cluster_summary(targets, null)
    null_reps = (random_counts > 0).mean(axis=1)
    a, b = float(real.mean()), float(null.mean())
    return {'pairs': n, 'targets': len(ids), 'real': a, 'null': b,
            'delta': a - b, 'ratio': a / b if b else None,
            'overlap_fraction_real': float(real_counts.mean() / k),
            'overlap_fraction_null': float(random_counts.mean() / k),
            'target_macro_real': float(np.mean(real_sums / den)),
            'target_macro_null': float(np.mean(null_sums / den)),
            'target_macro_delta': float(np.mean(sums / den)),
            'null_repeats': {'mean': float(null_reps.mean()),
                             'std': float(null_reps.std(ddof=1)),
                             'min': float(null_reps.min()), 'max': float(null_reps.max())}}


def screen(delta, user_ci, item_ci, coverage, matching, threshold=.005):
    if coverage < .95:
        return 'inconclusive_coverage'
    if matching < .8:
        return 'inconclusive_matching_support'
    if delta is None or user_ci is None or item_ci is None:
        return 'inconclusive'
    if delta >= threshold and min(user_ci[0], item_ci[0]) > 0:
        return 'candidate_signal'
    if max(user_ci[1], item_ci[1]) < threshold:
        return 'screen_stop'
    return 'inconclusive'
