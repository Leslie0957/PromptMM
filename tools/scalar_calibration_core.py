"""Convex frozen-score BPR scalar calibration; no I/O or dataset access."""
import numpy as np


def sample_pairs(train, count, seed, check=lambda: None):
    if count < 1:
        raise ValueError('Positive sample count required')
    rng = np.random.default_rng(seed)
    n, ni = train.shape
    pos = np.empty((n, count), dtype=np.int32)
    neg = np.empty_like(pos)
    for u in range(n):
        seen = train.indices[train.indptr[u]:train.indptr[u+1]]
        if not len(seen) or len(seen) >= ni:
            raise ValueError('Need observed and unobserved items for each user')
        pos[u] = rng.choice(seen, count, replace=True)
        # Bounded exact uniform complement sampling; no rejection loop.
        available = np.setdiff1d(np.arange(ni), seen, assume_unique=True)
        neg[u] = rng.choice(available, count, replace=True)
        if u % 256 == 0:
            check()
    return pos, neg


def objective(margin, delta, a, ridge):
    a = np.broadcast_to(np.asarray(a, dtype=np.float64), (len(margin),))
    z = margin + a[:, None] * delta
    return float(np.logaddexp(0., -z).mean() + ridge * np.mean(a*a))


def derivative(margin, delta, a, ridge):
    a = np.broadcast_to(np.asarray(a, dtype=np.float64), (len(margin),))
    z = margin + a[:, None] * delta
    prob = np.exp(-np.logaddexp(0., z))
    return (-delta * prob).mean(axis=1) + 2 * ridge * a


def solve(margin, delta, ridge=.01, iterations=40, per_user=True, check=lambda: None):
    margin = np.asarray(margin, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if (margin.ndim != 2 or margin.shape != delta.shape or margin.size == 0
            or not np.isfinite(margin).all() or not np.isin(delta, [-1,0,1]).all()
            or not np.isfinite(ridge) or ridge <= 0 or iterations < 1):
        raise ValueError('Invalid convex problem')
    bound = 1 / (2 * ridge)
    size = len(margin) if per_user else 1
    lo, hi = np.full(size, -bound), np.full(size, bound)
    trace = []
    for iteration in range(iterations):
        mid = (lo + hi) / 2
        grad = derivative(margin, delta, mid, ridge)
        if not per_user:
            grad = np.array([grad.mean()])
        lo = np.where(grad < 0, mid, lo)
        hi = np.where(grad > 0, mid, hi)
        lo = np.where(grad == 0, mid, lo)
        hi = np.where(grad == 0, mid, hi)
        trace.append({'iteration': iteration+1, 'max_bracket_width': float((hi-lo).max())})
        check()
    a = (lo+hi)/2
    grad = derivative(margin, delta, a, ridge)
    error = float(np.abs(grad if per_user else grad.mean()).max())
    return a, {'objective': objective(margin, delta, a, ridge),
               'max_stationarity_error': error, 'trace': trace}


def merge_signed(ids, scores, offset, k=20):
    ids, scores = np.asarray(ids), np.asarray(scores)
    n = len(ids)
    a = np.broadcast_to(np.asarray(offset, dtype=np.float64), (n,))
    if ids.shape != scores.shape or ids.shape != (n,2,k) or not np.isfinite(scores).all() or not np.isfinite(a).all():
        raise ValueError('Invalid cache/offset')
    pos = np.zeros((n,2), dtype=int); rows = np.arange(n)
    out = np.empty((n,k), dtype=np.int32)
    for j in range(k):
        x, y = ids[rows,0,pos[:,0]], ids[rows,1,pos[:,1]]
        if np.any((x<0)&(y<0)):
            raise ValueError('Insufficient candidates')
        diff = scores[rows,1,pos[:,1]].astype(np.float64)-scores[rows,0,pos[:,0]].astype(np.float64)
        take = (x>=0)&((y<0)|(a>diff)|((a==diff)&(x<y)))
        out[:,j] = np.where(take,x,y)
        pos[:,0] += take; pos[:,1] += ~take
    return out


def decision(rows):
    # New strict descriptive screen against BOTH B and G, no reused H tolerance.
    if all(r['low_U_minus_B'] > 0 and r['low_U_minus_G'] > 0
           and r['recall_U_minus_B'] >= 0 and r['recall_U_minus_G'] >= 0 for r in rows):
        return 'candidate_signal'
    return 'screen_stop'
