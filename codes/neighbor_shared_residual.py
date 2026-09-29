"""Fixed neighbor features and foldable BPR residual for the second study."""
import numpy as np
import torch
import torch.nn.functional as F


def features(anchor_items, queries, image_neighbors, text_neighbors):
    """Return X, per-modality RMS, with zero rows outside the valid low third."""
    q = np.asarray(anchor_items, dtype=np.float32)
    ids = np.asarray(queries, dtype=np.int64)
    if q.ndim != 2 or q.shape[1] != 64 or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid fixed table/queries')
    blocks, scales = [], []
    for graph in (image_neighbors, text_neighbors):
        graph = np.asarray(graph)
        if graph.shape != (len(ids), 20) or graph.min() < 0 or graph.max() >= len(q):
            raise ValueError('Invalid neighbor graph')
        v = q[graph].mean(axis=1) - q[ids]
        scale = float(np.sqrt(np.mean(np.sum(v.astype(np.float64) ** 2, axis=1))))
        if not np.isfinite(scale) or scale < 1e-12:
            raise ValueError('Uninformative or nonfinite neighborhood')
        blocks.append(v / scale / np.sqrt(2.0))
        scales.append(scale)
    x = np.zeros((len(q), 128), dtype=np.float32)
    x[ids] = np.concatenate(blocks, axis=1)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite feature')
    return x, scales


def effective_items(items, x, arm, extra):
    if arm == 'B':
        return items
    if arm == 'F':
        return items + extra[0] * x[:, :64] + extra[1] * x[:, 64:]
    if arm in ('N', 'R'):
        return items + F.linear(x, extra)
    raise ValueError('Unknown arm')


def triplet_loss(users, items, x, arm, extra, ids):
    u, i, j = ids
    p = users[u]
    if arm == 'B':
        pos, neg = items[i], items[j]
    elif arm == 'F':
        pos = items[i] + extra[0] * x[i, :64] + extra[1] * x[i, 64:]
        neg = items[j] + extra[0] * x[j, :64] + extra[1] * x[j, 64:]
    else:
        pos = items[i] + F.linear(x[i], extra)
        neg = items[j] + F.linear(x[j], extra)
    return -F.logsigmoid((p * pos).sum(dim=-1) - (p * neg).sum(dim=-1)).mean()


def group_metrics(top, users, val, groups):
    """Original user-normalized Recall contribution and positive-pair micro rate."""
    den = np.zeros(3, dtype=np.int64)
    hits = np.zeros(3, dtype=np.int64)
    contribution = np.zeros(3, dtype=np.float64)
    exposure = np.bincount(groups[top.ravel()], minlength=3)
    for row, uid in enumerate(users):
        positive = set(val.indices[val.indptr[uid]:val.indptr[uid + 1]].tolist())
        if not positive:
            raise ValueError('Validation user without positive')
        for item in positive:
            den[groups[item]] += 1
        for item in top[row]:
            if item in positive:
                g = groups[item]
                hits[g] += 1
                contribution[g] += 1 / len(positive) / len(users)
    if np.any(den == 0):
        raise ValueError('Empty frequency group')
    return {'denominator': den.tolist(), 'hits': hits.tolist(),
            'micro': (hits / den).tolist(), 'contribution': contribution.tolist(),
            'exposure': exposure.tolist(), 'recall20': float(contribution.sum())}
