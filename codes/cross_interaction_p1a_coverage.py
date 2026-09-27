"""Train-only P1a coverage planning. No model, teacher, or held-out imports."""

import random

import numpy as np


SEED = 31027
EXPOSED_P0_ITEMS = (1494, 1744, 1646)


def _negative(rng, positives, user, n_items):
    for _ in range(100000):
        item = int(rng.integers(n_items))
        if item not in positives[user]:
            return item
    raise RuntimeError('Negative rejection budget exhausted')


def coverage(matrix, seed=SEED, batch_size=1024, per_stratum=10, exclude=EXPOSED_P0_ITEMS):
    """Return deterministic metadata, not a launchable probe or utility plan.

    Selection is conditional on focus exposure in BOTH independent update
    contexts. All positive edges are partitioned before selection and every
    omitted stratum/item remains visible in the returned coverage counts.
    """
    if matrix.ndim != 2 or batch_size < 1 or per_stratum < 1:
        raise ValueError('Invalid matrix or planning limits')
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates()
    csr.eliminate_zeros()
    users, items = csr.nonzero()
    edges = np.column_stack((users, items)).astype(np.int64)
    n_users, n_items = csr.shape
    degree = np.bincount(users, minlength=n_users)
    frequency = np.bincount(items, minlength=n_items)
    positives = {int(u): set(csr.indices[csr.indptr[u]:csr.indptr[u + 1]].tolist())
                 for u in np.flatnonzero(degree)}
    if any(len(p) >= n_items for p in positives.values()):
        raise ValueError('Active user without a train-unobserved candidate')

    # Per-item shuffle and round-robin mirrors P0's separation, with a NEW seed.
    order = np.argsort(items, kind='stable')
    labels = np.empty(len(edges), dtype=np.int8)
    rng = np.random.default_rng(seed)
    start = 0
    for count in frequency:
        ids = order[start:start + count].copy()
        rng.shuffle(ids)
        labels[ids] = np.arange(count) % 4
        start += count
    pools = {name: edges[labels == ix] for ix, name in enumerate(('update', 'A', 'B', 'C'))}
    by_user = {}
    for u, p in pools['update']:
        by_user.setdefault(int(u), []).append(int(p))
    active_users = sorted(by_user)
    if not active_users:
        raise ValueError('Empty update pool')

    contexts = []
    for ctx in range(2):
        context_seed = seed + 100 + ctx
        py = random.Random(context_seed)
        chosen_users = (py.sample(active_users, batch_size) if batch_size <= len(active_users)
                        else py.choices(active_users, k=batch_size))
        nrng = np.random.default_rng(context_seed)
        batch = [[u, int(nrng.choice(by_user[u])), _negative(nrng, positives, u, n_items)]
                 for u in chosen_users]
        contexts.append(np.asarray(batch, dtype=np.int64))

    ranked = np.lexsort((np.arange(n_items), frequency))
    strata = np.array_split(ranked[frequency[ranked] > 0], 3)
    exposure = set(contexts[0][:, 1]) & set(contexts[1][:, 1])
    excluded = set(exclude)
    selected = []
    strata_info = []
    for layer, stratum in enumerate(strata):
        eligible = [int(i) for i in stratum if int(i) in exposure and int(i) not in excluded]
        pick = np.random.default_rng(seed + 200 + layer).choice(
            eligible, min(per_stratum, len(eligible)), replace=False).tolist() if eligible else []
        pick = sorted(int(i) for i in pick)
        selected.extend((layer, i) for i in pick)
        strata_info.append({'layer': layer, 'nonzero_items': len(stratum),
                            'exposed_both_unexcluded': len(eligible),
                            'selected': pick, 'shortfall': per_stratum - len(pick)})

    items_info = []
    for layer, item in selected:
        per_pool = {}
        for name in ('A', 'B', 'C'):
            pool = pools[name]
            mask = pool[:, 1] == item
            positive = pool[mask]
            # A negative-focus triple has n=item and requires (u,p) with
            # item absent from ALL of that user's training positives.
            valid_negative = sum(item not in positives[int(u)] for u in pool[:, 0])
            per_pool[name] = {'positive_edges': len(positive),
                              'positive_pairs': positive.tolist(),
                              'positive_users': len(set(positive[:, 0].tolist())),
                              'negative_context_edges': int(valid_negative)}
        items_info.append({'layer': layer, 'item': item, 'degree': int(frequency[item]),
                           'context_multiplicity': [int(sum(b[:, 1] == item)) for b in contexts],
                           'pools': per_pool})
    return {'kind': 'train_only_feasibility_not_launch_plan', 'seed': seed,
            'batch_size': batch_size, 'per_stratum': per_stratum,
            'shape': [n_users, n_items], 'edges': len(edges),
            'degree_0': int(sum(frequency == 0)),
            'degree_1_to_3': int(sum((frequency > 0) & (frequency < 4))),
            'pool_edges': {name: len(pool) for name, pool in pools.items()},
            'excluded_focus_items': sorted(int(i) for i in excluded if i < n_items),
            'exposure_intersection': len(exposure), 'strata': strata_info,
            'items': items_info,
            'contexts': [b.tolist() for b in contexts]}
