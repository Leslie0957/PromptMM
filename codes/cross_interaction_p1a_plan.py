"""Sealed train-only P1a U1 probes; no model or held-out imports."""

import hashlib
import json

import numpy as np

from cross_interaction_p1a_coverage import SEED, coverage, _negative


def _edges_and_positives(matrix, seed):
    csr = matrix.tocsr(copy=True)
    csr.sum_duplicates(); csr.eliminate_zeros()
    users, items = csr.nonzero()
    edges = np.column_stack((users, items)).astype(np.int64)
    degree = np.bincount(users, minlength=csr.shape[0])
    frequency = np.bincount(items, minlength=csr.shape[1])
    order = np.argsort(items, kind='stable')
    labels = np.empty(len(edges), dtype=np.int8)
    rng = np.random.default_rng(seed)
    start = 0
    for count in frequency:
        ids = order[start:start + count].copy()
        rng.shuffle(ids)
        labels[ids] = np.arange(count) % 4
        start += count
    pools = {name: edges[labels == k] for k, name in enumerate(('update', 'A', 'B', 'C'))}
    positives = {int(u): set(csr.indices[csr.indptr[u]:csr.indptr[u + 1]].tolist())
                 for u in np.flatnonzero(degree)}
    return pools, positives, degree, frequency


def _role_probe(edges, positives, degree, n_items, focus, role, seed):
    rng = np.random.default_rng(seed)
    weights = 1.0 / degree[edges[:, 0]] / len(positives)
    pool_mass = float(weights.sum())
    if role == 'positive':
        mask = edges[:, 1] == focus
        role_weights = weights[mask]
    elif role == 'negative':
        mask = np.fromiter((focus not in positives[int(u)] for u in edges[:, 0]),
                           dtype=bool, count=len(edges))
        role_weights = weights[mask] / (n_items - degree[edges[mask, 0]])
    else:
        raise ValueError(role)
    candidates = edges[mask]
    mass = float(role_weights.sum())
    base_count = min(8, len(candidates))
    if not base_count:
        return {'triples': [], 'mass': 0.0, 'pool_mass': pool_mass,
                'available_edges': 0, 'unique_edges': 0, 'base_draws': 0,
                'negative_repeats': 0}
    # IID conditional draws with replacement: sample mean targets this role law.
    indices = rng.choice(len(candidates), base_count, replace=True, p=role_weights / mass)
    triples = []
    for ix in indices:
        u, p = map(int, candidates[ix])
        if role == 'positive':
            # Three independent protocol negatives for the same base edge.
            triples.extend([[u, p, _negative(rng, positives, u, n_items)] for _ in range(3)])
        else:
            # n=focus is fixed. More "negative repeats" would copy the same
            # triple and manufacture sample size, so this role has one draw.
            triples.append([u, p, focus])
    return {'triples': triples, 'mass': mass / pool_mass, 'pool_mass': pool_mass,
            'available_edges': len(candidates),
            'unique_edges': len({tuple(candidates[i]) for i in indices}),
            'base_draws': base_count, 'negative_repeats': 3 if role == 'positive' else 1}


def make_plan(matrix, old_plan, seed=SEED):
    """Build deterministic plan, excluding P0 actually used positive edges from C."""
    cov = coverage(matrix, seed=seed)
    pools, positives, degree, frequency = _edges_and_positives(matrix, seed)
    exposed = {tuple(t[:2]) for t in old_plan['batch']}
    for groups in old_plan['probes'].values():
        for roles in groups.values():
            for probe in roles.values():
                exposed.update(tuple(t[:2]) for t in probe['triples'])
    # C uses fresh positive edges relative to the earlier *actual* P0 batch
    # and probes, not merely a new negative for a previously used edge.
    c_edges = pools['C']
    keep = np.fromiter((tuple(edge) not in exposed for edge in c_edges),
                       dtype=bool, count=len(c_edges))
    pools['C'] = c_edges[keep]
    items = []
    for row in cov['items']:
        item = row['item']
        contexts = []
        for ci in range(2):
            probes = {}
            for pi, name in enumerate(('A', 'B', 'C')):
                probes[name] = {
                    role: _role_probe(pools[name], positives, degree, matrix.shape[1], item,
                                      role, seed + 10000 + item * 100 + ci * 10 + pi * 2 + ri)
                    for ri, role in enumerate(('positive', 'negative'))
                }
            contexts.append(probes)
        items.append({'item': item, 'layer': row['layer'], 'degree': row['degree'],
                      'multiplicity': row['context_multiplicity'], 'probes': contexts})
    plan = {'protocol': 'sports_train_only_p1a_single_item_u1_v1', 'seed': seed,
            'cache_target': 'SPORTS_P0_WARM2022_EXISTING_STATE_V1',
            'raw_pool_edges': cov['pool_edges'], 'probe_C_edges': len(pools['C']),
            'old_exposed_edges_excluded_from_C':
            int(len(c_edges) - len(pools['C'])), 'strata': cov['strata'],
            'contexts': cov['contexts'], 'items': items,
            'states': ['cold2022', 'warm2022'], 'modalities': ['image', 'text'],
            'coefficients': {'image': 0.3 / 1.3, 'text': 0.09 / 1.3},
            'arms': ['full', 'half'], 'zero_pairs': 4,
            'max_nonzero_pairs': len(items) * 2 * 2 * 2 * 2}
    # The plan itself excludes identity hashes to make canonical regeneration
    # depend only on explicit train data and source-defined sampling.
    validate_plan(plan, positives, matrix.shape)
    return plan


def validate_plan(plan, positives, shape):
    n_users, n_items = shape
    if len(plan['contexts']) != 2 or len(plan['items']) > 30:
        raise ValueError('Planning cap exceeded')
    for batch in plan['contexts']:
        if len(batch) != 1024 and n_users == 35598:
            raise ValueError('Wrong BPR batch length')
        for u, p, n in batch:
            if not (0 <= u < n_users and 0 <= p < n_items and 0 <= n < n_items):
                raise ValueError('Out-of-range batch index')
            if p not in positives[u] or n in positives[u]:
                raise ValueError('Invalid BPR triple')
    for row in plan['items']:
        focus = row['item']
        if focus in (1494, 1744, 1646) or len(row['probes']) != 2:
            raise ValueError('P0 focus reused or context missing')
        for ci, probes in enumerate(row['probes']):
            if sum(t[1] == focus for t in plan['contexts'][ci]) != row['multiplicity'][ci]:
                raise ValueError('Focus exposure mismatch')
            for pool in probes.values():
                for role, probe in pool.items():
                    if probe['base_draws'] > 8 or len(probe['triples']) > (24 if role == 'positive' else 8):
                        raise ValueError('Probe cap exceeded')
                    for u, p, n in probe['triples']:
                        if p not in positives[u] or n in positives[u]:
                            raise ValueError('Invalid probe triple')
                        if (p == focus) != (role == 'positive') or (n == focus) != (role == 'negative'):
                            raise ValueError('Wrong focus role')


def canonical_bytes(plan):
    return json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(plan):
    return hashlib.sha256(canonical_bytes(plan)).hexdigest()
