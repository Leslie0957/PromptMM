"""Isolated train-only P0 measurement; no trainer, parser or evaluator imports."""
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn.functional as F

from td_distill_model_no_projection import TDDistillNoProjectionModel, bpr_loss

COEFFICIENTS = {'image': 0.3 / 1.3, 'text': 0.09 / 1.3}


def pools_from_train(matrix, seed=2022):
    """Round-robin shuffled positive edges per item: update/A/B/C, no overlap."""
    matrix = matrix.tocsr(copy=True)
    matrix.eliminate_zeros()
    matrix.sum_duplicates()
    users, items = matrix.nonzero()
    edges = np.column_stack((users, items)).astype(np.int64)
    degree = np.bincount(users, minlength=matrix.shape[0])
    frequency = np.bincount(items, minlength=matrix.shape[1])
    order = np.argsort(items, kind='stable')
    labels = np.empty(len(edges), dtype=np.int64)
    rng = np.random.default_rng(seed)
    start = 0
    for count in frequency:
        ids = order[start:start + count].copy()
        rng.shuffle(ids)
        labels[ids] = np.arange(count) % 4
        start += count
    pools = {name: edges[labels == k] for k, name in enumerate(('update', 'A', 'B', 'C'))}
    positives = {int(u): set(matrix.indices[matrix.indptr[u]:matrix.indptr[u + 1]].tolist())
                 for u in np.flatnonzero(degree)}
    if any(len(v) == matrix.shape[1] for v in positives.values()):
        raise ValueError('User without any train-unobserved candidate')
    return pools, positives, degree, frequency


def negative(rng, positives, user, n_items):
    for _ in range(100000):
        item = int(rng.integers(n_items))
        if item not in positives[int(user)]:
            return item
    raise RuntimeError('Negative rejection budget exhausted')


def update_batch(edges, positives, n_items, size=1024, seed=2022):
    """Original user-uniform law on the explicitly restricted update pool."""
    by_user = {}
    for u, p in edges:
        by_user.setdefault(int(u), []).append(int(p))
    users = sorted(by_user)
    py = random.Random(seed)
    chosen = py.sample(users, size) if size <= len(users) else py.choices(users, k=size)
    rng = np.random.default_rng(seed)
    return np.array([(u, int(rng.choice(by_user[u])), negative(rng, positives, u, n_items))
                     for u in chosen], dtype=np.int64)


def role_probe(edges, positives, degree, n_items, focus, role, seed, cap=8):
    """Direct conditional sampling under original sampler restricted to this pool.

    q(u,p)=1/(Nactive*degree[u]); negative candidate uniform on complement of
    ALL training positives. No importance weights are needed after direct sampling.
    """
    rng = np.random.default_rng(seed)
    weights = 1.0 / degree[edges[:, 0]] / len(positives)
    pool_mass = float(weights.sum())
    if role == 'positive':
        mask = edges[:, 1] == focus
        role_weights = weights[mask]
    elif role == 'negative':
        mask = np.array([focus not in positives[int(u)] for u in edges[:, 0]])
        role_weights = weights[mask] / (n_items - degree[edges[mask, 0]])
    else:
        raise ValueError(role)
    candidates = edges[mask]
    mass = float(role_weights.sum())
    count = min(cap, len(candidates))
    if not count:
        return {'triples': [], 'mass': 0.0, 'pool_mass': pool_mass,
                'available_edges': 0, 'unique_edges': 0, 'ess': 0}
    ix = rng.choice(len(candidates), count, replace=True, p=role_weights / mass)
    triples = [(int(u), int(p), focus if role == 'negative' else
                negative(rng, positives, u, n_items)) for u, p in candidates[ix]]
    return {'triples': triples, 'mass': mass / pool_mass, 'pool_mass': pool_mass,
            'available_edges': len(candidates), 'unique_edges': len(set((u, p) for u, p, _ in triples)),
            'ess': count, 'sampling': 'conditional iid with replacement; unit weights; dependent duplicates reported'}


def make_plan(matrix, seed=2022):
    pools, positives, degree, frequency = pools_from_train(matrix, seed)
    batch = update_batch(pools['update'], positives, matrix.shape[1], seed=seed)
    # Predeclared three frequency-rank strata; select only exposed items. Report
    # exposure selection bias and absent strata, never substitute a popular item.
    ranks = np.lexsort((np.arange(len(frequency)), frequency))
    strata = np.array_split(ranks[frequency[ranks] > 0], 3)
    exposed = set(batch[:, 1].tolist())
    chosen = []
    for k, stratum in enumerate(strata):
        eligible = [int(i) for i in stratum if int(i) in exposed]
        if eligible:
            chosen.append((k, int(np.random.default_rng(seed + k).choice(eligible))))
    probes = {}
    for k, focus in chosen:
        probes[str(focus)] = {
            name: {role: role_probe(edges, positives, degree, matrix.shape[1], focus, role,
                                   seed + 1000 + focus * 10 + pi * 2 + ri)
                   for ri, role in enumerate(('positive', 'negative'))}
            for pi, (name, edges) in enumerate((n, pools[n]) for n in ('A', 'B', 'C'))}
    coverage = {'shape': list(matrix.shape), 'edges': int(sum(degree)),
                'item_degree_0': int(sum(frequency == 0)),
                'item_degree_1_to_3': int(sum((frequency > 0) & (frequency < 4))),
                'items_four_pools': int(sum(frequency >= 4)),
                'pool_edges': {k: len(v) for k, v in pools.items()},
                'chosen': [{'stratum': k, 'item': i, 'degree': int(frequency[i]),
                            'multiplicity': int(sum(batch[:, 1] == i))} for k, i in chosen],
                'missing_strata': sorted(set(range(3)) - {k for k, _ in chosen})}
    # This detects cross-user rather than merely new-negative coverage.
    for focus, groups in probes.items():
        update_users = set(batch[batch[:, 1] == int(focus), 0].tolist())
        for group in groups.values():
            for probe in group.values():
                probe['users_shared_with_focus_update'] = len(update_users & {t[0] for t in probe['triples']})
    return {'seed': seed, 'batch': batch.tolist(), 'coverage': coverage, 'probes': probes}


def validate_checkpoint(checkpoint):
    state = checkpoint['model_state_dict']
    keys = ['user_id_embedding.weight', 'item_id_embedding.weight']
    if list(state) != keys or any(v.dtype != torch.float32 or v.ndim != 2 for v in state.values()):
        raise ValueError('Expected two float32 ID tables in original parameter order')
    opt = checkpoint['optimizer_state_dict']
    if len(opt['param_groups']) != 1 or opt['param_groups'][0]['params'] != [0, 1] or set(opt['state']) != {0, 1}:
        raise ValueError('Full AdamW state required, not infer-only or fresh optimizer')
    group = opt['param_groups'][0]
    for key, expected in {'lr': 6e-5, 'weight_decay': 0.01, 'eps': 1e-8,
                          'betas': (0.9, 0.999), 'amsgrad': False, 'maximize': False}.items():
        if group[key] != expected:
            raise ValueError('Optimizer mismatch: ' + key)
    for index, tensor in enumerate(state.values()):
        for name in ('exp_avg', 'exp_avg_sq'):
            if opt['state'][index][name].shape != tensor.shape:
                raise ValueError('Moment shape mismatch')
        if opt['state'][index]['step'].numel() != 1:
            raise ValueError('Invalid optimizer step')


def branch(checkpoint, device):
    validate_checkpoint(checkpoint)
    state = checkpoint['model_state_dict']
    nu, dim = state['user_id_embedding.weight'].shape
    ni = state['item_id_embedding.weight'].shape[0]
    model = TDDistillNoProjectionModel(nu, ni, dim, dim).to(device)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    # load_state_dict can alias moments: deep copy is essential for independent arms.
    optimizer.load_state_dict(deepcopy(checkpoint['optimizer_state_dict']))
    return model, optimizer


def per_loss(model, triples):
    users, pos, neg = triples.T
    u = model.user_id_embedding(users)
    p = model.item_id_embedding(pos)
    n = model.item_id_embedding(neg)
    return -F.logsigmoid((u * p).sum(-1) - (u * n).sum(-1))


def item_kd(model, batch, targets, focus):
    pos = batch[:, 1]
    selected = pos == focus
    student = model.item_id_embedding(pos[selected])
    # Keep original full batch x dimension denominator and all repeated positives.
    return ((F.normalize(student, dim=-1) - F.normalize(targets[pos[selected]].detach(), dim=-1)) ** 2).sum() / (len(pos) * student.shape[-1])


def paired_u1(checkpoint, targets, batch, probes, focus, coefficient, device='cpu'):
    """Full dense reference. Both arms start at saved S, exactly one step each."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device != 'cpu' else None
    arms = []
    for weight in (0.0, coefficient):
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        model, optimizer = branch(checkpoint, device)
        triples = torch.as_tensor(batch, dtype=torch.long, device=device)
        loss = per_loss(model, triples).mean() + weight * item_kd(model, triples, targets, focus)
        optimizer.zero_grad()
        loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite objective/gradient')
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite updated parameter')
        if not all(torch.isfinite(v).all() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)):
            raise RuntimeError('Nonfinite optimizer state')
        arms.append(model)
    a, b = arms
    if not torch.equal(a.user_id_embedding.weight, b.user_id_embedding.weight):
        raise RuntimeError('Unexpected user coupling at U1')
    delta = b.item_id_embedding.weight - a.item_id_embedding.weight
    outside = delta.detach().clone()
    outside[focus] = 0
    if torch.count_nonzero(outside):
        raise RuntimeError('Unexpected nonlocal item change')
    result = {'delta_norm': float(delta.detach().norm()), 'multiplicity': int(sum(t[1] == focus for t in batch)), 'pools': {}}
    with torch.no_grad():
        for name, roles in probes.items():
            result['pools'][name] = {}
            total = 0.0
            for role, probe in roles.items():
                if not probe['triples']:
                    result['pools'][name][role] = {'u1': None, 'reason': 'no positive-edge support'}
                    continue
                triple = torch.tensor(probe['triples'], device=device, dtype=torch.long)
                la, lb = per_loss(a, triple), per_loss(b, triple)
                diff = la.double() - lb.double()
                if not torch.isfinite(diff).all():
                    raise RuntimeError('Nonfinite probe')
                u = float(diff.mean())
                total += probe['mass'] * u
                resolution = float((8 * torch.finfo(la.dtype).eps * torch.maximum(la.abs(), lb.abs()).clamp_min(1)).max())
                result['pools'][name][role] = {'u1': u, 'variance': float(diff.var(unbiased=False)),
                    'differences': diff.cpu().tolist(), 'mass': probe['mass'],
                    'resolution_bound': resolution, 'resolved': abs(u) > resolution}
            result['pools'][name]['pool_distribution_contribution'] = total
    if coefficient == 0 and (result['delta_norm'] != 0 or any(
            v.get('u1', 0) not in (None, 0) for roles in result['pools'].values()
            for v in roles.values() if isinstance(v, dict))):
        raise RuntimeError('Zero intervention mismatch')
    return result
