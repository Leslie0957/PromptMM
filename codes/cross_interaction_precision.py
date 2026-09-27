"""Read-only CPU probe arithmetic on captured float32 A/B states."""
from decimal import Decimal, localcontext
import hashlib
import math

import torch
import torch.nn.functional as F

# Engineering tolerances, fixed before real precision-review execution.
STABLE_ATOL = 1e-15
STABLE_RTOL = 1e-10
DIRECT_SCALE = 5e-13
REFERENCE_ATOL = 1e-30
REFERENCE_RTOL = 1e-25


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def capture(a, b, probes, focus):
    triples = [t for roles in probes.values() for probe in roles.values() for t in probe['triples']]
    users = sorted({t[0] for t in triples})
    items = sorted({focus} | {i for t in triples for i in t[1:]})
    def take(model, side, ids):
        return getattr(model, side + '_id_embedding').weight[ids].detach().cpu().clone()
    result = {'user_ids': users, 'item_ids': items, 'focus': focus}
    for arm, model in (('a', a), ('b', b)):
        result[arm + '_users'] = take(model, 'user', users)
        result[arm + '_items'] = take(model, 'item', items)
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all()
           for v in result.values() if torch.is_tensor(v)):
        raise ValueError('Captured rows must be finite original float32 values')
    if not torch.equal(result['a_users'], result['b_users']):
        raise ValueError('Unexpected user change')
    nonfocus = [j for j, i in enumerate(items) if i != focus]
    if not torch.equal(result['a_items'][nonfocus], result['b_items'][nonfocus]):
        raise ValueError('Unexpected nonfocus change')
    return result


def decimal_difference(u, pa, na, pb, nb, precision):
    """Independent scalar dot/log/exp path; inputs are exact stored float32s."""
    with localcontext() as ctx:
        ctx.prec = precision
        rows = [[Decimal.from_float(float(x)) for x in row] for row in (u, pa, na, pb, nb)]
        du, dpa, dna, dpb, dnb = rows
        def dot(left, right):
            return sum((x * y for x, y in zip(left, right)), Decimal(0))
        ma = dot(du, dpa) - dot(du, dna)
        mb = dot(du, dpb) - dot(du, dnb)
        def loss(margin):
            # Stable softplus with independent Decimal arithmetic.
            x = -margin
            return max(x, Decimal(0)) + (Decimal(1) + (-abs(x)).exp()).ln()
        return loss(ma) - loss(mb)


def stable_difference(ma, delta):
    """Exact identity near zero; stable softplus fallback for large deltas."""
    if abs(delta) <= 0.5:
        q = math.exp(-ma) / (1 + math.exp(-ma)) if ma >= 0 else 1 / (1 + math.exp(ma))
        return -math.log1p(q * math.expm1(-delta))
    def loss(m):
        return max(-m, 0.0) + math.log1p(math.exp(-abs(m)))
    return loss(ma) - loss(ma + delta)


@torch.no_grad()
def evaluate_rows(rows, probes):
    user_index = {v: i for i, v in enumerate(rows['user_ids'])}
    item_index = {v: i for i, v in enumerate(rows['item_ids'])}
    hashes_before = {k: digest(v) for k, v in rows.items() if torch.is_tensor(v)}
    result = {'tolerances': dict(stable_atol=STABLE_ATOL, stable_rtol=STABLE_RTOL,
                                direct_scale=DIRECT_SCALE, reference_atol=REFERENCE_ATOL,
                                reference_rtol=REFERENCE_RTOL),
              'captured_row_hashes': hashes_before, 'consistent': True, 'pools': {}}
    for name, roles in probes.items():
        pool = {}; contribution = 0.0; contribution_tolerance = 0.0
        for role, probe in roles.items():
            samples = []
            for uid, pid, nid in probe['triples']:
                u = rows['a_users'][user_index[uid]]
                pa, na = (rows['a_items'][item_index[i]] for i in (pid, nid))
                pb, nb = (rows['b_items'][item_index[i]] for i in (pid, nid))
                if any(not torch.isfinite(v).all() for v in (u, pa, na, pb, nb)):
                    raise ValueError('Nonfinite captured row')
                u64, pa64, na64, pb64, nb64 = (v.double() for v in (u, pa, na, pb, nb))
                ma = float((u64 * pa64).sum() - (u64 * na64).sum())
                mb = float((u64 * pb64).sum() - (u64 * nb64).sum())
                delta = float((u64 * (pb64 - pa64)).sum() - (u64 * (nb64 - na64)).sum())
                # Unbatched float32 is supplemental; legacy batched GPU readings
                # remain in the parent result and are not replaced with this path.
                ma32 = (u * pa).sum() - (u * na).sum()
                mb32 = (u * pb).sum() - (u * nb).sum()
                cpu32 = float((-F.logsigmoid(ma32)).double() - (-F.logsigmoid(mb32)).double())
                direct = float(-F.logsigmoid(torch.tensor(ma, dtype=torch.float64)) +
                               F.logsigmoid(torch.tensor(mb, dtype=torch.float64)))
                stable = stable_difference(ma, delta)
                ref80 = decimal_difference(u, pa, na, pb, nb, 80)
                ref120 = decimal_difference(u, pa, na, pb, nb, 120)
                reference = float(ref120)
                reference_error = float(abs(ref80 - ref120))
                tolerance = STABLE_ATOL + STABLE_RTOL * abs(reference)
                direct_tolerance = DIRECT_SCALE * max(1, abs(ma), abs(mb))
                reference_tolerance = REFERENCE_ATOL + REFERENCE_RTOL * abs(reference)
                consistent = (reference_error <= reference_tolerance and
                              abs(stable - reference) <= tolerance and
                              abs(direct - reference) <= direct_tolerance)
                values = (ma, mb, delta, cpu32, direct, stable, reference, reference_error)
                if not all(math.isfinite(x) for x in values):
                    raise ValueError('Nonfinite arithmetic')
                result['consistent'] &= consistent
                samples.append(dict(triple=[uid, pid, nid], margin_a64=ma, margin_b64=mb,
                                    margin_delta64=delta, cpu32=cpu32, direct64=direct,
                                    stable64=stable, reference80=str(ref80), reference120=str(ref120),
                                    reference=reference, reference_precision_error=reference_error,
                                    stable_error=abs(stable-reference), direct_error=abs(direct-reference),
                                    stable_tolerance=tolerance, direct_tolerance=direct_tolerance,
                                    consistent=consistent))
            if not samples:
                pool[role] = {'u1': None, 'reason': 'no positive-edge support', 'samples': []}
                continue
            mean = math.fsum(x['stable64'] for x in samples) / len(samples)
            reference_mean = math.fsum(x['reference'] for x in samples) / len(samples)
            tolerance_mean = math.fsum(x['stable_tolerance'] for x in samples) / len(samples)
            variance = math.fsum((x['stable64'] - mean)**2 for x in samples) / len(samples)
            pool[role] = dict(u1=mean, reference_mean=reference_mean, variance=variance,
                              tolerance=tolerance_mean, resolved=abs(mean) > tolerance_mean,
                              mass=probe['mass'], samples=samples,
                              available_edges=probe.get('available_edges'), unique_edges=probe.get('unique_edges'))
            contribution += probe['mass'] * mean
            contribution_tolerance += probe['mass'] * tolerance_mean
        pool['pool_distribution_contribution'] = contribution
        pool['contribution_tolerance'] = contribution_tolerance
        result['pools'][name] = pool
    result['rows_unchanged'] = hashes_before == {k: digest(v) for k, v in rows.items() if torch.is_tensor(v)}
    if not result['rows_unchanged']:
        raise RuntimeError('Probe reader mutated captured states')
    focus_idx = item_index[rows['focus']]
    diff = rows['b_items'][focus_idx].double() - rows['a_items'][focus_idx].double()
    result['focus_changed_components'] = int(torch.count_nonzero(diff))
    result['focus_delta64_norm'] = float(diff.norm())
    return result
