"""Only pre-update A-pool features; never reads B/C outcomes."""

import math

import torch
import torch.nn.functional as F


def features(checkpoint, targets, focus, a_probes, multiplicity, frequency):
    state = checkpoint['model_state_dict']
    users = state['user_id_embedding.weight']
    items = state['item_id_embedding.weight']
    s = items[focus].detach().clone().requires_grad_(True)
    t = targets[focus].detach()
    kd = (F.normalize(s, dim=0) - F.normalize(t, dim=0)).square().mean()
    kd_grad = torch.autograd.grad(kd, s)[0].double() * (multiplicity / 1024)
    entries = {}
    for role in ('positive', 'negative'):
        values = []
        margins = []
        for uid, pid, nid in a_probes[role]['triples']:
            u = users[uid].double()
            margin = torch.dot(u, items[pid].double() - items[nid].double())
            factor = torch.sigmoid(-margin)
            probe_grad = (-factor if role == 'positive' else factor) * u
            dot = float(torch.dot(kd_grad, probe_grad))
            denom = float(kd_grad.norm() * probe_grad.norm())
            values.append((dot, dot / denom if denom > 0 else 0.0))
            margins.append(float(margin))
        entries[role] = {'dot': [v[0] for v in values], 'cos': [v[1] for v in values],
                         'margins': margins}
    def mean(v):
        return math.fsum(v) / len(v) if v else None
    pos, neg = entries['positive'], entries['negative']
    pm, nm = a_probes['positive']['mass'], a_probes['negative']['mass']
    # Match the actual triple budget for positive-only versus positive+negative:
    # 2k positive triples versus k positive + k negative triples. With sparse
    # support this can mean repeated positive edges with distinct negatives;
    # that dependence is retained in the report, not counted as new edges.
    k = min(len(pos['dot']) // 2, len(neg['dot']))
    result = {
        'local_dot': pos['dot'][0] if pos['dot'] else None,
        'local_cos': pos['cos'][0] if pos['cos'] else None,
        'positive_dot': mean(pos['dot'][:2*k]) if k else None,
        'positive_cos': mean(pos['cos'][:2*k]) if k else None,
        'both_dot': pm * mean(pos['dot'][:k]) + nm * mean(neg['dot'][:k]) if k else None,
        'both_cos': pm * mean(pos['cos'][:k]) + nm * mean(neg['cos'][:k]) if k else None,
        'hard_margin': -mean(pos['margins'][:2*k]) if k else None,
        'log_frequency': math.log1p(frequency),
        'representation_gap': float(1 - F.cosine_similarity(s.detach()[None], t[None])[0]),
        'constant': 0.0,
        'a_positive_triples': len(pos['dot']), 'a_negative_triples': len(neg['dot']),
        'matched_a_triples_per_rule': 2*k,
        'a_role_masses': {'positive': pm, 'negative': nm},
    }
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, float)):
        raise ValueError('Nonfinite A feature')
    return result
