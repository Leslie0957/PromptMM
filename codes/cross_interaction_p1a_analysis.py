"""Frozen additive single-item U1 screening analysis; no rankings or Test."""

import math
import random


PRACTICAL_GAIN = 1e-10  # Pool-probability-weighted BPR loss, one-step additive screen.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 3102701
FEATURES = ('local_dot', 'local_cos', 'positive_dot', 'positive_cos',
            'both_dot', 'both_cos', 'hard_margin', 'log_frequency',
            'representation_gap', 'constant')


def _gain(rows, selector):
    selected_total = 0.0
    random_total = 0.0
    half_total = 0.0
    for layer in (1, 2):
        layer_rows = [r for r in rows if r['layer'] == layer]
        if len(layer_rows) != 10 or any(r.get(selector) is None for r in layer_rows):
            return None
        k = 5
        chosen = sorted(layer_rows, key=lambda r: (-r[selector], r['item']))[:k]
        selected_total += math.fsum(r['c_full'] for r in chosen)
        random_total += k / len(layer_rows) * math.fsum(r['c_full'] for r in layer_rows)
        half_total += math.fsum(r['c_half'] for r in layer_rows)
    return {'selected': selected_total, 'random_expectation': random_total,
            'uniform_half': half_total,
            'gain_vs_random': selected_total - random_total,
            'gain_vs_uniform_half': selected_total - half_total,
            'gain_vs_best_control': selected_total - max(random_total, half_total)}


def _interval(rows, selector, seed):
    value = _gain(rows, selector)
    if value is None:
        return None
    rng = random.Random(seed)
    strata = {layer: [r for r in rows if r['layer'] == layer] for layer in (1, 2)}
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [rng.choice(strata[layer]) for layer in (1, 2)
                for _ in range(len(strata[layer]))]
        samples.append(_gain(draw, selector)['gain_vs_best_control'])
    samples.sort()
    lo = samples[int(.025 * BOOTSTRAP_REPLICATES)]
    hi = samples[int(.975 * BOOTSTRAP_REPLICATES) - 1]
    value['item_cluster_bootstrap_95pct'] = [lo, hi]
    value['decision'] = ('screen_pass' if lo > PRACTICAL_GAIN else
                         'screen_stop' if hi < PRACTICAL_GAIN else 'inconclusive')
    return value


def _contrast(rows, first, second, seed):
    a, b = _gain(rows, first), _gain(rows, second)
    if a is None or b is None:
        return None
    strata = {layer: [r for r in rows if r['layer'] == layer] for layer in (1, 2)}
    rng = random.Random(seed)
    values = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draw = [rng.choice(strata[layer]) for layer in (1, 2)
                for _ in range(len(strata[layer]))]
        values.append(_gain(draw, first)['selected'] - _gain(draw, second)['selected'])
    values.sort()
    return {'difference': a['selected'] - b['selected'],
            'item_cluster_bootstrap_95pct': [values[int(.025 * BOOTSTRAP_REPLICATES)],
                                              values[int(.975 * BOOTSTRAP_REPLICATES) - 1]],
            'interpretation': 'matched A triple count; duplicated positive edges remain dependent'}


def analyze(report):
    if report.get('status') not in ('running_analysis', 'completed') or len(report['pairs']) != 372:
        raise ValueError('Complete P1a pair set required before C analysis')
    index = {}
    for pair in report['pairs']:
        key = (pair['state'], pair['context'], pair['item'], pair['modality'], pair['arm'])
        if key in index:
            raise ValueError('Duplicate pair identity')
        index[key] = pair
    output = {'method': 'additive_single_item_U1_surrogate_not_joint_policy',
              'primary_estimand': 'pool_probability_weighted_BPR_loss_difference',
              'positive_and_negative_roles_reported_separately': True,
              'practical_gain': PRACTICAL_GAIN,
              'bootstrap_replicates': BOOTSTRAP_REPLICATES,
              'bootstrap_seed': BOOTSTRAP_SEED, 'groups': []}
    for state_i, state in enumerate(('cold2022', 'warm2022')):
        for modality_i, modality in enumerate(('image', 'text')):
            rows = []
            for item in sorted({p['item'] for p in report['pairs'] if p['arm'] == 'full'}):
                fulls = [index[(state, ci, item, modality, 'full')] for ci in (0, 1)]
                halfs = [index[(state, ci, item, modality, 'half')] for ci in (0, 1)]
                def contribution(pair, pool):
                    return pair['pools'][pool]['contribution']
                c_full = [contribution(p, 'C') for p in fulls]
                c_half = [contribution(p, 'C') for p in halfs]
                if (any(p['pools']['C']['positive']['u1'] is None for p in fulls + halfs)
                        or any(v is None for v in c_full + c_half)):
                    rows.append({'item': item, 'layer': fulls[0]['layer'], 'reason': 'C positive N/A'})
                    continue
                r = {'item': item, 'layer': fulls[0]['layer'],
                     'c_full': math.fsum(c_full) / 2, 'c_half': math.fsum(c_half) / 2,
                     'b_oracle': math.fsum(contribution(p, 'B') for p in fulls) / 2,
                     'b_half': math.fsum(contribution(p, 'B') for p in halfs) / 2,
                     'a_effect': math.fsum(contribution(p, 'A') for p in fulls) / 2}
                for feature in FEATURES:
                    values = [p['a_features'][feature] for p in fulls]
                    r[feature] = math.fsum(values) / 2 if all(v is not None for v in values) else None
                if not all(math.isfinite(v) for v in r.values() if isinstance(v, float)):
                    raise ValueError('Nonfinite analysis input')
                rows.append(r)
            gated = [r for r in rows if r['layer'] in (1, 2)]
            selectors = ('b_oracle', 'a_effect') + FEATURES
            results = {name: _interval(gated, name,
                                       BOOTSTRAP_SEED + state_i * 100 + modality_i * 10 + ix)
                       for ix, name in enumerate(selectors)}
            same_pool = _gain([{**r, 'c_full': r['b_oracle'], 'c_half': r['b_half']}
                               for r in gated], 'b_oracle')
            context_contrast = _contrast(gated, 'both_dot', 'positive_dot',
                                         BOOTSTRAP_SEED + 1000 + state_i * 10 + modality_i)
            random_seed = BOOTSTRAP_SEED + 2000 + state_i * 10 + modality_i
            random_rng = random.Random(random_seed)
            random_items = [r for layer in (1, 2) for r in random_rng.sample(
                [x for x in gated if x['layer'] == layer], 5)]
            output['groups'].append({'state': state, 'modality': modality,
                                     'low_layer': [r for r in rows if r['layer'] == 0],
                                     'mid_high_items': len(gated), 'selectors': results,
                                     'same_B_pool_optimism_not_gate': same_pool,
                                     'matched_A_context_vs_positive_contrast': context_contrast,
                                     'seeded_random_same_count': {
                                         'seed': random_seed, 'items': [r['item'] for r in random_items],
                                         'c_full_additive_sum': math.fsum(r['c_full'] for r in random_items)},
                                     'g1_screen': results['b_oracle']['decision'],
                                     'g2_screen': results['positive_dot']['decision']
                                     if results['positive_dot'] is not None else 'not_estimable'})
    return output
