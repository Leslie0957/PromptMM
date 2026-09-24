"""Compare pinned Sports item-target gradients at matched aggregate scale, without updates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/process0/initial_tensors.pt'
BATCHES = ROOT / 'exp/gradient_checks/sports_sharedinit_seed2022_v2/batches.json'
PRIOR = ROOT / 'exp/gradient_checks/sports_sharedinit_seed2022_v2/report.json'
LOSS_SOURCE = ROOT / 'codes/td_distill_model_no_projection.py'
OUT = ROOT / 'exp/gradient_checks/sports_matched_target_direction_seed2022_v1'
EXPECTED = {
    ASSET: 'e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20',
    BATCHES: 'd0489b97de4733f14b9dbd10bbea82dffdd1cbbc4420181a91681cf0ccfa75f0',
    PRIOR: '5219ef3e3ccb7399bb484f42dd7edb7bc0dfe0281f7e3a555accbb68c48b74ff',
    LOSS_SOURCE: '13fd9e55c2ab6bbe7c87aa478935e902feee12b1b3aee953b7f66698e0fdaf1b',
}
ALPHA_A = 0.3


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cosine(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    product = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / product) if product else None


def calibrated_alpha(norm_a, norm_b):
    a2 = sum(float(x) ** 2 for x in norm_a)
    b2 = sum(float(x) ** 2 for x in norm_b)
    if a2 <= 0 or b2 <= 0:
        raise ValueError('Cannot match a zero aggregate gradient.')
    return ALPHA_A * math.sqrt(a2 / b2)


def self_test():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 6.0]
    alpha_b = calibrated_alpha(a, b)
    assert math.isclose(alpha_b, ALPHA_A / 2, rel_tol=1e-14)
    assert math.isclose(sum((ALPHA_A * x) ** 2 for x in a),
                        sum((alpha_b * x) ** 2 for x in b), rel_tol=1e-14)
    assert math.isclose(cosine(torch.tensor([1., 2.]), torch.tensor([2., 4.])), 1.0)
    assert cosine(torch.zeros(2), torch.ones(2)) is None
    print('synthetic fixed-alpha RMS and cosine checks passed')


def item_batch_gradients(initial, image, text, batch, bpr_loss, directional_loss):
    users = initial['users'].clone().requires_grad_(True)
    items = initial['items'].clone().requires_grad_(True)
    user_ids, positive_ids, negative_ids = (
        torch.tensor(side, dtype=torch.long) for side in batch
    )
    image_loss = directional_loss(items[positive_ids], image[positive_ids])
    text_loss = directional_loss(items[positive_ids], text[positive_ids])
    loss_a = (image_loss + 0.3 * text_loss) / 1.3
    loss_b = (image_loss + text_loss) / 2.0
    loss_bpr = bpr_loss(users[user_ids], items[positive_ids], items[negative_ids])
    grad_a, = torch.autograd.grad(loss_a, items, retain_graph=True)
    grad_b, = torch.autograd.grad(loss_b, items, retain_graph=True)
    grad_user_bpr, grad_item_bpr = torch.autograd.grad(loss_bpr, (users, items))
    if users.grad is not None or items.grad is not None:
        raise AssertionError('autograd.grad wrote model .grad buffers.')
    if not torch.equal(users.detach(), initial['users']) or not torch.equal(items.detach(), initial['items']):
        raise AssertionError('Initial parameters changed.')
    for value in (loss_a, loss_b, loss_bpr, grad_a, grad_b, grad_user_bpr, grad_item_bpr):
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite objective or gradient.')
    return {'a': grad_a.detach(), 'b': grad_b.detach(), 'bpr': grad_item_bpr.detach(),
            'user_bpr_norm': float(grad_user_bpr.detach().double().norm()),
            'loss_a': float(loss_a.detach()), 'loss_b': float(loss_b.detach()),
            'loss_bpr': float(loss_bpr.detach()), 'positives': int(len(positive_ids))}


def quantiles(values):
    if values.numel() == 0:
        return None
    values = values.double()
    return {'p10': float(torch.quantile(values, 0.10)),
            'median': float(torch.quantile(values, 0.50)),
            'p90': float(torch.quantile(values, 0.90))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.describe:
        print('CPU; 8 pinned batches; A rate 1:0.3 alpha 0.3; B rate 1:1 alpha fixed by pooled item-gradient RMS; zero updates/ranking; exclusive output:', OUT)
        return
    if args.self_test:
        self_test()
        return

    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError('Input/source fingerprint mismatch: ' + str(path))
    if OUT.exists():
        raise FileExistsError('Exclusive output already exists: ' + str(OUT))
    with BATCHES.open(encoding='utf-8') as stream:
        batches = json.load(stream)
    with PRIOR.open(encoding='utf-8') as stream:
        prior = json.load(stream)
    if len(batches) != 8 or prior['status'] != 'completed' or len(prior['rows']) != 8:
        raise ValueError('Prior eight-batch input is incomplete.')
    initial = torch.load(ASSET, map_location='cpu', weights_only=True)
    if set(initial) != {'users', 'items', 'image_items', 'text_items', 'image_users', 'text_users'}:
        raise ValueError('Shared tensor keys mismatch.')
    if initial['users'].shape != (35598, 64) or initial['items'].shape != (18357, 64):
        raise ValueError('Shared initialization shape mismatch.')
    if not all(torch.isfinite(value).all() for value in initial.values()):
        raise ValueError('Shared initial tensor nonfinite.')
    sys.path.insert(0, str(ROOT / 'codes'))
    from td_distill_model_no_projection import bpr_loss, directional_distillation_loss

    raw = []
    for index, batch in enumerate(batches):
        if len(batch) != 3 or not all(len(side) == 1024 for side in batch):
            raise ValueError('Saved batch shape mismatch.')
        result = item_batch_gradients(initial, initial['image_items'], initial['text_items'],
                                      batch, bpr_loss, directional_distillation_loss)
        old = prior['rows'][index]['gradients']['items']
        old_a_norm = old['weighted_distill']['norm']
        old_bpr_norm = old['bpr']['norm']
        if not math.isclose(ALPHA_A * float(result['a'].double().norm()), old_a_norm,
                            rel_tol=2e-4, abs_tol=1e-10):
            raise AssertionError('Current A gradient differs from historical batch ' + str(index))
        if not math.isclose(float(result['bpr'].double().norm()), old_bpr_norm,
                            rel_tol=2e-4, abs_tol=1e-10):
            raise AssertionError('Current BPR gradient differs from historical batch ' + str(index))
        raw.append(result)
    norms_a = [float(row['a'].double().norm()) for row in raw]
    norms_b = [float(row['b'].double().norm()) for row in raw]
    alpha_b = calibrated_alpha(norms_a, norms_b)
    a_rms = math.sqrt(sum((ALPHA_A * n) ** 2 for n in norms_a) / len(raw))
    b_rms = math.sqrt(sum((alpha_b * n) ** 2 for n in norms_b) / len(raw))
    if not math.isclose(a_rms, b_rms, rel_tol=1e-12, abs_tol=1e-14):
        raise AssertionError('Aggregate item-gradient RMS did not match.')

    rows = []
    stacked_dot = 0.0
    stacked_a2 = 0.0
    stacked_b2 = 0.0
    sampled_row_ratios = []
    unmatched_row_counts = []
    for index, result in enumerate(raw):
        a = ALPHA_A * result['a']
        b = alpha_b * result['b']
        bpr = result['bpr']
        na = float(a.double().norm())
        nb = float(b.double().norm())
        stacked_dot += float(torch.sum(a.double() * b.double()))
        stacked_a2 += na * na
        stacked_b2 += nb * nb
        an = a.double().norm(dim=1)
        bn = b.double().norm(dim=1)
        both = (an > 1e-20) & (bn > 1e-20)
        only_one = (an > 1e-20) ^ (bn > 1e-20)
        unmatched_row_counts.append(int(only_one.sum()))
        sampled_row_ratios.append((bn[both] / an[both]))
        rows.append({
            'batch': index, 'positive_count': result['positives'],
            'loss_a_unscaled': result['loss_a'], 'loss_b_unscaled': result['loss_b'],
            'bpr_loss': result['loss_bpr'],
            'a_item_gradient_norm': na, 'b_item_gradient_norm': nb,
            'b_over_a_item_norm': nb / na,
            'a_b_item_gradient_cosine': cosine(a, b),
            'a_bpr_item_gradient_cosine': cosine(a, bpr),
            'b_bpr_item_gradient_cosine': cosine(b, bpr),
            'bpr_item_gradient_norm': float(bpr.double().norm()),
            'bpr_user_gradient_norm': result['user_bpr_norm'],
            'a_over_bpr_item_norm': na / float(bpr.double().norm()),
            'b_over_bpr_item_norm': nb / float(bpr.double().norm()),
            'sampled_item_rows_with_both_nonzero': int(both.sum()),
            'sampled_item_rows_with_only_one_nonzero': int(only_one.sum()),
            'sampled_item_row_b_over_a_norm_quantiles': quantiles(bn[both] / an[both]),
        })
    all_ratios = torch.cat(sampled_row_ratios)
    stacked_cosine = stacked_dot / math.sqrt(stacked_a2 * stacked_b2)
    norm_ratios = [row['b_over_a_item_norm'] for row in rows]
    report = {
        'status': 'completed', 'kind': 'nonformal_zero_update_matched_target_direction',
        'source_sha256': sha256(Path(__file__)),
        'input_source_sha256': {str(path.relative_to(ROOT)): value for path, value in EXPECTED.items()},
        'torch': torch.__version__, 'numpy': np.__version__,
        'shared_initialization': 'SHARED_TD_TENSORS_PROCESS0_V1',
        'fixed_batches': 8, 'batch_size': 1024, 'student_dimension': 64,
        'a_target': '(Dimage+0.3*Dtext)/1.3', 'alpha_a': ALPHA_A,
        'b_target': '(Dimage+Dtext)/2', 'alpha_b': alpha_b,
        'alpha_b_rule': 'alpha_a*sqrt(sum_b||grad_item(A_unscaled,b)||^2/sum_b||grad_item(B_unscaled,b)||^2); one global coefficient',
        'a_item_gradient_rms': a_rms, 'b_item_gradient_rms': b_rms,
        'rms_relative_difference': abs(a_rms - b_rms) / a_rms,
        'stacked_eight_batch_a_b_item_gradient_cosine': stacked_cosine,
        'mean_batch_a_b_item_gradient_cosine': float(np.mean([row['a_b_item_gradient_cosine'] for row in rows])),
        'batch_b_over_a_norm_min': min(norm_ratios), 'batch_b_over_a_norm_max': max(norm_ratios),
        'sampled_item_row_b_over_a_norm_quantiles': quantiles(all_ratios),
        'sampled_item_rows_with_only_one_nonzero_total': sum(unmatched_row_counts),
        'mean_a_bpr_item_gradient_cosine': float(np.mean([row['a_bpr_item_gradient_cosine'] for row in rows])),
        'mean_b_bpr_item_gradient_cosine': float(np.mean([row['b_bpr_item_gradient_cosine'] for row in rows])),
        'rows': rows, 'initial_tensors_unchanged': True, 'grad_buffers_untouched': True,
        'optimizer_steps': 0, 'teacher_forwards': 0, 'new_sampling': 0,
        'validation_split_loaded': False, 'validation_rankings': 0,
        'test_split_loaded': False, 'test_rankings': 0,
        'caveat': 'This is a local fixed-state raw-gradient direction comparison. Global RMS matching does not match each batch or item and says nothing by itself about AdamW updates or ranking quality.',
    }
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in (
        'alpha_a', 'alpha_b', 'a_item_gradient_rms', 'b_item_gradient_rms',
        'stacked_eight_batch_a_b_item_gradient_cosine',
        'mean_a_bpr_item_gradient_cosine', 'mean_b_bpr_item_gradient_cosine')}, indent=2))
    print('Report:', OUT / 'report.json')


if __name__ == '__main__':
    main()
