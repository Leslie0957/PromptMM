"""CPU-only, zero-update geometry and gradient audit for the pinned Sports student."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/process0/initial_tensors.pt'
TRAIN = ROOT / 'data/sports/train_mat'
BATCHES = ROOT / 'exp/gradient_checks/sports_sharedinit_seed2022_v2/batches.json'
OLD_REPORT = ROOT / 'exp/gradient_checks/sports_sharedinit_seed2022_v2/report.json'
OUT = ROOT / 'exp/gradient_checks/sports_weighted_directions_seed2022_v1'
EXPECTED = {
    ASSET: 'e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20',
    TRAIN: '5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8',
    BATCHES: 'd0489b97de4733f14b9dbd10bbea82dffdd1cbbc4420181a91681cf0ccfa75f0',
    OLD_REPORT: '5219ef3e3ccb7399bb484f42dd7edb7bc0dfe0281f7e3a555accbb68c48b74ff',
}
ALPHA = 0.3
TEXT_RATE = 0.3


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def positive_probabilities(matrix):
    """Exact marginal of Data.sample's uniform-user, uniform-positive draw."""
    matrix = matrix.tocsr()
    if not matrix.has_canonical_format or not np.all(matrix.data != 0):
        raise ValueError('Training CSR has duplicate or stored-zero entries.')
    degrees = np.diff(matrix.indptr)
    active = degrees > 0
    if not active.any():
        raise ValueError('Training matrix has no positive users.')
    rows = np.repeat(np.arange(matrix.shape[0]), degrees)
    weights = 1.0 / (int(active.sum()) * degrees[rows])
    probability = np.bincount(matrix.indices, weights=weights, minlength=matrix.shape[1])
    if not np.isclose(probability.sum(), 1.0, rtol=0, atol=1e-12):
        raise AssertionError('Positive-item probabilities do not sum to one.')
    return probability, int(active.sum()), degrees


def occurrence_gradient(initial_items, image_items, text_items):
    """Gradient of alpha*(D_image+0.3*D_text)/1.3 for one positive draw."""
    norm = initial_items.norm(dim=1, keepdim=True)
    if torch.any(norm <= 1e-12):
        raise ValueError('Initial item vector is at the normalization epsilon.')
    n = F.normalize(initial_items, dim=1)
    image = F.normalize(image_items, dim=1)
    text = F.normalize(text_items, dim=1)
    q = (image + TEXT_RATE * text) / (1.0 + TEXT_RATE)
    tangent = q - (n * q).sum(dim=1, keepdim=True) * n
    return -2.0 * ALPHA / initial_items.shape[1] * tangent / norm, q, tangent


def weighted_mean(values, probability):
    return float(np.dot(values.detach().double().numpy(), probability))


def self_test():
    toy = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]]))
    p, users, _ = positive_probabilities(toy)
    np.testing.assert_allclose(p, [0.25, 0.75, 0.0], atol=1e-15)
    assert users == 2
    initial = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64, requires_grad=True)
    image = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    text = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    pos = torch.tensor([0, 1, 1])
    loss = ALPHA * (F.mse_loss(F.normalize(initial[pos], dim=1), F.normalize(image[pos], dim=1))
                    + TEXT_RATE * F.mse_loss(F.normalize(initial[pos], dim=1), F.normalize(text[pos], dim=1))) / (1 + TEXT_RATE)
    actual, = torch.autograd.grad(loss, initial)
    per_occurrence, _, _ = occurrence_gradient(initial.detach(), image, text)
    expected = torch.zeros_like(initial).index_add_(0, pos, per_occurrence[pos] / len(pos))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-14)
    print('synthetic probability and autograd checks passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.describe:
        print('p(item)=mean over active train users of 1/degree(user); pinned F and image/text targets; alpha=0.3; CPU autograd; no updates/ranking; exclusive output:', OUT)
        return
    if args.self_test:
        self_test()
        return

    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError('Input fingerprint mismatch: ' + str(path))
    with TRAIN.open('rb') as stream:
        matrix = pickle.load(stream)
    if not sp.issparse(matrix) or matrix.shape != (35598, 18357):
        raise ValueError('Training matrix type/shape mismatch.')
    p_np, active_users, degrees = positive_probabilities(matrix)
    tensors = torch.load(ASSET, map_location='cpu', weights_only=True)
    if set(tensors) != {'users', 'items', 'image_items', 'text_items', 'image_users', 'text_users'}:
        raise ValueError('Shared tensor key mismatch.')
    initial = tensors['items']
    image = tensors['image_items']
    text = tensors['text_items']
    if initial.shape != (18357, 64) or image.shape != initial.shape or text.shape != initial.shape:
        raise ValueError('Shared item tensor shape mismatch.')
    if not all(torch.isfinite(x).all() for x in (initial, image, text)):
        raise ValueError('Nonfinite shared item tensor.')
    with BATCHES.open(encoding='utf-8') as stream:
        batches = json.load(stream)
    with OLD_REPORT.open(encoding='utf-8') as stream:
        old = json.load(stream)
    if old['status'] != 'completed' or len(batches) != 8 or len(old['rows']) != 8:
        raise ValueError('Prior fixed-batch diagnostic incomplete.')

    occurrence, q, tangent = occurrence_gradient(initial, image, text)
    n = F.normalize(initial, dim=1)
    image_n = F.normalize(image, dim=1)
    text_n = F.normalize(text, dim=1)
    equal_q = F.normalize(image_n + text_n, dim=1)
    q_direction = F.normalize(q, dim=1)
    positive = torch.from_numpy(p_np > 0)
    nonzero_target = (q.norm(dim=1) > 1e-12)
    valid = positive & nonzero_target
    probability = np.where(valid.numpy(), p_np, 0.0)
    probability /= probability.sum()
    cos_actual = (n * q_direction).sum(dim=1)
    cos_equal = (n * equal_q).sum(dim=1)
    expected_gradient = occurrence * torch.from_numpy(p_np).to(occurrence.dtype)[:, None]
    norm_occurrence = occurrence.norm(dim=1)
    tangent_norm = tangent.norm(dim=1)
    if not torch.isfinite(expected_gradient).all():
        raise ValueError('Nonfinite expected gradient.')
    if torch.max(torch.abs((expected_gradient * n).sum(dim=1))) > 2e-10:
        raise AssertionError('Expected directional gradient has a radial component.')

    sys.path.insert(0, str(ROOT / 'codes'))
    from td_distill_model_no_projection import directional_distillation_loss
    pos0 = torch.tensor(batches[0][1], dtype=torch.long)
    state = initial.clone().requires_grad_(True)
    objective = ALPHA * (
        directional_distillation_loss(state[pos0], image[pos0])
        + TEXT_RATE * directional_distillation_loss(state[pos0], text[pos0])
    ) / (1 + TEXT_RATE)
    actual, = torch.autograd.grad(objective, state)
    analytic = torch.zeros_like(initial).index_add_(0, pos0, occurrence[pos0] / len(pos0))
    difference = actual - analytic
    torch.testing.assert_close(actual, analytic, rtol=2e-5, atol=2e-9)
    if state.grad is not None or not torch.equal(state.detach(), initial):
        raise AssertionError('Diagnostic mutated initial parameters or grad buffer.')
    old_norm = old['rows'][0]['gradients']['items']['weighted_distill']['norm']
    if not np.isclose(float(actual.norm()), old_norm, rtol=2e-4, atol=1e-10):
        raise AssertionError('Autograd norm disagrees with prior fixed-batch diagnostic.')

    eight = []
    for row, batch in zip(old['rows'], batches):
        pos = torch.tensor(batch[1], dtype=torch.long)
        gradient = torch.zeros_like(initial).index_add_(0, pos, occurrence[pos] / len(pos))
        historical = row['gradients']['items']['weighted_distill']['norm']
        if not np.isclose(float(gradient.norm()), historical, rtol=2e-4, atol=1e-10):
            raise AssertionError('Analytic norm disagrees with prior batch ' + str(row['batch']))
        eight.append({'batch': row['batch'], 'analytic_item_gradient_norm': float(gradient.norm()),
                      'prior_autograd_item_gradient_norm': historical,
                      'prior_item_gradient_relative_to_bpr': row['gradients']['items']['weighted_distill']['relative_to_bpr'],
                      'prior_item_gradient_cosine_to_bpr': row['gradients']['items']['weighted_distill']['cosine_to_bpr']})

    report = {
        'status': 'completed', 'kind': 'nonformal_zero_update_direction_diagnostic',
        'input_sha256': {str(path.relative_to(ROOT)): value for path, value in EXPECTED.items()},
        'source_sha256': sha256(Path(__file__)), 'torch': torch.__version__, 'numpy': np.__version__,
        'student_alpha': ALPHA, 'item_image_rate': 1.0, 'item_text_rate': TEXT_RATE,
        'sample_rule': 'uniform active training user, then uniform positive item of that user; exact one-position marginal',
        'active_training_users': active_users, 'train_interactions': int(matrix.nnz),
        'items_with_positive_probability': int((p_np > 0).sum()),
        'zero_modality_target_items': int((~nonzero_target).sum()),
        'positive_probability_on_zero_targets': float(p_np[(~nonzero_target).numpy()].sum()),
        'positive_probability_sum': float(p_np.sum()),
        'positive_probability_min_nonzero': float(p_np[p_np > 0].min()),
        'positive_probability_max': float(p_np.max()),
        'sample_weighted_nonzero_target': {
            'cos_F_target_1_to_0_3': weighted_mean(cos_actual, probability),
            'cos_F_equal_image_text': weighted_mean(cos_equal, probability),
            'fraction_probability_cos_F_target_below_0_5': float(probability[(cos_actual < 0.5).numpy()].sum()),
            'norm_F': weighted_mean(initial.norm(dim=1), probability),
            'norm_target_combination': weighted_mean(q.norm(dim=1), probability),
            'tangent_target_component_norm': weighted_mean(tangent_norm, probability),
            'per_positive_directional_gradient_norm': weighted_mean(norm_occurrence, probability),
        },
        'unweighted_nonzero_target': {
            'item_count': int(valid.sum()),
            'cos_F_target_1_to_0_3': float(cos_actual[valid].mean()),
            'cos_F_equal_image_text': float(cos_equal[valid].mean()),
        },
        'expected_item_gradient_matrix_norm': float(expected_gradient.norm()),
        'first_batch_autograd': {'loss': float(objective), 'item_gradient_norm': float(actual.norm()),
                                 'prior_item_gradient_norm': old_norm,
                                 'analytic_max_abs_error': float(difference.abs().max()),
                                 'analytic_rms_error': float(difference.square().mean().sqrt()),
                                 'parameters_unchanged': True, 'grad_buffer_untouched': True},
        'eight_fixed_batch_comparison': eight,
        'optimizer_steps': 0, 'teacher_forwards': 0,
        'validation_split_loaded': False, 'validation_rankings': 0,
        'test_split_loaded': False, 'test_rankings': 0,
        'caveat': 'Expected item gradient is a one-draw marginal under Data.sample, not a realized batch, an AdamW update, or a full training trajectory. Geometry is not ranking quality.',
    }
    if not np.isfinite(report['sample_weighted_nonzero_target']['cos_F_target_1_to_0_3']):
        raise ValueError('Nonfinite summary.')
    OUT.mkdir(parents=True, exist_ok=False)
    with (OUT / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'report': str(OUT / 'report.json'),
                      'weighted': report['sample_weighted_nonzero_target'],
                      'first_batch_autograd': report['first_batch_autograd']}, indent=2))


if __name__ == '__main__':
    main()
