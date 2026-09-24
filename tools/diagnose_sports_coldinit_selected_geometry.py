"""Read-only selected-state geometry of the nine audited Sports cold-init students."""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'exp/audit/sports300_audit.json'
AUDIT_SHA = 'a0a382825ca17838a1d201461cf6cd58a00d589c353f9a1ef1cc84c605cd4f43'
TRAIN = ROOT / 'data/sports/train_mat'
TRAIN_SHA = '5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8'
REFERENCE = ROOT / 'exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/process0/initial_tensors.pt'
REFERENCE_SHA = 'e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20'
OUT = ROOT / 'exp/representation_checks/sports_coldinit_three_seed_selected_v1'
ARMS = ('bpr', 'image_matched', 'full')
SEEDS = (2022, 2023, 2024)
N_ITEMS = 18357
DIM = 64


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def within_root(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Audited artifact path is outside the repository: ' + str(path))
    return path


def positive_probabilities(matrix):
    matrix = matrix.tocsr()
    if matrix.shape != (35598, N_ITEMS) or not matrix.has_canonical_format or np.any(matrix.data == 0):
        raise ValueError('Unexpected training CSR.')
    degrees = np.diff(matrix.indptr)
    active = degrees > 0
    rows = np.repeat(np.arange(len(degrees)), degrees)
    p = np.bincount(matrix.indices,
                    weights=1.0 / (int(active.sum()) * degrees[rows]),
                    minlength=N_ITEMS)
    if not math.isclose(float(p.sum()), 1.0, abs_tol=1e-12):
        raise ValueError('Positive probability does not sum to one.')
    return p, int(active.sum())


def weighted_cos(student, target, weights):
    values = (student * target).sum(dim=1).double().numpy()
    value = float(np.dot(weights, values))
    if not math.isfinite(value) or abs(value) > 1.000001:
        raise ValueError('Invalid weighted cosine.')
    return value


def self_test():
    toy = sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]]))
    # The standalone probability routine expects the pinned Sports shape; test
    # the same single-position marginal independently on this toy matrix.
    degree = np.diff(toy.indptr)
    rows = np.repeat(np.arange(3), degree)
    p = np.bincount(toy.indices, weights=1 / (2 * degree[rows]), minlength=3)
    np.testing.assert_allclose(p, [0.25, 0.75, 0.0], atol=1e-15)
    student = torch.tensor([[1., 0.], [0., 1.], [1., 0.]])
    target = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    assert math.isclose(weighted_cos(student, target, p), 0.25, abs_tol=1e-15)
    print('Synthetic probability and weighted-cosine checks passed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    for path, expected in ((AUDIT, AUDIT_SHA), (TRAIN, TRAIN_SHA), (REFERENCE, REFERENCE_SHA)):
        if sha256(path) != expected:
            raise RuntimeError('Pinned input changed: ' + str(path))
    audit = json.loads(AUDIT.read_text(encoding='utf-8'))
    if audit['checks'] != 'passed' or len(audit['rows']) != 9:
        raise ValueError('Nine-run audit incomplete.')
    with TRAIN.open('rb') as stream:
        train = pickle.load(stream)
    if not sp.issparse(train):
        raise ValueError('Training matrix is not sparse.')
    p, active_users = positive_probabilities(train)
    reference = torch.load(REFERENCE, map_location='cpu', weights_only=True)
    if set(reference) != {'users', 'items', 'image_items', 'text_items', 'image_users', 'text_users'}:
        raise ValueError('Reference tensor keys changed.')
    targets = {name: reference[name] for name in ('items', 'image_items', 'text_items')}
    if any(t.shape != (N_ITEMS, DIM) or not torch.isfinite(t).all() for t in targets.values()):
        raise ValueError('Reference item tensor invalid.')
    image = F.normalize(targets['image_items'], dim=1)
    text = F.normalize(targets['text_items'], dim=1)
    teacher_final = F.normalize(targets['items'], dim=1)
    text_residual = text - (text * image).sum(dim=1, keepdim=True) * image
    residual_norm = text_residual.norm(dim=1)
    base_mask = (p > 0)
    for target in targets.values():
        base_mask &= target.norm(dim=1).numpy() > 1e-12
    base_mask &= residual_norm.numpy() > 1e-8
    retained_mass = float(p[base_mask].sum())
    if retained_mass <= 0.99:
        raise ValueError('Too much sampling probability excluded.')
    weights = np.where(base_mask, p / retained_mass, 0.0)
    residual = F.normalize(text_residual, dim=1)
    directions = {'image': image, 'text': text, 'teacher_final': teacher_final,
                  'text_orthogonal_to_image': residual}
    rows = []
    seen = set()
    for row in audit['rows']:
        profile, seed, arm = row['profile'], row['seed'], row['arm']
        if (seed, arm) in seen or seed not in SEEDS or arm not in ARMS:
            raise ValueError('Unexpected or duplicate run in audit.')
        seen.add((seed, arm))
        if profile != f'sports_student_{arm}_seed{seed}_val300_v1':
            raise ValueError('Profile mismatch in audit.')
        manifest_record = next(x for x in row['artifacts'] if 'run_manifest__' in x['path'])
        checkpoint_record = next(x for x in row['artifacts'] if 'td_distill_full__' in x['path'])
        manifest_path = within_root(manifest_record['path'])
        checkpoint_path = within_root(checkpoint_record['path'])
        for record, path in ((manifest_record, manifest_path), (checkpoint_record, checkpoint_path)):
            if sha256(path) != record['sha256']:
                raise RuntimeError('Audited artifact changed: ' + str(path))
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        cfg = manifest['resolved_arguments']
        if (manifest['status'] != 'validation_completed' or manifest['student_config_profile'] != profile
                or manifest['final_test_performed'] is not False
                or manifest['teacher_final_test_performed'] is not False
                or manifest['paper_ready_eligible'] is not False
                or cfg['seed'] != seed or cfg['epoch'] != 300
                or cfg['td_init_from_teacher'] is not False
                or manifest['teacher_reuse_authorized_sha256'] != '57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea'
                or manifest['dataset_preflight']['matrices']['train']['sha256'] != TRAIN_SHA
                or manifest['best_selection_recall'] != row['recall']
                or manifest['best_selection_epoch'] != row['epoch']
                or Path(manifest['td_full_checkpoint']).resolve() != checkpoint_path):
            raise ValueError('Run identity or protocol mismatch: ' + profile)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        items = checkpoint['model_state_dict']['item_id_embedding.weight']
        if (checkpoint['student_config_profile'] != profile or checkpoint['best_epoch'] != row['epoch']
                or checkpoint['best_selection_recall'] != row['recall']
                or items.shape != (N_ITEMS, DIM) or not torch.isfinite(items).all()):
            raise ValueError('Selected checkpoint mismatch: ' + profile)
        if bool((items.norm(dim=1).numpy()[base_mask] <= 1e-12).any()):
            raise ValueError('Zero sampled selected item vector: ' + profile)
        student = F.normalize(items, dim=1)
        metrics = {key: weighted_cos(student, direction, weights)
                   for key, direction in directions.items()}
        rows.append({'seed': seed, 'arm': arm, 'profile': profile,
                     'best_epoch_0_based': row['epoch'], 'validation_recall20': row['recall'],
                     'weighted_cosine': metrics,
                     'weighted_item_norm': float(np.dot(weights, items.norm(dim=1).double().numpy())),
                     'manifest': str(manifest_path), 'manifest_sha256': manifest_record['sha256'],
                     'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_record['sha256']})
    if seen != {(seed, arm) for seed in SEEDS for arm in ARMS}:
        raise ValueError('Missing seed/arm run.')
    paired = []
    for seed in SEEDS:
        by_arm = {row['arm']: row for row in rows if row['seed'] == seed}
        paired.append({'seed': seed, 'full_minus_image_matched': {
            key: by_arm['full']['weighted_cosine'][key] - by_arm['image_matched']['weighted_cosine'][key]
            for key in directions}, 'full_minus_bpr': {
            key: by_arm['full']['weighted_cosine'][key] - by_arm['bpr']['weighted_cosine'][key]
            for key in directions}})
    report = {'status': 'completed', 'kind': 'read_only_coldinit_selected_state_geometry',
              'source_sha256': sha256(__file__), 'input_sha256': {
                  'sports300_audit': AUDIT_SHA, 'train_mat': TRAIN_SHA,
                  'common_teacher_reference': REFERENCE_SHA},
              'reference_boundary': 'common later-saved teacher process0 tensors; exact historical in-process targets unproven',
              'positive_rule': 'uniform active train user then uniform train-positive item',
              'active_training_users': active_users, 'train_interactions': int(train.nnz),
              'positive_items': int((p > 0).sum()), 'positive_probability_sum': float(p.sum()),
              'retained_items': int(base_mask.sum()), 'retained_positive_probability': retained_mass,
              'reference_image_text_cosine': weighted_cos(image, text, weights),
              'rows': rows, 'paired': paired, 'optimizer_steps': 0, 'model_forwards': 0,
              'validation_split_loaded': False, 'validation_rankings': 0,
              'test_split_loaded': False, 'test_rankings': 0,
              'parameters_mutated': False}
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / 'report.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print('Report:', path)
    for row in rows:
        print(row['seed'], row['arm'], row['weighted_cosine'])
    print('Retained sampling mass:', retained_mass)


if __name__ == '__main__':
    main()
