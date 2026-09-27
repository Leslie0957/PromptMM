"""Isolated, train-tensor-only four-arm core. No real-asset loader or CLI.

The caller must supply already validated train triplets, pinned initial tables,
semantic targets and a Validation-only callback. This module never opens a split.
"""

from dataclasses import dataclass
from pathlib import Path
import math
import os
import sys


ARMS = ('R0', 'R1', 'T0', 'T1')
FULL_RATES = {'item_image': 1.0, 'item_text': 0.3,
              'user_image': 0.0, 'user_text': 0.0}


@dataclass(frozen=True)
class Protocol:
    epochs: int = 300
    batches_per_epoch: int = 214
    batch_size: int = 1024
    n_users: int = 35598
    n_items: int = 18357
    dim: int = 64
    lr: float = 0.00006
    weight_decay: float = 0.01
    alpha_full: float = 0.3
    early_stopping_patience: int = 300
    primary_k: int = 20
    run_final_test: bool = False


def arm_config(arm, protocol=Protocol()):
    if arm not in ARMS:
        raise ValueError('Unregistered arm')
    return {'arm': arm,
            'initial': 'paired_random_initial' if arm[0] == 'R' else 'shared_teacher_final_tables',
            'objective': 'BPR' if arm[1] == '0' else 'BPR+Full',
            'alpha': 0.0 if arm[1] == '0' else protocol.alpha_full,
            'rates': dict(FULL_RATES),
            'optimizer': {'type': 'AdamW', 'lr': protocol.lr,
                          'weight_decay': protocol.weight_decay, 'fresh_step': 0},
            'epochs': protocol.epochs, 'batches_per_epoch': protocol.batches_per_epoch,
            'batch_size': protocol.batch_size, 'run_final_test': False}


def assert_train_val_path(path, dataset_root):
    """Reject Test names under the declared dataset root, including symlink targets."""
    root = Path(dataset_root).resolve()
    for target in (Path(path).absolute(), Path(path).resolve()):
        if target != root and root in target.parents:
            if any(part.casefold().startswith('test') for part in target.relative_to(root).parts):
                raise PermissionError('Test split access prohibited: ' + str(target))


def install_test_read_denial(dataset_root):
    """Process-wide audit hook: must be installed before any loader import/read."""
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            assert_train_val_path(args[0], dataset_root)
    sys.addaudithook(guard)


def reserve_attempt(directory):
    """Exclusive new namespace; an existing/partial attempt is never reused."""
    Path(directory).mkdir(parents=True, exist_ok=False)


def validate_inputs(random_initial, teacher_tables, semantics, tape, protocol):
    import torch
    required = {'user', 'item'}
    if set(random_initial) != required or set(teacher_tables) != required:
        raise ValueError('Initial table keys mismatch')
    for tables in (random_initial, teacher_tables):
        for key, shape in (('user', (protocol.n_users, protocol.dim)),
                           ('item', (protocol.n_items, protocol.dim))):
            value = tables[key]
            if value.shape != shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('Initial table shape/dtype/finite mismatch: ' + key)
    if set(semantics) != {'item_image', 'item_text', 'user_image', 'user_text'}:
        raise ValueError('Semantic target keys mismatch')
    for key, value in semantics.items():
        rows = protocol.n_items if key.startswith('item_') else protocol.n_users
        if value.shape != (rows, protocol.dim) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Semantic shape/dtype/finite mismatch: ' + key)
    if tuple(tape.shape) != (protocol.epochs, protocol.batches_per_epoch, 3, protocol.batch_size):
        raise ValueError('Triplet tape shape mismatch')
    if str(tape.dtype) != 'int32':
        raise ValueError('Triplet tape dtype mismatch')
    if tape.min() < 0 or tape[:, :, 0, :].max() >= protocol.n_users or tape[:, :, 1:, :].max() >= protocol.n_items:
        raise ValueError('Triplet tape index out of range')


def select_epochs(recall):
    if not recall or any(not math.isfinite(float(x)) for x in recall):
        raise ValueError('Nonfinite or empty Validation curve')
    return {'fixed_final_index': len(recall) - 1,
            'best_index': max(range(len(recall)), key=lambda i: recall[i])}


def run_arm(arm, random_initial, teacher_tables, semantics, tape, validation_callback,
            protocol=Protocol()):
    """Train one arm in memory for a caller-supplied tape and Val-only callback.

    No checkpoint, data or report I/O. A future real adapter must enforce the
    declared asset hashes, source identity, wall-clock and output budget.
    """
    import torch
    from td_distill_model_no_projection import (
        TDDistillNoProjectionModel, bpr_loss, directional_distillation_loss)

    validate_inputs(random_initial, teacher_tables, semantics, tape, protocol)
    config = arm_config(arm, protocol)
    source = random_initial if arm[0] == 'R' else teacher_tables
    model = TDDistillNoProjectionModel(protocol.n_users, protocol.n_items,
                                      protocol.dim, protocol.dim,
                                      item_head_names=('image', 'text'), user_head_names=())
    model.init_user_item_embed(source['user'], source['item'])
    for key, weight in (('user', model.user_id_embedding.weight),
                        ('item', model.item_id_embedding.weight)):
        if not torch.equal(weight.detach(), source[key]):
            raise RuntimeError('Initial pair copy mismatch: ' + key)
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol.lr,
                                  weight_decay=protocol.weight_decay)
    if optimizer.state:
        raise RuntimeError('AdamW state not fresh')
    # Epoch 0 is descriptive and never competes for best checkpoint.
    initial_metric = validation_callback(model, 0)
    curve = []
    for epoch in range(protocol.epochs):
        model.train()
        for batch in range(protocol.batches_per_epoch):
            ids = torch.as_tensor(tape[epoch, batch], dtype=torch.long)
            u = model.user_id_embedding(ids[0])
            pos = model.item_id_embedding(ids[1])
            neg = model.item_id_embedding(ids[2])
            loss = bpr_loss(u, pos, neg)
            if config['alpha']:
                image = directional_distillation_loss(pos, semantics['item_image'][ids[1]])
                text = directional_distillation_loss(pos, semantics['item_text'][ids[1]])
                loss = loss + config['alpha'] * (image + 0.3 * text) / 1.3
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metric = validation_callback(model, epoch + 1)
        if 'recall20' not in metric or 'ndcg20' not in metric:
            raise ValueError('Validation callback missing fixed endpoints')
        curve.append(metric)
    selection = select_epochs([x['recall20'] for x in curve])
    return {'arm': arm, 'config': config, 'initial_validation': initial_metric,
            'curve': curve, 'selection': selection,
            'optimizer_steps': protocol.epochs * protocol.batches_per_epoch,
            'model': model, 'optimizer': optimizer}


def interaction(last_recall):
    if set(last_recall) != set(ARMS):
        raise ValueError('Four final arm recalls required')
    delta_r = last_recall['R1'] - last_recall['R0']
    delta_t = last_recall['T1'] - last_recall['T0']
    return {'delta_R': delta_r, 'delta_T': delta_t, 'I': delta_r - delta_t}


def run_cohort(random_initial, teacher_tables, semantics, tape, validation_callback,
               protocol=Protocol()):
    """Execute exactly R0/R1/T0/T1 against one immutable in-memory input set."""
    import torch
    validate_inputs(random_initial, teacher_tables, semantics, tape, protocol)
    originals = {('R', key): value.clone() for key, value in random_initial.items()}
    originals.update({('T', key): value.clone() for key, value in teacher_tables.items()})
    results = {}
    for arm in ARMS:
        results[arm] = run_arm(arm, random_initial, teacher_tables, semantics,
                               tape, validation_callback, protocol)
        for key, value in random_initial.items():
            if not torch.equal(value, originals['R', key]):
                raise RuntimeError('Random initial asset mutated')
        for key, value in teacher_tables.items():
            if not torch.equal(value, originals['T', key]):
                raise RuntimeError('Teacher initial asset mutated')
    finals = {arm: results[arm]['curve'][-1]['recall20'] for arm in ARMS}
    return {'arms': results, 'interaction': interaction(finals)}
