"""Prepared one-shot seed2022 B/R/A/M cohort and isolated non-formal smoke."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
PROFILE = ROOT / 'docs/research/innovation2/MINIMAL_RANKING_SUPERVISION_PROFILE_V1.json'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def source_gate(profile):
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    if branch != profile['branch'] or profile['launch_source_rule'] != 'committed_head':
        raise RuntimeError('Branch/source rule mismatch')
    if any(not line.startswith('?? archive/reviews/') for line in dirty):
        raise RuntimeError('Formal source is dirty')
    return head


def test_denial(root):
    from initialization_kd_interaction import assert_train_val_path
    attempts = {'denied_test_open_attempts': 0}
    def guard(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            try:
                assert_train_val_path(args[0], root)
            except PermissionError:
                attempts['denied_test_open_attempts'] += 1
                raise
    sys.addaudithook(guard)
    return attempts


def monitor(output, started, caps, process, torch, samples, force=False):
    now = time.monotonic()
    if not force and samples and now - started - samples[-1]['elapsed_seconds'] < caps['sample_interval_seconds']:
        return
    row = {'elapsed_seconds': now - started, 'rss_bytes': process.memory_info().rss,
           'cuda_allocator_bytes': torch.cuda.memory_reserved('cuda:0'),
           'free_disk_bytes': shutil.disk_usage(output).free,
           'output_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
    samples.append(row)
    save_json(output / 'resource_samples.json', samples)
    if (row['elapsed_seconds'] > caps['wall_seconds'] or row['rss_bytes'] > caps['rss_bytes']
            or row['cuda_allocator_bytes'] > caps['cuda_allocator_bytes']
            or row['output_bytes'] > caps['output_bytes']
            or row['free_disk_bytes'] < caps['minimum_free_disk_bytes']):
        raise RuntimeError('Resource cap breached: ' + json.dumps(row))


def load_train(profile):
    import numpy as np
    import scipy.sparse as sparse
    asset = profile['assets']['train']
    path = ROOT / asset['path']
    if sha(path) != asset['sha256']:
        raise RuntimeError('Train SHA mismatch')
    with path.open('rb') as stream:
        train = pickle.load(stream)
    if not sparse.issparse(train) or train.shape != (35598, 18357):
        raise ValueError('Train sparse identity mismatch')
    train = train.tocsr(copy=True)
    train.sum_duplicates()
    train.sort_indices()
    if not np.isfinite(train.data).all() or (train.data <= 0).any():
        raise ValueError('Train values invalid')
    return train


def load_teacher(profile):
    import torch
    asset = profile['assets']['teacher_tables']
    if sha(ROOT / asset['path']) != asset['sha256']:
        raise RuntimeError('Teacher table SHA mismatch')
    cache = torch.load(ROOT / asset['path'], map_location='cpu', weights_only=True)
    if set(cache) != {'users', 'items', 'image_items', 'text_items', 'image_users', 'text_users'}:
        raise ValueError('Shared cache keys mismatch')
    user, item = cache['users'], cache['items']
    for table, shape in ((user, (35598, 64)), (item, (18357, 64))):
        if tuple(table.shape) != shape or table.dtype != torch.float32 or not torch.isfinite(table).all():
            raise ValueError('Teacher table invalid')
    return user, item


def load_tape(profile):
    import numpy as np
    asset = profile['assets']['triplet_tape']
    if sha(ROOT / asset['path']) != asset['sha256']:
        raise RuntimeError('Tape SHA mismatch')
    tape = np.load(ROOT / asset['path'], mmap_mode='r', allow_pickle=False)
    if tape.shape != (300, 214, 3, 1024) or tape.dtype != np.int32:
        raise ValueError('Tape identity mismatch')
    return tape


def validate_profile(profile):
    if sha(ROOT / profile['anchor_manifest']) != profile['anchor_manifest_sha256']:
        raise RuntimeError('Original launch manifest SHA mismatch')
    anchor = json.loads((ROOT / profile['anchor_manifest']).read_text(encoding='utf-8'))
    expected_anchors = {
        2022: ('484a4bef36030e0d76a5d77e4da14d84aea46589',
               'b9372db495735f9af235ecf7dfa41745872f585aa8d5d52987d169e27e85d543',
               0.095151699965148426),
        2023: ('32c89f119275f3d9001466721452f25e4a35e41e',
               'da3c34aea7f2569cfa99173ebd04fc35ad7a474cee51058f1c9b41f55006dce8',
               0.094343178530468197),
        2024: ('5450185776df81e77bfcc8b92ab54b0f134fd37c',
               'ae86caee1c1f8451cb6254579cc3292b036eaf422fceb8703e31be9f5b34b6ac',
               0.094615509685152296),
    }
    expected = expected_anchors.get(profile['seed'])
    if (profile['dataset'] != 'sports' or expected is None
            or profile['arms'] != ['B', 'R', 'A']
            or profile['training'] != {'epochs': 300, 'batches_per_epoch': 214,
                'batch_size': 1024, 'optimizer': 'AdamW', 'lr': 0.00006,
                'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-8,
                'bpr': 'mean(-logsigmoid(dot(u,p)-dot(u,n)))',
                'rank_kl_weight': 1.0, 'anchor_weight': 1.0,
                'smoke_steps_per_arm': 10}
            or profile['evaluation'] != {'split': 'Validation_only', 'test_access': False,
                'candidate_exclusion': 'Train_only', 'k': 20,
                'primary': 'epoch300_recall20', 'secondary': 'ndcg20',
                'calls': 904, 'user_batch': 256, 'dtype': 'float32', 'tf32': False,
                'B_epoch0_expected': 0.094184495190847733,
                'B_epoch300_expected': expected[2],
                'regression_tolerance': 1e-6, 'mixture_teacher_fraction': 0.5}
            or profile['candidates'] != {'teacher_top': 32, 'uniform_random': 32,
                'numpy_rng_seed': 20260930, 'user_order': 'ascending_train_user_id',
                'tie_rule': 'descending_score_ascending_item_id', 'smoke_train_users_max': 256}
            or anchor['source_commit'] != expected[0]
            or anchor['config_digest'] != expected[1]):
        raise ValueError('Fixed protocol/anchor mismatch')
    for new, old in (('teacher_tables', 'shared_cache'), ('triplet_tape', 'triplet_tape'),
                     ('train', 'train_mat_record_only'), ('validation', 'val_mat_record_only')):
        if profile['assets'][new] != {key: anchor['assets'][old][key] for key in ('path', 'sha256')}:
            raise ValueError('Asset anchor mismatch: ' + new)
    return anchor


def model_from_teacher(user, item, device):
    import torch
    from td_distill_model_no_projection import TDDistillNoProjectionModel
    model = TDDistillNoProjectionModel(len(user), len(item), user.shape[1], item.shape[1],
                                      item_head_names=(), user_head_names=())
    model.init_user_item_embed(user, item)
    if not torch.equal(model.user_id_embedding.weight.detach(), user) or not torch.equal(model.item_id_embedding.weight.detach(), item):
        raise RuntimeError('Initial table copy mismatch')
    return model.to(device)


def train_step(model, optimizer, arm, ids, candidates, initial, scales,
               record_gradient=False):
    import torch
    from td_distill_model_no_projection import bpr_loss
    from minimal_ranking_supervision import anchor_loss, rank_kl
    u, p, n = (ids[i] for i in range(3))
    user = model.user_id_embedding(u)
    pos = model.item_id_embedding(p)
    neg = model.item_id_embedding(n)
    bpr = bpr_loss(user, pos, neg)
    auxiliary = None
    if arm == 'R':
        row = candidates['row_for_user'][u]
        if (row < 0).any():
            raise RuntimeError('Tape user missing fixed candidates')
        auxiliary = rank_kl(user, model.item_id_embedding(candidates['ids'][row]),
                            candidates['logits'][row], candidates['mu'][row],
                            candidates['scale'][row])
    elif arm == 'A':
        auxiliary = anchor_loss(user, pos, neg, initial['user'][u], initial['item'][p],
                                initial['item'][n], *scales)
    loss = bpr if auxiliary is None else bpr + auxiliary
    if not torch.isfinite(loss) or (auxiliary is not None and not torch.isfinite(auxiliary)):
        raise RuntimeError('Nonfinite objective')
    gradient_norm = None
    if record_gradient and auxiliary is not None:
        gradients = torch.autograd.grad(auxiliary, tuple(model.parameters()), retain_graph=True)
        gradient_norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(not torch.isfinite(param.grad).all() for param in model.parameters()):
        raise RuntimeError('Nonfinite gradient')
    optimizer.step()
    if any(not torch.isfinite(param).all() for param in model.parameters()):
        raise RuntimeError('Nonfinite parameter')
    return {'bpr': float(bpr.detach()), 'auxiliary': float(auxiliary.detach()) if auxiliary is not None else 0.,
            'total': float(loss.detach()), 'auxiliary_gradient_norm': gradient_norm}


def save_top20(output, name, user_table, item_table, users, train, check):
    import numpy as np
    import torch
    from initialization_kd_fast_eval import exact_topk
    top = np.empty((len(users), 20), dtype=np.int32)
    item = item_table.to('cuda:0')
    with torch.no_grad():
        for start in range(0, len(users), 256):
            end = min(start + 256, len(users))
            selected = torch.as_tensor(users[start:end], device=user_table.device, dtype=torch.long)
            scores = (user_table[selected].to('cuda:0') @ item.T).cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError('Nonfinite final ranking score')
            for row, uid in enumerate(users[start:end], start):
                seen = set(train.indices[train.indptr[uid]:train.indptr[uid + 1]].tolist())
                top[row] = exact_topk(scores[row - start], seen, 20)
            check()
    np.savez_compressed(output / (name + '_final_top20.npz'), users=users, top20=top)
    check(True)


def run(profile, output, formal, started, manifest):
    import numpy as np
    import psutil
    import torch
    from minimal_ranking_supervision import candidate_rows, initial_scales, mixed_tables

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    np.random.seed(profile['seed'])
    torch.manual_seed(profile['seed'])
    torch.cuda.manual_seed_all(profile['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    caps = profile['caps'] if formal else dict(profile['caps'], wall_seconds=600,
                                               output_bytes=134217728)
    torch.cuda.set_per_process_memory_fraction(caps['cuda_allocator_bytes'] /
        torch.cuda.get_device_properties('cuda:0').total_memory, device='cuda:0')
    process = psutil.Process()
    samples = []
    check = lambda force=False: monitor(output, started, caps, process, torch, samples, force)
    test_attempts = test_denial(ROOT / 'data/sports')
    check(True)
    anchor = validate_profile(profile)
    manifest['original_source_commit'] = anchor['source_commit']
    manifest['original_config_digest'] = anchor['config_digest']
    manifest['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'psutil': psutil.__version__, 'cuda_device': torch.cuda.get_device_name(0)}
    manifest['numerics'] = {'numpy_global_seed': profile['seed'], 'numpy_candidate_rng_seed':
        profile['candidates']['numpy_rng_seed'], 'torch_cpu_cuda_seed': profile['seed'],
        'deterministic_algorithms': True, 'tf32': False, 'cublas_workspace_config': ':4096:8',
        'torch_cpu_threads': 4}
    save_json(output / 'manifest.json', manifest)
    train = load_train(profile)
    teacher_user, teacher_item = load_teacher(profile)
    tape = load_tape(profile)
    check(True)
    candidate_started = time.monotonic()
    if formal:
        prepared = profile['prepared_candidates']
        source = ROOT / prepared['path']
        if sha(source) != prepared['sha256']:
            raise RuntimeError('Prepared candidate file SHA mismatch')
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {'users', 'ids', 'logits', 'mu', 'scale'}:
                raise ValueError('Prepared candidate keys mismatch')
            users, ids, logits, mu, scale = (archive[key] for key in ('users', 'ids', 'logits', 'mu', 'scale'))
        if (ids.shape != tuple(prepared['shape']) or len(users) != prepared['users']
                or not np.array_equal(users, np.flatnonzero(np.diff(train.indptr)))
                or logits.shape != ids.shape or mu.shape != users.shape or scale.shape != users.shape
                or ids.dtype != np.int32 or logits.dtype != np.float32
                or not np.isfinite(logits).all() or not np.isfinite(mu).all()
                or not np.isfinite(scale).all() or (scale <= 0).any()
                or ids.min() < 0 or ids.max() >= train.shape[1]):
            raise ValueError('Prepared candidate structure mismatch')
        for row, uid in enumerate(users):
            if (len(np.unique(ids[row])) != 64 or set(ids[row]) &
                    set(train.indices[train.indptr[uid]:train.indptr[uid + 1]])):
                raise ValueError('Prepared candidate exclusion/uniqueness mismatch')
            if row % 256 == 0:
                check()
        shutil.copyfile(source, output / 'candidates.npz')
    else:
        users, ids, logits, mu, scale = candidate_rows(train, teacher_user, teacher_item,
            limit=profile['candidates']['smoke_train_users_max'],
            seed=profile['candidates']['numpy_rng_seed'], check=check)
        np.savez_compressed(output / 'candidates.npz', users=users, ids=ids,
                            logits=logits, mu=mu, scale=scale)
    candidate_hash = hashlib.sha256(ids.tobytes() + logits.tobytes() + mu.tobytes() + scale.tobytes()).hexdigest()
    if formal and candidate_hash != profile['prepared_candidates']['logical_sha256']:
        raise RuntimeError('Prepared candidate logical SHA mismatch')
    manifest['candidate_sha256'] = candidate_hash
    manifest['candidate_users'] = len(users)
    manifest['candidate_preparation_seconds'] = time.monotonic() - candidate_started
    save_json(output / 'manifest.json', manifest)
    check(True)
    row_for_user = np.full(35598, -1, dtype=np.int32)
    row_for_user[users] = np.arange(len(users), dtype=np.int32)
    candidate_device = {'row_for_user': torch.as_tensor(row_for_user, device='cuda:0'),
        'ids': torch.as_tensor(ids.astype(np.int64), device='cuda:0'),
        'logits': torch.as_tensor(logits, device='cuda:0'),
        'mu': torch.as_tensor(mu, device='cuda:0'),
        'scale': torch.as_tensor(scale, device='cuda:0')}
    initial = {'user': teacher_user.to('cuda:0'), 'item': teacher_item.to('cuda:0')}
    scales = initial_scales(initial['user'], initial['item'])
    report = {'status': 'partial', 'mode': 'formal' if formal else 'non_formal_resource_smoke',
        'arms': {}, 'validation_calls': 0, 'test_denied_open_attempts': 0,
        'candidate_users': len(users), 'candidate_sha256': candidate_hash,
        'initial_scales': scales, 'source_commit': manifest['source_commit']}
    save_json(output / 'report.json', report)
    if formal:
        from initialization_kd_adapter import load_train_val
        from initialization_kd_fast_eval import evaluate_validation_fast
        val_asset = profile['assets']['validation']
        train, val = load_train_val(ROOT / 'data/sports', profile['assets']['train']['sha256'],
                                    val_asset['sha256'], 35598, 18357)
        validation_users = np.flatnonzero(np.diff(val.indptr))
        if len(validation_users) != 35598:
            raise RuntimeError('Validation users mismatch')
        def evaluate(model):
            check()
            model.eval()
            with torch.no_grad():
                metric = evaluate_validation_fast(model, train, val, ks=(20,), user_batch=256)
            report['validation_calls'] += 1
            check()
            return {'recall20': metric['recall20'], 'ndcg20': metric['ndcg20']}
    else:
        evaluate = None
    for arm in profile['arms']:
        model = model_from_teacher(teacher_user, teacher_item, 'cuda:0')
        optimizer = torch.optim.AdamW(model.parameters(), lr=profile['training']['lr'],
            weight_decay=profile['training']['weight_decay'],
            betas=tuple(profile['training']['betas']), eps=profile['training']['eps'])
        if optimizer.state:
            raise RuntimeError('Optimizer moments not fresh')
        record = {'optimizer_steps': 0, 'curve': [], 'epoch_diagnostics': []}
        if formal:
            record['epoch0'] = evaluate(model)
            expected = profile['evaluation']['B_epoch0_expected']
            if abs(record['epoch0']['recall20'] - expected) > 1e-6:
                raise RuntimeError(arm + ' epoch0 numeric regression failed')
        else:
            record['epoch0'] = 'skipped_no_validation'
        report['arms'][arm] = record
        save_json(output / 'report.json', report)
        epochs = 300 if formal else 1
        batches = 214 if formal else profile['training']['smoke_steps_per_arm']
        best = -1.
        for epoch in range(epochs):
            model.train()
            epoch_sums = {'bpr': 0., 'auxiliary': 0., 'total': 0.}
            for batch in range(batches):
                check()
                batch_ids = tape[epoch, batch].copy()
                if not formal:
                    keep = np.isin(batch_ids[0], users)
                    batch_ids = batch_ids[:, keep]
                    if batch_ids.shape[1] == 0:
                        raise RuntimeError('Smoke batch has no selected Train user')
                tensor_ids = torch.as_tensor(batch_ids, device='cuda:0', dtype=torch.long)
                values = train_step(model, optimizer, arm, tensor_ids,
                    candidate_device, initial, scales, record_gradient=(batch == batches - 1))
                for key in epoch_sums:
                    epoch_sums[key] += values[key]
                record['optimizer_steps'] += 1
                check()
            with torch.no_grad():
                drift_user = float((model.user_id_embedding.weight - initial['user']).square().mean().sqrt())
                drift_item = float((model.item_id_embedding.weight - initial['item']).square().mean().sqrt())
            record['epoch_diagnostics'].append({'epoch': epoch + 1, 'last_step': values,
                'mean_step_losses': {key: value / batches for key, value in epoch_sums.items()},
                'user_rms_drift': drift_user, 'item_rms_drift': drift_item})
            if formal:
                metric = evaluate(model)
                record['curve'].append({'epoch': epoch + 1, **metric})
                save_json(output / (arm + '_curve.json'), record['curve'])
                for label, needed in (('best', metric['recall20'] > best), ('final', epoch == 299)):
                    if needed:
                        target = output / (arm + '_' + label + '.pt')
                        temporary = target.with_suffix('.tmp')
                        torch.save({'arm': arm, 'epoch': epoch + 1, 'metric': metric,
                            'source_commit': manifest['source_commit'],
                            'profile_sha256': manifest['profile_sha256'],
                            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                            'optimizer': optimizer.state_dict()}, temporary)
                        temporary.replace(target)
                best = max(best, metric['recall20'])
                if arm == 'B' and epoch == 299 and abs(metric['recall20'] -
                    profile['evaluation']['B_epoch300_expected']) > 1e-6:
                    raise RuntimeError('B epoch300 numeric regression failed; stop before R/A')
            save_json(output / 'report.json', report)
            check(True)
        if formal and arm == 'B':
            b_user = model.user_id_embedding.weight.detach().cpu().clone()
            b_item = model.item_id_embedding.weight.detach().cpu().clone()
        if formal:
            save_top20(output, arm, model.user_id_embedding.weight.detach(),
                       model.item_id_embedding.weight.detach(), validation_users, train, check)
        del model, optimizer
        torch.cuda.empty_cache()
    if formal:
        mixed_user, mixed_item = mixed_tables(b_user, b_item, teacher_user, teacher_item)
        mix_model = model_from_teacher(mixed_user, mixed_item, 'cuda:0')
        report['mixture'] = evaluate(mix_model)
        save_top20(output, 'M', mix_model.user_id_embedding.weight.detach(),
                   mix_model.item_id_embedding.weight.detach(), validation_users, train, check)
        report['mixture_table_dimension'] = 128
        report['B_teacher_table_rms'] = {
            'B_user': float(b_user.square().mean().sqrt()),
            'B_item': float(b_item.square().mean().sqrt()),
            'teacher_user': float(teacher_user.square().mean().sqrt()),
            'teacher_item': float(teacher_item.square().mean().sqrt())}
        del mix_model
        if report['validation_calls'] != 904:
            raise RuntimeError('Evaluation call count mismatch')
        if source_gate(profile) != manifest['source_commit']:
            raise RuntimeError('Source changed during formal cohort')
        b = report['arms']['B']['curve'][-1]['recall20']
        r = report['arms']['R']['curve'][-1]['recall20']
        a = report['arms']['A']['curve'][-1]['recall20']
        m = report['mixture']['recall20']
        delta = profile['screening']['delta']
        eps = profile['screening']['simple_alternative_epsilon']
        report['screen'] = ('screen_stop' if r - b < delta else
            'simple_alternative_or_unresolved' if r - max(a, m) < eps else
            'candidate_for_replication')
        report['deltas'] = {'R_minus_B': r-b, 'A_minus_B': a-b, 'M_minus_B': m-b}
    report['status'] = 'completed'
    report['test_denied_open_attempts'] = test_attempts['denied_test_open_attempts']
    report['elapsed_seconds'] = time.monotonic() - started
    report['peak_sampled_rss_bytes'] = max(row['rss_bytes'] for row in samples)
    report['peak_cuda_allocator_bytes'] = torch.cuda.max_memory_reserved('cuda:0')
    save_json(output / 'report.json', report)
    if formal and (set(report['arms']) != set(profile['arms']) or
            any(record['optimizer_steps'] != 64200 or len(record['curve']) != 300
                or [row['epoch'] for row in record['curve']] != list(range(1, 301))
                for record in report['arms'].values())):
        raise RuntimeError('Formal curve/step acceptance failed')
    save_json(output / 'resource.json', {'samples_file': 'resource_samples.json',
        'sample_interval_seconds': caps['sample_interval_seconds'], 'os_hard_isolation': False,
        'peak_sampled_rss_bytes': report['peak_sampled_rss_bytes'],
        'peak_cuda_allocator_bytes': report['peak_cuda_allocator_bytes'],
        'elapsed_seconds': report['elapsed_seconds']})
    check(True)
    artifacts = ['candidates.npz', 'report.json', 'resource.json', 'resource_samples.json']
    if formal:
        artifacts += [f'{arm}_{label}.pt' for arm in profile['arms'] for label in ('best', 'final')]
        artifacts += [f'{arm}_curve.json' for arm in profile['arms']]
        artifacts += [f'{arm}_final_top20.npz' for arm in (*profile['arms'], 'M')]
    save_json(output / 'acceptance.json', {'status': 'passed', 'mode': report['mode'],
        'test_denied_open_attempts': report['test_denied_open_attempts'],
        'validation_calls': report['validation_calls'], 'optimizer_steps':
        {arm: report['arms'][arm]['optimizer_steps'] for arm in profile['arms']},
        'artifact_sha256': {name: sha(output / name) for name in artifacts}})


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--formal', action='store_true')
    group.add_argument('--resource-smoke', action='store_true')
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    validate_profile(profile)
    if args.formal:
        if profile['status'] != 'launchable_after_explicit_authorization':
            raise RuntimeError('Formal cohort not authorized/activated in committed profile')
        head = source_gate(profile)
    else:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    output = ROOT / (profile['output'] if args.formal else profile['smoke_output'])
    if output.exists():
        raise RuntimeError('Output namespace already exists; no retry/overwrite')
    if shutil.disk_usage(ROOT).free < profile['caps']['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient free disk')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {'source_commit': head, 'source_state': 'clean_committed' if args.formal else
        'non_formal_uncommitted_preparation', 'profile': str(PROFILE.relative_to(ROOT)).replace('\\', '/'),
        'profile_sha256': sha(PROFILE), 'mode': 'formal' if args.formal else 'non_formal_resource_smoke',
        'command': profile['command'] if args.formal else 'python -B tools/run_minimal_ranking_supervision.py --resource-smoke',
        'assets': profile['assets'], 'caps': profile['caps'], 'test_policy': 'Test0'}
    manifest['source_files_sha256'] = {name: sha(ROOT / name) for name in (
        'codes/minimal_ranking_supervision.py', 'tools/run_minimal_ranking_supervision.py',
        'codes/td_distill_model_no_projection.py', 'codes/initialization_kd_adapter.py',
        'codes/initialization_kd_fast_eval.py')}
    save_json(output / 'manifest.json', manifest)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = '2022'
    status = 'failed'
    with (output / 'stdout.log').open('w', encoding='utf-8') as stdout, \
         (output / 'stderr.log').open('w', encoding='utf-8') as stderr, \
         contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            run(profile, output, args.formal, started, manifest)
            status = 'completed'
        except BaseException:
            traceback.print_exc()
            save_json(output / 'acceptance.json', {'status': 'failed',
                'reason': 'See stderr.log; partial assets preserved, no retry'})
        finally:
            save_json(output / 'exit.json', {'status': status,
                'exit_code': 0 if status == 'completed' else 1,
                'elapsed_seconds': time.monotonic() - started})
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
