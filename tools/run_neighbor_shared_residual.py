"""One-shot serial B/N/F/R x three-seed cohort; --smoke is isolated and capped."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from neighbor_shared_residual import effective_items, features, group_metrics, triplet_loss
from run_minimal_ranking_supervision import load_teacher, load_tape, load_train, save_json, sha, test_denial

PROFILE = ROOT / 'docs/research/innovation2/NEIGHBOR_SHARED_RESIDUAL_PROFILE_V1.json'


def run_serial_arms(arms, call):
    for arm in arms:
        call(arm)


def status_gate(profile, formal):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    if branch != profile['branch'] or (formal and any(not x.startswith('?? archive/reviews/') for x in dirty)):
        raise RuntimeError('Wrong branch or dirty launch source')
    if formal and profile['status'] != 'manual_launch_ready':
        raise RuntimeError('Profile is not launchable')
    return head


def verify_assets(profile):
    for name, spec in profile['assets'].items():
        if sha(ROOT / spec['path']) != spec['sha256']:
            raise RuntimeError(name + ' SHA mismatch')
    for seed in profile['seeds']:
        for name in ('anchor_manifest', 'triplet_tape'):
            spec = profile['seeds'][seed][name]
            if sha(ROOT / spec['path']) != spec['sha256']:
                raise RuntimeError(seed + ' ' + name + ' SHA mismatch')


def check_profile(p):
    if (p['ordered_seeds'] != [2022, 2023, 2024] or p['arms'] != ['B', 'N', 'F', 'R']
            or p['test_policy'] != 'Test0' or p['attempts'] != 1
            or p['training'] != {'epochs': 300, 'batches_per_epoch': 214, 'batch_size': 1024,
                'lr': 0.00006, 'weight_decay': 0.01, 'betas': [0.9, 0.999], 'eps': 1e-8}
            or p['evaluation']['primary'] != 'epoch300_validation_recall20'
            or p['evaluation']['curve_epochs'] != [0, 50, 100, 200, 300]
            or p['evaluation']['screen'] != {'N_minus_B_low_micro_min': .001,
                'N_minus_B_recall_min': -.0002, 'N_minus_F_low_micro_min': .0005,
                'N_minus_R_low_micro_min': .0005, 'gate_queue': False}
            or p['evaluation']['test_access'] is not False):
        raise ValueError('Frozen protocol mismatch')


def monitor(root, start, caps, process, torch, samples):
    now = time.monotonic() - start
    size = (sum(f.stat().st_size for f in root.rglob('*') if f.is_file())
            if not samples or now - samples[-1]['seconds'] >= caps['sample_interval_seconds']
            else samples[-1]['output_bytes'])
    row = {'seconds': now, 'rss_bytes': process.memory_info().rss,
           'cuda_allocator_bytes': torch.cuda.memory_reserved('cuda:0'),
           'output_bytes': size,
           'free_disk_bytes': shutil.disk_usage(root).free}
    if not samples or row['seconds'] - samples[-1]['seconds'] >= caps['sample_interval_seconds']:
        samples.append(row)
    if (row['seconds'] > caps['wall_seconds'] or row['rss_bytes'] > caps['rss_bytes']
            or row['cuda_allocator_bytes'] > caps['cuda_allocator_bytes']
            or row['output_bytes'] > caps['output_bytes']
            or row['free_disk_bytes'] < caps['minimum_free_disk_bytes']):
        raise RuntimeError('Resource cap breached: ' + json.dumps(row))
    return row


def load_graphs(profile, anchor_items, train):
    import numpy as np
    degree = np.bincount(train.indices, minlength=18357)
    order = np.lexsort((np.arange(len(degree)), degree))
    low = order[:len(degree) // 3]
    pair = []
    for modality in ('image', 'text'):
        spec = profile['assets'][modality + '_graph']
        with np.load(ROOT / spec['path'], allow_pickle=False) as z:
            q, real, random, d = (z[k] for k in ('queries', 'real', 'random', 'degree'))
        if (len(q) != 6114 or not np.array_equal(d, degree)
                or not np.array_equal(np.sort(q), np.sort(np.intersect1d(low, q)))
                or real.shape != (6114, 20) or random.shape != (100, 6114, 20)):
            raise RuntimeError('Frozen graph/Train/low group mismatch')
        pair.append((q, real, random[0]))
    if not np.array_equal(pair[0][0], pair[1][0]):
        raise RuntimeError('Modality query mismatch')
    x_real, scales_real = features(anchor_items, pair[0][0], pair[0][1], pair[1][1])
    x_random, scales_random = features(anchor_items, pair[0][0], pair[0][2], pair[1][2])
    groups = np.empty(len(degree), dtype=np.int8)
    for g, part in enumerate(np.array_split(order, 3)):
        groups[part] = g
    return x_real, x_random, groups, {'valid_low': len(pair[0][0]),
            'low_total': int((groups == 0).sum()), 'real_scales': scales_real,
            'random_scales': scales_random}


def rank(model, train, val, groups, check, full):
    import numpy as np
    import torch
    from initialization_kd_fast_eval import exact_topk
    users = np.flatnonzero(np.diff(val.indptr))
    top = np.empty((len(users), 20), dtype=np.int32) if full else None
    total = 0.
    with torch.no_grad():
        for start in range(0, len(users), 256):
            chunk = users[start:start + 256]
            scores = (model.user_id_embedding.weight[torch.as_tensor(chunk, device='cuda:0')]
                      @ model.item_id_embedding.weight.T).detach().cpu().numpy()
            for row, uid in enumerate(chunk):
                seen = set(train.indices[train.indptr[uid]:train.indptr[uid + 1]].tolist())
                selected = exact_topk(scores[row], seen, 20)
                positive = set(val.indices[val.indptr[uid]:val.indptr[uid + 1]].tolist())
                total += sum(item in positive for item in selected) / len(positive) / len(users)
                if full:
                    top[start + row] = selected
            check()
    result = {'recall20': total}
    if full:
        result['groups'] = group_metrics(top, users, val, groups)
        if abs(result['groups']['recall20'] - total) > 1e-10:
            raise RuntimeError('Group contribution denominator mismatch')
    return result, top


def one_arm(arm, seed, profile, folder, teacher, xreal, xrandom, groups, train, val,
            tape, formal, source, check, samples):
    import numpy as np
    import torch
    from run_minimal_ranking_supervision import model_from_teacher
    arm_started = time.monotonic()
    sample_start = len(samples)
    x = torch.as_tensor(xrandom if arm == 'R' else xreal, device='cuda:0')
    model = model_from_teacher(*teacher, 'cuda:0')
    extra = None if arm == 'B' else torch.nn.Parameter(torch.zeros((2,) if arm == 'F' else (64, 128), device='cuda:0'))
    params = list(model.parameters()) + ([] if extra is None else [extra])
    optimizer = torch.optim.AdamW(params, lr=profile['training']['lr'],
        weight_decay=profile['training']['weight_decay'],
        betas=tuple(profile['training']['betas']), eps=profile['training']['eps'])
    if optimizer.state or not torch.equal(model.user_id_embedding.weight.detach().cpu(), teacher[0]) or not torch.equal(model.item_id_embedding.weight.detach().cpu(), teacher[1]):
        raise RuntimeError('Initial ID or optimizer mismatch')
    record = {'status': 'running', 'steps': 0, 'curve': [], 'checkpoints': {},
              'parameter_count_extra': 0 if extra is None else extra.numel()}
    save_json(folder / 'status.json', record)
    def evaluate(epoch):
        with torch.no_grad():
            item = effective_items(model.item_id_embedding.weight, x, arm, extra)
        view = SimpleNamespace(user_id_embedding=model.user_id_embedding,
                 item_id_embedding=SimpleNamespace(weight=item))
        full = epoch in profile['evaluation']['curve_epochs']
        result, top = rank(view, train, val, groups, check, full)
        if full and result['groups']['denominator'] != [6347, 6389, 25163]:
            raise RuntimeError('Validation group denominators changed')
        if full:
            np.savez_compressed(folder / f'top20_epoch{epoch}.npz',
                users=np.flatnonzero(np.diff(val.indptr)), top20=top)
        return result
    if formal:
        epoch0 = evaluate(0)
        record['epoch0'] = epoch0
        if abs(epoch0['recall20'] - profile['evaluation']['B_epoch0_expected']) > 1e-6:
            raise RuntimeError(arm + ' epoch0 score regression')
    else:
        record['epoch0'] = 'Validation0'
    epochs = 300 if formal else 1
    batches = 214 if formal else 20
    best = -1.
    for epoch in range(1, epochs + 1):
        extra_before = None if extra is None else extra.detach().clone()
        loss_sum = 0.
        for batch in range(batches):
            check()
            ids = torch.as_tensor(np.array(tape[epoch - 1, batch], copy=True), device='cuda:0', dtype=torch.long)
            loss = triplet_loss(model.user_id_embedding.weight, model.item_id_embedding.weight,
                                x, arm, extra, ids)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
                raise RuntimeError('Nonfinite/missing gradient')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in params):
                raise RuntimeError('Nonfinite parameter')
            loss_sum += float(loss.detach())
            record['steps'] += 1
            check()
        row = {'epoch': epoch, 'mean_loss': loss_sum / batches,
               'residual_norm': 0. if extra is None else float(extra.detach().norm()),
               'extra_update_norm': 0. if extra is None else float((extra.detach() - extra_before).norm()),
               'item_drift_norm': float((model.item_id_embedding.weight.detach() - teacher[1].to('cuda:0')).norm())}
        with torch.no_grad():
            row['residual_output_norm'] = float((effective_items(model.item_id_embedding.weight, x, arm, extra)
                                                 - model.item_id_embedding.weight).norm())
        if formal:
            metric = evaluate(epoch)
            row.update(metric)
            if metric['recall20'] > best or epoch == 300:
                for label in (('best' if metric['recall20'] > best else None),
                              ('final' if epoch == 300 else None)):
                    if label:
                        path = folder / f'{label}.pt'
                        torch.save({'arm': arm, 'seed': seed, 'epoch': epoch, 'source': source,
                          'metric': metric, 'user': model.user_id_embedding.weight.detach().cpu(),
                          'item': model.item_id_embedding.weight.detach().cpu(),
                          'extra': None if extra is None else extra.detach().cpu(),
                          'optimizer': optimizer.state_dict()}, path)
                        record['checkpoints'][label] = {'epoch': epoch, 'sha256': sha(path)}
            best = max(best, metric['recall20'])
            if arm == 'B' and epoch == 300 and abs(metric['recall20'] - profile['seeds'][str(seed)]['B_epoch300_expected']) > 1e-6:
                raise RuntimeError('B epoch300 regression')
        record['curve'].append(row)
        save_json(folder / 'status.json', record)
    if formal:
        exported = effective_items(model.item_id_embedding.weight, x, arm, extra).detach()
        with torch.no_grad():
            probe = torch.arange(16, device='cuda:0')
            direct_items = model.item_id_embedding.weight[probe] if arm == 'B' else (
                model.item_id_embedding.weight[probe] +
                (extra[0] * x[probe, :64] + extra[1] * x[probe, 64:] if arm == 'F'
                 else torch.nn.functional.linear(x[probe], extra)))
            direct = model.user_id_embedding.weight[:16] @ direct_items.T
            folded = model.user_id_embedding.weight[:16] @ exported[probe].T
            if not torch.allclose(direct, folded, rtol=1e-6, atol=1e-6):
                raise RuntimeError('Export score mismatch')
        torch.save({'user': model.user_id_embedding.weight.detach().cpu(), 'item': exported.cpu()}, folder / 'export.pt')
    record['status'] = 'completed'
    arm_samples = samples[sample_start:] or samples[-1:]
    record['resource_peak_sampled'] = {key: max(s[key] for s in arm_samples)
        for key in ('rss_bytes','cuda_allocator_bytes','output_bytes')}
    record['arm_seconds'] = time.monotonic() - arm_started
    save_json(folder / 'status.json', record)
    required = ['status.json'] + (['best.pt', 'final.pt', 'export.pt', 'top20_epoch300.npz'] if formal else [])
    save_json(folder / 'acceptance.json', {'status': 'passed', 'steps': record['steps'],
        'curve_rows': len(record['curve']), 'test_access': 0,
        'files_sha256': {name: sha(folder / name) for name in required}})
    del model, optimizer, extra
    torch.cuda.empty_cache()
    return record


def summarize(output, profile):
    import numpy as np
    rows = {}
    for seed in profile['ordered_seeds']:
        rows[str(seed)] = {}
        for arm in profile['arms']:
            path = output / f'seed{seed}' / arm / 'status.json'
            row = json.loads(path.read_text())
            acceptance = json.loads((path.parent / 'acceptance.json').read_text())
            if acceptance['status'] != 'passed' or acceptance['steps'] != 64200:
                raise RuntimeError('Arm acceptance absent or failed')
            if row['status'] != 'completed' or row['steps'] != 64200 or len(row['curve']) != 300:
                raise RuntimeError('Incomplete arm cannot enter full summary')
            rows[str(seed)][arm] = row['curve'][-1]
    def difference(left, right):
        a, b = left['groups'], right['groups']
        return {'recall20': left['recall20'] - right['recall20'],
            'micro': [x-y for x,y in zip(a['micro'], b['micro'])],
            'hits': [x-y for x,y in zip(a['hits'], b['hits'])],
            'contribution': [x-y for x,y in zip(a['contribution'], b['contribution'])],
            'exposure': [x-y for x,y in zip(a['exposure'], b['exposure'])]}
    paired = {str(seed): {name: difference(rows[str(seed)][a], rows[str(seed)][b])
        for name, a, b in [('N_minus_B','N','B'), ('F_minus_B','F','B'),
                           ('R_minus_B','R','B'), ('N_minus_F','N','F'),
                           ('N_minus_R','N','R')]} for seed in profile['ordered_seeds']}
    screens = {}
    for seed in profile['ordered_seeds']:
        r = rows[str(seed)]
        screens[str(seed)] = ('candidate_signal' if
            paired[str(seed)]['N_minus_B']['micro'][0] >= .001 and
            paired[str(seed)]['N_minus_B']['recall20'] >= -.0002 and
            paired[str(seed)]['N_minus_F']['micro'][0] >= .0005 and
            paired[str(seed)]['N_minus_R']['micro'][0] >= .0005
            else 'screen_stop')
    aggregate = {arm: {key: {'mean': float(np.mean(vals)), 'min': float(np.min(vals)),
                               'max': float(np.max(vals)), 'values': vals}
        for key, vals in {'recall20': [paired[str(s)][arm+'_minus_B']['recall20'] for s in profile['ordered_seeds']],
                          'low_micro': [paired[str(s)][arm+'_minus_B']['micro'][0] for s in profile['ordered_seeds']]}.items()}
        for arm in ('N','F','R')}
    save_json(output / 'summary.json', {'status': 'complete_unreviewed', 'rows': rows,
        'paired': paired, 'three_seed_descriptive': aggregate, 'screens': screens,
        'cohort_screen': 'all_three_candidate_signal' if all(v == 'candidate_signal' for v in screens.values()) else 'screen_stop',
        'interpretation': 'descriptive only; no significance, equivalence or noninferiority'})


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--formal', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    check_profile(profile)
    source = status_gate(profile, args.formal)
    verify_assets(profile)
    output = ROOT / profile['output' if args.formal else 'smoke_output']
    if output.exists():
        raise RuntimeError('Output exists; no overwrite/retry')
    caps = profile['caps' if args.formal else 'smoke_caps']
    if shutil.disk_usage(ROOT).free < caps['minimum_free_disk_bytes']:
        raise RuntimeError('Disk floor failed')
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    state = {'status': 'running', 'source_commit': source, 'profile_sha256': sha(PROFILE),
        'mode': 'formal' if args.formal else 'non_formal_smoke', 'seed_status':
        {str(s): 'not_started' for s in profile['ordered_seeds']}, 'test_access': 0,
        'arm_status': {str(s): {a: 'not_started' for a in profile['arms']}
                       for s in profile['ordered_seeds']},
        'assets': profile['assets'], 'seeds': profile['seeds'], 'caps': caps,
        'environment': {'python': platform.python_version()}, 'arms': profile['arms'],
        'source_files_sha256': {p: sha(ROOT / p) for p in (
            'codes/neighbor_shared_residual.py', 'tools/run_neighbor_shared_residual.py',
            'codes/td_distill_model_no_projection.py', 'codes/initialization_kd_fast_eval.py')}}
    save_json(output / 'manifest.json', state)
    try:
        import numpy as np
        import psutil
        import torch
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(caps['cuda_allocator_bytes'] /
            torch.cuda.get_device_properties(0).total_memory, device='cuda:0')
        state['environment'].update({'torch': torch.__version__, 'numpy': np.__version__,
                                    'cuda': torch.cuda.get_device_name(0)})
        denied = test_denial(ROOT / 'data/sports')
        process = psutil.Process()
        samples = []
        def check():
            monitor(output, start, caps, process, torch, samples)
        check()
        common = dict(assets={'teacher_tables': profile['assets']['teacher'],
                              'train': profile['assets']['train']})
        teacher = load_teacher(common)
        train = load_train(common)
        xreal, xrandom, groups, graph_info = load_graphs(profile, teacher[1].numpy(), train)
        graph_info['feature_sha256'] = {
            'N_F': hashlib.sha256(xreal.tobytes()).hexdigest(),
            'R': hashlib.sha256(xrandom.tobytes()).hexdigest()}
        state['graphs'] = graph_info
        save_json(output / 'manifest.json', state)
        if args.formal:
            from initialization_kd_adapter import load_train_val
            train, val = load_train_val(ROOT / 'data/sports', profile['assets']['train']['sha256'],
                    profile['assets']['validation']['sha256'], 35598, 18357)
        else:
            val = None
        current_seed = None
        current_arm = None
        for seed in (profile['ordered_seeds'] if args.formal else [2022]):
            current_seed = seed
            state['seed_status'][str(seed)] = 'running'
            save_json(output / 'manifest.json', state)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            spec = profile['seeds'][str(seed)]
            tape = load_tape({'assets': {'triplet_tape': spec['triplet_tape']}})
            def launch_arm(arm):
                nonlocal current_arm
                current_arm = arm
                state['arm_status'][str(seed)][arm] = 'running'
                save_json(output / 'manifest.json', state)
                folder = output / f'seed{seed}' / arm
                folder.mkdir(parents=True, exist_ok=False)
                arm_start = time.monotonic()
                def check_arm():
                    check()
                    if args.formal and time.monotonic() - arm_start > 28800:
                        raise RuntimeError('Eight-hour arm wall cap breached')
                one_arm(arm, seed, profile, folder, teacher, xreal, xrandom, groups,
                        train, val, tape, args.formal, source, check_arm, samples)
                check()
                if args.formal and status_gate(profile, True) != source:
                    raise RuntimeError('Source changed during cohort')
                state['arm_status'][str(seed)][arm] = 'completed'
                save_json(output / 'manifest.json', state)
            run_serial_arms(profile['arms'], launch_arm)
            state['seed_status'][str(seed)] = 'completed'
            save_json(output / 'manifest.json', state)
        if args.formal:
            summarize(output, profile)
        state['status'] = 'completed'
        state['test_access'] = denied['denied_test_open_attempts']
        save_json(output / 'resource.json', {'samples': samples, 'peak_rss_bytes': max(s['rss_bytes'] for s in samples),
            'peak_cuda_allocator_bytes': max(s['cuda_allocator_bytes'] for s in samples),
            'seconds': time.monotonic() - start})
        save_json(output / 'manifest.json', state)
        code = 0
    except BaseException:
        state['status'] = 'failed'
        if 'current_seed' in locals() and current_seed is not None:
            state['seed_status'][str(current_seed)] = 'failed'
            if current_arm is not None:
                state['arm_status'][str(current_seed)][current_arm] = 'failed'
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        if 'samples' in locals() and samples:
            save_json(output / 'resource.json', {'samples': samples, 'status': 'partial_failed',
                'peak_rss_bytes_sampled': max(s['rss_bytes'] for s in samples),
                'peak_cuda_allocator_bytes_sampled': max(s['cuda_allocator_bytes'] for s in samples),
                'seconds': time.monotonic() - start})
        save_json(output / 'manifest.json', state)
        code = 1
    save_json(output / 'exit.json', {'status': state['status'], 'exit_code': code,
        'seconds': time.monotonic() - start, 'seed_status': state['seed_status']})
    (output / 'HANDOFF.md').write_text(
        '# Neighbor shared residual cohort handoff\n\n'
        f"Status: {state['status']}; source: `{source}`; mode: {state['mode']}; Test access: {state['test_access']}.\n\n"
        'Review manifest.json, exit.json, per-arm status/acceptance/curve/checkpoints/top20, '
        'resource.json, summary.json and completeness.json when present. '
        'All raw files in this directory must be preserved. Machine summaries are not '
        'an independent audit. Do not rerun, resume, or infer significance, equivalence, '
        'or noninferiority from three seeds.\n', encoding='utf-8')
    files = {str(p.relative_to(output)).replace('\\','/'): sha(p) for p in output.rglob('*')
             if p.is_file() and p.name != 'completeness.json'}
    save_json(output / 'completeness.json', {'status': 'complete_unreviewed' if code == 0 else 'partial_failed',
        'files_sha256': files, 'test_access': state['test_access'], 'independent_audit': False,
        'excluded_self_hash': 'completeness.json'})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
