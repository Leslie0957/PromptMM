"""One-shot, read-only Sports Validation ranking-gap cohort."""
import contextlib
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def source_gate(profile):
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    if branch != profile['source_branch'] or profile['source_rule'] != 'committed_head':
        raise RuntimeError('Source branch/rule mismatch')
    if any(not line.startswith('?? archive/reviews/') for line in dirty):
        raise RuntimeError('Tracked source or unexpected path is dirty: ' + repr(dirty))
    return head


def monitor(output, started, caps, torch, device, process, telemetry, force=False):
    now = time.monotonic()
    if not force and telemetry and now - telemetry[-1]['elapsed_seconds'] - started < caps['telemetry_interval_seconds']:
        return
    row = {'elapsed_seconds': now - started, 'rss_bytes': process.memory_info().rss,
           'cuda_allocator_bytes': torch.cuda.memory_reserved(device),
           'free_disk_bytes': shutil.disk_usage(output).free,
           'output_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
    telemetry.append(row)
    write_json(output / 'resource.json', {'samples': telemetry, 'interval_seconds': 1,
        'os_hard_isolation': False})
    if (row['elapsed_seconds'] > caps['wall_seconds'] or row['rss_bytes'] > caps['rss_bytes']
            or row['cuda_allocator_bytes'] > caps['cuda_allocator_bytes']
            or row['output_bytes'] > caps['output_bytes']
            or row['free_disk_bytes'] < caps['minimum_free_disk_bytes']):
        raise RuntimeError('Resource cap exceeded: ' + json.dumps(row))


def score_state(name, user_table, item_table, users, train, val, profile,
                output, started, torch, device, process, telemetry):
    import numpy as np
    from initialization_kd_fast_eval import exact_topk

    dim = profile['dimensions']
    for table, shape in ((user_table, (dim['users'], dim['dim'])),
                         (item_table, (dim['items'], dim['dim']))):
        if tuple(table.shape) != shape or table.dtype != torch.float32 or not torch.isfinite(table).all():
            raise ValueError(name + ' table shape/dtype/finite mismatch')
    batch = profile['evaluation']['user_batch']
    if batch > 256:
        raise ValueError('User batch cap exceeded')
    top = np.empty((len(users), 20), dtype=np.int32)
    hits = np.zeros((len(users), 20), dtype=np.bool_)
    tie_boundary_rows = 0
    item_gpu = item_table.to(device)
    with torch.no_grad():
        for begin in range(0, len(users), batch):
            end = min(begin + batch, len(users))
            selected = torch.as_tensor(users[begin:end], dtype=torch.long)
            scores = (user_table[selected].to(device) @ item_gpu.T).cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite score')
            for local, uid in enumerate(users[begin:end]):
                seen = set(train.indices[train.indptr[uid]:train.indptr[uid + 1]].tolist())
                ranked = exact_topk(scores[local], seen, 20)
                if len(set(ranked)) != 20 or set(ranked) & seen or min(ranked) < 0 or max(ranked) >= dim['items']:
                    raise ValueError('Invalid top20')
                top[begin + local] = ranked
                positives = set(val.indices[val.indptr[uid]:val.indptr[uid + 1]].tolist())
                hits[begin + local] = [item in positives for item in ranked]
                candidate_scores = scores[local].copy()
                candidate_scores[list(seen)] = -np.inf
                threshold = candidate_scores[ranked[-1]]
                if np.count_nonzero(candidate_scores == threshold) > 1:
                    tie_boundary_rows += 1
            monitor(output, started, profile['caps'], torch, device, process, telemetry)
    del item_gpu
    torch.cuda.empty_cache()
    recall = float(np.mean(hits.sum(axis=1) / np.diff(val.indptr)[users], dtype=np.float64))
    np.savez_compressed(output / (name + '_per_user.npz'), users=users, top20=top,
                        hit20=hits, validation_denominator=np.diff(val.indptr)[users])
    monitor(output, started, profile['caps'], torch, device, process, telemetry, True)
    return top, {'recall20': recall, 'tie_boundary_rows': tie_boundary_rows,
                 'users': len(users), 'output': name + '_per_user.npz'}


def execute(profile, output, manifest, started):
    import numpy as np
    import psutil
    import scipy
    import torch
    from initialization_kd_adapter import load_train_val
    from initialization_kd_interaction import install_test_read_denial
    from ranking_gap_diagnostic import thirds, decompose

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(
        profile['caps']['cuda_allocator_bytes'] / torch.cuda.get_device_properties(device).total_memory,
        device=device)
    process = psutil.Process()
    manifest['environment'].update({'torch': torch.__version__, 'numpy': np.__version__,
                                    'scipy': scipy.__version__, 'psutil': psutil.__version__,
                                    'cuda_device': torch.cuda.get_device_name(device)})
    telemetry = []
    install_test_read_denial(ROOT / 'data/sports')
    monitor(output, started, profile['caps'], torch, device, process, telemetry, True)
    phase = {}
    phase_start = time.monotonic()
    assets = [profile['train'], profile['validation'], profile['teacher'], *profile['states']]
    for asset in assets:
        if sha(ROOT / asset['path']) != asset['sha256']:
            raise RuntimeError('SHA mismatch: ' + asset['path'])
        monitor(output, started, profile['caps'], torch, device, process, telemetry)
    train, val = load_train_val(ROOT / 'data/sports', profile['train']['sha256'],
                                profile['validation']['sha256'], 35598, 18357)
    users = np.flatnonzero(np.diff(val.indptr))
    if len(users) != 35598:
        raise ValueError('Validation user count mismatch')
    user_groups = thirds(np.diff(train.indptr))
    item_groups = thirds(np.bincount(train.indices, minlength=train.shape[1]))
    phase['load_seconds'] = time.monotonic() - phase_start
    manifest['assets'] = assets
    manifest['shapes'] = {'train': train.shape, 'validation': val.shape,
                          'teacher_user': [35598, 64], 'teacher_item': [18357, 64]}
    manifest['dtypes'] = {'score': 'float32', 'checkpoint_tables': 'float32'}
    manifest['data_mapping'] = 'row user_id / column item_id shared by original Sports sparse matrices and ID tables'
    write_json(output / 'manifest.json', manifest)
    report = {'status': 'partial', 'states': {}, 'comparisons': {}, 'test_file_reads': 0,
              'model_updates': 0, 'phase_seconds': phase, 'source_commit': manifest['source_commit']}
    write_json(output / 'report.json', report)
    phase_start = time.monotonic()
    cache = torch.load(ROOT / profile['teacher']['path'], map_location='cpu', weights_only=True)
    if set(cache) != {'users', 'items', 'image_items', 'text_items', 'image_users', 'text_users'}:
        raise ValueError('Teacher cache keys mismatch')
    teacher_top, metric = score_state('teacher', cache['users'], cache['items'], users, train, val,
                                      profile, output, started, torch, device, process, telemetry)
    del cache
    report['states']['teacher'] = metric
    write_json(output / 'report.json', report)
    if abs(metric['recall20'] - profile['teacher']['expected_recall20']) > 1e-6:
        raise ValueError('Teacher numeric regression failed')
    all_arrays = {}
    for spec in profile['states']:
        name = str(spec['seed']) + '_' + spec['arm']
        checkpoint = torch.load(ROOT / spec['path'], map_location='cpu', weights_only=True)
        original_manifest = json.loads((ROOT / spec['path']).parents[1].joinpath('launch_manifest.json').read_text(encoding='utf-8'))
        if (original_manifest['config']['seed'] != spec['seed']
                or original_manifest['assets']['shared_cache']['sha256'] != profile['teacher']['sha256']
                or original_manifest['assets']['train_mat_record_only']['sha256'] != profile['train']['sha256']
                or original_manifest['assets']['val_mat_record_only']['sha256'] != profile['validation']['sha256']):
            raise ValueError(name + ' original asset binding mismatch')
        if (checkpoint['arm'] != spec['arm'] or checkpoint['epoch'] != 300
                or checkpoint['source_commit'] != original_manifest['source_commit']
                or checkpoint['config_digest'] != original_manifest['config_digest']
                or checkpoint['metric']['recall20'] != spec['expected_recall20']):
            raise ValueError(name + ' checkpoint provenance mismatch')
        manifest.setdefault('state_provenance', {})[name] = {
            'original_source_commit': original_manifest['source_commit'],
            'original_config_digest': original_manifest['config_digest'],
            'original_manifest': str((ROOT / spec['path']).parents[1].joinpath('launch_manifest.json').relative_to(ROOT)).replace('\\', '/')}
        write_json(output / 'manifest.json', manifest)
        model = checkpoint['model']
        if set(model) != {'user_id_embedding.weight', 'item_id_embedding.weight'}:
            raise ValueError(name + ' model keys mismatch')
        top, metric = score_state(name, model['user_id_embedding.weight'],
                                  model['item_id_embedding.weight'], users, train, val,
                                  profile, output, started, torch, device, process, telemetry)
        del checkpoint, model
        report['states'][name] = metric
        write_json(output / 'report.json', report)
        if abs(metric['recall20'] - spec['expected_recall20']) > 1e-6:
            raise ValueError(name + ' numeric regression failed')
        arrays, means, user_rows, item_rows = decompose(
            teacher_top, top, val, users, user_groups, item_groups)
        if abs(means['recall_student'] - metric['recall20']) > 1e-10:
            raise ValueError(name + ' per-user recall mismatch')
        report['comparisons'][name] = {'metrics': means, 'user_groups': user_rows,
                                       'item_groups': item_rows}
        all_arrays[name] = arrays
        write_json(output / 'report.json', report)
        print(name, 'Recall', metric['recall20'], 'G', means['g'], 'L', means['l'], flush=True)
        monitor(output, started, profile['caps'], torch, device, process, telemetry, True)
    phase['score_and_aggregate_seconds'] = time.monotonic() - phase_start
    phase_start = time.monotonic()
    rng = np.random.default_rng(profile['bootstrap']['rng_seed'])
    keys = list(all_arrays)
    replicates = {name: np.empty((2000, 3), dtype=np.float64) for name in keys}
    for r in range(2000):
        sample = rng.integers(0, len(users), size=len(users))
        for name in keys:
            values = all_arrays[name]
            replicates[name][r] = (values['g'][sample].mean(),
                                    values['l'][sample].mean(),
                                    (values['g'][sample] - values['l'][sample]).mean())
        if r % 20 == 0:
            monitor(output, started, profile['caps'], torch, device, process, telemetry)
    for name in keys:
        bounds = np.quantile(replicates[name], [.025, .975], axis=0)
        report['comparisons'][name]['bootstrap95'] = {
            metric: [float(bounds[0, i]), float(bounds[1, i])]
            for i, metric in enumerate(('G', 'L', 'N'))}
    phase['bootstrap_seconds'] = time.monotonic() - phase_start
    t0 = [report['comparisons'][str(seed) + '_T0']['metrics']['g'] for seed in (2022, 2023, 2024)]
    delta = profile['screen_delta']
    report['screen'] = ('opportunity_present' if all(x >= delta for x in t0) else
                        'screen_stop' if all(x < delta for x in t0) else 'inconclusive')
    report['status'] = 'completed'
    report['elapsed_seconds'] = time.monotonic() - started
    report['peak_sampled_rss_bytes'] = max(x['rss_bytes'] for x in telemetry)
    report['peak_cuda_allocator_bytes'] = torch.cuda.max_memory_reserved(device)
    write_json(output / 'report.json', report)
    if source_gate(profile) != manifest['source_commit']:
        raise RuntimeError('Source changed during diagnostic')
    write_json(output / 'resource.json', {'samples': telemetry, 'interval_seconds': 1,
        'peak_sampled_rss_bytes': report['peak_sampled_rss_bytes'],
        'peak_cuda_allocator_bytes': report['peak_cuda_allocator_bytes'],
        'elapsed_seconds': report['elapsed_seconds'], 'os_hard_isolation': False})
    monitor(output, started, profile['caps'], torch, device, process, telemetry, True)
    write_json(output / 'acceptance.json', {'status': 'passed', 'seven_states': len(report['states']) == 7,
        'six_comparisons': len(report['comparisons']) == 6, 'numeric_tolerance': 1e-6,
        'identity_tolerance': 1e-10, 'test_file_reads': 0, 'model_updates': 0,
        'screen': report['screen']})


def main():
    profile_path = ROOT / 'docs/research/innovation2/RANKING_GAP_PROFILE_V1.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    head = source_gate(profile)
    output = ROOT / profile['output']
    if output.exists():
        raise RuntimeError('Output namespace exists; no retry or overwrite')
    if shutil.disk_usage(ROOT).free < profile['caps']['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient free disk')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {'source_commit': head, 'branch': profile['source_branch'],
        'command': profile['command'], 'profile': str(profile_path.relative_to(ROOT)).replace('\\', '/'),
        'profile_sha256': sha(profile_path), 'comparison_anchor': profile['comparison_anchor'],
        'environment': {'python': platform.python_version(), 'executable': sys.executable},
        'protocol': profile['evaluation'], 'screen_delta': profile['screen_delta'],
        'caps': profile['caps'], 'bootstrap': profile['bootstrap'], 'test_policy': 'Test0'}
    write_json(output / 'manifest.json', manifest)
    status = 'failed'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = '2022'
    with (output / 'stdout.log').open('w', encoding='utf-8') as stdout, \
         (output / 'stderr.log').open('w', encoding='utf-8') as stderr, \
         contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            execute(profile, output, manifest, started)
            status = 'completed'
        except BaseException:
            traceback.print_exc()
            write_json(output / 'acceptance.json', {'status': 'failed',
                'reason': 'See stderr.log and partial report; no retry authorized',
                'test_file_reads': 0})
        finally:
            write_json(output / 'exit.json', {'status': status,
                'exit_code': 0 if status == 'completed' else 1,
                'elapsed_seconds': time.monotonic() - started})
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
