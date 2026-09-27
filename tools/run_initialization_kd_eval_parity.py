"""User-manual evaluation-only parity/timing; no optimizer or training entry."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from initialization_kd_adapter import require_hash, load_train_val, check_budget, sha256
from initialization_kd_interaction import install_test_read_denial, reserve_attempt, Protocol
from initialization_kd_runtime import digest_json, bind_worker, resolve_protocol
from initialization_kd_eval_parity import compare_evaluators
from run_initialization_kd_resource_smoke import load_spec as smoke_spec
from run_initialization_kd_interaction import _json, _source_gate, _supervise


def load_spec():
    delta = json.loads((ROOT / 'docs/research/INITIALIZATION_KD_EVAL_PARITY_CONFIG_2026-09-27.json').read_text(encoding='utf-8'))
    base = smoke_spec()
    resolve_protocol(base)
    spec = dict(base, **delta)
    for key in ('common', 'seed', 'arms', 'environment', 'numerics', 'hard_caps'):
        if spec[key] != base[key]:
            raise RuntimeError('Undeclared common/resource change: ' + key)
    if spec['mode'] != 'evaluation_parity' or spec['comparison'] != {
            'passes': 2, 'order': ['reference', 'fast'], 'user_batch': 256,
            'ks': [10, 20, 40, 50], 'score_identity': 'all_float32_score_row_bytes_equal',
            'ranking_identity': 'all_users_ordered_top50_equal',
            'metric_identity': 'exact_all_report_fields', 'training_steps': 0, 'test_reads': 0}:
        raise RuntimeError('Parity contract mismatch')
    return spec


def accept(spec, output):
    manifest = json.loads((output / 'launch_manifest.json').read_text(encoding='utf-8'))
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    receipt = json.loads((output / 'acceptance.json').read_text(encoding='utf-8'))
    if (receipt.get('report_sha256') != sha256(output / 'report.json') or receipt.get('config_digest') != digest_json(spec)
            or report.get('status') != 'completed' or report.get('source_commit') != manifest['source_commit']
            or report.get('config_digest') != digest_json(spec)
            or report.get('training_steps') != 0 or report.get('test_reads') != 0):
        raise RuntimeError('Incomplete parity report/source/hash')
    parity = report['parity']
    import numpy as np
    ranks = []
    for name in ('reference_top50.npy', 'fast_top50.npy'):
        require_hash(output / name, receipt['ranks'][name])
        value = np.load(output / name, allow_pickle=False)
        if value.shape != (35598, 50) or value.dtype != np.dtype('int32'):
            raise RuntimeError('Saved rank matrix invalid')
        ranks.append(value)
    if not np.array_equal(*ranks) or hashlib.sha256(ranks[1].tobytes()).hexdigest() != parity['rank_sha256']:
        raise RuntimeError('Saved rank parity/hash mismatch')
    if (parity['status'] != 'passed' or parity['users_checked'] != 35598 or parity['k'] != 50
            or not all(parity[k] is True for k in ('score_rows_exact', 'ordered_top50_exact', 'metrics_exact'))
            or parity['reference_metrics'] != parity['fast_metrics']):
        raise RuntimeError('Parity acceptance failed')
    for key in ('reference_seconds', 'fast_seconds', 'speedup_observed'):
        if not math.isfinite(parity[key]) or parity[key] <= 0:
            raise RuntimeError('Invalid timing')
    caps = spec['hard_caps']
    if report['peak_cuda_reserved_bytes'] > caps['cuda_allocator_bytes'] or report['peak_sampled_rss_bytes'] > caps['process_rss_bytes']:
        raise RuntimeError('Resource cap exceeded')
    return receipt


def worker(spec, output, manifest):
    import torch
    import psutil
    install_test_read_denial(ROOT / 'data/sports')
    checkpoint_path = ROOT / spec['checkpoint']['path']
    require_hash(checkpoint_path, spec['checkpoint']['sha256'])
    saved = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    if saved['source_commit'] != spec['checkpoint']['source_commit'] or saved['arm'] != 'T1' or saved['epoch'] != 1:
        raise RuntimeError('Checkpoint source/identity mismatch')
    state = saved['model']
    del saved  # do not create or reuse any optimizer
    p = Protocol()
    shapes = {'user_id_embedding.weight': (p.n_users, p.dim), 'item_id_embedding.weight': (p.n_items, p.dim)}
    if set(state) != set(shapes):
        raise RuntimeError('Checkpoint keys mismatch')
    for name, shape in shapes.items():
        if state[name].shape != shape or state[name].dtype != torch.float32 or not torch.isfinite(state[name]).all():
            raise RuntimeError('Checkpoint tensor invalid')
    assets = spec['common']['assets']
    train, val = load_train_val(ROOT / 'data/sports', assets['train_mat_record_only']['sha256'],
                               assets['val_mat_record_only']['sha256'], p.n_users, p.n_items)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda:0')
    caps = spec['hard_caps']
    torch.cuda.set_per_process_memory_fraction(caps['cuda_allocator_bytes'] / torch.cuda.get_device_properties(device).total_memory, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    model = SimpleNamespace(user_id_embedding=SimpleNamespace(weight=state['user_id_embedding.weight'].to(device)),
                            item_id_embedding=SimpleNamespace(weight=state['item_id_embedding.weight'].to(device)))
    started = time.monotonic()
    last = [0.0]
    peak_rss = [0]
    def check():
        if time.monotonic() - last[0] < 1:
            return
        measured = check_budget(started, output, caps, psutil.Process().memory_info().rss,
                                torch.cuda.memory_reserved(device), shutil.disk_usage(output).free)
        peak_rss[0] = max(peak_rss[0], measured['process_rss_bytes'])
        _json(output / 'telemetry.json', measured)
        last[0] = time.monotonic()
    print('Reference then fast: 2 complete Validation passes; training steps 0.', flush=True)
    def capture(reference, fast):
        import numpy as np
        for name, value in (('reference_top50.npy', reference), ('fast_top50.npy', fast)):
            with (output / name).open('xb') as stream:
                np.save(stream, value, allow_pickle=False)
    parity = compare_evaluators(model, train, val, torch.cuda.synchronize, check, capture)
    last[0] = 0
    check()
    if _source_gate(spec) != manifest['source_commit']:
        raise RuntimeError('Source changed during parity check')
    report = {'status': 'completed', 'source_commit': manifest['source_commit'],
              'config_digest': digest_json(spec), 'checkpoint': spec['checkpoint'],
              'training_steps': 0, 'test_reads': 0, 'parity': parity,
              'device': torch.cuda.get_device_name(device),
              'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device),
              'peak_sampled_rss_bytes': peak_rss[0]}
    _json(output / 'report.json', report)
    _json(output / 'acceptance.json', {'report_sha256': sha256(output / 'report.json'),
                                      'config_digest': digest_json(spec),
                                      'ranks': {name: sha256(output / name) for name in ('reference_top50.npy', 'fast_top50.npy')}})
    accept(spec, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec = load_spec()
    head = _source_gate(spec)
    output = ROOT / spec['output_namespace']
    if args.worker is not None:
        if sys.stdin.isatty():
            raise RuntimeError('Internal worker requires parent pipe')
        token = sys.stdin.readline().strip()
        manifest = bind_worker(args.worker, output, spec, head, token, os.getppid())
        try:
            worker(spec, output, manifest)
        except Exception as exc:
            _json(output / 'failure.json', {'status': 'failed', 'reason': repr(exc)})
            raise
    else:
        if shutil.disk_usage(ROOT).free < spec['hard_caps']['minimum_free_disk_bytes']:
            raise RuntimeError('Insufficient free disk')
        started = time.monotonic()
        reserve_attempt(output)
        token = secrets.token_hex(32)
        _json(output / 'launch_manifest.json', {'source_commit': head, 'config': spec,
              'config_digest': digest_json(spec), 'parent_pid': os.getpid(),
              'token_sha256': hashlib.sha256(token.encode()).hexdigest()})
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(output)]
        try:
            _supervise(spec, output, command=command, token=token, started=started, completion_check=accept)
        except BaseException as exc:
            if not (output / 'supervisor.json').exists():
                _json(output / 'supervisor.json', {'status': 'failed', 'reason': repr(exc)})
            raise


if __name__ == '__main__':
    main()
