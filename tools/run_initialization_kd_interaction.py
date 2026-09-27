"""Sealed four-arm launcher; disabled until a later committed declaration.

No real input is opened by import or by an invocation while config is in
preparation_only_not_launchable status. Never imports legacy data/evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from initialization_kd_adapter import (  # noqa: E402
    check_budget, evaluate_validation, load_pinned_tensors, load_train_val,
    require_hash)
from initialization_kd_interaction import (  # noqa: E402
    ARMS, Protocol, install_test_read_denial, interaction, reserve_attempt,
    run_arm)

CONFIG = ROOT / 'docs/research/INITIALIZATION_KD_INTERACTION_CONFIG_2026-09-27.json'


def _json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def _source_gate(spec):
    if spec['status'] != 'launchable' or not spec.get('launch_command'):
        raise RuntimeError('Preparation only: no authorized launch command')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if branch != spec['launch_branch'] or spec.get('launch_source_rule') != 'committed_head':
        raise RuntimeError('Launch branch/source rule mismatch')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if any(line and line[3:] != 'check/' for line in dirty.splitlines()):
        raise RuntimeError('Launch source is dirty')
    return head


def _asset_gate(spec):
    assets = spec['common']['assets']
    for name in ('shared_cache', 'random_initial', 'triplet_tape',
                 'teacher_checkpoint_record_only', 'train_mat_record_only',
                 'val_mat_record_only'):
        require_hash(ROOT / assets[name]['path'], assets[name]['sha256'])


def _budget_from_spec(spec):
    return spec['hard_caps']


def _worker(spec, output):
    import torch
    import psutil

    data_root = ROOT / 'data/sports'
    install_test_read_denial(data_root)
    caps = _budget_from_spec(spec)
    if not torch.cuda.is_available():
        raise RuntimeError('Declared CUDA device unavailable')
    device = torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(
        caps['cuda_allocator_bytes'] / torch.cuda.get_device_properties(device).total_memory,
        device=device)
    protocol = Protocol()
    assets = spec['common']['assets']
    train, val = load_train_val(
        data_root, assets['train_mat_record_only']['sha256'],
        assets['val_mat_record_only']['sha256'], protocol.n_users, protocol.n_items)
    random, teacher, semantics, tape = load_pinned_tensors(ROOT, spec, protocol)
    started = time.monotonic()
    process = psutil.Process()
    last_check = [0.0]
    def budget():
        current = time.monotonic()
        if current - last_check[0] < 1.0:
            return
        measured = check_budget(started, output, caps, process.memory_info().rss,
                                torch.cuda.memory_reserved(device),
                                shutil.disk_usage(output).free)
        _json(output / 'telemetry.json', measured)
        last_check[0] = current

    final = {}
    status = {'status': 'partial', 'arms': {}, 'test_file_reads': 0,
              'selection_split': 'Validation', 'source_commit': _source_gate(spec)}
    _json(output / 'report.json', status)
    for arm in ARMS:
        arm_dir = output / arm
        arm_dir.mkdir()
        def validation(model, epoch):
            budget()
            result = evaluate_validation(model, train, val)
            budget()
            return result
        curve = []
        def epoch_record(name, epoch, metric, model, optimizer, best, last):
            budget()
            curve.append({'epoch': epoch, 'validation': metric})
            _json(arm_dir / 'curve.json', curve)
            for label, save in (('best', best), ('final', last)):
                if save:
                    target = arm_dir / (label + '.pt')
                    temporary = target.with_name(target.name + '.tmp')
                    torch.save({'arm': name, 'epoch': epoch, 'metric': metric,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, temporary)
                    temporary.replace(target)
            budget()
        result = run_arm(arm, random, teacher, semantics, tape, validation,
                         protocol, device=device, epoch_callback=epoch_record,
                         budget_check=budget)
        status['arms'][arm] = {
            'initial_validation': result['initial_validation'],
            'selection': result['selection'],
            'final_validation': result['curve'][-1],
            'optimizer_steps': result['optimizer_steps'],
            'best_checkpoint': str((arm_dir / 'best.pt').relative_to(output)),
            'final_checkpoint': str((arm_dir / 'final.pt').relative_to(output))}
        _json(output / 'report.json', status)
        final[arm] = result['curve'][-1]['recall20']
        del result
        torch.cuda.empty_cache()
    status['interaction'] = interaction(final)
    status['status'] = 'completed'
    _json(output / 'report.json', status)


def _supervise(spec, output, command=None, poll_seconds=1.0):
    import psutil

    caps = _budget_from_spec(spec)
    if command is None:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(output)]
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=ROOT)
    child = psutil.Process(proc.pid)
    failure = None
    try:
        while proc.poll() is None:
            try:
                rss = child.memory_info().rss
                telemetry = output / 'telemetry.json'
                cuda = json.loads(telemetry.read_text(encoding='utf-8')).get('cuda_allocator_bytes', 0) if telemetry.exists() else 0
                check_budget(started, output, caps, rss, cuda, shutil.disk_usage(output).free)
            except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
                failure = str(exc)
                proc.kill()
                break
            time.sleep(poll_seconds)
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if failure or code:
        _json(output / 'supervisor.json', {'status': 'failed', 'exit_code': code,
                                           'reason': failure or 'worker exited nonzero'})
        raise RuntimeError(failure or 'Worker exited nonzero; partial attempt preserved')
    _json(output / 'supervisor.json', {'status': 'completed', 'exit_code': code})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec = json.loads(CONFIG.read_text(encoding='utf-8'))
    head = _source_gate(spec)
    if args.worker is None:
        _asset_gate(spec)
        output = ROOT / spec['output_namespace']
        if shutil.disk_usage(ROOT).free < spec['hard_caps']['minimum_free_disk_bytes']:
            raise RuntimeError('Insufficient free disk')
        reserve_attempt(output)
        _json(output / 'launch_manifest.json', {'source_commit': head,
                                                'assets': spec['common']['assets'],
                                                'config': spec})
        _supervise(spec, output)
    else:
        _worker(spec, args.worker)


if __name__ == '__main__':
    main()
