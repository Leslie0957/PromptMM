"""Sealed four-arm launcher; disabled until a later committed declaration.

No real input is opened by import or by an invocation while config is in
preparation_only_not_launchable status. Never imports legacy data/evaluation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import secrets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from initialization_kd_adapter import (  # noqa: E402
    check_budget, evaluate_validation, load_pinned_tensors, load_train_val,
    require_hash, sha256)
from initialization_kd_interaction import (  # noqa: E402
    ARMS, Protocol, install_test_read_denial, interaction, reserve_attempt,
    run_arm)
from initialization_kd_runtime import (bind_worker, digest_json, resolve_protocol,
                                       validate_artifacts, validate_environment)

CONFIG = ROOT / 'docs/research/INITIALIZATION_KD_INTERACTION_CONFIG_2026-09-27.json'


def _json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    # Windows readers may briefly deny replacement while holding the old file.
    # Retry publication of these same bytes only, never the worker or experiment.
    for attempt in range(11):
        try:
            temporary.replace(path)
            return
        except OSError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 10:
                raise
            time.sleep(0.05)  # at most 0.5 seconds of requested waiting


def _source_gate(spec):
    if spec['status'] not in ('launchable', 'manual_resource_smoke_ready') or not spec.get('launch_command'):
        raise RuntimeError('Preparation only: no authorized launch command')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if branch != spec['launch_branch'] or spec.get('launch_source_rule') != 'committed_head':
        raise RuntimeError('Launch branch/source rule mismatch')
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if any(line and line != '?? check/' for line in dirty.splitlines()):
        raise RuntimeError('Launch source is dirty')
    validate_environment(spec['environment'])
    resolve_protocol(spec)
    return head


def _asset_gate(spec):
    assets = spec['common']['assets']
    for name in ('shared_cache', 'random_initial', 'triplet_tape',
                 'teacher_checkpoint_record_only', 'train_mat_record_only',
                 'val_mat_record_only'):
        require_hash(ROOT / assets[name]['path'], assets[name]['sha256'])


def _budget_from_spec(spec):
    return spec['hard_caps']


def _worker(spec, output, manifest):
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
    _asset_gate(spec)
    protocol = resolve_protocol(spec)
    torch.manual_seed(spec['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    assets = spec['common']['assets']
    train, val = load_train_val(
        data_root, assets['train_mat_record_only']['sha256'],
        assets['val_mat_record_only']['sha256'], protocol.n_users, protocol.n_items)
    random, teacher, semantics, tape = load_pinned_tensors(ROOT, spec, Protocol())
    smoke = spec.get('mode') == 'resource_smoke'
    arms = ('T1',) if smoke else ARMS
    if smoke:
        tape = tape[:1, :8]
    started = time.monotonic()
    process = psutil.Process()
    last_check = [0.0]
    peak_rss = [0]
    def budget():
        current = time.monotonic()
        if current - last_check[0] < 1.0:
            return
        measured = check_budget(started, output, caps, process.memory_info().rss,
                                torch.cuda.memory_reserved(device),
                                shutil.disk_usage(output).free)
        _json(output / 'telemetry.json', measured)
        peak_rss[0] = max(peak_rss[0], measured['process_rss_bytes'])
        last_check[0] = current

    final = {}
    status = {'status': 'partial', 'arms': {}, 'test_file_reads': 0,
              'selection_split': 'Validation', 'source_commit': manifest['source_commit'],
              'config_digest': digest_json(spec), 'mode': spec.get('mode', 'four_arm'),
              'device': torch.cuda.get_device_name(device), 'validation_calls': 0,
              'timings': {}}
    _json(output / 'report.json', status)
    for arm in arms:
        arm_dir = output / arm
        arm_dir.mkdir()
        def validation(model, epoch):
            if smoke and epoch == 0:
                return {'skipped': 'resource_smoke_no_epoch_zero_ranking'}
            budget()
            torch.cuda.synchronize()
            evaluation_started = time.monotonic()
            result = evaluate_validation(model, train, val)
            torch.cuda.synchronize()
            status['timings'][f'{arm}_validation_epoch{epoch}'] = time.monotonic() - evaluation_started
            status['validation_calls'] += 1
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
                                'source_commit': manifest['source_commit'],
                                'config_digest': digest_json(spec),
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, temporary)
                    temporary.replace(target)
            budget()
        arm_started = time.monotonic()
        print(f'{arm}: starting {protocol.epochs} epoch(s), {protocol.batches_per_epoch} batches/epoch', flush=True)
        result = run_arm(arm, random, teacher, semantics, tape, validation,
                         protocol, device=device, epoch_callback=epoch_record,
                         budget_check=budget)
        torch.cuda.synchronize()
        status['timings'][arm + '_total_seconds'] = time.monotonic() - arm_started
        status['arms'][arm] = {
            'initial_validation': result['initial_validation'],
            'selection': result['selection'],
            'final_validation': result['curve'][-1],
            'optimizer_steps': result['optimizer_steps'],
            'best_checkpoint': (arm_dir / 'best.pt').relative_to(output).as_posix(),
            'final_checkpoint': (arm_dir / 'final.pt').relative_to(output).as_posix()}
        _json(output / 'report.json', status)
        final[arm] = result['curve'][-1]['recall20']
        del result
        torch.cuda.empty_cache()
    if not smoke:
        status['interaction'] = interaction(final)
    status['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
    status['peak_sampled_rss_bytes'] = peak_rss[0]
    status['runtime_seconds'] = time.monotonic() - started
    status['status'] = 'completed'
    _json(output / 'report.json', status)
    # Inspect checkpoint contents in the supervised worker, then let the parent
    # independently verify hashes. A failed acceptance never becomes completed.
    hashes = validate_artifacts(output, arms, protocol, manifest['source_commit'], digest_json(spec))
    if _source_gate(spec) != manifest['source_commit']:
        raise RuntimeError('Source changed during worker execution')
    _json(output / 'acceptance.json', {'status': 'passed', 'hashes': hashes,
                                      'source_commit': manifest['source_commit'],
                                      'config_digest': digest_json(spec)})
    last_check[0] = 0.0
    budget()


def _accept_completion(spec, output):
    manifest = json.loads((output / 'launch_manifest.json').read_text(encoding='utf-8'))
    acceptance = json.loads((output / 'acceptance.json').read_text(encoding='utf-8'))
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    arms = ('T1',) if spec.get('mode') == 'resource_smoke' else ARMS
    expected = {'report.json'} | {f'{arm}/{name}' for arm in arms for name in ('curve.json', 'best.pt', 'final.pt')}
    if (acceptance.get('status') != 'passed' or set(acceptance.get('hashes', {})) != expected
            or acceptance.get('source_commit') != manifest['source_commit']
            or acceptance.get('config_digest') != digest_json(spec)
            or report.get('status') != 'completed'
            or report.get('source_commit') != manifest['source_commit']
            or report.get('config_digest') != digest_json(spec)
            or set(report.get('arms', {})) != set(arms)):
        raise RuntimeError('Missing or mismatched artifact acceptance')
    expected_calls = 1 if spec.get('mode') == 'resource_smoke' else 1204
    if report.get('validation_calls') != expected_calls:
        raise RuntimeError('Validation call count mismatch')
    if (report.get('peak_cuda_reserved_bytes', float('inf')) > spec['hard_caps']['cuda_allocator_bytes']
            or report.get('peak_sampled_rss_bytes', float('inf')) > spec['hard_caps']['process_rss_bytes']):
        raise RuntimeError('Peak resource cap exceeded')
    for relative, expected_hash in acceptance['hashes'].items():
        require_hash(output / relative, expected_hash)
    return acceptance


def _supervise(spec, output, command=None, poll_seconds=1.0, token=None, started=None,
               completion_check=None):
    import psutil

    caps = _budget_from_spec(spec)
    if command is None:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', str(output)]
    if started is None:
        started = time.monotonic()
    environment_vars = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONHASHSEED='2022')
    log = (output / 'worker.log').open('x', encoding='utf-8')
    try:
        proc = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                stdout=log, stderr=subprocess.STDOUT,
                                env=environment_vars, text=True)
    except BaseException:
        log.close()
        raise
    failure = None
    last_progress = started
    try:
        if token is not None:
            proc.stdin.write(token + '\n')
            proc.stdin.flush()
        proc.stdin.close()
        while proc.poll() is None:
            if time.monotonic() - last_progress >= 30:
                print(f'Resource smoke/cohort running: {time.monotonic() - started:.0f}s; details in worker.log', flush=True)
                last_progress = time.monotonic()
            try:
                rss = psutil.Process(proc.pid).memory_info().rss
                telemetry = output / 'telemetry.json'
                cuda = json.loads(telemetry.read_text(encoding='utf-8')).get('cuda_allocator_bytes', 0) if telemetry.exists() else 0
                check_budget(started, output, caps, rss, cuda, shutil.disk_usage(output).free)
            except psutil.NoSuchProcess:
                break
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
        log.close()
    if not failure and code == 0:
        try:
            check_budget(started, output, caps, 0, 0, shutil.disk_usage(output).free)
            (completion_check or _accept_completion)(spec, output)
            check_budget(started, output, caps, 0, 0, shutil.disk_usage(output).free)
        except Exception as exc:
            failure = 'Final acceptance failed: ' + str(exc)
    if failure or code:
        _json(output / 'supervisor.json', {'status': 'failed', 'exit_code': code,
                                           'reason': failure or 'worker exited nonzero'})
        raise RuntimeError(failure or 'Worker exited nonzero; partial attempt preserved')
    _json(output / 'supervisor.json', {'status': 'completed', 'exit_code': code,
                                      'elapsed_seconds': time.monotonic() - started,
                                      'artifact_acceptance': 'passed'})
    print('Completed; artifact acceptance passed. No next stage will start.', flush=True)


def main(spec=None, entry_path=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if spec is None:
        spec = json.loads(CONFIG.read_text(encoding='utf-8'))
    head = _source_gate(spec)
    output = ROOT / spec['output_namespace']
    if args.worker is None:
        started = time.monotonic()
        if shutil.disk_usage(ROOT).free < spec['hard_caps']['minimum_free_disk_bytes']:
            raise RuntimeError('Insufficient free disk')
        reserve_attempt(output)
        token = secrets.token_hex(32)
        _json(output / 'launch_manifest.json', {'source_commit': head,
                                                'assets': spec['common']['assets'],
                                                'config': spec, 'config_digest': digest_json(spec),
                                                'parent_pid': os.getpid(),
                                                'token_sha256': hashlib.sha256(token.encode()).hexdigest()})
        command = [sys.executable, '-B', str(entry_path or Path(__file__).resolve()), '--worker', str(output)]
        print('Reserved one attempt: ' + str(output), flush=True)
        try:
            _supervise(spec, output, command=command, token=token, started=started)
        except BaseException as exc:
            if not (output / 'supervisor.json').exists():
                _json(output / 'supervisor.json', {'status': 'failed', 'reason': repr(exc)})
            raise
    else:
        if sys.stdin.isatty():
            raise RuntimeError('Internal worker requires parent pipe')
        token = sys.stdin.readline().strip()
        manifest = bind_worker(args.worker, output, spec, head, token, os.getppid())
        try:
            _worker(spec, args.worker, manifest)
        except Exception as exc:
            _json(output / 'failure.json', {'status': 'failed', 'reason': repr(exc)})
            raise


if __name__ == '__main__':
    main()
