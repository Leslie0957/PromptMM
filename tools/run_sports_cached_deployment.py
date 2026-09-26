"""Manual, one-attempt, serial cached-deployment benchmark. --dry-run is read-only."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
import cached_deployment_benchmark as bench

WORKER_TIMEOUT = 1800
PREFLIGHT_TIMEOUT = 600
SOURCE_PATHS = ('tools/run_sports_cached_deployment.py',
                'codes/cached_deployment_benchmark.py', 'codes/Models_mmlight.py',
                'codes/utility/norm.py', 'codes/utility/hard_token_cache.py',
                'codes/utility/experiment_protocol.py')


def now():
    return datetime.now().astimezone().isoformat()


def source_state(expected=None):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Clean committed source required')
    if expected is not None and expected != head:
        raise RuntimeError('Launch HEAD changed')
    return head


def schedule():
    conditions = []
    for round_id in range(3):
        arms = ('T', 'F', 'B')[round_id:] + ('T', 'F', 'B')[:round_id]
        batches = bench.BATCHES[round_id:] + bench.BATCHES[:round_id]
        for arm in arms:
            conditions.append(dict(round=round_id, arm=arm, phase='offline', batch_size=0))
        for size in batches:
            for arm in arms:
                conditions.append(dict(round=round_id, arm=arm, phase='online', batch_size=size))
    return conditions


def condition_name(c):
    return f"round{c['round']}_{c['phase']}_{c['arm']}_b{c['batch_size']}"


def child_command(index, head):
    return [str(bench.PYTHON), '-B', str(Path(__file__).resolve()),
            '--worker', str(index), '--head', head]


def hardware_snapshot():
    cmd = ['nvidia-smi', '--query-gpu=index,name,driver_version,temperature.gpu,clocks.sm,power.draw,utilization.gpu,memory.used',
           '--format=csv,noheader,nounits']
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        return dict(time=now(), returncode=r.returncode, csv=r.stdout.strip(), error=r.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(time=now(), unavailable=repr(error))


def telemetry(stop, path):
    with path.open('x', encoding='utf-8') as stream:
        while not stop.is_set():
            stream.write(json.dumps(hardware_snapshot()) + '\n')
            stream.flush()
            stop.wait(2)


def configure():
    torch = bench.torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU fallback is not this protocol')
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)
    return dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                numpy=bench.np.__version__, gpu=torch.cuda.get_device_name(0),
                cpu=platform.processor(), platform=platform.platform(),
                num_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                autocast=False, compile=False, hardware=hardware_snapshot())


def worker(index, head):
    source_state(head)
    bench.install_io_guard(bench.OUT)
    batch = json.loads((bench.OUT / 'batch.json').read_text(encoding='utf-8'))
    if batch['launch_commit'] != head or batch['status'] != 'started' or batch['active_worker'] != index:
        raise RuntimeError('Worker not declared by active serial parent')
    for path, expected in batch['source'].items():
        bench.check_hash(ROOT / path, expected)
    c = schedule()[index]
    folder = bench.OUT / condition_name(c)
    folder.mkdir(exist_ok=False)
    path = folder / 'report.json'
    report = dict(status='started', condition=c, started_at=now(), launch_commit=head,
                  optimizer_steps=0, validation_accesses=0, test_accesses=0,
                  quality_evaluations=0, teacher_forwards=0)
    bench.save_json(path, report)
    try:
        report['environment'] = configure()
        bench.check_hash(bench.OUT / 'preflight.json', batch['preflight_sha256'])
        pre = json.loads((bench.OUT / 'preflight.json').read_text(encoding='utf-8'))
        export = pre['exports'][c['arm']]
        bench.check_hash(export['path'], export['sha256'])
        start = time.perf_counter()
        saved = bench.torch.load(export['path'], map_location='cpu', weights_only=True)
        tables = (saved['users'], saved['items'])
        bench.tables_ok(tables)
        report['load_tables_cpu_seconds'] = time.perf_counter() - start
        before = bench.tensor_digest(tables)
        if before != export['tensor_sha256']:
            raise ValueError('Table tensor digest mismatch')
        report['input_export'] = export
        start = time.perf_counter()
        # Offline teacher reference remains CPU-only; not part of GPU residency.
        if not (c['phase'] == 'offline' and c['arm'] == 'T'):
            tables = tuple(t.to('cuda:0') for t in tables)
        bench.torch.cuda.synchronize()
        report['table_transfer_seconds'] = time.perf_counter() - start
        del saved
        with bench.torch.inference_mode():
            if c['phase'] == 'online':
                bench.check_hash(bench.TRAIN, bench.TRAIN_SHA)
                start = time.perf_counter()
                with bench.TRAIN.open('rb') as f:
                    train = bench.pickle.load(f).tocsr()
                report['load_train_seconds'] = time.perf_counter() - start
                bench.check_hash(bench.OUT / 'request_order.npy', pre['request_sha256'])
                order = bench.np.load(bench.OUT / 'request_order.npy', allow_pickle=False)
                if not bench.np.array_equal(order, bench.request_order(train)):
                    raise ValueError('Request order differs from train-only rule')
                bench.measure_online(tables, train, order, c['batch_size'], report, path)
            else:
                bench.measure_offline(c['arm'], tables, report, path, folder)
        if bench.tensor_digest(tables) != before:
            raise RuntimeError('Input tables mutated')
        report.update(status='completed', input_tables_unchanged=True)
        source_state(head)
        if any(name in sys.modules for name in ('main_mmlight', 'utility.batch_test', 'utility.load_data')):
            raise RuntimeError('Forbidden data/training module imported')
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_at'] = now()
        report['working_set_end'] = bench.working_set()
        bench.save_json(path, report)


def preflight_worker(head):
    source_state(head)
    bench.install_io_guard(bench.OUT)
    batch = json.loads((bench.OUT / 'batch.json').read_text(encoding='utf-8'))
    if (batch['launch_commit'] != head or batch['status'] != 'started'
            or batch['active_worker'] is not None or (bench.OUT / 'preflight.json').exists()):
        raise RuntimeError('Preflight is not a fresh declared attempt')
    bench.save_json(bench.OUT / 'preflight.json', bench.preflight())
    source_state(head)


def aggregate(records):
    groups = {}
    for record in records:
        c = record['condition']
        key = f"{c['phase']}_{c['arm']}_b{c['batch_size']}"
        groups.setdefault(key, []).append(record)
    result = {}
    for key, values in groups.items():
        if len(values) != 3:
            raise ValueError('Missing measurement round')
        medians = [v['summary']['median_ms'] for v in values]
        result[key] = dict(round_median_ms=medians, median_of_round_medians_ms=float(bench.np.median(medians)),
                           min_round_median_ms=min(medians), max_round_median_ms=max(medians),
                           round_p95_ms=[v['summary']['p95_ms'] for v in values])
        if values[0]['condition']['phase'] == 'online':
            result[key]['round_users_per_second'] = [v['summary']['users_per_second'] for v in values]
    return result


def verify_worker(c, path):
    r = json.loads(path.read_text(encoding='utf-8'))
    if r['status'] != 'completed' or r['condition'] != c or not r['input_tables_unchanged']:
        raise RuntimeError('Invalid worker outcome')
    if any(r[k] != 0 for k in ('optimizer_steps', 'validation_accesses', 'test_accesses', 'quality_evaluations')):
        raise RuntimeError('Unexpected update or evaluation')
    count = bench.ONLINE_RUNS if c['phase'] == 'online' else bench.OFFLINE_RUNS
    if len(r['raw_wall_ms']) != count:
        raise RuntimeError('Incomplete timing measurements')
    bench.summary(r['raw_wall_ms'])
    if c['phase'] == 'online':
        if len(r['gemm_only_ms']) != bench.ONLINE_RUNS or r['teacher_forwards'] != 0:
            raise RuntimeError('Invalid online measurement')
        bench.summary(r['gemm_only_ms'])
    elif not all(x['allclose'] for x in r['reference_comparison']):
        raise RuntimeError('Offline output/reference mismatch')
    return r


def run(confirmed):
    if not confirmed:
        raise RuntimeError('Use --conditions-confirmed only after applying the documented desktop conditions')
    if Path(sys.executable).resolve() != bench.PYTHON.resolve():
        raise RuntimeError('Declared run_5060 Python required')
    head = source_state()
    bench.OUT.mkdir(parents=True, exist_ok=False)
    bench.install_io_guard(bench.OUT)
    path = bench.OUT / 'batch.json'
    report = dict(status='started', launch_commit=head, launch_dirty=False,
                  branch=subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
                  started_at=now(), python=sys.executable, active_worker=None,
                  source={p: bench.sha256(ROOT / p) for p in SOURCE_PATHS},
                  conditions_confirmed_at_launch=True,
                  uninterrupted_conditions_verified=False,
                  conditions='AC power, fixed performance mode, terminal visible/foreground, no other fullscreen app or GPU job',
                  schedule=schedule(), completed=[], worker_timeout_seconds=WORKER_TIMEOUT,
                  preflight_timeout_seconds=PREFLIGHT_TIMEOUT,
                  interpretation='Descriptive timings only; inspect telemetry and report disturbances before claiming speedup')
    bench.save_json(path, report)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    stop = threading.Event()
    monitor = None
    try:
        precommand = [str(bench.PYTHON), '-B', str(Path(__file__).resolve()), '--preflight-worker', '--head', head]
        report['preflight_command'] = precommand
        bench.save_json(path, report)
        with (bench.OUT / 'preflight_console.txt').open('x', encoding='utf-8') as log:
            subprocess.run(precommand, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=PREFLIGHT_TIMEOUT)
        report['preflight_sha256'] = bench.sha256(bench.OUT / 'preflight.json')
        source_state(head)
        monitor = threading.Thread(target=telemetry, args=(stop, bench.OUT / 'telemetry.jsonl'), daemon=True)
        monitor.start()
        records = []
        for index, c in enumerate(schedule()):
            source_state(head)
            command = child_command(index, head)
            report.update(active_worker=index, active_command=command)
            bench.save_json(path, report)
            print(f"[{index+1}/36] {condition_name(c)}", flush=True)
            with (bench.OUT / (condition_name(c) + '_console.txt')).open('x', encoding='utf-8') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=WORKER_TIMEOUT)
            source_state(head)
            result_path = bench.OUT / condition_name(c) / 'report.json'
            record = verify_worker(c, result_path)
            if c['phase'] == 'online':
                peers = [r for r in records if r['condition']['phase'] == 'online' and
                         r['condition']['batch_size'] == c['batch_size']]
                if any(r['request_tensor_sha256'] != record['request_tensor_sha256'] for r in peers):
                    raise RuntimeError('Request identities differ across arms/rounds')
            records.append(record)
            report['completed'].append(dict(condition=c, report=str(result_path), sha256=bench.sha256(result_path)))
            bench.save_json(path, report)
        pre = json.loads((bench.OUT / 'preflight.json').read_text(encoding='utf-8'))
        for asset, expected in pre['assets'].items():
            bench.check_hash(asset, expected)
        for export in pre['exports'].values():
            bench.check_hash(export['path'], export['sha256'])
        bench.check_hash(bench.OUT / 'request_order.npy', pre['request_sha256'])
        bench.check_hash(bench.OUT / 'preflight.json', report['preflight_sha256'])
        report['summary'] = aggregate(records)
        report.update(status='completed', active_worker=None, source_unchanged=True,
                      final_assets_unchanged=True)
        source_state(head)
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=8)
        telemetry_path = bench.OUT / 'telemetry.jsonl'
        if telemetry_path.exists() and (monitor is None or not monitor.is_alive()):
            report['telemetry'] = dict(path=str(telemetry_path), sha256=bench.sha256(telemetry_path))
        else:
            report['telemetry'] = dict(unavailable_or_incomplete=True)
        report['finished_at'] = now()
        bench.save_json(path, report)
        print('Preserved batch report:', path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--conditions-confirmed', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(36), help=argparse.SUPPRESS)
    parser.add_argument('--preflight-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--head', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dry_run:
        print('No assets opened or output created; 1 CPU preflight + 36 isolated GPU workers.')
        for index, c in enumerate(schedule()):
            print(index, condition_name(c))
    elif args.preflight_worker:
        if not args.head:
            parser.error('Internal worker requires launch HEAD')
        preflight_worker(args.head)
    elif args.worker is not None:
        if not args.head:
            parser.error('Internal worker requires launch HEAD')
        worker(args.worker, args.head)
    else:
        run(args.conditions_confirmed)


if __name__ == '__main__':
    main()
