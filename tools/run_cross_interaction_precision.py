"""Manual P0 precision review of the same float32 updates; no retry or next stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/cross_interaction/p0_precision_warm2022_v1'
OLD = ROOT / 'exp/cross_interaction/p0_warm2022_v1'
OLD_HASHES = {
    'plan.json': '11130dd5cc71ee2da45a485bb04c3946a6d5e9f310940019cc219fe1d1557393',
    'report.json': '9eee8804963aa2db9691d53472d94350b8dafeb95d56000e68e80219d9fac02a',
    'supervisor.json': 'd35150d6308a6bfd4362ff67a18285ad0fbd0f867eef4d9d7ffd57968ed77e47',
}
ANCHOR = ROOT / 'docs/research/CROSS_INTERACTION_P0_ANCHOR.json'


def deny_heldout(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace('\\', '/').rsplit('/', 1)[-1].lower()
        if name in {'val_mat', 'test_mat', 'val.json', 'test.json'}:
            raise RuntimeError('P0 forbids held-out split access: ' + name)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source():
    # User review originals are the sole permitted unrelated untracked files.
    lines = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).splitlines()
    if any(not line.startswith('?? check/') for line in lines):
        raise RuntimeError('Committed source required; only untracked check/ is exempt')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def save(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def identities(anchor):
    for name, asset in anchor['assets'].items():
        if sha(ROOT / asset['path']) != asset['sha256']:
            raise RuntimeError('Identity mismatch: ' + name)


def worker(head):
    # Only invoked by supervised parent after exclusive output creation.
    started = time.monotonic()
    report = {'status': 'started', 'launch_commit': head, 'tracked_source_clean': True,
              'untracked_review_exemption': 'check/', 'teacher_forwards': 0,
              'validation_split_reads': 0, 'test_split_reads': 0, 'ranking_evaluations': 0,
              'pairs': [], 'seed': 2022, 'protocol': 'train_only_precision_u1_p0_v1'}
    save(OUT / 'report.json', report)
    try:
        sys.addaudithook(deny_heldout)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        sys.path.insert(0, str(ROOT / 'codes'))
        from cross_interaction_p0 import paired_u1, validate_checkpoint, COEFFICIENTS
        from cross_interaction_precision import capture, evaluate_rows
        anchor = json.loads(ANCHOR.read_text(encoding='utf-8'))
        identities(anchor)
        if source() != head:
            raise RuntimeError('Source changed')
        if torch.__version__ != anchor['torch'] or not torch.cuda.is_available():
            raise RuntimeError('Pinned CUDA environment required')
        torch.set_num_threads(2)
        torch.manual_seed(2022)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(0.25, 2 * 1024**3 / total_memory), 0)
        torch.cuda.reset_peak_memory_stats()
        def guard():
            torch.cuda.synchronize()
            if time.monotonic() - started > 570:
                raise RuntimeError('Worker time budget reached')
            if torch.cuda.max_memory_reserved() > 2 * 1024**3:
                raise RuntimeError('2 GiB CUDA allocator budget reached')
        assets = anchor['assets']
        for name, expected in OLD_HASHES.items():
            if sha(OLD / name) != expected:
                raise RuntimeError('Original P0 identity mismatch: ' + name)
        plan_bytes = (OLD / 'plan.json').read_bytes()
        plan = json.loads(plan_bytes)
        old_report = json.loads((OLD / 'report.json').read_text())
        if plan['coverage'] != anchor['expected_coverage']:
            raise RuntimeError('Pinned plan coverage mismatch')
        (OUT / 'plan.json').write_bytes(plan_bytes)
        report['old_artifacts'] = OLD_HASHES
        checkpoint = torch.load(ROOT / assets['warm_full']['path'], map_location='cpu', weights_only=True)
        validate_checkpoint(checkpoint)
        if checkpoint['best_epoch'] != 282 or checkpoint['student_config_profile'] != 'sports_student_full_sharedteacherinit_seed2022_val300_v1':
            raise RuntimeError('Warm-state profile mismatch')
        for tensor in checkpoint['model_state_dict'].values():
            if not torch.isfinite(tensor).all():
                raise RuntimeError('Nonfinite saved weights')
        steps = [float(s['step']) for s in checkpoint['optimizer_state_dict']['state'].values()]
        if steps != [283 * 214, 283 * 214]:
            raise RuntimeError('Selected optimizer step mismatch')
        for state in checkpoint['optimizer_state_dict']['state'].values():
            if not all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
                raise RuntimeError('Nonfinite saved moments')
        targets = torch.load(ROOT / assets['cache']['path'], map_location='cpu', weights_only=True)
        targets = {m: targets[m + '_items'].to('cuda:0') for m in COEFFICIENTS}
        if any(t.shape != (18357, 64) or t.dtype != torch.float32 or not torch.isfinite(t).all() for t in targets.values()):
            raise RuntimeError('Semantic cache schema/value mismatch')
        report.update(anchor=anchor, optimizer_steps_at_start=steps,
                      plan_sha256=sha(OUT / 'plan.json'), environment={
                          'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                          'gpu': torch.cuda.get_device_name(0), 'deterministic': True, 'tf32': False})
        chosen = plan['coverage']['chosen']
        if not chosen:
            raise RuntimeError('No exposed focus items')
        # One zero-control pair plus at most 3 x 2 interventions = 7 pairs.
        jobs = [(chosen[0]['item'], 'image', 0.0)] + [
            (row['item'], modality, coefficient) for row in chosen for modality, coefficient in COEFFICIENTS.items()]
        for index, (focus, modality, coefficient) in enumerate(jobs):
            guard()
            def observer(a, b, probes, item):
                rows = capture(a, b, probes, item)
                row_path = OUT / ('rows_%02d.pt' % index)
                torch.save(rows, row_path)
                precision = evaluate_rows(rows, probes)
                precision['rows_artifact'] = dict(path=row_path.name, sha256=sha(row_path))
                return precision
            result = paired_u1(checkpoint, targets[modality], plan['batch'],
                               plan['probes'][str(focus)], focus, coefficient, 'cuda:0', observer=observer)
            for pool_name, roles in result['precision']['pools'].items():
                for role in ('positive', 'negative'):
                    samples = roles[role]['samples']
                    legacy = result['pools'][pool_name][role].get('differences', [])
                    if len(samples) != len(legacy):
                        raise RuntimeError('Probe order/length mismatch')
                    for sample, value in zip(samples, legacy):
                        sample['legacy_gpu32'] = value
                        sample['legacy_gpu32_error'] = abs(value - sample['reference'])
            guard()
            report['pairs'].append(dict(item=focus, modality=modality, coefficient=coefficient, **result))
            report['elapsed_seconds'] = time.monotonic() - started
            report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            expected = {k: v for k, v in old_report['pairs'][index].items()
                        if k not in ('item', 'modality', 'coefficient')}
            actual = {k: v for k, v in result.items() if k != 'precision'}
            report['pairs'][-1]['legacy_readout_exact_match'] = actual == expected
            save(OUT / 'report.json', report)
            if actual != expected:
                raise RuntimeError('Old float32 update/readout mismatch; do not interpret new effects')
            if not result['precision']['consistent']:
                raise RuntimeError('Precision/reference agreement failed; rows and report preserved')
            if coefficient == 0 and any(v.get('u1') not in (None, 0) for pool in result['precision']['pools'].values()
                                        for v in pool.values() if isinstance(v, dict)):
                raise RuntimeError('High-precision zero-control mismatch')
            print('Precision pair %d/%d complete' % (index + 1, len(jobs)), flush=True)
        for name, expected in OLD_HASHES.items():
            if sha(OLD / name) != expected:
                raise RuntimeError('Original P0 artifact changed')
        identities(anchor)
        if source() != head:
            raise RuntimeError('Launch source changed during diagnostic')
        report.update(status='completed', source_and_assets_unchanged=True,
                      meaning='same-state precision check only; numeric resolution is not statistical or P1 acceptance')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        save(OUT / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.describe:
        print('P0 precision warm seed2022; 7 same-plan float32 pairs + CPU64/Decimal probes; 600s; 2GiB; '
              'train only; no teacher forward; exclusive output: ' + str(OUT))
        return
    if args.worker:
        if os.environ.get('PROMPTMM_P0_PRECISION_SUPERVISED') != args.worker or not (OUT / 'launch.json').exists():
            raise RuntimeError('Use supervised manual entry only')
        worker(args.worker)
        return
    head = source()
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / 'launch.json', {'commit': head, 'command': [sys.executable, '-B', str(Path(__file__).resolve())],
                             'max_seconds': 600, 'max_pairs': 7, 'no_retry': True})
    env = dict(os.environ, PROMPTMM_P0_PRECISION_SUPERVISED=head)
    print('P0 precision started; log: ' + str(OUT / 'worker.log'), flush=True)
    status = {'status': 'failed', 'launch_commit': head}
    try:
        with open(OUT / 'worker.log', 'x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', head],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        status.update(exit_code=result.returncode, status='completed' if result.returncode == 0 else 'failed')
    except subprocess.TimeoutExpired:
        status.update(status='failed', reason='hard 600s timeout; child killed; partial report preserved; no retry')
    finally:
        save(OUT / 'supervisor.json', status)
    if status['status'] != 'completed':
        raise SystemExit('P0 precision stopped; preserve artifacts; see worker.log/report.json; do not retry.')
    print('P0 precision completed (7 pairs). Audit report: ' + str(OUT / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
