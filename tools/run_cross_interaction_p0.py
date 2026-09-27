"""One manual, non-formal P0 resource check. No retry, resume or next stage."""
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
OUT = ROOT / 'exp/cross_interaction/p0_warm2022_v1'
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
              'pairs': [], 'seed': 2022, 'protocol': 'train_only_conditional_u1_p0_v1'}
    save(OUT / 'report.json', report)
    try:
        sys.addaudithook(deny_heldout)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import pickle
        import torch
        sys.path.insert(0, str(ROOT / 'codes'))
        from cross_interaction_p0 import make_plan, paired_u1, validate_checkpoint, COEFFICIENTS
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
        # Exact train_mat only. Never import Data, batch_test, parser or trainer.
        with open(ROOT / assets['train']['path'], 'rb') as stream:
            matrix = pickle.load(stream).tocsr()
        if matrix.shape != (35598, 18357) or matrix.nnz != 218409:
            raise RuntimeError('Train schema mismatch')
        plan = make_plan(matrix)
        if plan['coverage'] != anchor['expected_coverage']:
            raise RuntimeError('Training coverage/selection changed')
        save(OUT / 'plan.json', plan)
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
        for focus, modality, coefficient in jobs:
            guard()
            result = paired_u1(checkpoint, targets[modality], plan['batch'],
                               plan['probes'][str(focus)], focus, coefficient, 'cuda:0')
            guard()
            report['pairs'].append(dict(item=focus, modality=modality, coefficient=coefficient, **result))
            report['elapsed_seconds'] = time.monotonic() - started
            report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            save(OUT / 'report.json', report)
        identities(anchor)
        if source() != head:
            raise RuntimeError('Launch source changed during diagnostic')
        report.update(status='completed', source_and_assets_unchanged=True,
                      meaning='measurement/resource check only; no P1 gate or scientific effect acceptance')
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
        print('P0 warm seed2022; <=7 dense paired U1 steps; 600s parent timeout; 2GiB CUDA allocator; '
              'train only; no teacher forward; exclusive output: ' + str(OUT))
        return
    if args.worker:
        if os.environ.get('PROMPTMM_P0_SUPERVISED') != args.worker or not (OUT / 'launch.json').exists():
            raise RuntimeError('Use supervised manual entry only')
        worker(args.worker)
        return
    head = source()
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT / 'launch.json', {'commit': head, 'command': [sys.executable, '-B', str(Path(__file__).resolve())],
                             'max_seconds': 600, 'max_pairs': 7, 'no_retry': True})
    env = dict(os.environ, PROMPTMM_P0_SUPERVISED=head)
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
        raise SystemExit('P0 stopped; preserve all artifacts and audit before any newly authorized execution.')


if __name__ == '__main__':
    main()
