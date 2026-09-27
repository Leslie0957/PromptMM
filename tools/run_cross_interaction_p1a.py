"""One user-manual P1a single-item U1 screen; no retry or next stage."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/research/CROSS_INTERACTION_P1A_PLAN.json'
PLAN_SHA = '3dc983a5eff66df500493d75aa4773ef893fa58a3af2964f874cc085ba6e4dd8'
ANCHOR = ROOT / 'docs/research/CROSS_INTERACTION_P0_ANCHOR.json'
OLD_PLAN = ROOT / 'exp/cross_interaction/p0_warm2022_v1/plan.json'
OLD_PLAN_SHA = '11130dd5cc71ee2da45a485bb04c3946a6d5e9f310940019cc219fe1d1557393'
OUT = ROOT / 'exp/cross_interaction/p1a_commoncache_seed2022_u1_v1'
COLD = ROOT / 'Model/sports/td_distill/td_distill_full__val_test_once_v1__2026-09-24 18_22_52.151219_sports_light_init_pid5012.pth'
COLD_SHA = 'd2648d9f1f6c593b8720b27ad226e81152373537f7cb4d3fa18de07acf20b4a2'
COLD_MANIFEST = ROOT / 'exp/runs/sports/run_manifest__2026-09-24 18_22_52.151219_sports_light_init_pid5012.json'
COLD_MANIFEST_SHA = 'd31cf835fd68b575b7312e1328f8e439ad0d459302e65c9587b524ce760d6ddf'
PARENT_SECONDS = 1800
WORKER_SECONDS = 1740
OUTPUT_BYTES = 512 * 1024 * 1024
RSS_BYTES = 6 * 1024 * 1024 * 1024


def deny_heldout(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0]).replace('\\', '/').rsplit('/', 1)[-1].lower()
        if name in {'val_mat', 'test_mat', 'val.json', 'test.json'}:
            raise RuntimeError('P1a forbids held-out split access: ' + name)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source():
    lines = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'],
                                    cwd=ROOT, text=True).splitlines()
    if any(not line.startswith('?? check/') for line in lines):
        raise RuntimeError('Committed source required; only untracked check/ exempt')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def verify_inputs(anchor):
    checks = {str(ROOT / asset['path']): asset['sha256'] for asset in anchor['assets'].values()}
    checks.update({str(COLD): COLD_SHA, str(COLD_MANIFEST): COLD_MANIFEST_SHA,
                   str(PLAN): PLAN_SHA, str(OLD_PLAN): OLD_PLAN_SHA})
    for path, expected in checks.items():
        if sha(path) != expected:
            raise RuntimeError('Pinned input identity mismatch: ' + path)
    return checks


def _checkpoint(checkpoint, label, torch):
    from cross_interaction_p0 import validate_checkpoint
    validate_checkpoint(checkpoint)
    epoch = 293 if label == 'cold2022' else 282
    profile = ('sports_student_full_pairedcold_seed2022_val300_v1' if label == 'cold2022'
               else 'sports_student_full_sharedteacherinit_seed2022_val300_v1')
    if checkpoint['best_epoch'] != epoch or checkpoint['student_config_profile'] != profile:
        raise RuntimeError('Checkpoint epoch/profile mismatch: ' + label)
    steps = [float(s['step']) for s in checkpoint['optimizer_state_dict']['state'].values()]
    if steps != [(epoch + 1) * 214] * 2:
        raise RuntimeError('Full optimizer step mismatch: ' + label)
    for value in checkpoint['model_state_dict'].values():
        if not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite checkpoint parameter')
    for state in checkpoint['optimizer_state_dict']['state'].values():
        if not all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
            raise RuntimeError('Nonfinite checkpoint optimizer state')
    return steps


def worker(head):
    started = time.monotonic()
    report = {'status': 'started', 'launch_commit': head, 'protocol': 'sports_train_only_p1a_single_item_u1_v1',
              'cache_interpretation': 'common pinned warm semantic target for historical cold and warm student states',
              'historical_cold_cache_identity_proven': False,
              'teacher_forwards': 0, 'validation_split_reads': 0, 'test_split_reads': 0,
              'ranking_evaluations': 0, 'pairs': []}
    save(OUT / 'report.json', report)
    try:
        sys.addaudithook(deny_heldout)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import torch
        import psutil
        sys.path.insert(0, str(ROOT / 'codes'))
        from cross_interaction_p0 import paired_u1
        from cross_interaction_p1a_plan import make_plan, canonical_bytes
        from cross_interaction_p1a_signal import features
        from cross_interaction_p1a_analysis import analyze
        from cross_interaction_precision import capture, evaluate_rows
        anchor = json.loads(ANCHOR.read_text(encoding='utf-8'))
        identities = verify_inputs(anchor)
        if source() != head:
            raise RuntimeError('Launch source changed before worker')
        if torch.__version__ != anchor['torch'] or not torch.cuda.is_available():
            raise RuntimeError('Pinned CUDA environment required')
        torch.set_num_threads(2)
        torch.manual_seed(31027)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(.25, 2 * 1024**3 / total_memory), 0)
        torch.cuda.reset_peak_memory_stats()
        def guard():
            torch.cuda.synchronize()
            if time.monotonic() - started > WORKER_SECONDS:
                raise RuntimeError('Worker time budget reached')
            if torch.cuda.max_memory_reserved() > 2 * 1024**3:
                raise RuntimeError('CUDA allocator budget reached')
            if psutil.Process().memory_info().rss > RSS_BYTES:
                raise RuntimeError('Process RSS budget reached')
            size = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
            if size > OUTPUT_BYTES:
                raise RuntimeError('Output byte budget reached')

        with open(ROOT / anchor['assets']['train']['path'], 'rb') as stream:
            matrix = pickle.load(stream).tocsr()
        if matrix.shape != (35598, 18357) or matrix.nnz != 218409:
            raise RuntimeError('Train schema mismatch')
        saved_bytes = PLAN.read_bytes()
        old = json.loads(OLD_PLAN.read_text(encoding='utf-8'))
        regenerated = canonical_bytes(make_plan(matrix, old))
        if regenerated != saved_bytes:
            raise RuntimeError('Plan does not regenerate exactly from pinned train/P0 inputs')
        plan = json.loads(saved_bytes)
        if len(plan['items']) != 23 or plan['max_nonzero_pairs'] != 368 or plan['zero_pairs'] != 4:
            raise RuntimeError('Plan count mismatch')
        report.update(plan_sha256=PLAN_SHA, train_sha256=anchor['assets']['train']['sha256'],
                      environment={'python': sys.version, 'torch': torch.__version__,
                                   'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                                   'deterministic': True, 'tf32': False},
                      planned_pairs=372, planned_steps=744)
        save(OUT / 'report.json', report)
        targets_cpu = torch.load(ROOT / anchor['assets']['cache']['path'],
                                 map_location='cpu', weights_only=True)
        targets = {m: targets_cpu[m + '_items'].to('cuda:0') for m in ('image', 'text')}
        if any(t.shape != (18357, 64) or t.dtype != torch.float32 or not torch.isfinite(t).all()
               for t in targets.values()):
            raise RuntimeError('Common semantic cache invalid')
        pair_index = 0
        for state_label, path in [('cold2022', COLD),
                                  ('warm2022', ROOT / anchor['assets']['warm_full']['path'])]:
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            steps = _checkpoint(checkpoint, state_label, torch)
            report.setdefault('optimizer_steps_at_start', {})[state_label] = steps
            for ci, batch in enumerate(plan['contexts']):
                guard()
                row0 = plan['items'][0]
                jobs = [(row0, 'image', 'zero', 0.0)]
                for row in plan['items']:
                    for modality in ('image', 'text'):
                        full = plan['coefficients'][modality]
                        jobs.extend([(row, modality, 'full', full),
                                     (row, modality, 'half', full / 2)])
                for row, modality, arm, weight in jobs:
                    guard()
                    focus = row['item']
                    probes = row['probes'][ci]
                    def observer(a, b, actual_probes, actual_focus):
                        rows = capture(a, b, actual_probes, actual_focus)
                        row_path = OUT / 'rows' / ('rows_%03d.pt' % pair_index)
                        torch.save(rows, row_path)
                        result = evaluate_rows(rows, actual_probes)
                        result['rows_artifact'] = {'path': str(row_path.relative_to(OUT)),
                                                   'sha256': sha(row_path)}
                        return result
                    outcome = paired_u1(checkpoint, targets[modality], batch,
                                        probes, focus, weight, 'cuda:0', observer=observer)
                    if not outcome['precision']['consistent'] or not outcome['precision']['rows_unchanged']:
                        raise RuntimeError('P1a precision reference mismatch')
                    if arm == 'zero' and (outcome['delta_norm'] != 0 or any(
                            r.get('u1') not in (None, 0) for p in outcome['precision']['pools'].values()
                            for r in p.values() if isinstance(r, dict))):
                        raise RuntimeError('Zero control mismatch')
                    raw_path = OUT / 'pairs' / ('pair_%03d.json' % pair_index)
                    save(raw_path, outcome)
                    short = {'index': pair_index, 'state': state_label, 'context': ci,
                             'item': focus, 'layer': row['layer'], 'degree': row['degree'],
                             'modality': modality, 'arm': arm, 'coefficient': weight,
                             'multiplicity': outcome['multiplicity'], 'delta_norm': outcome['delta_norm'],
                             'artifact': {'path': str(raw_path.relative_to(OUT)), 'sha256': sha(raw_path)},
                             'rows_artifact': outcome['precision']['rows_artifact'], 'pools': {}}
                    for pool_name, pool in outcome['precision']['pools'].items():
                        short['pools'][pool_name] = {
                            role: {key: pool[role].get(key) for key in
                                   ('u1', 'variance', 'tolerance', 'resolved', 'mass',
                                    'available_edges', 'unique_edges')}
                            for role in ('positive', 'negative')}
                        short['pools'][pool_name]['contribution'] = pool['pool_distribution_contribution']
                    forbidden_users = {u for u, p, _ in batch if p == focus}
                    forbidden_users.update(t[0] for name in ('A', 'B')
                                           for role in ('positive', 'negative')
                                           for t in probes[name][role]['triples'])
                    short['c_unshared_user_sensitivity'] = {}
                    for role in ('positive', 'negative'):
                        samples = outcome['precision']['pools']['C'][role]['samples']
                        kept = [sample['stable64'] for sample in samples
                                if sample['triple'][0] not in forbidden_users]
                        short['c_unshared_user_sensitivity'][role] = {
                            'retained': len(kept), 'total': len(samples),
                            'mean': math.fsum(kept) / len(kept) if kept else None}
                    if arm == 'full':
                        short['a_features'] = features(checkpoint, targets_cpu[modality + '_items'],
                                                       focus, probes['A'], row['multiplicity'][ci],
                                                       row['degree'])
                    report['pairs'].append(short)
                    pair_index += 1
                    report['elapsed_seconds'] = time.monotonic() - started
                    report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
                    save(OUT / 'report.json', report)
                    guard()
                    print('P1a pair %d/372 complete' % pair_index, flush=True)
            del checkpoint
        if pair_index != 372:
            raise RuntimeError('P1a pair count mismatch')
        for path, expected in identities.items():
            if sha(path) != expected:
                raise RuntimeError('Pinned input changed during run: ' + path)
        if source() != head:
            raise RuntimeError('Launch source changed during run')
        report['status'] = 'running_analysis'
        save(OUT / 'report.json', report)
        analysis = analyze(report)
        save(OUT / 'analysis.json', analysis)
        report.update(status='completed', source_and_assets_unchanged=True,
                      analysis_sha256=sha(OUT / 'analysis.json'),
                      meaning='single-item U1 screening only; no joint policy or ranking evidence')
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
        print('P1a common-cache cold/warm seed2022 U1 screen; 372 pairs / 744 independent steps; '
              '1800s parent; 2GiB CUDA allocator; 6GiB RSS; 512MiB output; '
              'train only; no teacher forward; '
              'exclusive output: ' + str(OUT))
        return
    if args.worker:
        if os.environ.get('PROMPTMM_P1A_SUPERVISED') != args.worker or not (OUT / 'launch.json').exists():
            raise RuntimeError('Use supervised manual entry only')
        worker(args.worker)
        return
    head = source()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'pairs').mkdir()
    (OUT / 'rows').mkdir()
    save(OUT / 'launch.json', {'commit': head, 'command': [sys.executable, '-B', str(Path(__file__).resolve())],
                              'plan_sha256': PLAN_SHA, 'max_seconds': PARENT_SECONDS,
                              'max_pairs': 372, 'no_retry': True})
    env = dict(os.environ, PROMPTMM_P1A_SUPERVISED=head)
    print('P1a started; log: ' + str(OUT / 'worker.log'), flush=True)
    status = {'status': 'failed', 'launch_commit': head}
    try:
        with open(OUT / 'worker.log', 'x', encoding='utf-8') as log:
            result = subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--worker', head],
                                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=PARENT_SECONDS)
        status.update(exit_code=result.returncode,
                      status='completed' if result.returncode == 0 else 'failed')
    except subprocess.TimeoutExpired:
        status.update(status='failed', reason='Hard 1800s timeout; child killed; no retry')
    finally:
        save(OUT / 'supervisor.json', status)
    if status['status'] != 'completed':
        raise SystemExit('P1a stopped; preserve partial artifacts; see worker.log/report.json; do not retry.')
    print('P1a completed (372 pairs). Audit report: ' + str(OUT / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
