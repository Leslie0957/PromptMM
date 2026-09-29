"""One bounded CPU diagnostic. Preparation profile refuses real-data execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'docs/research/innovation2/MODAL_NEIGHBOR_PROFILE_V1.json'


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


class DataGuard:
    def __init__(self, root, validation):
        self.root = root.resolve()
        self.validation = validation.resolve()
        self.graph_sealed = False
        self.test_denied = 0
        self.validation_denied = 0

    def __call__(self, event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        raw = os.fsdecode(args[0])
        p = Path(raw).resolve()
        # Restrict Test detection to project data; dependencies may have test modules.
        if p.is_relative_to(self.root / 'data') and 'test' in str(p).lower():
            self.test_denied += 1
            raise PermissionError('Test access denied')
        if p == self.validation and not self.graph_sealed:
            self.validation_denied += 1
            raise PermissionError('Validation access before graph sealing')


def limit_reason(sample, caps):
    for key in ('seconds', 'rss', 'output_bytes'):
        if sample[key] > caps[key]:
            return key
    if sample['free_disk'] < caps['minimum_free_disk']:
        return 'free_disk'
    return None


def supervise(command, out, caps, env):
    """Parent remains outside numerical worker; bounds include startup/hash work."""
    import psutil
    start = time.monotonic()
    peak = 0
    reason = None
    with (out / 'worker.log').open('w', encoding='utf-8') as log, \
            (out / 'resource.jsonl').open('w', encoding='utf-8') as telemetry:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        proc = psutil.Process(child.pid)
        try:
            while True:
                try:
                    rss = proc.memory_info().rss + sum(p.memory_info().rss for p in proc.children(recursive=True))
                except psutil.NoSuchProcess:
                    rss = 0
                rss += psutil.Process().memory_info().rss
                sample = {'seconds': time.monotonic() - start, 'rss': rss,
                          'output_bytes': sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
                          'free_disk': shutil.disk_usage(out).free}
                peak = max(peak, rss)
                telemetry.write(json.dumps(sample) + '\n')
                telemetry.flush()
                reason = limit_reason(sample, caps)
                if reason:
                    try:
                        for p in proc.children(recursive=True):
                            p.kill()
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
                    break
                if child.poll() is not None:
                    break
                time.sleep(caps['sample_seconds'])
        except BaseException:
            try:
                for p in proc.children(recursive=True):
                    p.kill()
                proc.kill()
            except psutil.NoSuchProcess:
                pass
            child.wait(timeout=10)
            write(out / 'supervisor_failure.json', {'traceback': traceback.format_exc()})
            raise
        code = child.wait(timeout=10)
    result = {'exit_code': code, 'limit': reason, 'seconds': time.monotonic() - start,
              'sampled_peak_combined_rss': peak, 'sampling_seconds': caps['sample_seconds'],
              'status': 'completed' if code == 0 and reason is None else 'failed'}
    write(out / 'supervisor.json', result)
    return result


def logical_sha(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(list(array.shape)).encode())
    h.update(array.tobytes(order='C'))
    return h.hexdigest()


def worker(profile, out):
    # Install guard before importing/loading serialization dependencies.
    guard = DataGuard(ROOT, ROOT / profile['inputs']['validation']['path'])
    sys.addaudithook(guard)
    import pickle
    import numpy as np
    import scipy
    import torch
    from modal_neighbor_core import (binary_matrix, normalize, neighbors, matched_random,
                                     associations, measures, bootstrap, cluster_summary, screen)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.time()
    write(out / 'environment.json', {'numpy': np.__version__, 'scipy': scipy.__version__,
          'torch': torch.__version__, 'torch_threads': torch.get_num_threads(),
          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})

    def load_matrix(key, strict=False):
        asset = profile['inputs'][key]
        path = ROOT / asset['path']
        if sha(path) != asset['sha256']:
            raise ValueError('Input SHA mismatch: ' + key)
        with path.open('rb') as stream:
            matrix = pickle.load(stream)
        return binary_matrix(matrix, profile['shape'], strict)

    try:
        train, duplicates = load_matrix('train')
        asset = profile['inputs']['teacher']
        if sha(ROOT / asset['path']) != asset['sha256']:
            raise ValueError('Teacher SHA mismatch')
        cache = torch.load(ROOT / asset['path'], map_location='cpu', weights_only=True)
        vectors, valid = [], []
        for key in ('image_items', 'text_items'):
            t = cache[key]
            if tuple(t.shape) != (profile['shape'][1], 64):
                raise ValueError('Modality shape mismatch')
            x, ok = normalize(t.detach().cpu().numpy())
            vectors.append(x)
            valid.append(ok)
        del cache
        good = valid[0] & valid[1]
        degree = np.bincount(train.indices, minlength=train.shape[1])
        low = np.lexsort((np.arange(len(degree)), degree))[:len(degree) // 3]
        if len(low) != profile['expected_low_items'] or degree[low].max() != profile['expected_low_max_degree']:
            raise ValueError('Low-third anchor mismatch')
        queries = low[good[low]]
        candidates = np.flatnonzero(good & (degree > 0))
        if not len(queries):
            raise ValueError('No valid queries')
        graphs = []
        graph_files = {}
        graph_logical = {}
        for mi, name in enumerate(('image', 'text')):
            real, scores = neighbors(vectors[mi], queries, candidates, profile['k'], profile['block'])
            random, variable, pools = matched_random(queries, real, candidates, degree,
                profile['random_repeats'], mi, profile['random_seed'])
            arrays = dict(queries=queries, candidates=candidates, degree=degree, real=real,
                          similarities=scores, random=random, variable=variable, pool_sizes=pools)
            path = out / (name + '_graphs.npz')
            np.savez(path, **arrays)
            graph_files[path.name] = sha(path)
            graph_logical[name] = {k: logical_sha(v) for k, v in arrays.items()}
            graphs.append((real, random, variable, pools))
        write(out / 'graph_seal.json', {'time': time.time(), 'file_hashes': graph_files,
                                       'array_hashes': graph_logical, 'validation_loaded': False})
        # This durable seal is created before even hashing the Validation file.
        guard.graph_sealed = True
        validation_started = time.time()
        val, _ = load_matrix('validation', strict=True)
        if train.multiply(val).nnz:
            raise ValueError('Train/Validation overlap')
        vr, vi = val.nonzero()
        original_mask = np.isin(vi, low)
        original_pairs = int(original_mask.sum())
        if original_pairs != profile['expected_low_validation_pairs']:
            raise ValueError('Low-third Validation denominator mismatch')
        index = np.full(train.shape[1], -1, dtype=np.int32)
        index[queries] = np.arange(len(queries))
        keep = index[vi] >= 0
        users, targets = vr[keep], vi[keep]
        qi = index[targets]
        coverage = len(users) / original_pairs
        report = {'interpretation': 'Exploratory reused-Validation neighbor association, not Recall or semantic causality',
                  'support': {'low_queries': len(low), 'valid_queries': len(queries),
                    'candidates': len(candidates), 'original_pairs': original_pairs,
                    'pairs': len(users), 'excluded_pairs': original_pairs - len(users),
                    'coverage': coverage, 'users': len(np.unique(users)),
                    'targets': len(np.unique(targets)), 'train_duplicates_deduplicated': duplicates,
                    'zero_image_queries': int((~valid[0][low]).sum()),
                    'zero_text_queries': int((~valid[1][low]).sum()),
                    'zero_image_items': int((~valid[0]).sum()),
                    'zero_text_items': int((~valid[1]).sum()),
                    'zero_degree_candidates_excluded': int((degree == 0).sum()),
                    'empty_history_pairs': int((np.diff(train.indptr)[users] == 0).sum())},
                  'modalities': {}, 'validation_started': validation_started}
        _, target_support = np.unique(targets, return_counts=True)
        report['support']['target_positive_count_quantiles'] = (
            np.quantile(target_support, [0, .25, .5, .75, 1]).tolist() if len(target_support) else None)
        for mi, name in enumerate(('image', 'text')):
            real, random, variable, pools = graphs[mi]
            rc = associations(train, users, qi, real)
            nc = np.stack([associations(train, users, qi, r) for r in random])
            delta = (rc > 0).astype(float) - (nc > 0).mean(axis=0)
            row = measures(rc, nc, targets, profile['k'])
            uc = bootstrap(users, delta, profile['bootstrap_repeats'], profile['user_bootstrap_seed'])
            ic = bootstrap(targets, delta, profile['bootstrap_repeats'], profile['item_bootstrap_seed'])
            matching = float((variable[qi] >= .8).mean()) if len(qi) else 0.
            overlap = np.array([[np.isin(r[j], real[j]).mean() for j in range(len(queries))] for r in random])
            row.update(user_ci=uc, item_ci=ic, matching_support=matching,
                screen=screen(row['delta'], uc, ic, coverage, matching, profile['threshold']),
                variable_quantiles=np.quantile(variable, [0, .25, .5, .75, 1]).tolist(),
                pool_size_quantiles=np.quantile(pools, [0, .25, .5, .75, 1]).tolist(),
                real_random_overlap_mean=float(overlap.mean()),
                real_random_overlap_range=[float(overlap.min()), float(overlap.max())])
            row['frequency_groups'] = {}
            for d in range(6):
                mask = degree[targets] == d
                row['frequency_groups'][str(d)] = measures(rc[mask], nc[:, mask], targets[mask], profile['k'])
            mask = degree[targets] > 0
            row['excluding_zero_degree'] = measures(rc[mask], nc[:, mask], targets[mask], profile['k'])
            arrays = {}
            for label, ids in [('user', users), ('target', targets)]:
                unique, sums, denominators = cluster_summary(ids, delta)
                _, real_sums, _ = cluster_summary(ids, rc > 0)
                _, null_sums, _ = cluster_summary(ids, (nc > 0).mean(axis=0))
                arrays.update({label + '_ids': unique, label + '_delta_sum': sums,
                               label + '_denominator': denominators, label + '_real_sum': real_sums,
                               label + '_null_sum': null_sums})
            arrays['null_repeat_rates'] = (nc > 0).mean(axis=1) if len(rc) else np.zeros(len(nc))
            np.savez(out / (name + '_summaries.npz'), **arrays)
            report['modalities'][name] = row
        write(out / 'report.json', report)
        write(out / 'acceptance.json', {'status': 'passed', 'test_denied_attempts': guard.test_denied,
              'validation_early_denied_attempts': guard.validation_denied,
              'graph_seal_precedes_validation': True, 'training': 0, 'cuda': False,
              'hashes': {p.name: sha(p) for p in out.iterdir() if p.suffix in ('.npz', '.json')},
              'seconds': time.time() - started})
    except BaseException:
        write(out / 'worker_failure.json', {'traceback': traceback.format_exc(),
              'test_denied_attempts': guard.test_denied,
              'validation_early_denied_attempts': guard.validation_denied})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    if profile['status'] != 'authorized_once' or not args.formal:
        raise SystemExit('Preparation only: real-data launch requires a committed authorization declaration and profile activation.')
    out = ROOT / profile['output']
    if args.worker:
        if os.environ.get('MODAL_NEIGHBOR_PARENT') != str(os.getppid()):
            raise SystemExit('Worker requires supervisor')
        worker(profile, out)
        return
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).splitlines()
    if any(not s.startswith('?? archive/reviews/') for s in dirty):
        raise SystemExit('Dirty source')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != profile['branch']:
        raise SystemExit('Branch mismatch')
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', MODAL_NEIGHBOR_PARENT=str(os.getpid()))
    write(out / 'manifest.json', {'source': source, 'branch': branch, 'profile': profile,
          'profile_sha': sha(PROFILE), 'python': sys.version, 'executable': sys.executable,
          'command': [sys.executable, '-B', str(Path(__file__).resolve()), '--formal'],
          'environment': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
          'clean_source': True, 'allowed_untracked': dirty})
    try:
        result = supervise([sys.executable, '-B', str(Path(__file__).resolve()), '--formal', '--worker'],
                           out, profile['caps'], env)
    except BaseException:
        write(out / 'exit.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
    if result['status'] == 'completed' and not (out / 'acceptance.json').exists():
        result.update(status='failed', reason='Missing worker acceptance')
    result['hashes'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write(out / 'exit.json', result)
    raise SystemExit(0 if result['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
