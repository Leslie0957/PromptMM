"""One-shot fixed-checkpoint intervention; explicit --formal, no training code."""
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
PROFILE = ROOT / 'docs/research/innovation2/OWN_ANCHOR_CONTROL_PROFILE_V1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def data_guard(event, args, counts):
    if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(ROOT / 'data'):
            allowed = [ROOT / 'data/sports/train_mat', ROOT / 'data/sports/val_mat']
            if p not in allowed:
                counts['denied_data_opens'] += 1
                raise PermissionError('Only Sports Train/Validation permitted')
            counts[p.name + '_opens'] += 1


def control_features(q, ids, graphs):
    import numpy as np
    from neighbor_shared_residual import features
    xr, scales = features(q, ids, *graphs)
    centers = [q[g].mean(axis=1).mean(axis=0) for g in graphs]
    xc = np.zeros_like(xr)
    xc[ids] = np.concatenate([(mu - q[ids]) / a / np.sqrt(2.0)
                             for mu, a in zip(centers, scales)], axis=1)
    require(np.isfinite(xc).all(), 'Nonfinite control features')
    return xr, xc, scales, centers


def affine_residual(q, ids, w, scales, centers):
    import numpy as np
    w = w.astype(np.float64)
    a = -(w[:, :64] / scales[0] + w[:, 64:] / scales[1]) / np.sqrt(2.0)
    b = (w[:, :64] @ centers[0].astype(np.float64) / scales[0]
         + w[:, 64:] @ centers[1].astype(np.float64) / scales[1]) / np.sqrt(2.0)
    out = np.zeros(q.shape, dtype=np.float64)
    out[ids] = q[ids].astype(np.float64) @ a.T + b
    return out


def paired(top_r, top_c, users, val, nitems):
    import numpy as np
    per_user = np.zeros((len(users), 4), dtype=np.int64)  # gain, loss, overlap, positives
    per_item = np.zeros((nitems, 2), dtype=np.int64)
    for k, uid in enumerate(users):
        positives = set(val.indices[val.indptr[uid]:val.indptr[uid + 1]])
        r, c = set(top_r[k]), set(top_c[k])
        gain, loss = (c - r) & positives, (r - c) & positives
        per_user[k] = [len(gain), len(loss), len(r & c), len(positives)]
        for i in gain:
            per_item[i, 0] += 1
        for i in loss:
            per_item[i, 1] += 1
    require((per_user[:, 3] > 0).all(), 'Empty evaluation user')
    return per_user, per_item


def worker(p, out):
    import numpy as np
    import psutil
    import torch
    from initialization_kd_adapter import load_train_val
    from run_neighbor_shared_residual import rank
    counts = {'denied_data_opens': 0, 'train_mat_opens': 0, 'val_mat_opens': 0}
    sys.addaudithook(lambda e, a: data_guard(e, a, counts))
    start = time.monotonic()
    samples = []
    result = {'status': 'running', 'states': {}, 'comparisons': {}, 'data_access': counts,
              'parameter_updates': 0, 'test_policy': 'Test0'}
    def check():
        row = {'seconds': time.monotonic() - start,
               'rss_bytes': psutil.Process().memory_info().rss,
               'cuda_reserved': torch.cuda.memory_reserved()}
        samples.append(row)
        require(row['seconds'] < p['caps']['wall_seconds'] and
                row['rss_bytes'] < p['caps']['rss_bytes'] and
                row['cuda_reserved'] <= p['caps']['cuda_allocator_bytes'], 'Worker resource cap')
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        require(torch.cuda.is_available(), 'CUDA required for original scoring protocol')
        torch.cuda.set_per_process_memory_fraction(p['caps']['cuda_allocator_bytes'] /
                torch.cuda.get_device_properties(0).total_memory)
        result['environment'] = {'torch': torch.__version__, 'numpy': np.__version__,
                                 'cuda': torch.cuda.get_device_name(0)}
        for spec in [p['anchor_manifest'], p['anchor_completeness'], *p['assets'].values(),
                     *p['inputs'].values()]:
            require(sha(ROOT / spec['path']) == spec['sha256'], 'Input SHA: ' + spec['path'])
            check()
        anchor = read(ROOT / p['anchor_manifest']['path'])
        require(anchor['source_commit'] == 'b8be549613498e5cbb98412ffa766c62279ad7a6', 'Original source')
        q = torch.load(ROOT / p['assets']['teacher']['path'], map_location='cpu', weights_only=True)['items'].numpy()
        require(q.shape == (18357, 64) and np.isfinite(q).all(), 'Teacher shape/finite')
        graphs, ids, degrees = [], None, None
        for m in ['image', 'text']:
            with np.load(ROOT / p['assets'][m + '_graph']['path'], allow_pickle=False) as z:
                current = z['queries'].copy()
                if ids is not None:
                    require(np.array_equal(current, ids) and np.array_equal(degrees, z['degree']), 'Graph identity')
                ids, degrees = current, z['degree'].copy()
                graphs.append(z['random'][0].copy())
        require(len(ids) == 6114, 'Query count')
        xr, xc, scales, centers = control_features(q, ids, graphs)
        require(hashlib.sha256(xr.tobytes()).hexdigest() == anchor['graphs']['feature_sha256']['R'], 'R X regression')
        save(out / 'intervention.json', {'scales': scales, 'centers': [x.tolist() for x in centers],
              'xr_sha256': hashlib.sha256(xr.tobytes()).hexdigest(),
              'xc_sha256': hashlib.sha256(xc.tobytes()).hexdigest()})
        # All intervention statistics fixed before reading interaction labels.
        train, val = load_train_val(ROOT / 'data/sports', p['assets']['train']['sha256'],
                                   p['assets']['validation']['sha256'], 35598, 18357)
        degree = np.bincount(train.indices, minlength=len(q))
        require(np.array_equal(degree, degrees), 'Train degree identity')
        order = np.lexsort((np.arange(len(q)), degree))
        require(np.isin(ids, order[:6119]).all(), 'Targets outside low group')
        groups = np.empty(len(q), dtype=np.int8)
        for g, part in enumerate(np.array_split(order, 3)):
            groups[part] = g
        users = np.flatnonzero(np.diff(val.indptr))
        mask = np.ones(len(q), dtype=bool)
        mask[ids] = False
        def bound_path(seed, rel):
            return ROOT / p['inputs'][f'seed{seed}/{rel}']['path']
        with torch.inference_mode():
            for seed in p['seeds']:
                ck = torch.load(bound_path(seed, 'R/final.pt'), map_location='cpu', weights_only=False)
                require(ck['epoch'] == 300 and ck['arm'] == 'R' and ck['seed'] == seed
                        and ck['source'] == anchor['source_commit'], 'Wrong checkpoint identity')
                for key, shape in [('user', (35598,64)), ('item', (18357,64)), ('extra', (64,128))]:
                    require(tuple(ck[key].shape) == shape and ck[key].dtype == torch.float32
                            and bool(torch.isfinite(ck[key]).all()), 'Checkpoint tensor ' + key)
                old = read(bound_path(seed, 'R/status.json'))['curve'][-1]
                base = read(bound_path(seed, 'B/status.json'))['curve'][-1]
                require(old['epoch'] == base['epoch'] == 300, 'Status epoch')
                scores = {}
                tops = {}
                u, item, w = (ck[k].to('cuda:0') for k in ['user','item','extra'])
                for state, x in [('R', xr), ('C', xc)]:
                    residual = torch.nn.functional.linear(torch.from_numpy(x).to('cuda:0'), w)
                    effective = item + residual
                    require(torch.equal(effective[mask], item[mask]), 'Non-target change')
                    if state == 'C':
                        expected = affine_residual(q, ids, ck['extra'].numpy(), scales, centers)
                        require(np.allclose(residual.cpu().numpy(), expected, rtol=1e-5, atol=1e-5), 'Affine identity')
                    view = SimpleNamespace(user_id_embedding=SimpleNamespace(weight=u),
                                           item_id_embedding=SimpleNamespace(weight=effective))
                    metric, top = rank(view, train, val, groups, check, True)
                    require(metric['groups']['denominator'] == [6347,6389,25163], 'Group denominator')
                    require(all(len(set(row)) == 20 for row in top), 'Duplicate candidates')
                    scores[state], tops[state] = metric, top
                    np.savez_compressed(out / f'seed{seed}_{state}_top20.npz', users=users, top20=top)
                    result['states'][f'{seed}_{state}'] = metric
                    if state == 'R':
                        with np.load(bound_path(seed, 'R/top20_epoch300.npz'), allow_pickle=False) as z:
                            require(np.array_equal(z['users'], users) and np.array_equal(z['top20'], top), 'R exact Top20 regression')
                        require(abs(metric['recall20'] - old['recall20']) <= 1e-6, 'R Recall regression')
                        require(metric['groups']['hits'] == old['groups']['hits'], 'R hits regression')
                    save(out / 'report.json', result)
                pu, pi = paired(tops['R'], tops['C'], users, val, len(q))
                delta = scores['C']['recall20'] - scores['R']['recall20']
                require(abs(float(np.mean((pu[:,0]-pu[:,1])/pu[:,3])) - delta) < 1e-10, 'Recall paired identity')
                diff = np.array(scores['C']['groups']['hits']) - np.array(scores['R']['groups']['hits'])
                require(np.array_equal(diff, [sum((pi[:,0]-pi[:,1])[groups == g]) for g in range(3)]), 'Group paired identity')
                retention = (scores['C']['groups']['hits'][0] - base['groups']['hits'][0]) / (old['groups']['hits'][0] - base['groups']['hits'][0])
                result['comparisons'][str(seed)] = {'low_gain_retention': retention,
                    'recall_C_minus_R': delta, 'recall_C_minus_B': scores['C']['recall20']-base['recall20'],
                    'low_hits_C_minus_R': int(diff[0]),
                    'low_hits_C_minus_B': scores['C']['groups']['hits'][0]-base['groups']['hits'][0],
                    'screen_pass': bool(retention >= .8 and delta >= -.0002)}
                np.savez_compressed(out / f'seed{seed}_paired.npz', users=users,
                                    per_user_gain_loss_overlap_den=pu, per_item_gain_loss=pi)
                save(out / 'report.json', result)
                require(all(torch.equal(t.cpu(), ck[k]) for t,k in [(u,'user'),(item,'item'),(w,'extra')]), 'Parameter mutation')
                del ck, u, item, w, residual, effective, view
                check()
        require(len(result['states']) == 6 and counts['denied_data_opens'] == 0, 'Incomplete or forbidden read attempt')
        result['status'] = 'completed'
        result['screen'] = 'pass' if all(x['screen_pass'] for x in result['comparisons'].values()) else 'screen_stop'
        save(out / 'acceptance.json', {'hard_acceptance': True, 'states': 6, 'Test0': True})
    except BaseException:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        save(out / 'acceptance.json', {'hard_acceptance': False, 'error': result['error']})
        raise
    finally:
        save(out / 'report.json', result)
        save(out / 'worker_resources.json', samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.formal, 'Explicit --formal required; default does nothing')
    p = read(PROFILE)
    out = ROOT / p['output']
    if args.worker:
        require(os.environ.get('OWN_ANCHOR_WORKER') == str(os.getppid()), 'Worker must be supervised')
        worker(p, out)
        return
    from run_minimal_ranking_supervision import source_gate
    source = source_gate(dict(p, launch_source_rule='committed_head'))
    require(p['seeds'] == [2022,2023,2024] and p['states'] == ['R','C'] and p['user_batch'] == 256
            and p['test_policy'] == 'Test0' and p['screen'] == {'low_gain_retention_min':.8,'recall_C_minus_R_min':-.0002}, 'Frozen protocol')
    out.mkdir(parents=True, exist_ok=False)  # consumed once, even if execution fails
    save(out / 'manifest.json', {'source_commit': source, 'profile_sha256': sha(PROFILE),
                                'profile': p, 'command': sys.argv, 'status': 'launched'})
    import psutil
    started = time.monotonic()
    env = dict(os.environ, OWN_ANCHOR_WORKER=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    samples, reason = [], None
    with (out / 'console.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--formal', '--worker'],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:
                rss = psutil.Process(proc.pid).memory_info().rss
            except psutil.NoSuchProcess:
                break
            row = {'seconds':time.monotonic()-started, 'rss_bytes':rss,
                   'output_bytes':sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
                   'free_disk_bytes':shutil.disk_usage(out).free}
            samples.append(row)
            caps = p['caps']
            if row['seconds'] > caps['wall_seconds'] or rss > caps['rss_bytes'] or row['output_bytes'] > caps['output_bytes'] or row['free_disk_bytes'] < caps['minimum_free_disk_bytes']:
                reason = 'External resource limit'
                proc.terminate()
                break
            time.sleep(1)
        code = proc.wait()
    save(out / 'supervisor.json', {'exit_code':code,'termination_reason':reason,'samples':samples,
                                   'elapsed_seconds':time.monotonic()-started})
    save(out / 'completeness.json', {'files_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name != 'completeness.json'}})
    require(code == 0 and reason is None, 'Diagnostic failed; preserve artifacts, no retry')
    print(str(out / 'report.json'))


if __name__ == '__main__':
    main()
