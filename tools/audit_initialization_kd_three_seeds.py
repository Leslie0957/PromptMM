"""Read-only raw-artifact audit; stdout JSON only, no project imports or ranking.

Run from any directory with the CPU-capable run_5060 Python. Only existing cohort
outputs and committed JSON/source blobs are read; data splits are never opened.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def deny_data(event, args):
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        p = Path(args[0]).resolve()
        if ROOT / 'data' in p.parents:
            raise PermissionError('Audit cannot open any data split')


sys.addaudithook(deny_data)
import torch
torch.set_num_threads(4)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def blob(commit, path):
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], cwd=ROOT)


def interaction(values):
    r, t = values['R1'] - values['R0'], values['T1'] - values['T0']
    return {'delta_R': r, 'delta_T': t, 'I': r - t}


def audit(seed):
    out = ROOT / f'exp/initialization_kd_interaction/sports_init_kd_interaction_seed{seed}_v1'
    m, r, a, s, telemetry = [read(out / n) for n in
        ('launch_manifest.json', 'report.json', 'acceptance.json', 'supervisor.json', 'telemetry.json')]
    cfg, commit = m['config'], m['source_commit']
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    for obj in (m, r, a):
        require(obj['source_commit'] == commit and obj['config_digest'] == digest, 'identity')
    require(r['status'] == s['status'] == 'completed' and s['exit_code'] == 0
            and s['artifact_acceptance'] == a['status'] == 'passed', 'completion')
    require(not (out / 'failure.json').exists(), 'failure artifact')
    require(r['test_file_reads'] == 0 and r['selection_split'] == 'Validation'
            and r['validation_calls'] == 1204 and r['evaluator'] == 'exact_topk_v1', 'protocol')
    # Reconstruct sealed configuration from launch-commit blobs, without importing runners.
    base_path = 'docs/research/INITIALIZATION_KD_COHORT_CONFIG_2026-09-27.json'
    delta = json.loads(blob(commit, base_path))
    base = json.loads(blob(commit, delta['base_config']))
    require(sha(ROOT / delta['base_config']) == delta['base_config_sha256'], 'base byte anchor')
    spec = dict(base, **delta)
    if seed != 2022:
        d = json.loads(blob(commit, f'docs/research/INITIALIZATION_KD_SEED{seed}_COHORT_CONFIG_2026-09-28.json'))
        require(sha(ROOT / d['base_cohort_config']) == d['base_cohort_sha256'], 'cohort anchor')
        require(sha(ROOT / d['asset_preflight']) == d['asset_preflight_sha256'], 'preflight anchor')
        preflight = json.loads(blob(commit, d['asset_preflight']))
        for name in ('random_initial', 'triplet_tape'):
            require(all(d[name][k] == preflight['assets'][name][k] for k in ('path', 'sha256')), 'preflight assets')
            spec['common']['assets'][name].update(d[name])
        for key in ('seed', 'run_id', 'output_namespace', 'launch_command'):
            spec[key] = d[key]
    require(spec == cfg and cfg['seed'] == seed and m['assets'] == cfg['common']['assets'], 'resolved config')
    sources = ('codes/initialization_kd_interaction.py', 'codes/initialization_kd_adapter.py',
               'tools/run_initialization_kd_interaction.py', 'codes/td_distill_model_no_projection.py')
    source_hashes = {p: hashlib.sha256(blob(commit, p)).hexdigest() for p in sources}
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    require(len(a['hashes']) == 13 and all(hashes[k] == v for k, v in a['hashes'].items()), 'accepted hashes')
    arms, curves = {}, {}
    require(set(r['arms']) == {'R0', 'R1', 'T0', 'T1'}, 'arms')
    for arm in ('R0', 'R1', 'T0', 'T1'):
        c, ar = read(out / arm / 'curve.json'), r['arms'][arm]
        require([v['epoch'] for v in c] == list(range(1, 301)), '300 epochs')
        for v in [ar['initial_validation']] + [row['validation'] for row in c]:
            require(v['validation_users'] == 35598, 'validation population')
            require(v['recall20'] == v['recall'][1] and v['ndcg20'] == v['ndcg'][1], 'K20 mapping')
            require(all(math.isfinite(x) and 0 <= x <= 1 for x in v['recall'] + v['ndcg']), 'metric finite')
        vals = [v['validation']['recall20'] for v in c]
        best = max(range(300), key=lambda i: vals[i])
        require(ar['selection'] == {'best_index': best, 'fixed_final_index': 299}, 'earliest max')
        require(ar['optimizer_steps'] == 64200 and ar['final_validation'] == c[-1]['validation'], 'final')
        require(all(f'{arm}_validation_epoch{e}' in r['timings'] for e in range(301)), '1204 timing records')
        for label, index in (('best', best), ('final', 299)):
            rel = f'{arm}/{label}.pt'
            require(ar[label + '_checkpoint'] == rel, 'checkpoint pointer')
            ck = torch.load(out / rel, map_location='cpu', weights_only=True)
            require(ck['arm'] == arm and ck['epoch'] == index + 1 and ck['metric'] == c[index]['validation']
                    and ck['source_commit'] == commit and ck['config_digest'] == digest, 'checkpoint binding')
            shapes = {'user_id_embedding.weight': (35598, 64), 'item_id_embedding.weight': (18357, 64)}
            require(set(ck['model']) == set(shapes), 'ID-only parameters')
            for k, shape in shapes.items():
                t = ck['model'][k]
                require(tuple(t.shape) == shape and t.dtype == torch.float32 and torch.isfinite(t).all().item(), 'model tensor')
            opt = ck['optimizer']
            require(len(opt['param_groups']) == 1 and len(opt['state']) == 2, 'AdamW state count')
            group = opt['param_groups'][0]
            for k, v in dict(lr=6e-5, weight_decay=.01, eps=1e-8, betas=(.9, .999), amsgrad=False).items():
                require(group[k] == v, 'AdamW ' + k)
            require(len(group['params']) == 2 and set(group['params']) == set(opt['state']), 'parameter binding')
            for pid, shape in zip(group['params'], shapes.values()):
                state = opt['state'][pid]
                require(float(state['step']) == (index + 1) * 214, 'AdamW step')
                for k in ('exp_avg', 'exp_avg_sq'):
                    require(tuple(state[k].shape) == shape and state[k].dtype == torch.float32
                            and torch.isfinite(state[k]).all().item(), 'moments')
                require((state['exp_avg_sq'] >= 0).all().item(), 'negative second moment')
            del ck
        curves[arm] = vals
        arms[arm] = dict(initial=ar['initial_validation']['recall20'], final=vals[-1],
                         final_ndcg20=ar['final_validation']['ndcg20'], best_epoch=best + 1,
                         best=vals[best], last20_mean=sum(vals[-20:])/20,
                         last50_gain=vals[-1]-vals[249], steps=ar['optimizer_steps'])
    require(r['arms']['R0']['initial_validation'] == r['arms']['R1']['initial_validation']
            and r['arms']['T0']['initial_validation'] == r['arms']['T1']['initial_validation'], 'paired epoch0')
    primary = interaction({k:v['final'] for k,v in arms.items()})
    require(primary == r['interaction'], 'primary interaction')
    cap = cfg['hard_caps']
    require(cap['parent_wall_seconds'] is None and cap['attempts'] == 1, 'caps')
    require(r['peak_cuda_reserved_bytes'] <= cap['cuda_allocator_bytes'] and
            r['peak_sampled_rss_bytes'] <= cap['process_rss_bytes'] and
            telemetry['output_bytes'] <= cap['output_bytes'] and
            telemetry['free_disk_bytes'] >= cap['minimum_free_disk_bytes'], 'resources')
    diff = [x-y for x,y in zip(curves['T1'], curves['T0'])]
    return dict(seed=seed, status='passed', source_commit=commit, config_digest=digest,
        source_blob_sha256=source_hashes, assets=m['assets'], command=cfg['launch_command'],
        arms=arms, primary=primary, best=interaction({k:v['best'] for k,v in arms.items()}),
        last20=interaction({k:v['last20_mean'] for k,v in arms.items()}),
        warm_epoch_sign_counts=dict(positive=sum(x>0 for x in diff), negative=sum(x<0 for x in diff), zero=sum(x==0 for x in diff)),
        validation_calls=r['validation_calls'], test_file_reads=r['test_file_reads'],
        parent_seconds=s['elapsed_seconds'], worker_seconds=r['runtime_seconds'],
        peak_cuda_bytes=r['peak_cuda_reserved_bytes'], peak_sampled_rss_bytes=r['peak_sampled_rss_bytes'],
        final_telemetry=telemetry, file_sha256=hashes)


if __name__ == '__main__':
    runs = [audit(seed) for seed in (2022, 2023, 2024)]
    normalized = []
    for seed in (2022, 2023, 2024):
        cfg = read(ROOT / f'exp/initialization_kd_interaction/sports_init_kd_interaction_seed{seed}_v1/launch_manifest.json')['config']
        for key in ('seed', 'run_id', 'output_namespace', 'launch_command'):
            del cfg[key]
        for key in ('random_initial', 'triplet_tape'):
            del cfg['common']['assets'][key]
        normalized.append(cfg)
    require(all(c == normalized[0] for c in normalized), 'unexpected cross-seed configuration delta')
    for key in ('shared_cache', 'teacher_checkpoint_record_only', 'train_mat_record_only', 'val_mat_record_only'):
        require(all(r['assets'][key] == runs[0]['assets'][key] for r in runs), 'common anchor')
    for key in ('random_initial', 'triplet_tape'):
        require(len({r['assets'][key]['sha256'] for r in runs}) == 3, 'distinct seed assets')
    require(all(r['source_blob_sha256'] == runs[0]['source_blob_sha256'] for r in runs), 'shared core source')
    print(json.dumps(dict(status='passed', audit_mode='CPU raw artifacts only; no data split or model forward',
        asset_scope='manifest/committed anchor comparison; unchanged large inputs not rehashed',
        runs=runs, mean_primary={k:sum(r['primary'][k] for r in runs)/3 for k in ('delta_R','delta_T','I')}), indent=2))
