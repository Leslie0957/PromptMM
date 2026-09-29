"""Read-only artifact verification; no dataset, teacher, scoring or resampling."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = ROOT / 'exp/innovation2/modal_neighbor_v1'
    read = lambda name: json.loads((out / name).read_text(encoding='utf-8'))
    exit_record, acceptance = read('exit.json'), read('acceptance.json')
    report, manifest, seal = read('report.json'), read('manifest.json'), read('graph_seal.json')
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    check('exit/acceptance', exit_record['status'] == 'completed' and acceptance['status'] == 'passed')
    for record_name, record in [('exit', exit_record), ('acceptance', acceptance), ('seal', {'hashes': seal['file_hashes']})]:
        for name, value in record['hashes'].items():
            check(record_name + ':' + name, sha(out / name) == value)
    check('profile', manifest['profile_sha'] == sha(ROOT / 'docs/research/innovation2/MODAL_NEIGHBOR_PROFILE_V1.json'))
    check('chronology', seal['time'] <= report['validation_started'] and not seal['validation_loaded'])
    check('guards', acceptance['test_denied_attempts'] == acceptance['validation_early_denied_attempts'] == 0)
    samples = [json.loads(x) for x in (out / 'resource.jsonl').read_text().splitlines()]
    caps = manifest['profile']['caps']
    for key in ['seconds', 'rss', 'output_bytes']:
        check('resource:' + key, max(s[key] for s in samples) <= caps[key])
    check('disk', min(s['free_disk'] for s in samples) >= caps['minimum_free_disk'])
    check('final_output', sum(p.stat().st_size for p in out.iterdir() if p.is_file()) <= caps['output_bytes'])
    for mode in ['image', 'text']:
        with np.load(out / (mode + '_graphs.npz'), allow_pickle=False) as z:
            q, c, degree = z['queries'], z['candidates'], z['degree']
            real, random = z['real'], z['random']
            check(mode + ':shapes', real.shape == (len(q), 20) and random.shape == (100, len(q), 20))
            for key in z.files:
                a = z[key]
                h = hashlib.sha256()
                h.update(str(a.dtype).encode()); h.update(json.dumps(list(a.shape)).encode()); h.update(a.tobytes(order='C'))
                check(mode + ':logical:' + key, h.hexdigest() == seal['array_hashes'][mode][key])
                check(mode + ':finite:' + key, np.isfinite(a).all())
            check(mode + ':candidate_positive_degree', (degree[c] > 0).all())
            check(mode + ':real_valid', np.isin(real, c).all() and not (real == q[:, None]).any())
            check(mode + ':real_unique', (np.diff(np.sort(real, axis=1), axis=1) != 0).all())
            check(mode + ':real_order', (np.diff(z['similarities'], axis=1) <= 0).all())
            expected = np.sort(degree[real], axis=1)
            for rep, graph in enumerate(random):
                check(f'{mode}:random:{rep}', np.isin(graph, c).all() and not (graph == q[:, None]).any()
                      and (np.diff(np.sort(graph, axis=1), axis=1) != 0).all()
                      and np.array_equal(np.sort(degree[graph], axis=1), expected))
            variables = []
            for i, row in zip(q, real):
                ds, n = np.unique(degree[row], return_counts=True)
                sizes = np.array([np.sum((degree[c] == d) & (c != i)) for d in ds])
                variables.append(n[sizes > n].sum() / 20)
            check(mode + ':variable', np.array_equal(variables, z['variable']))
        row = report['modalities'][mode]
        with np.load(out / (mode + '_summaries.npz'), allow_pickle=False) as z:
            for group in ['user', 'target']:
                den = z[group + '_denominator']; n = den.sum()
                check(mode + ':' + group + ':den', n == report['support']['pairs'])
                for field, key in [('real', 'real_sum'), ('null', 'null_sum'), ('delta', 'delta_sum')]:
                    check(mode + ':' + group + ':' + field, abs(z[group + '_' + key].sum()/n - row[field]) < 1e-12)
                if group == 'target':
                    check(mode + ':target_macro', abs(np.mean(z['target_delta_sum']/den)-row['target_macro_delta']) < 1e-12)
            check(mode + ':random_mean', abs(z['null_repeat_rates'].mean()-row['null']) < 1e-12)
        cov, matching = report['support']['coverage'], row['matching_support']
        u, i = row['user_ci'], row['item_ci']
        if cov < .95:
            decision = 'inconclusive_coverage'
        elif matching < .8:
            decision = 'inconclusive_matching_support'
        elif row['delta'] >= .005 and u[0] > 0 and i[0] > 0:
            decision = 'candidate_signal'
        elif u[1] < .005 and i[1] < .005:
            decision = 'screen_stop'
        else:
            decision = 'inconclusive'
        check(mode + ':screen', decision == row['screen'])
    print(json.dumps({'status': 'passed', 'checks': len(checks), 'source': manifest['source'],
                      'report_sha': sha(out / 'report.json'),
                      'scope': 'Saved artifacts only; no independent reconstruction from Train/Validation or cosine/bootstrap rerun'}, indent=2))


if __name__ == '__main__':
    main()
