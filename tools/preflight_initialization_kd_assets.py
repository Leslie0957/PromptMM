"""One CPU-only, read-only asset preflight; never constructs a model or ranks."""
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from initialization_kd_interaction import Protocol, install_test_read_denial
from initialization_kd_adapter import sha256, load_pinned_tensors, load_train_val


def main():
    import numpy as np
    import psutil
    import torch

    started = time.monotonic()
    install_test_read_denial(ROOT / 'data/sports')
    config = ROOT / 'docs/research/INITIALIZATION_KD_INTERACTION_CONFIG_2026-09-27.json'
    spec = json.loads(config.read_text(encoding='utf-8'))
    output = ROOT / 'docs/research/INITIALIZATION_KD_ASSET_PREFLIGHT_2026-09-27.json'
    # No overwrite/resume of an earlier preflight report.
    with output.open('x', encoding='utf-8') as stream:
        stream.write('{}\n')
    report = {'status': 'partial', 'config_sha256': sha256(config),
              'assets': {}, 'model_instances': 0, 'rankings': 0,
              'test_reads': 0, 'cuda_allocations': 0}
    try:
        for name, asset in spec['common']['assets'].items():
            path = ROOT / asset['path']
            actual = sha256(path)
            report['assets'][name] = {'path': asset['path'], 'sha256': actual,
                                      'bytes': path.stat().st_size,
                                      'matches_expected': actual == asset['sha256']}
            if actual != asset['sha256']:
                raise RuntimeError('Asset identity conflict: ' + name)
        p = Protocol()
        random, teacher, semantics, tape = load_pinned_tensors(ROOT, spec, p)
        train, val = load_train_val(ROOT / 'data/sports',
            spec['common']['assets']['train_mat_record_only']['sha256'],
            spec['common']['assets']['val_mat_record_only']['sha256'],
            p.n_users, p.n_items)
        report['tensors'] = {}
        for group, tables in (('random', random), ('teacher', teacher), ('semantics', semantics)):
            for name, tensor in tables.items():
                report['tensors'][group + '/' + name] = {
                    'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                    'finite': bool(torch.isfinite(tensor).all()), 'device': str(tensor.device)}
        users = np.flatnonzero(np.diff(val.indptr))
        report['matrices'] = {'shape': list(train.shape), 'train_nnz': int(train.nnz),
                              'val_nnz': int(val.nnz), 'val_users': int(len(users)),
                              'overlap_nnz': int(train.multiply(val).nnz),
                              'minimum_candidates': int((p.n_items - np.diff(train.indptr)[users]).min())}
        if report['matrices']['overlap_nnz'] or report['matrices']['minimum_candidates'] < 50:
            raise RuntimeError('Train/Val overlap or insufficient candidates')
        positives = negatives = 0
        for epoch in range(p.epochs):
            if time.monotonic() - started > 120:
                raise RuntimeError('Preflight 120-second wall budget exceeded')
            u, pos, neg = (tape[epoch, :, i, :].reshape(-1) for i in range(3))
            positives += int(np.count_nonzero(np.asarray(train[u, pos]).ravel() <= 0))
            negatives += int(np.count_nonzero(np.asarray(train[u, neg]).ravel() != 0))
        report['tape'] = {'shape': list(tape.shape), 'dtype': str(tape.dtype),
                           'writeable': bool(tape.flags.writeable),
                           'checked_triplets': int(p.epochs * p.batches_per_epoch * p.batch_size),
                           'positive_not_in_train': positives, 'negative_in_train': negatives}
        if positives or negatives:
            raise RuntimeError('Replay membership conflict')
        report['resources'] = {'disk_free_bytes': shutil.disk_usage(ROOT).free,
                                'available_ram_bytes': psutil.virtual_memory().available,
                                'preflight_rss_bytes': psutil.Process().memory_info().rss,
                                'gpu_gate': 'not_executed'}
        report['formal_output_exists'] = (ROOT / spec['output_namespace']).exists()
        if report['formal_output_exists']:
            raise RuntimeError('Formal output namespace already exists')
        report['status'] = 'passed_cpu_asset_preflight_only'
    except Exception as exc:
        report['status'] = 'failed'
        report['failure'] = str(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': report['status'], 'report': str(output),
                          'elapsed_seconds': report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
