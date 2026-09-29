"""One read-only pinned epoch-zero Validation regression; no optimization/Test."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from run_neighbor_shared_residual import (PROFILE, ROOT, check_profile, load_teacher,
    load_train, rank, save_json, test_denial, verify_assets)


def main():
    import numpy as np
    import psutil
    import torch
    from initialization_kd_adapter import load_train_val
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    check_profile(profile)
    verify_assets(profile)
    output = ROOT / 'exp/innovation2/neighbor_shared_residual_epoch0_preflight_v1'
    if output.exists():
        raise RuntimeError('Preflight namespace already exists')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    denied = test_denial(ROOT / 'data/sports')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    process = psutil.Process()
    def check():
        if (time.monotonic() - started > 600 or process.memory_info().rss > 6442450944
                or torch.cuda.memory_reserved('cuda:0') > 2147483648):
            raise RuntimeError('Preflight resource cap')
    try:
        user, item = load_teacher({'assets': {'teacher_tables': profile['assets']['teacher']}})
        train, val = load_train_val(ROOT / 'data/sports', profile['assets']['train']['sha256'],
                profile['assets']['validation']['sha256'], 35598, 18357)
        degree = np.bincount(train.indices, minlength=18357)
        order = np.lexsort((np.arange(len(degree)), degree))
        groups = np.empty(18357, dtype=np.int8)
        for g, part in enumerate(np.array_split(order, 3)):
            groups[part] = g
        view = SimpleNamespace(user_id_embedding=SimpleNamespace(weight=user.to('cuda:0')),
                               item_id_embedding=SimpleNamespace(weight=item.to('cuda:0')))
        metric, _ = rank(view, train, val, groups, check, True)
        expected = profile['evaluation']['B_epoch0_expected']
        if (abs(metric['recall20'] - expected) > 1e-6
                or metric['groups']['denominator'] != [6347, 6389, 25163]
                or denied['denied_test_open_attempts'] != 0):
            raise RuntimeError('Epoch0 score/denominator/Test regression')
        save_json(output / 'report.json', {'status': 'passed', 'metric': metric,
            'expected': expected, 'validation_calls': 1, 'training_steps': 0,
            'test_access': 0, 'seconds': time.monotonic() - started,
            'sampled_rss_bytes': process.memory_info().rss,
            'cuda_allocator_bytes': torch.cuda.memory_reserved('cuda:0')})
        return 0
    except BaseException as exc:
        save_json(output / 'report.json', {'status': 'failed', 'error': str(exc),
            'training_steps': 0, 'test_access': denied['denied_test_open_attempts']})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
