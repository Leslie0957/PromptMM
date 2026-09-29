"""One-shot full candidate preparation from Train and the pinned fused teacher."""
import contextlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'tools'))
from run_minimal_ranking_supervision import (PROFILE, load_teacher, load_train,
    save_json, sha, test_denial, validate_profile)
from minimal_ranking_supervision import candidate_rows


def main():
    import numpy as np
    import psutil
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    validate_profile(profile)
    output = ROOT / 'exp/innovation2/minimal_ranking_candidate_prepare_seed2022_v1'
    if output.exists():
        raise RuntimeError('Candidate preparation namespace exists; no overwrite/retry')
    if shutil.disk_usage(ROOT).free < profile['caps']['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient free disk')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    process = psutil.Process()
    limits = {'wall_seconds': 600, 'rss_bytes': 3221225472,
              'output_bytes': 134217728, 'minimum_free_disk_bytes': 4294967296}
    samples = []
    status = 'failed'
    with (output / 'stdout.log').open('w', encoding='utf-8') as stdout, \
         (output / 'stderr.log').open('w', encoding='utf-8') as stderr, \
         contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            attempts = test_denial(ROOT / 'data/sports')
            def check():
                elapsed = time.monotonic() - started
                row = {'elapsed_seconds': elapsed, 'rss_bytes': process.memory_info().rss,
                    'free_disk_bytes': shutil.disk_usage(output).free,
                    'output_bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}
                samples.append(row)
                save_json(output / 'resource_samples.json', samples)
                if (elapsed > limits['wall_seconds'] or row['rss_bytes'] > limits['rss_bytes']
                        or row['output_bytes'] > limits['output_bytes']
                        or row['free_disk_bytes'] < limits['minimum_free_disk_bytes']):
                    raise RuntimeError('Preparation resource cap breached')
            check()
            train = load_train(profile)
            user, item = load_teacher(profile)
            users, ids, logits, mu, scale = candidate_rows(train, user, item,
                seed=profile['candidates']['numpy_rng_seed'], check=check)
            if len(users) != int((np.diff(train.indptr) > 0).sum()):
                raise RuntimeError('Train user count mismatch')
            logical_sha = __import__('hashlib').sha256(
                ids.tobytes() + logits.tobytes() + mu.tobytes() + scale.tobytes()).hexdigest()
            candidate_path = output / 'candidates.npz'
            np.savez_compressed(candidate_path, users=users, ids=ids, logits=logits,
                                mu=mu, scale=scale)
            check()
            save_json(output / 'manifest.json', {'mode': 'full_candidate_preparation_no_training',
                'source_commit_at_preparation': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'profile_sha256_at_preparation': sha(PROFILE), 'inputs':
                {key: profile['assets'][key] for key in ('train', 'teacher_tables')},
                'users': len(users), 'candidate_shape': list(ids.shape),
                'candidate_file_sha256': sha(candidate_path), 'candidate_logical_sha256': logical_sha,
                'test_denied_open_attempts': attempts['denied_test_open_attempts'],
                'validation_reads': 0, 'training_steps': 0, 'limits': limits})
            save_json(output / 'acceptance.json', {'status': 'passed', 'users': len(users),
                'candidate_file_sha256': sha(candidate_path), 'candidate_logical_sha256': logical_sha,
                'test_denied_open_attempts': attempts['denied_test_open_attempts'],
                'validation_reads': 0, 'training_steps': 0})
            status = 'completed'
        except BaseException:
            traceback.print_exc()
            save_json(output / 'acceptance.json', {'status': 'failed',
                'reason': 'See stderr.log; no automatic retry'})
        finally:
            save_json(output / 'exit.json', {'status': status,
                'exit_code': 0 if status == 'completed' else 1,
                'elapsed_seconds': time.monotonic() - started})
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
