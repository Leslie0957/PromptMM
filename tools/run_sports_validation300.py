"""User-invoked, one-shot serial execution of the nine declared Sports runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'exp' / 'runs' / 'sports'
PROFILES = [f'sports_student_{arm}_seed{seed}_val300_v1'
            for seed in (2022, 2023, 2024)
            for arm in ('bpr', 'full', 'image_matched')]


def verify_outcome(manifest, profile):
    args = manifest.get('resolved_arguments', {})
    if not (manifest.get('status') == 'validation_completed'
            and manifest.get('student_config_profile') == profile
            and manifest.get('final_test_performed') is False
            and manifest.get('teacher_final_test_performed') is False
            and manifest.get('student_config_overrides') == {}
            and manifest.get('dataset_config_overrides') == {}
            and args.get('epoch') == 300
            and args.get('early_stopping_patience') == 300
            and args.get('if_train_teacher') is False):
        raise RuntimeError(f'Run did not satisfy declared completion gates: {profile}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    commands = [[sys.executable, '-B', 'codes/main_mmlight.py', '--dataset', 'sports',
                 '--student_profile', profile, '--gpu_id', '0'] for profile in PROFILES]
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if dirty.strip():
        raise RuntimeError('Working tree is dirty; do not launch.')
    for path in RUNS.glob('run_manifest__*.json'):
        existing = json.loads(path.read_text(encoding='utf-8'))
        if existing.get('student_config_profile') in PROFILES:
            raise RuntimeError(f'Existing 300-epoch run found; no automatic retry: {path}')
    RUNS.mkdir(parents=True, exist_ok=True)
    record_path = RUNS / 'serial_validation300_seed2022_2024.json'
    record = dict(started_at=datetime.now().astimezone().isoformat(), launch_commit=head,
                  launch_dirty=False, python=sys.executable, commands=commands,
                  status='started', completed_profiles=[])
    with record_path.open('x', encoding='utf-8') as file:
        json.dump(record, file, indent=2)
    try:
        for profile, command in zip(PROFILES, commands):
            if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head:
                raise RuntimeError('Source commit changed during batch.')
            if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
                raise RuntimeError('Working tree changed during batch.')
            before = set(RUNS.glob('run_manifest__*.json'))
            record['active_profile'] = profile
            record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            print(f'\nStarting {profile}', flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            created = set(RUNS.glob('run_manifest__*.json')) - before
            if len(created) != 1:
                raise RuntimeError(f'Expected exactly one new manifest, got {len(created)}')
            path = created.pop()
            verify_outcome(json.loads(path.read_text(encoding='utf-8')), profile)
            record['completed_profiles'].append(dict(profile=profile, manifest=str(path)))
        record['status'] = 'completed'
    except BaseException as error:
        record['status'] = 'stopped'
        record['error'] = repr(error)
        raise
    finally:
        record['updated_at'] = datetime.now().astimezone().isoformat()
        record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
