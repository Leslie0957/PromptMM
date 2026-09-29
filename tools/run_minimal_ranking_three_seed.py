"""One manually launched, serial three-seed B/R/A/M formal cohort."""
import argparse
import contextlib
import copy
import json
import os
import shutil
import time
import traceback

from run_minimal_ranking_supervision import ROOT, run, save_json, sha, source_gate, validate_profile

PROFILE = ROOT / 'docs/research/innovation2/MINIMAL_RANKING_THREE_SEED_PROFILE_V1.json'


def build_seed_profiles(cohort):
    if (cohort['status'] != 'manual_launch_ready' or cohort['ordered_seeds'] != [2022, 2023, 2024]
            or cohort['test_policy'] != 'Test0'
            or cohort['failure_policy'] != 'stop_preserve_no_retry_no_next_seed'
            or cohort['cohort_caps'] != {'wall_seconds': 86400, 'output_bytes': 6442450944,
                                         'minimum_free_disk_bytes': 4294967296, 'attempts': 1}
            or set(cohort['seeds']) != {'2022', '2023', '2024'}):
        raise ValueError('Fixed cohort protocol mismatch')
    base_path = ROOT / cohort['base_profile']
    if sha(base_path) != cohort['base_profile_sha256']:
        raise RuntimeError('Base profile SHA mismatch')
    base = json.loads(base_path.read_text(encoding='utf-8'))
    profiles = []
    for seed in cohort['ordered_seeds']:
        delta = cohort['seeds'][str(seed)]
        profile = copy.deepcopy(base)
        profile['seed'] = seed
        profile['anchor_manifest'] = delta['anchor_manifest']
        profile['anchor_manifest_sha256'] = delta['anchor_manifest_sha256']
        profile['assets']['triplet_tape'] = delta['triplet_tape']
        profile['evaluation']['B_epoch300_expected'] = delta['B_epoch300_expected']
        profile['branch'] = cohort['branch']
        profile['launch_source_rule'] = cohort['launch_source_rule']
        profile['command'] = cohort['command']
        profile['output'] = cohort['output'] + f'/seed{seed}'
        validate_profile(profile)
        profiles.append(profile)
    return profiles


def preflight_assets(profiles):
    shared = profiles[0]['prepared_candidates']
    if sha(ROOT / shared['path']) != shared['sha256']:
        raise RuntimeError('Prepared candidate SHA mismatch')
    for profile in profiles:
        tape = profile['assets']['triplet_tape']
        if sha(ROOT / tape['path']) != tape['sha256']:
            raise RuntimeError(f"Seed {profile['seed']} tape SHA mismatch")
    for asset in ('teacher_tables', 'train', 'validation'):
        if not (ROOT / profiles[0]['assets'][asset]['path']).is_file():
            raise RuntimeError(f'Missing shared asset: {asset}')


def run_ordered(profiles, run_one):
    for profile in profiles:
        run_one(profile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal', action='store_true', required=True)
    parser.parse_args()
    cohort = json.loads(PROFILE.read_text(encoding='utf-8'))
    profiles = build_seed_profiles(cohort)
    head = source_gate(cohort)
    preflight_assets(profiles)
    output = ROOT / cohort['output']
    if output.exists():
        raise RuntimeError('Cohort output exists; no retry/overwrite')
    if shutil.disk_usage(ROOT).free < cohort['cohort_caps']['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient free disk')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {'source_commit': head, 'source_state': 'clean_committed',
                'profile': str(PROFILE.relative_to(ROOT)).replace('\\', '/'),
                'profile_sha256': sha(PROFILE), 'base_profile_sha256': cohort['base_profile_sha256'],
                'command': cohort['command'], 'ordered_seeds': cohort['ordered_seeds'],
                'test_policy': cohort['test_policy'], 'cohort_caps': cohort['cohort_caps'],
                'seed_status': {str(seed): 'not_started' for seed in cohort['ordered_seeds']}}
    save_json(output / 'manifest.json', manifest)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = '2022'

    def check_cohort():
        size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if (time.monotonic() - started > cohort['cohort_caps']['wall_seconds']
                or size > cohort['cohort_caps']['output_bytes']
                or shutil.disk_usage(ROOT).free < cohort['cohort_caps']['minimum_free_disk_bytes']):
            raise RuntimeError('Cohort resource cap breached')
        if source_gate(cohort) != head:
            raise RuntimeError('Source changed during cohort')

    def run_one(profile):
        check_cohort()
        seed = profile['seed']
        folder = output / f'seed{seed}'
        folder.mkdir(exist_ok=False)
        seed_started = time.monotonic()
        seed_manifest = {'source_commit': head, 'source_state': 'clean_committed',
                         'profile': manifest['profile'], 'profile_sha256': manifest['profile_sha256'],
                         'effective_profile': profile, 'command': cohort['command'],
                         'assets': profile['assets'], 'caps': profile['caps'], 'test_policy': 'Test0',
                         'source_files_sha256': {name: sha(ROOT / name) for name in (
                             'codes/minimal_ranking_supervision.py',
                             'tools/run_minimal_ranking_supervision.py',
                             'tools/run_minimal_ranking_three_seed.py',
                             'codes/td_distill_model_no_projection.py',
                             'codes/initialization_kd_adapter.py',
                             'codes/initialization_kd_fast_eval.py')}}
        save_json(folder / 'manifest.json', seed_manifest)
        manifest['seed_status'][str(seed)] = 'running'
        save_json(output / 'manifest.json', manifest)
        with (folder / 'stdout.log').open('w', encoding='utf-8') as stdout, \
             (folder / 'stderr.log').open('w', encoding='utf-8') as stderr, \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                run(profile, folder, True, seed_started, seed_manifest)
                check_cohort()
                status = 'completed'
            except BaseException:
                traceback.print_exc()
                status = 'failed'
                save_json(folder / 'acceptance.json', {'status': 'failed',
                          'reason': 'See stderr.log; partial assets preserved, no retry'})
            finally:
                save_json(folder / 'exit.json', {'status': status,
                    'exit_code': 0 if status == 'completed' else 1,
                    'elapsed_seconds': time.monotonic() - seed_started})
        manifest['seed_status'][str(seed)] = status
        save_json(output / 'manifest.json', manifest)
        if status != 'completed':
            raise RuntimeError(f'Seed {seed} failed; no later seed launched')

    try:
        run_ordered(profiles, run_one)
        status = 'completed'
    except BaseException:
        status = 'failed'
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    save_json(output / 'exit.json', {'status': status, 'exit_code': 0 if status == 'completed' else 1,
              'elapsed_seconds': time.monotonic() - started,
              'seed_status': manifest['seed_status']})
    return 0 if status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
