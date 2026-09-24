"""User-launched, one-attempt serial Sports Full/image paired-cold cohort."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from sports_paired_cold import sha256
from utility.dataset_profiles import SPORTS_PAIRED_COLD_PROFILES, SPORTS_STUDENT_PROFILES
from run_sports_validation300 import verify_outcome

PROFILES = [f'sports_student_{arm}_pairedcold_seed{seed}_val300_v1'
            for seed in (2022, 2023, 2024)
            for arm in ('full', 'image_matched')]
REPORT = ROOT / 'exp' / 'paired_coldinit' / 'sports_three_seed_v1' / 'batch.json'
RUNS = ROOT / 'exp' / 'runs' / 'sports'
TEACHER_SHA = '57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea'
TRAIN_SHA = '5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8'


def source_head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def check_source(head):
    if source_head() != head or subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Committed clean launch source changed')


def commands(python):
    return [[str(python), '-B', 'codes/main_mmlight.py', '--dataset', 'sports',
             '--student_profile', profile, '--gpu_id', '0'] for profile in PROFILES]


def verify(path, profile):
    import torch
    manifest = json.loads(path.read_text(encoding='utf-8'))
    verify_outcome(manifest, profile)
    args = manifest['resolved_arguments']
    for name, value in SPORTS_STUDENT_PROFILES[profile]['defaults'].items():
        if args.get(name) != value:
            raise RuntimeError('Resolved argument mismatch: ' + name)
    seed, arm = SPORTS_PAIRED_COLD_PROFILES[profile]
    if (manifest.get('paired_cold_seed'), manifest.get('paired_cold_arm')) != (seed, arm):
        raise RuntimeError('Paired identity mismatch')
    if manifest.get('paper_ready_eligible') is not False or args.get('run_final_test') is not False:
        raise RuntimeError('Unexpected Test or eligibility policy')
    teacher = manifest['teacher_checkpoint_fingerprint']
    if teacher['sha256'] != TEACHER_SHA:
        raise RuntimeError('Teacher identity mismatch')
    initial = manifest['paired_cold_initial']
    tape = manifest['paired_cold_triplets']
    if sha256(initial['path']) != initial['sha256'] or sha256(tape['path']) != tape['sha256']:
        raise RuntimeError('Paired asset changed')
    if tape['shape'] != [300, 214, 3, 1024] or tape['triplet_batches_consumed'] != 64200:
        raise RuntimeError('Incomplete or wrong-shape triplet replay')
    for fingerprint in manifest['code_fingerprints'].values():
        if sha256(fingerprint['path']) != fingerprint['sha256']:
            raise RuntimeError('Run source fingerprint changed')
    with Path(manifest['artifacts']['converge_run']).open('rb') as stream:
        curve = pickle.load(stream)
    for key in ('td_batch_loss_List', 'td_bpr_loss_List', 'td_distill_loss_List',
                'selection_recall20_List', 'selection_ndcg20_List'):
        if len(curve[key]) != 300 or not all(math.isfinite(float(x)) for x in curve[key]):
            raise RuntimeError('Incomplete or nonfinite curve: ' + key)
    best = max(range(300), key=lambda i: curve['selection_recall20_List'][i])
    if manifest['best_selection_epoch'] != best or not math.isclose(
            manifest['best_selection_recall'], float(curve['selection_recall20_List'][best]),
            rel_tol=0, abs_tol=1e-12):
        raise RuntimeError('Validation checkpoint selection mismatch')
    checkpoint = Path(manifest['td_full_checkpoint'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['best_epoch'] != best or state['td_init_from_teacher'] is not False:
        raise RuntimeError('Checkpoint identity mismatch')
    if not all(torch.isfinite(t).all() for t in state['model_state_dict'].values()):
        raise RuntimeError('Nonfinite checkpoint')
    return {'profile': profile, 'seed': seed, 'arm': arm, 'manifest': str(path),
            'manifest_sha256': sha256(path), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': sha256(checkpoint), 'best_epoch': best,
            'best_recall20': float(curve['selection_recall20_List'][best]),
            'initial': initial, 'triplets': tape}


def run(python=sys.executable):
    if Path(python).resolve() != Path('D:/miniconda/envs/run_5060/python.exe').resolve():
        raise RuntimeError('Declared run_5060 Python environment required')
    head = source_head()
    check_source(head)
    teacher = ROOT / SPORTS_STUDENT_PROFILES[PROFILES[0]]['defaults']['teacher_checkpoint']
    if sha256(teacher) != TEACHER_SHA or sha256(ROOT / 'data/sports/train_mat') != TRAIN_SHA:
        raise RuntimeError('Shared teacher or training matrix anchor changed')
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise RuntimeError('At least 4 GiB free disk needed for three triplet tapes')
    if REPORT.exists():
        raise FileExistsError('Existing batch attempt; no retry: ' + str(REPORT))
    for path in RUNS.glob('run_manifest__*.json'):
        if json.loads(path.read_text(encoding='utf-8')).get('student_config_profile') in PROFILES:
            raise FileExistsError('Existing paired run attempt: ' + str(path))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'started', 'started_at': datetime.now().astimezone().isoformat(),
              'launch_commit': head, 'launch_dirty': False, 'python': str(python),
              'commands': commands(python), 'completed': []}

    def save():
        REPORT.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')

    with REPORT.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    try:
        for profile, command in zip(PROFILES, commands(python)):
            check_source(head)
            if sha256(teacher) != TEACHER_SHA or sha256(ROOT / 'data/sports/train_mat') != TRAIN_SHA:
                raise RuntimeError('Shared teacher or training matrix changed during batch')
            record['active_profile'] = profile
            save()
            before = set(RUNS.glob('run_manifest__*.json'))
            print('Starting', profile, flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            check_source(head)
            if sha256(teacher) != TEACHER_SHA or sha256(ROOT / 'data/sports/train_mat') != TRAIN_SHA:
                raise RuntimeError('Shared teacher or training matrix changed during batch')
            created = set(RUNS.glob('run_manifest__*.json')) - before
            if len(created) != 1:
                raise RuntimeError('Expected one new run manifest')
            result = verify(created.pop(), profile)
            if result['arm'] == 'image_matched':
                first = record['completed'][-1]
                if (first['seed'] != result['seed'] or
                    first['initial'] != result['initial'] or
                    first['triplets'] != result['triplets']):
                    raise RuntimeError('Paired initial or triplet identity differs')
            record['completed'].append(result)
            save()
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error=repr(error))
        raise
    finally:
        record['updated_at'] = datetime.now().astimezone().isoformat()
        save()
        print('Batch report:', REPORT, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        for command in commands(sys.executable):
            print(subprocess.list2cmdline(command))
    else:
        run()


if __name__ == '__main__':
    main()
