"""One declared, fail-fast user-run cohort: Baby release 3 + selected final Test 12."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / 'exp/formal_closeout_cohort/innovation1_fixed_v1'
SEEDS = (2022, 2023, 2024)
SLOTS = tuple(f'{arm}_{seed}' for arm in ('b3', 's1', 's2', 's3') for seed in SEEDS)
V1_BATCH_SHA256 = '0ce638d4a963de18e521730ee67766da8e8e5643ec7d71b8a140de7886a6fbef'
V1_PREFLIGHT_SHA256 = '6bf4cc712f7185b1edd99c58127f3f5763493a0bd7dfba071cb3c45c392a6fed'


def now():
    return datetime.now().astimezone().isoformat()


def plan(cohort_id=None):
    python = sys.executable
    baby = ROOT / 'codes/promptmm_release_baby_formal.py'
    evaluator = ROOT / 'codes/formal_closeout_eval.py'
    steps = []
    for seed in SEEDS:
        command = [python, '-B', str(baby), '--baby_release_formal', '--seed', str(seed)]
        if cohort_id:
            command += ['--cohort-id', cohort_id]
        steps.append(dict(name=f'train_b3_{seed}', phase='baby_training',
                          command=command))
    for slot in SLOTS:
        command = [python, '-B', str(evaluator), '--preflight', '--slot', slot]
        if cohort_id:
            command += ['--cohort-id', cohort_id]
        steps.append(dict(name=f'preflight_{slot}', phase='no_test_validation',
                          command=command))
    for slot in SLOTS:
        command = [python, '-B', str(evaluator), '--final-test-once', '--slot', slot]
        if cohort_id:
            command += ['--cohort-id', cohort_id]
        steps.append(dict(name=f'final_test_{slot}', phase='final_test_once',
                          command=command))
    return steps


def clean_head():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Clean committed source required.')
    return head


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_step_artifacts(step, cohort_id=None):
    """Treat the child's exit code as necessary, not sufficient, evidence."""
    name, phase = step['name'], step['phase']
    if phase == 'baby_training':
        slot = name.removeprefix('train_')
        seed = int(slot.rsplit('_', 1)[1])
        baby_root = ROOT / 'exp/promptmm_release_baby'
        if cohort_id:
            baby_root = baby_root / cohort_id
        directory = baby_root / f'baby_promptmm_release_cap1000_patience7_seed{seed}_lr6e5_v1'
        path = directory / 'report.json'
    elif phase == 'no_test_validation':
        slot = name.removeprefix('preflight_')
        preflight_root = ROOT / 'exp/formal_closeout_preflight'
        path = ((preflight_root / cohort_id / slot) if cohort_id else
                (preflight_root / slot)) / 'report.json'
    elif phase == 'final_test_once':
        slot = name.removeprefix('final_test_')
        eval_root = ROOT / 'exp/formal_closeout_eval'
        path = ((eval_root / cohort_id / slot) if cohort_id else
                (eval_root / slot)) / 'report.json'
    else:
        raise ValueError(f'Unknown cohort phase: {phase}')
    if not path.is_file():
        raise RuntimeError(f'{name} exited 0 without expected report: {path}')
    report = json.loads(path.read_text(encoding='utf-8'))
    if cohort_id and report.get('cohort_id') != cohort_id:
        raise RuntimeError(f'{name} report cohort identity mismatch')
    expected_status = {'baby_training': 'validation_completed',
                       'no_test_validation': 'passed',
                       'final_test_once': 'completed'}[phase]
    if report.get('status') != expected_status:
        raise RuntimeError(f'{name} report status is {report.get("status")!r}; expected {expected_status!r}')
    if phase == 'baby_training':
        checkpoint = directory / 'best.pt'
        if (report.get('config', {}).get('seed') != seed or
                report.get('test_split_loaded') is not False or
                report.get('test_evaluations') != 0 or
                not checkpoint.is_file() or
                sha256(checkpoint) != report.get('best_checkpoint_sha256')):
            raise RuntimeError(f'{name} Baby identity, Test-zero or checkpoint verification failed')
    elif phase == 'no_test_validation':
        if (report.get('slot') != slot or
                report.get('test_access_started') is not False or
                report.get('test_split_loaded') is not False or
                report.get('student_test_evaluations') != 0 or
                report.get('teacher_test_evaluations') != 0):
            raise RuntimeError(f'{name} preflight identity or Test-zero verification failed')
    else:
        if (report.get('slot') != slot or
                report.get('test_access_started') is not True or
                report.get('student_test_attempts') != 1 or
                report.get('student_test_evaluations') != 1 or
                report.get('teacher_test_evaluations') != 0 or
                report.get('selected_checkpoint_sha256') != report.get('evaluated_checkpoint_sha256')):
            raise RuntimeError(f'{name} Test chronology or checkpoint verification failed')
    return path


def check_expected_absent(cohort_id=None):
    for seed in SEEDS:
        baby_root = ROOT / 'exp/promptmm_release_baby'
        if cohort_id:
            baby_root = baby_root / cohort_id
        target = baby_root / f'baby_promptmm_release_cap1000_patience7_seed{seed}_lr6e5_v1'
        if target.exists():
            raise FileExistsError(f'Baby run already attempted: {target}')
    for family in ('formal_closeout_preflight', 'formal_closeout_eval'):
        for slot in SLOTS:
            family_root = ROOT / 'exp' / family
            target = family_root / cohort_id / slot if cohort_id else family_root / slot
            if target.exists():
                raise FileExistsError(f'Closeout slot already attempted: {target}')
    cohort_path = ROOT / 'exp/formal_closeout_cohort' / cohort_id if cohort_id else COHORT
    if cohort_path.exists():
        raise FileExistsError(f'Cohort already attempted: {cohort_path}')


def check_v1_recovery_anchor():
    """Reject recovery if the audited zero-Test v1 state changed."""
    records = ((ROOT/'exp/formal_closeout_cohort/innovation1_fixed_v1/batch.json',
                V1_BATCH_SHA256),
               (ROOT/'exp/formal_closeout_preflight/b3_2022/report.json',
                V1_PREFLIGHT_SHA256))
    for path, expected in records:
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f'Preserved v1 failure artifact drift: {path}')
    for slot in SLOTS:
        if (ROOT/'exp/formal_closeout_eval'/slot).exists():
            raise RuntimeError(f'Unexpected v1 Test attempt before recovery: {slot}')


def run(cohort_id=None):
    head = clean_head()
    if cohort_id:
        check_v1_recovery_anchor()
        check_expected_absent(cohort_id)
    else:
        check_expected_absent()
    cohort_path = ROOT / 'exp/formal_closeout_cohort' / cohort_id if cohort_id else COHORT
    cohort_path.mkdir(parents=True, exist_ok=False)
    batch_path = cohort_path / 'batch.json'
    report = dict(status='started', cohort=cohort_id or 'innovation1_fixed_v1',
                  source_commit=head, started_at=now(),
                  fixed_slots=list(SLOTS), steps=[],
                  planned_trainings=3, planned_student_tests=12,
                  teacher_test_evaluations=0, automatic_retries=0)
    atomic_json(batch_path, report)
    try:
        steps = plan(cohort_id) if cohort_id else plan()
        for index, step in enumerate(steps, start=1):
            if clean_head() != head:
                raise RuntimeError('Launch source changed during cohort.')
            entry = dict(index=index, name=step['name'], phase=step['phase'],
                         command=step['command'], status='started', started_at=now(),
                         log_path=str(cohort_path / (step['name'] + '.log')))
            report['steps'].append(entry)
            atomic_json(batch_path, report)
            print(f'[{index}/{len(steps)}] {step["name"]}', flush=True)
            with (cohort_path / (step['name'] + '.log')).open('x', encoding='utf-8') as log:
                process = subprocess.Popen(step['command'], cwd=ROOT,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT,
                                           text=True, bufsize=1)
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                process.stdout.close()
                returncode = process.wait()
            if returncode:
                entry.update(status='failed', exit_code=returncode, finished_at=now())
                atomic_json(batch_path, report)
                raise RuntimeError(f'Fail-fast cohort stopped at {step["name"]}: exit={returncode}')
            try:
                verified_path = verify_step_artifacts(step, cohort_id)
            except BaseException as exc:
                entry.update(status='failed', exit_code=returncode,
                             artifact_verification_error=repr(exc), finished_at=now())
                atomic_json(batch_path, report)
                raise RuntimeError(f'Fail-fast cohort stopped at {step["name"]}: {exc}') from exc
            entry.update(status='completed', exit_code=returncode,
                         verified_report_path=str(verified_path), finished_at=now())
            atomic_json(batch_path, report)
        report['status'] = 'completed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['finished_at'] = now()
        atomic_json(batch_path, report)
        print('Cohort report:', batch_path, flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--describe', action='store_true', help='Print 27 commands without launching.')
    group.add_argument('--conditions-confirmed', action='store_true',
                       help='Run the uniquely declared cohort once; stops at first failure.')
    parser.add_argument('--cohort-id', choices=['innovation1_fixed_v2'])
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(plan(args.cohort_id), indent=2))
        return 0
    return run(args.cohort_id)


if __name__ == '__main__':
    raise SystemExit(main())
