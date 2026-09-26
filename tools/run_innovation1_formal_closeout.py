"""One declared, fail-fast user-run cohort: Baby release 3 + selected final Test 12."""
import argparse
from datetime import datetime
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


def now():
    return datetime.now().astimezone().isoformat()


def plan():
    python = sys.executable
    baby = ROOT / 'codes/promptmm_release_baby_formal.py'
    evaluator = ROOT / 'codes/formal_closeout_eval.py'
    steps = []
    for seed in SEEDS:
        steps.append(dict(name=f'train_b3_{seed}', phase='baby_training',
                          command=[python, '-B', str(baby), '--baby_release_formal', '--seed', str(seed)]))
    for slot in SLOTS:
        steps.append(dict(name=f'preflight_{slot}', phase='no_test_validation',
                          command=[python, '-B', str(evaluator), '--preflight', '--slot', slot]))
    for slot in SLOTS:
        steps.append(dict(name=f'final_test_{slot}', phase='final_test_once',
                          command=[python, '-B', str(evaluator), '--final-test-once', '--slot', slot]))
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


def check_expected_absent():
    for seed in SEEDS:
        target = ROOT / 'exp/promptmm_release_baby' / (
            f'baby_promptmm_release_cap1000_patience7_seed{seed}_lr6e5_v1')
        if target.exists():
            raise FileExistsError(f'Baby run already attempted: {target}')
    for family in ('formal_closeout_preflight', 'formal_closeout_eval'):
        for slot in SLOTS:
            target = ROOT / 'exp' / family / slot
            if target.exists():
                raise FileExistsError(f'Closeout slot already attempted: {target}')
    if COHORT.exists():
        raise FileExistsError(f'Cohort already attempted: {COHORT}')


def run():
    head = clean_head()
    check_expected_absent()
    COHORT.mkdir(parents=True, exist_ok=False)
    batch_path = COHORT / 'batch.json'
    report = dict(status='started', cohort='innovation1_fixed_v1',
                  source_commit=head, started_at=now(),
                  fixed_slots=list(SLOTS), steps=[],
                  planned_trainings=3, planned_student_tests=12,
                  teacher_test_evaluations=0, automatic_retries=0)
    atomic_json(batch_path, report)
    try:
        for index, step in enumerate(plan(), start=1):
            if clean_head() != head:
                raise RuntimeError('Launch source changed during cohort.')
            entry = dict(index=index, name=step['name'], phase=step['phase'],
                         command=step['command'], status='started', started_at=now(),
                         log_path=str(COHORT / (step['name'] + '.log')))
            report['steps'].append(entry)
            atomic_json(batch_path, report)
            print(f'[{index}/{len(plan())}] {step["name"]}', flush=True)
            with (COHORT / (step['name'] + '.log')).open('x', encoding='utf-8') as log:
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
            entry.update(status='completed' if returncode == 0 else 'failed',
                         exit_code=returncode, finished_at=now())
            atomic_json(batch_path, report)
            if returncode:
                raise RuntimeError(f'Fail-fast cohort stopped at {step["name"]}: exit={returncode}')
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
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(plan(), indent=2))
        return 0
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
