"""One proposed eval-only recovery cohort; user launch requires separate authorization."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

from run_innovation1_formal_closeout import (SLOTS, atomic_json, clean_head,
                                             sha256, verify_step_artifacts,
                                             check_v1_recovery_anchor)

ROOT = Path(__file__).resolve().parents[1]
COHORT_ID = 'innovation1_eval_recovery_v1'
SOURCE_COMMIT = 'da2371ebf6de2a5a0daf06444c7f1fd32837ce37'
V2_BATCH_SHA256 = 'e7febbd5ba054996a53c1dcce96eac5b6f1ab97238bfdee5a61faf5dc167b82f'
V2_FAILED_PREFLIGHT_SHA256 = '5589100c9de6593adcb494a199e18116f9d3343b4af74776c38d18962084df40'
BABY_SHA256 = {
    2022: ('53ea4868f6a6109f31a86a590e9d0adb506a4c515085d7067b20868771a73384',
           'de1cbc51d5150d4b6d42a9a747d9fe14afd58998379eb409a3d030c2934bf3ac'),
    2023: ('d5edac3a1719c874872735f55ffea87e6eb46831cf054308b77962a350e7189d',
           '2526c852b2bd3459e11b71a3542dc6407342c8b8c913f4e81a74bb3593323c9f'),
    2024: ('e4140d87c587cfa47801de7c54e6d7d8bd1f9e9f94f5ec858602d9abbebb266e',
           '27f8546df27f2a9de4781398595782de7d55a94edefde6571cfb9ecb83b6548d'),
}
COHORT = ROOT / 'exp/formal_closeout_cohort' / COHORT_ID


def now():
    return datetime.now().astimezone().isoformat()


def plan():
    evaluator = ROOT / 'codes/formal_closeout_eval.py'
    steps = []
    for slot in SLOTS:
        steps.append(dict(name=f'preflight_{slot}', phase='no_test_validation',
                          command=[sys.executable, '-B', str(evaluator),
                                   '--preflight', '--slot', slot, '--cohort-id', COHORT_ID]))
    for slot in SLOTS:
        steps.append(dict(name=f'final_test_{slot}', phase='final_test_once',
                          command=[sys.executable, '-B', str(evaluator),
                                   '--final-test-once', '--slot', slot, '--cohort-id', COHORT_ID]))
    return steps


def check_source_anchor():
    check_v1_recovery_anchor()
    batch = ROOT / 'exp/formal_closeout_cohort/innovation1_fixed_v2/batch.json'
    failed = ROOT / 'exp/formal_closeout_preflight/innovation1_fixed_v2/s3_2022/report.json'
    if sha256(batch) != V2_BATCH_SHA256 or sha256(failed) != V2_FAILED_PREFLIGHT_SHA256:
        raise RuntimeError('Audited v2 failed cohort fingerprint drift.')
    recorded = json.loads(batch.read_text(encoding='utf-8'))
    if (recorded.get('status') != 'failed' or recorded.get('source_commit') != SOURCE_COMMIT
            or [x.get('status') for x in recorded.get('steps', [])] !=
            ['completed'] * 12 + ['failed']):
        raise RuntimeError('Unexpected v2 batch chronology.')
    for seed, (report_hash, checkpoint_hash) in BABY_SHA256.items():
        base = ROOT / 'exp/promptmm_release_baby/innovation1_fixed_v2' / (
            f'baby_promptmm_release_cap1000_patience7_seed{seed}_lr6e5_v1')
        report_path, checkpoint_path = base/'report.json', base/'best.pt'
        if sha256(report_path) != report_hash or sha256(checkpoint_path) != checkpoint_hash:
            raise RuntimeError(f'Audited Baby source fingerprint drift: {seed}')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if (report.get('status') != 'validation_completed'
                or report.get('cohort_id') != 'innovation1_fixed_v2'
                or report.get('launch_commit') != SOURCE_COMMIT
                or report.get('config', {}).get('seed') != seed
                or report.get('best_checkpoint_sha256') != checkpoint_hash
                or report.get('test_evaluations') != 0
                or report.get('test_split_loaded') is not False):
            raise RuntimeError(f'Audited Baby source status/identity mismatch: {seed}')
    if (ROOT/'exp/formal_closeout_eval/innovation1_fixed_v2').exists():
        raise RuntimeError('Unexpected v2 Test attempt; recovery authority must be reviewed.')


def check_expected_absent():
    if COHORT.exists():
        raise FileExistsError(f'Recovery cohort already attempted: {COHORT}')
    for family in ('formal_closeout_preflight', 'formal_closeout_eval'):
        root = ROOT / 'exp' / family / COHORT_ID
        if root.exists():
            raise FileExistsError(f'Recovery output already attempted: {root}')


def verify_recovery_artifact(step):
    path = verify_step_artifacts(step, COHORT_ID)
    report = json.loads(path.read_text(encoding='utf-8'))
    if step['phase'] == 'no_test_validation':
        delta = report.get('validation_recall_delta')
        if (report.get('mode') != 'preflight' or
                not isinstance(delta, (int, float)) or
                not math.isfinite(delta) or abs(delta) > 1e-10):
            raise RuntimeError(f'Invalid recovery preflight parity: {step["name"]}')
    else:
        metrics = report.get('test_result')
        if (report.get('mode') != 'final-test-once' or
                report.get('test_split_loaded') is not True or
                not isinstance(metrics, dict) or
                not all(key in metrics and len(metrics[key]) == 4 and
                        all(math.isfinite(x) for x in metrics[key])
                        for key in ('recall', 'precision', 'ndcg', 'hit_ratio'))):
            raise RuntimeError(f'Incomplete recovery Test report: {step["name"]}')
    return path


def run():
    head = clean_head()
    check_source_anchor()
    check_expected_absent()
    COHORT.mkdir(parents=True, exist_ok=False)
    batch_path = COHORT / 'batch.json'
    steps = plan()
    report = dict(status='started', cohort=COHORT_ID, source_commit=head,
                  training_source_commit=SOURCE_COMMIT, started_at=now(),
                  fixed_slots=list(SLOTS), steps=[], planned_trainings=0,
                  planned_preflights=12, planned_student_tests=12,
                  teacher_test_evaluations=0, automatic_retries=0)
    atomic_json(batch_path, report)
    try:
        for index, step in enumerate(steps, start=1):
            if clean_head() != head:
                raise RuntimeError('Launch source changed during recovery cohort.')
            entry = dict(index=index, name=step['name'], phase=step['phase'],
                         command=step['command'], status='started', started_at=now(),
                         log_path=str(COHORT/(step['name'] + '.log')))
            report['steps'].append(entry)
            atomic_json(batch_path, report)
            print(f'[{index}/{len(steps)}] {step["name"]}', flush=True)
            with (COHORT/(step['name'] + '.log')).open('x', encoding='utf-8') as log:
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
                raise RuntimeError(f'Fail-fast recovery stopped at {step["name"]}: exit={returncode}')
            try:
                verified = verify_recovery_artifact(step)
            except BaseException as exc:
                entry.update(status='failed', exit_code=returncode,
                             artifact_verification_error=repr(exc), finished_at=now())
                atomic_json(batch_path, report)
                raise RuntimeError(f'Fail-fast recovery stopped at {step["name"]}: {exc}') from exc
            entry.update(status='completed', exit_code=returncode,
                         verified_report_path=str(verified), finished_at=now())
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
        print('Recovery cohort report:', batch_path, flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--describe', action='store_true')
    group.add_argument('--conditions-confirmed', action='store_true')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(plan(), indent=2))
        return 0
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
