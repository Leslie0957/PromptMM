"""One manual serial batch: fixed Sports release seeds2023/2024 at lr6e-5,300 epochs."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2023, 2024)
BATCH_ID = 'sports_promptmm_release_validation300_lr6e5_remaining_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(seed):
    return f'sports_promptmm_release_validation300_seed{seed}_lr6e5_v1'


def verify_completed(run_dir, seed, head):
    report_path = run_dir/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    expected = dict(status='completed', run_id=identity(seed), epochs=300,
                    optimizer_steps_completed=64200, validation_evaluations=301,
                    test_evaluations=0, teacher_test_evaluations=0, test_split_loaded=False,
                    launch_commit=head, launch_dirty=False, early_stopping=False,
                    teacher_unchanged=True, prompt_unchanged=True, alias_preserved=True,
                    student_updated=True, best_checkpoint_roundtrip=True,
                    completion_source_unchanged=True, finite_updates_checked=True,
                    paper_ready_eligible=False)
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f'Incomplete/invalid run {seed}: {key}')
    if report['config']['seed'] != seed:
        raise RuntimeError('Resolved seed mismatch.')
    if report['config']['learning_rate'] != 6e-5:
        raise RuntimeError('Resolved learning rate mismatch.')
    curve = report['curve']
    if [e['epoch'] for e in curve] != list(range(1,301)):
        raise RuntimeError('Incomplete curve.')
    for row in curve:
        if not all(math.isfinite(v) for v in row['losses_mean'].values()):
            raise RuntimeError('Nonfinite loss.')
        if not all(math.isfinite(v) for values in row['validation'].values() for v in values):
            raise RuntimeError('Nonfinite metric.')
    best = max(curve, key=lambda e:e['validation']['recall'][1])
    if report['best_epoch'] != best['epoch'] or report['best_validation'] != best['validation']:
        raise RuntimeError('Selection mismatch.')
    if digest(run_dir/'best.pt') != report['best_checkpoint_sha256']:
        raise RuntimeError('Checkpoint fingerprint mismatch.')
    return dict(seed=seed,run_id=identity(seed),report_sha256=digest(report_path),
                best_epoch=report['best_epoch'],best_validation=report['best_validation'],
                best_checkpoint_sha256=report['best_checkpoint_sha256'],
                input_sha256=report['input_sha256'],config=report['config'])


def run_batch(root=ROOT, python=sys.executable):
    root=Path(root)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('Clean committed source required before batch.')
    base=root/'exp/promptmm_release'
    # Check every target before the first launch; never silently skip or retry.
    for seed in SEEDS:
        if (base/identity(seed)).exists():
            raise FileExistsError(f'Existing run; batch not started: {identity(seed)}')
    batch=base/BATCH_ID
    batch.mkdir(parents=True,exist_ok=False)
    summary=dict(status='started',launch_commit=head,started_at=datetime.now().astimezone().isoformat(),
                 planned_seeds=list(SEEDS),epochs=300,completed=[])
    def save():
        tmp=batch/'batch.json.tmp'
        tmp.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
        tmp.replace(batch/'batch.json')
    save()
    try:
        for seed in SEEDS:
            if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=head:
                raise RuntimeError('Batch launch commit changed.')
            if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
                raise RuntimeError('Dirty source between seeds.')
            command=[str(python),'-B','codes/main_mmlight.py','--promptmm_release_validation',
                     '--dataset','sports','--seed',str(seed),'--epochs','300','--student_lr','6e-5','--gpu_id','0']
            summary.update(active_seed=seed,active_command=command)
            save()
            print(f'=== Starting seed {seed},300 epochs ===',flush=True)
            result=subprocess.run(command,cwd=root)
            if result.returncode:
                raise RuntimeError(f'Seed {seed} exited {result.returncode}; no retry or next seed.')
            outcome=verify_completed(base/identity(seed),seed,head)
            if summary['completed']:
                first=summary['completed'][0]
                if outcome['input_sha256']!=first['input_sha256']:
                    raise RuntimeError('Cross-seed input identity mismatch.')
                if {k:v for k,v in outcome['config'].items() if k!='seed'}!={k:v for k,v in first['config'].items() if k!='seed'}:
                    raise RuntimeError('Cross-seed common configuration mismatch.')
            summary['completed'].append(outcome)
            save()
        summary['status']='completed'
    except BaseException as exc:
        summary.update(status='failed',error=repr(exc))
        raise
    finally:
        summary['finished_at']=datetime.now().astimezone().isoformat()
        save()
        print('Batch report:',batch/'batch.json',flush=True)
    return 0


if __name__=='__main__':
    if len(sys.argv)!=1:
        raise SystemExit('No overrides supported; run this fixed batch once.')
    raise SystemExit(run_batch())
