"""One-shot manual seed2022 initialization pair; stops on any failed audit."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'codes'))
from utility.dataset_profiles import SPORTS_TD_TEACHER_INIT_PROFILE as PROFILE
from promptmm_release_validation import atomic_json, run_identity

RELEASE_ID = run_identity(2022, 300, 6e-5, 'random')
BATCH_ID = 'sports_initialization_pair_seed2022_val300_v1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def commands(python):
    return [[str(python), '-B', 'codes/main_mmlight.py', '--dataset', 'sports',
             '--student_profile', PROFILE, '--gpu_id', '0'],
            [str(python), '-B', 'codes/main_mmlight.py', '--promptmm_release_validation',
             '--dataset', 'sports', '--seed', '2022', '--epochs', '300',
             '--student_lr', '6e-5', '--student_initialization', 'random', '--gpu_id', '0']]


def check_source(root, head):
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=head:
        raise RuntimeError('Launch commit changed.')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise RuntimeError('Clean unchanged source required.')


def verify_td(root, path, profile=PROFILE):
    import torch
    from run_sports_validation300 import verify_outcome
    from utility.dataset_profiles import SPORTS_STUDENT_PROFILES
    m=json.loads(path.read_text(encoding='utf-8'))
    verify_outcome(m, profile)
    for k,v in SPORTS_STUDENT_PROFILES[profile]['defaults'].items():
        if m['resolved_arguments'].get(k)!=v:raise RuntimeError('TD argument mismatch: '+k)
    if not m['initialization']['independent_storage'] or m['initialization']['mode']!='teacher':
        raise RuntimeError('TD initialization mismatch.')
    if m.get('initial_validation_selected') is not False:raise RuntimeError('Initial selection policy mismatch.')
    if m.get('paper_ready_eligible') is not False:raise RuntimeError('Unexpected eligibility.')
    with Path(m['artifacts']['converge_run']).open('rb') as f:curve=pickle.load(f)
    for k in ('td_batch_loss_List','td_bpr_loss_List','td_distill_loss_List',
              'selection_recall20_List','selection_recall50_List','selection_ndcg20_List','selection_ndcg50_List'):
        if len(curve[k])!=300 or not all(math.isfinite(float(x)) for x in curve[k]):
            raise RuntimeError('Incomplete/nonfinite TD curve: '+k)
    best=max(range(300),key=lambda i:curve['selection_recall20_List'][i])
    if m['best_selection_epoch']!=best or m['best_selection_recall']!=curve['selection_recall20_List'][best]:
        raise RuntimeError('TD selection mismatch.')
    checkpoint=Path(m['td_full_checkpoint'])
    c=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if c['best_epoch']!=best or c['initialization']!=m['initialization'] or not c['td_init_from_teacher']:
        raise RuntimeError('TD checkpoint mismatch.')
    if not all(torch.isfinite(v).all() for v in c['model_state_dict'].values()):
        raise RuntimeError('Nonfinite TD weights.')
    for fp in m['code_fingerprints'].values():
        if digest(fp['path'])!=fp['sha256']:raise RuntimeError('TD source changed.')
    return dict(arm='td_teacher_init',manifest=str(path),manifest_sha256=digest(path),
                checkpoint_sha256=digest(checkpoint),best_epoch=best,
                best_recall=m['best_selection_recall'])


def verify_release(root, head):
    import math
    run_dir=root/'exp/promptmm_release'/RELEASE_ID
    r=json.loads((root/'exp/promptmm_release'/RELEASE_ID/'report.json').read_text())
    if r['student_initialization']!='random' or r['initialization']['source']!='constructor_xavier_uniform':
        raise RuntimeError('Release initialization mismatch.')
    expected=dict(status='completed',run_id=RELEASE_ID,epochs=300,optimizer_steps_completed=64200,
                  validation_evaluations=301,test_evaluations=0,teacher_test_evaluations=0,
                  test_split_loaded=False,launch_commit=head,launch_dirty=False,early_stopping=False,
                  teacher_unchanged=True,prompt_unchanged=True,alias_preserved=True,student_updated=True,
                  best_checkpoint_roundtrip=True,completion_source_unchanged=True,finite_updates_checked=True,
                  paper_ready_eligible=False,initial_validation_selected=False)
    for key,value in expected.items():
        if r.get(key)!=value:raise RuntimeError('Release completion mismatch: '+key)
    if r['config']['seed']!=2022 or r['config']['learning_rate']!=6e-5:
        raise RuntimeError('Release config mismatch.')
    curve=r['curve']
    if [v['epoch'] for v in curve]!=list(range(1,301)):raise RuntimeError('Incomplete release curve.')
    for row in curve:
        values=list(row['losses_mean'].values())+[v for vs in row['validation'].values() for v in vs]
        if not all(math.isfinite(v) for v in values):raise RuntimeError('Nonfinite release curve.')
    best=max(curve,key=lambda row:row['validation']['recall'][1])
    if best['epoch']!=r['best_epoch'] or best['validation']!=r['best_validation']:
        raise RuntimeError('Release selection mismatch.')
    if digest(run_dir/'best.pt')!=r['best_checkpoint_sha256']:raise RuntimeError('Release checkpoint changed.')
    return dict(arm='release_random_init',report=str(run_dir/'report.json'),
                report_sha256=digest(run_dir/'report.json'),best_epoch=r['best_epoch'],
                best_validation=r['best_validation'],best_checkpoint_sha256=r['best_checkpoint_sha256'])


def run_batch(root=ROOT, python=sys.executable):
    root=Path(root); runs=root/'exp/runs/sports'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    check_source(root,head)
    for p in runs.glob('run_manifest__*.json'):
        if json.loads(p.read_text(encoding='utf-8')).get('student_config_profile')==PROFILE:
            raise FileExistsError('Existing TD attempt; no retry: '+str(p))
    if (root/'exp/promptmm_release'/RELEASE_ID).exists():raise FileExistsError('Existing release attempt.')
    out=root/'exp/initialization_checks'/BATCH_ID
    out.mkdir(parents=True,exist_ok=False)
    cmds=commands(python)
    record=dict(status='started',launch_commit=head,launch_dirty=False,commands=cmds,
                started_at=datetime.now().astimezone().isoformat(),completed=[])
    def save():atomic_json(out/'batch.json',record)
    save()
    try:
        for index,cmd in enumerate(cmds):
            check_source(root,head)
            before=set(runs.glob('run_manifest__*.json'))
            record.update(active_arm=index,active_command=cmd);save()
            print('=== Starting initialization arm',index+1,'of 2 ===',flush=True)
            subprocess.run(cmd,cwd=root,check=True)
            check_source(root,head)
            if index==0:
                new=set(runs.glob('run_manifest__*.json'))-before
                if len(new)!=1:raise RuntimeError('Expected one TD manifest.')
                result=verify_td(root,new.pop())
            else:result=verify_release(root,head)
            record['completed'].append(result);save()
        record['status']='completed'
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));raise
    finally:
        record['finished_at']=datetime.now().astimezone().isoformat();save()
        print('Batch report:',out/'batch.json',flush=True)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.dry_run:
        for command in commands(sys.executable):print(subprocess.list2cmdline(command))
        return 0
    return run_batch()


if __name__=='__main__':raise SystemExit(main())
