"""One manual teacher-initialized BPR-only control, no retries or Test ranking."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import subprocess
import sys
from run_sports_initialization_pair import ROOT, check_source, digest, verify_td
from promptmm_release_validation import atomic_json
from utility.dataset_profiles import SPORTS_BPR_TEACHER_INIT_PROFILE as PROFILE
from initialization_audit import require_same_td_initial

RUN_ID='sports_bpr_teacherinit_seed2022_val300_v1'
REFERENCE=Path('exp/runs/sports/run_manifest__2026-09-22 20_57_40.716760_sports_light_init_pid8964.json')
REFERENCE_SHA='7f863a7eec52d5af7eec2e6a611d05820653d32790e3a344813d536f795c9e59'


def command(python):
    return [str(python),'-B','codes/main_mmlight.py','--dataset','sports','--student_profile',PROFILE,'--gpu_id','0']


def verify(root,path):
    outcome=verify_td(root,path,profile=PROFILE)
    m=json.loads(path.read_text(encoding='utf-8'))
    require_same_td_initial(m['initialization'])
    ref=json.loads((root/REFERENCE).read_text(encoding='utf-8'))
    if m['initial_validation']!=ref['initial_validation']:raise RuntimeError('Initial Validation differs from reference.')
    with Path(m['artifacts']['converge_run']).open('rb') as f:curve=pickle.load(f)
    if any(float(v)!=0. for v in curve['td_distill_loss_List']):raise RuntimeError('Distillation not disabled.')
    if curve['td_batch_loss_List']!=curve['td_bpr_loss_List']:raise RuntimeError('Objective not pure BPR.')
    outcome['arm']='td_teacher_init_bpr_only'
    return outcome


def run(root=ROOT,python=sys.executable):
    root=Path(root)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    check_source(root,head)
    if digest(root/REFERENCE)!=REFERENCE_SHA:raise RuntimeError('Reference manifest mismatch.')
    require_same_td_initial(json.loads((root/REFERENCE).read_text())['initialization'])
    runs=root/'exp/runs/sports'
    for p in runs.glob('run_manifest__*.json'):
        if json.loads(p.read_text()).get('student_config_profile')==PROFILE:raise FileExistsError('Existing attempt; no retry.')
    out=root/'exp/initialization_checks'/RUN_ID;out.mkdir(parents=True,exist_ok=False)
    record=dict(status='started',launch_commit=head,launch_dirty=False,command=command(python),
                reference_manifest_sha256=REFERENCE_SHA,started_at=datetime.now().astimezone().isoformat())
    def save():atomic_json(out/'run.json',record)
    save()
    try:
        before=set(runs.glob('run_manifest__*.json'))
        subprocess.run(command(python),cwd=root,check=True)
        check_source(root,head)
        if digest(root/REFERENCE)!=REFERENCE_SHA:raise RuntimeError('Reference changed.')
        new=set(runs.glob('run_manifest__*.json'))-before
        if len(new)!=1:raise RuntimeError('Expected exactly one manifest.')
        record['outcome']=verify(root,new.pop());record['status']='completed'
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));raise
    finally:
        record['finished_at']=datetime.now().astimezone().isoformat();save()
        print('Run report:',out/'run.json',flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:print(subprocess.list2cmdline(command(sys.executable)));return 0
    return run()


if __name__=='__main__':raise SystemExit(main())
