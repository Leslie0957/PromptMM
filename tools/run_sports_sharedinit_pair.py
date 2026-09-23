"""Manual full/BPR pair with exactly shared initial vectors and teacher targets."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import subprocess
import sys
from run_sports_initialization_pair import ROOT,check_source,digest,verify_td
from promptmm_release_validation import atomic_json
from utility.dataset_profiles import SPORTS_SHARED_INIT_PROFILES as PROFILES
from shared_initialization import ASSET,ASSET_SHA,load_shared

RUN_ID='sports_sharedteacherinit_pair_seed2022_val300_v1'


def commands(python):
    return [[str(python),'-B','codes/main_mmlight.py','--dataset','sports','--student_profile',p,'--gpu_id','0'] for p in PROFILES]


def verify(root,path,profile,identity):
    outcome=verify_td(root,path,profile=profile)
    m=json.loads(path.read_text(encoding='utf-8'))
    if m['shared_initialization_asset']!=identity:raise RuntimeError('Shared asset metadata mismatch.')
    if m['initialization'].get('shared_asset_sha256')!=ASSET_SHA:raise RuntimeError('Missing pinned initialization.')
    for side in ('users','items'):
        if m['initialization'][side]!=identity['tensors'][side]:raise RuntimeError('Initial tensor mismatch.')
    with Path(m['artifacts']['converge_run']).open('rb') as f:curve=pickle.load(f)
    if profile==PROFILES[1]:
        if any(float(v)!=0. for v in curve['td_distill_loss_List']):raise RuntimeError('BPR distillation not disabled.')
        if curve['td_batch_loss_List']!=curve['td_bpr_loss_List']:raise RuntimeError('Not BPR-only.')
    outcome.update(arm=profile,initial_validation=m['initial_validation'],initialization=m['initialization'])
    return outcome


def run(root=ROOT,python=sys.executable):
    root=Path(root);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    check_source(root,head)
    tensors,identity=load_shared(root,'cpu',35598,18357);del tensors
    runs=root/'exp/runs/sports'
    for p in runs.glob('run_manifest__*.json'):
        if json.loads(p.read_text()).get('student_config_profile') in PROFILES:raise FileExistsError('Existing shared-pair attempt; no retry.')
    out=root/'exp/initialization_checks'/RUN_ID;out.mkdir(parents=True,exist_ok=False)
    record=dict(status='started',launch_commit=head,launch_dirty=False,commands=commands(python),
                shared_initialization_asset=identity,started_at=datetime.now().astimezone().isoformat(),completed=[])
    def save():atomic_json(out/'batch.json',record)
    save()
    try:
        for profile,cmd in zip(PROFILES,commands(python)):
            check_source(root,head)
            if digest(root/ASSET)!=ASSET_SHA:raise RuntimeError('Shared asset changed.')
            before=set(runs.glob('run_manifest__*.json'))
            record['active_profile']=profile;save()
            print('=== Starting',profile,'===',flush=True)
            subprocess.run(cmd,cwd=root,check=True)
            check_source(root,head)
            if digest(root/ASSET)!=ASSET_SHA:raise RuntimeError('Shared asset changed.')
            new=set(runs.glob('run_manifest__*.json'))-before
            if len(new)!=1:raise RuntimeError('Expected exactly one new manifest.')
            result=verify(root,new.pop(),profile,identity)
            if record['completed']:
                first=record['completed'][0]
                if result['initialization']!=first['initialization'] or result['initial_validation']!=first['initial_validation']:
                    raise RuntimeError('Paired initial state/metrics mismatch.')
            record['completed'].append(result);save()
        record['status']='completed'
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));raise
    finally:
        record['finished_at']=datetime.now().astimezone().isoformat();save()
        print('Batch report:',out/'batch.json',flush=True)
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:
        for cmd in commands(sys.executable):print(subprocess.list2cmdline(cmd))
        return 0
    return run()


if __name__=='__main__':raise SystemExit(main())
