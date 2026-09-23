"""One manual alpha3 run; no retries or additional arms."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from run_sports_sharedinit_pair import verify,load_shared,ASSET,ASSET_SHA
from run_sports_initialization_pair import ROOT,check_source,digest
from promptmm_release_validation import atomic_json
from utility.dataset_profiles import SPORTS_ALPHA3_PROFILE as PROFILE
RUN_ID='sports_sharedteacherinit_alpha3_seed2022_val300_v1'
ANCHOR=Path('exp/initialization_checks/sports_sharedteacherinit_pair_seed2022_val300_v1/batch.json')
ANCHOR_SHA='4c2e0ba811f8085a69247a1b4339ed2e120b295097db79ed2f901404aed3ac26'

def command(python):
    return [str(python),'-B','codes/main_mmlight.py','--dataset','sports','--student_profile',PROFILE,'--gpu_id','0']

def run(root=ROOT,python=sys.executable):
    root=Path(root);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    check_source(root,head)
    if digest(root/ANCHOR)!=ANCHOR_SHA:raise RuntimeError('Comparison anchor changed.')
    anchor=json.loads((root/ANCHOR).read_text())
    if anchor['status']!='completed':raise RuntimeError('Comparison incomplete.')
    tensors,identity=load_shared(root,'cpu',35598,18357);del tensors
    if identity!=anchor['shared_initialization_asset']:raise RuntimeError('Shared asset mismatch.')
    runs=root/'exp/runs/sports'
    for p in runs.glob('run_manifest__*.json'):
        if json.loads(p.read_text()).get('student_config_profile')==PROFILE:raise FileExistsError('Existing attempt; no retry.')
    out=root/'exp/initialization_checks'/RUN_ID;out.mkdir(parents=True,exist_ok=False)
    record=dict(status='started',launch_commit=head,launch_dirty=False,command=command(python),
                comparison_anchor_sha256=ANCHOR_SHA,shared_initialization_asset=identity,
                started_at=datetime.now().astimezone().isoformat())
    def save():atomic_json(out/'run.json',record)
    save()
    try:
        before=set(runs.glob('run_manifest__*.json'))
        subprocess.run(command(python),cwd=root,check=True)
        check_source(root,head)
        if digest(root/ASSET)!=ASSET_SHA or digest(root/ANCHOR)!=ANCHOR_SHA:raise RuntimeError('Input/anchor changed.')
        new=set(runs.glob('run_manifest__*.json'))-before
        if len(new)!=1:raise RuntimeError('Expected one manifest.')
        result=verify(root,new.pop(),PROFILE,identity)
        first=anchor['completed'][0]
        if result['initialization']!=first['initialization'] or result['initial_validation']!=first['initial_validation']:
            raise RuntimeError('Initial state/metrics differ from comparison.')
        record.update(status='completed',result=result)
    except BaseException as exc:
        record.update(status='failed',error=repr(exc));raise
    finally:
        record['finished_at']=datetime.now().astimezone().isoformat();save();print('Run report:',out/'run.json')
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if a.dry_run:print(subprocess.list2cmdline(command(sys.executable)));return 0
    return run()

if __name__=='__main__':raise SystemExit(main())
