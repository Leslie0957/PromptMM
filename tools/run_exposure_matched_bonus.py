"""Fixed B score bonus; prediction-only calibration before Validation access."""
import os
import sys
import time
import traceback
from pathlib import Path
from run_own_anchor_control import ROOT, read, save, sha, require, paired
from exposure_bonus_core import merge, calibrate, decision

PROFILE = ROOT / 'docs/research/innovation2/EXPOSURE_MATCHED_BONUS_PROFILE_V1.json'


def access_guard(event, args, state):
    if event != 'open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.is_relative_to(ROOT/'data'):
        allowed = {ROOT/'data/sports/train_mat'}
        if state['sealed']:
            allowed.add(ROOT/'data/sports/val_mat')
        if path not in allowed:
            state['denied'] += 1
            raise PermissionError('Dataset access outside sealed phase')
        state['opens'].append({'file':path.name,'sealed':state['sealed'],'time_ns':time.time_ns()})


def worker(p, out):
    import numpy as np
    import torch
    import psutil
    from run_minimal_ranking_supervision import load_train
    from initialization_kd_adapter import load_train_val
    from initialization_kd_fast_eval import exact_topk
    from neighbor_shared_residual import group_metrics
    access={'sealed':False,'denied':0,'opens':[]}
    sys.addaudithook(lambda e,a:access_guard(e,a,access))
    started=time.monotonic();samples=[]
    result={'status':'running','states':{},'comparisons':{},'data_access':access,'parameter_updates':0,'test_policy':'Test0'}
    def check():
        row={'seconds':time.monotonic()-started,'rss_bytes':psutil.Process().memory_info().rss,
             'cuda_reserved':torch.cuda.memory_reserved()}
        samples.append(row)
        require(row['seconds']<p['caps']['wall_seconds'] and row['rss_bytes']<p['caps']['rss_bytes']
                and row['cuda_reserved']<=p['caps']['cuda_allocator_bytes'],'Worker resource cap')
    def path(seed,tail):
        return ROOT/p['inputs'][f'seed{seed}/{tail}']['path']
    def lists(file):
        with np.load(file,allow_pickle=False) as z:
            u,t=z['users'].copy(),z['top20'].copy()
        require(np.array_equal(u,np.arange(35598)) and t.shape==(35598,20),'List identity')
        require(t.min()>=0 and t.max()<18357 and all(len(set(x))==20 for x in t),'List IDs')
        return u,t
    try:
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        require(torch.cuda.is_available(),'CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(p['caps']['cuda_allocator_bytes']/torch.cuda.get_device_properties(0).total_memory)
        result['environment']={'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.cuda.get_device_name(0)}
        # Validation bytes and status metrics remain unopened during calibration.
        specs=[p['anchor_manifest'],p['anchor_completeness']]
        specs += [v for k,v in p['assets'].items() if k!='validation']
        specs += [v for k,v in p['inputs'].items() if not k.endswith('status.json')]
        for spec in specs:
            require(sha(ROOT/spec['path'])==spec['sha256'],'Asset SHA '+spec['path']);check()
        anchor=read(ROOT/p['anchor_manifest']['path'])
        require(anchor['source_commit']=='b8be549613498e5cbb98412ffa766c62279ad7a6','Anchor source')
        train=load_train(p)
        degree=np.bincount(train.indices,minlength=18357)
        target_ids=None
        for m in ['image','text']:
            with np.load(ROOT/p['assets'][m+'_graph']['path'],allow_pickle=False) as z:
                ids=z['queries'].copy()
                require(np.array_equal(z['degree'],degree),'Graph degree')
            if target_ids is not None:
                require(np.array_equal(ids,target_ids),'Modality queries')
            target_ids=ids
        require(len(target_ids)==6114 and len(np.unique(target_ids))==6114,'Target count')
        mask=np.zeros(18357,dtype=bool);mask[target_ids]=True
        order=np.lexsort((np.arange(18357),degree));groups=np.empty(18357,dtype=np.int8)
        require(np.isin(target_ids,order[:6119]).all(),'Target low membership')
        for g,part in enumerate(np.array_split(order,3)):groups[part]=g
        all_cal={}; group_ids=[np.flatnonzero(mask),np.flatnonzero(~mask)]
        with torch.inference_mode():
            for seed in p['seeds']:
                users,r=lists(path(seed,'R/top20_epoch300.npz'))
                _,b=lists(path(seed,'B/top20_epoch300.npz'))
                ck=torch.load(path(seed,'B/final.pt'),map_location='cpu',weights_only=False)
                require(ck['arm']=='B' and ck['epoch']==300 and ck['seed']==seed and ck['source']==anchor['source_commit'],'Checkpoint identity')
                for key,shape in [('user',(35598,64)),('item',(18357,64))]:
                    require(tuple(ck[key].shape)==shape and ck[key].dtype==torch.float32 and bool(torch.isfinite(ck[key]).all()),'Tensor identity')
                u,it=ck['user'].cuda(),ck['item'].cuda()
                ci=np.full((len(users),2,20),-1,dtype=np.int32)
                cs=np.zeros((len(users),2,20),dtype=np.float32)
                maximum=0.
                for start in range(0,len(users),256):
                    batch=users[start:start+256]
                    scores=(u[torch.as_tensor(batch,device='cuda:0')]@it.T).cpu().numpy()
                    require(np.isfinite(scores).all(),'Nonfinite score')
                    for j,uid in enumerate(batch):
                        seen=train.indices[train.indptr[uid]:train.indptr[uid+1]]
                        valid=np.ones(18357,dtype=bool);valid[seen]=False
                        maximum=max(maximum,float(np.abs(scores[j,valid]).max()))
                        for g,gi in enumerate(group_ids):
                            candidates=gi[valid[gi]];k=min(20,len(candidates))
                            if k:
                                chosen=candidates[exact_topk(scores[j,candidates],set(),k)]
                                ci[start+j,g,:k]=chosen;cs[start+j,g,:k]=scores[j,chosen]
                    check()
                zero,_=merge(ci,cs,0.)
                require(np.array_equal(zero,b),'B Top20 exact regression')
                require(torch.equal(u.cpu(),ck['user']) and torch.equal(it.cpu(),ck['item']),'Parameter mutation')
                np.savez_compressed(out/f'seed{seed}_candidates.npz',users=users,ids=ci,scores=cs,targets=target_ids)
                cal=calibrate(ci,cs,int(mask[r].sum()),maximum,check)
                q,qe=merge(ci,cs,cal['beta']);re=mask[r].sum(axis=1)
                cal.update(maximum_valid_score_abs=maximum,
                           cache_sha256=sha(out/f'seed{seed}_candidates.npz'),sealed_time_ns=time.time_ns())
                all_cal[str(seed)]=cal
                np.savez_compressed(out/f'seed{seed}_lists.npz',users=users,B=b,Q=q,R=r,
                    target_exposure_Q=qe,target_exposure_R=re,
                    item_exposure_Q=np.bincount(q.ravel(),minlength=18357),
                    item_exposure_R=np.bincount(r.ravel(),minlength=18357))
                del ck,u,it,ci,cs,scores
                check()
        save(out/'calibration.json',all_cal)
        seal={'calibration_sha256':sha(out/'calibration.json'),'time_ns':time.time_ns(),
              'lists_sha256':{str(s):sha(out/f'seed{s}_lists.npz') for s in p['seeds']}}
        save(out/'calibration_seal.json',seal)
        require(access['denied']==0,'Forbidden calibration access')
        access['sealed']=True
        _,val=load_train_val(ROOT/'data/sports',p['assets']['train']['sha256'],p['assets']['validation']['sha256'],35598,18357)
        require(np.array_equal(np.flatnonzero(np.diff(val.indptr)),users),'Validation users')
        for seed in p['seeds']:
            with np.load(out/f'seed{seed}_lists.npz',allow_pickle=False) as z:
                tops={s:z[s].copy() for s in ['B','Q','R']}
                qe,re=z['target_exposure_Q'].copy(),z['target_exposure_R'].copy()
            metrics={s:group_metrics(t,users,val,groups) for s,t in tops.items()}
            for s,gm in metrics.items():
                require(gm['denominator']==[6347,6389,25163] and np.isfinite(gm['recall20']),'Metric validity')
                for row,uid in enumerate(users):
                    require(not np.isin(tops[s][row],train.indices[train.indptr[uid]:train.indptr[uid+1]]).any(),'Seen recommendation')
                result['states'][f'{seed}_{s}']=gm
            for arm in ['B','R']:
                spec=p['inputs'][f'seed{seed}/{arm}/status.json']
                require(sha(ROOT/spec['path'])==spec['sha256'],'Status SHA')
                old=read(ROOT/spec['path'])['curve'][-1]
                require(old['epoch']==300 and abs(old['recall20']-metrics[arm]['recall20'])<=1e-6
                        and old['groups']['hits']==metrics[arm]['hits'],'Metric regression')
            for ref in ['B','R']:
                pu,pi=paired(tops[ref],tops['Q'],users,val,18357)
                delta=metrics['Q']['recall20']-metrics[ref]['recall20']
                require(abs(float(np.mean((pu[:,0]-pu[:,1])/pu[:,3]))-delta)<1e-10,'Paired recall')
                require(np.array_equal([int((pi[:,0]-pi[:,1])[groups==g].sum()) for g in range(3)],
                        np.array(metrics['Q']['hits'])-metrics[ref]['hits']),'Paired group')
                np.savez_compressed(out/f'seed{seed}_Q_minus_{ref}.npz',users=users,per_user=pu,per_item=pi)
            result['comparisons'][str(seed)]={'matched':all_cal[str(seed)]['matched'],
                'low_Q_minus_R':metrics['Q']['hits'][0]-metrics['R']['hits'][0],
                'recall_Q_minus_R':metrics['Q']['recall20']-metrics['R']['recall20'],
                'low_Q_minus_B':metrics['Q']['hits'][0]-metrics['B']['hits'][0],
                'recall_Q_minus_B':metrics['Q']['recall20']-metrics['B']['recall20'],
                'mean_target_slot_delta':float(np.mean(qe-re)),
                'mean_absolute_target_slot_delta':float(np.mean(np.abs(qe-re))),
                'Q_target_slots_hist':np.bincount(qe,minlength=21).tolist(),
                'R_target_slots_hist':np.bincount(re,minlength=21).tolist()}
            check();save(out/'report.json',result)
        require(sha(out/'calibration.json')==seal['calibration_sha256'] and all(sha(out/f'seed{s}_lists.npz')==seal['lists_sha256'][str(s)] for s in p['seeds']),'Seal changed')
        require(access['denied']==0 and all(x['sealed'] and x['time_ns']>=seal['time_ns'] for x in access['opens'] if x['file']=='val_mat'),'Label chronology')
        result['status']='completed';result['decision']=decision(list(result['comparisons'].values()))
        save(out/'acceptance.json',{'hard_acceptance':True,'new_scoring_states':6,'saved_R_references':3,'Test0':True})
    except BaseException:
        result['status']='failed';result['error']=traceback.format_exc()
        save(out/'acceptance.json',{'hard_acceptance':False,'error':result['error']})
        raise
    finally:
        save(out/'report.json',result);save(out/'worker_resources.json',samples)


import argparse
import shutil
import subprocess

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.formal, 'Explicit --formal required; default does nothing')
    p = read(PROFILE)
    out = ROOT / p['output']
    if args.worker:
        require(os.environ.get('EXPOSURE_BONUS_WORKER') == str(os.getppid()), 'Worker must be supervised')
        worker(p, out)
        return
    from run_minimal_ranking_supervision import source_gate
    source = source_gate(dict(p, launch_source_rule='committed_head'))
    require(p['seeds'] == [2022,2023,2024] and p['states'] == ['B','Q'] and p['user_batch'] == 256
            and p['test_policy'] == 'Test0' and p['screen'] == {'low_hit_tolerance':3,'recall_tolerance':.0002}, 'Frozen protocol')
    out.mkdir(parents=True, exist_ok=False)  # consumed once, even if execution fails
    save(out / 'manifest.json', {'source_commit': source, 'profile_sha256': sha(PROFILE),
                                'profile': p, 'command': sys.argv, 'status': 'launched'})
    import psutil
    started = time.monotonic()
    env = dict(os.environ, EXPOSURE_BONUS_WORKER=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    samples, reason = [], None
    with (out / 'console.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--formal', '--worker'],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:
                rss = psutil.Process(proc.pid).memory_info().rss
            except psutil.NoSuchProcess:
                break
            row = {'seconds':time.monotonic()-started, 'rss_bytes':rss,
                   'output_bytes':sum(f.stat().st_size for f in out.rglob('*') if f.is_file()),
                   'free_disk_bytes':shutil.disk_usage(out).free}
            samples.append(row)
            caps = p['caps']
            if row['seconds'] > caps['wall_seconds'] or rss > caps['rss_bytes'] or row['output_bytes'] > caps['output_bytes'] or row['free_disk_bytes'] < caps['minimum_free_disk_bytes']:
                reason = 'External resource limit'
                proc.terminate()
                break
            time.sleep(1)
        code = proc.wait()
    save(out / 'supervisor.json', {'exit_code':code,'termination_reason':reason,'samples':samples,
                                   'elapsed_seconds':time.monotonic()-started})
    save(out / 'completeness.json', {'files_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name != 'completeness.json'}})
    require(code == 0 and reason is None, 'Diagnostic failed; preserve artifacts, no retry')
    print(str(out / 'report.json'))


if __name__ == '__main__':
    main()
