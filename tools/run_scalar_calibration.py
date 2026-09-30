"""One-shot CPU Train-only G/U scalar fit on frozen B; Test0."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import time
import traceback
from pathlib import Path
from run_own_anchor_control import ROOT, read, save, sha, require, paired
from scalar_calibration_core import sample_pairs, solve, objective, merge_signed, decision
from run_exposure_matched_bonus import access_guard

PROFILE = ROOT/'docs/research/innovation2/SCALAR_CALIBRATION_PROFILE_V1.json'


def worker(p, out):
    import numpy as np
    import torch
    import psutil
    from run_minimal_ranking_supervision import load_train
    from initialization_kd_adapter import load_train_val
    from run_per_user_slot_control import metrics, construct
    access = {'sealed':False, 'denied':0, 'opens':[]}
    sys.addaudithook(lambda e,a: access_guard(e,a,access))
    started=time.monotonic(); samples=[]
    result={'status':'running','states':{},'comparisons':{},'fit':{},'data_access':access,
            'test_policy':'Test0','base_parameter_updates':0,'environment':{'numpy':np.__version__,'torch':torch.__version__,'device':'cpu'}}
    def check():
        row={'seconds':time.monotonic()-started,'rss_bytes':psutil.Process().memory_info().rss}
        samples.append(row)
        require(row['seconds']<p['caps']['wall_seconds'] and row['rss_bytes']<p['caps']['rss_bytes'],'Worker resource cap')
    def inp(key):return ROOT/p['inputs'][key]['path']
    try:
        torch.set_num_threads(4)
        for key,sp in p['inputs'].items():
            if not key.endswith('status.json'):
                require(sha(ROOT/sp['path'])==sp['sha256'],'Input SHA '+key);check()
        sp=p['anchor_manifest']; require(sha(ROOT/sp['path'])==sp['sha256'],'Anchor SHA')
        anchor=read(ROOT/sp['path'])
        require(anchor['source_commit']=='b8be549613498e5cbb98412ffa766c62279ad7a6','Anchor source')
        train=load_train(p);degree=np.bincount(train.indices,minlength=18357)
        target=None
        for mod in ['image','text']:
            sp=p['assets'][mod+'_graph'];require(sha(ROOT/sp['path'])==sp['sha256'],'Graph SHA')
            with np.load(ROOT/sp['path'],allow_pickle=False) as z:
                require(np.array_equal(z['degree'],degree),'Degree')
                t=z['queries'].copy()
            if target is not None:require(np.array_equal(target,t),'S agreement')
            target=t
        require(len(target)==6114 and len(np.unique(target))==6114,'S identity')
        mask=np.zeros(18357,dtype=bool);mask[target]=True
        order=np.lexsort((np.arange(18357),degree));groups=np.empty(18357,dtype=int)
        for g,part in enumerate(np.array_split(order,3)):groups[part]=g
        require((groups[target]==0).all(),'S low membership')
        users=np.arange(35598)
        pos,neg=sample_pairs(train,p['fit']['pairs_per_user'],p['fit']['tape_seed'],check)
        delta=mask[pos].astype(np.int8)-mask[neg].astype(np.int8)
        np.savez_compressed(out/'train_pairs.npz',users=users,positive=pos,negative=neg,delta=delta)
        result['sampling']={'positive_target_pairs':int((delta==1).sum()),'negative_target_pairs':int((delta==-1).sum()),
            'same_group_pairs':int((delta==0).sum()),'users_no_cross_group':int((delta==0).all(1).sum())}
        sealed={}
        for seed in p['seeds']:
            ck=torch.load(inp(f'seed{seed}/B/final.pt'),map_location='cpu',weights_only=False)
            require(ck['arm']=='B' and ck['epoch']==300 and ck['seed']==seed and ck['source']==anchor['source_commit'],'Checkpoint identity')
            for key,shape in [('user',(35598,64)),('item',(18357,64))]:
                require(tuple(ck[key].shape)==shape and ck[key].dtype==torch.float32 and bool(torch.isfinite(ck[key]).all()),'Tensor identity')
            u=ck['user'].detach().numpy().astype(np.float64);it=ck['item'].detach().numpy().astype(np.float64)
            margin=np.empty(pos.shape,dtype=np.float64)
            for start in range(0,len(users),256):
                end=min(start+256,len(users))
                margin[start:end]=np.einsum('ud,upd->up',u[start:end],it[pos[start:end]]-it[neg[start:end]])
                check()
            aG,fitG=solve(margin,delta,p['fit']['ridge'],p['fit']['iterations'],False,check)
            aU,fitU=solve(margin,delta,p['fit']['ridge'],p['fit']['iterations'],True,check)
            base_obj=objective(margin,delta,0.,p['fit']['ridge'])
            require(fitU['objective']<=fitG['objective']+1e-10 and fitG['objective']<=base_obj+1e-10,'Convex objective ordering')
            require(max(fitG['max_stationarity_error'],fitU['max_stationarity_error'])<1e-8,'Solver accuracy')
            require(np.array_equal(u,ck['user'].detach().numpy()) and np.array_equal(it,ck['item'].detach().numpy()),'Frozen B changed')
            with np.load(inp(f'seed{seed}_candidates.npz'),allow_pickle=False) as z:
                ids,scores=z['ids'].copy(),z['scores'].copy()
                require(np.array_equal(z['users'],users) and np.array_equal(z['targets'],target),'Cache identities')
            with np.load(inp(f'seed{seed}/B/top20_epoch300.npz'),allow_pickle=False) as z:
                b=z['top20'].copy();require(np.array_equal(z['users'],users),'B users')
            # Validate cached group membership, padding, order, tie IDs via existing pure constructor.
            construct(ids,scores,b,mask)
            require(np.array_equal(merge_signed(ids,scores,0.),b),'B exact regression')
            tops={'B':b,'G':merge_signed(ids,scores,aG),'U':merge_signed(ids,scores,aU)}
            for arm,top in tops.items():
                require(top.shape==(35598,20) and top.min()>=0 and top.max()<18357 and all(len(set(x))==20 for x in top),'Top IDs')
                for row in range(len(users)):
                    require(not np.isin(top[row],train.indices[train.indptr[row]:train.indptr[row+1]]).any(),'Seen recommendation')
                check()
            np.savez_compressed(out/f'seed{seed}_fit.npz',users=users,global_offset=aG,user_offset=aU,targets=target)
            np.savez_compressed(out/f'seed{seed}_lists.npz',users=users,**tops)
            for suffix in ['fit','lists']:sealed[f'seed{seed}_{suffix}.npz']=sha(out/f'seed{seed}_{suffix}.npz')
            result['fit'][str(seed)]={'B_objective':base_obj,'G':fitG,'U':fitU,
                'U_min':float(aU.min()),'U_max':float(aU.max()),'U_mean':float(aU.mean()),'U_positive':int((aU>0).sum()),'G_offset':float(aG[0])}
            del ck,u,it,margin,ids,scores,tops
            check()
        sealed['train_pairs.npz']=sha(out/'train_pairs.npz')
        seal={'time_ns':time.time_ns(),'files_sha256':sealed};save(out/'fit_seal.json',seal)
        access['sealed']=True
        _,val=load_train_val(ROOT/'data/sports',p['assets']['train']['sha256'],p['assets']['validation']['sha256'],35598,18357)
        require(np.array_equal(np.flatnonzero(np.diff(val.indptr)),users),'Val users')
        for seed in p['seeds']:
            with np.load(out/f'seed{seed}_lists.npz',allow_pickle=False) as z:tops={a:z[a].copy() for a in ['B','G','U']}
            ms={a:metrics(t,users,val,groups,mask) for a,t in tops.items()}
            for arm,v in ms.items():
                require(v['denominator']==[6347,6389,25163] and np.isfinite(v['recall20']),'Metrics')
                result['states'][f'{seed}_{arm}']=v
            sp=p['inputs'][f'seed{seed}/B/status.json'];require(sha(ROOT/sp['path'])==sp['sha256'],'Status SHA')
            old=read(ROOT/sp['path'])['curve'][-1]
            require(old['epoch']==300 and abs(old['recall20']-ms['B']['recall20'])<1e-6 and old['groups']['hits']==ms['B']['hits'],'B metrics regression')
            row={}
            for a,b in [('U','B'),('U','G'),('G','B')]:
                pu,pi=paired(tops[b],tops[a],users,val,18357)
                dr=ms[a]['recall20']-ms[b]['recall20'];dh=np.array(ms[a]['hits'])-ms[b]['hits']
                require(abs(np.mean((pu[:,0]-pu[:,1])/pu[:,3])-dr)<1e-10,'Paired Recall')
                require(np.array_equal([(pi[:,0]-pi[:,1])[groups==g].sum() for g in range(3)],dh),'Paired hits')
                np.savez_compressed(out/f'seed{seed}_{a}_minus_{b}.npz',users=users,per_user=pu,per_item=pi)
                row[f'low_{a}_minus_{b}']=int(dh[0]);row[f'recall_{a}_minus_{b}']=float(dr)
            result['comparisons'][str(seed)]=row;check();save(out/'report.json',result)
        require(all(sha(out/f)==h for f,h in sealed.items()),'Seal changed')
        require(access['denied']==0 and all(x['time_ns']>=seal['time_ns'] for x in access['opens'] if x['file']=='val_mat'),'Val chronology')
        result['status']='completed';result['decision']=decision(list(result['comparisons'].values()))
        save(out/'acceptance.json',{'hard_acceptance':True,'fitted_states':6,'metric_states':9,'Test0':True})
    except BaseException:
        result['status']='failed';result['error']=traceback.format_exc()
        save(out/'acceptance.json',{'hard_acceptance':False,'error':result['error']});raise
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
        require(os.environ.get('SCALAR_CALIBRATION_WORKER') == str(os.getppid()), 'Worker must be supervised')
        worker(p, out)
        return
    from run_minimal_ranking_supervision import source_gate
    source = source_gate(dict(p, launch_source_rule='committed_head'))
    require(p['seeds']==[2022,2023,2024] and p['test_policy']=='Test0' and p['fit']=={'pairs_per_user':64,'tape_seed':8721,'ridge':.01,'iterations':40}, 'Frozen protocol')
    out.mkdir(parents=True, exist_ok=False)  # consumed once, even if execution fails
    save(out / 'manifest.json', {'source_commit': source, 'profile_sha256': sha(PROFILE),
                                'profile': p, 'command': sys.argv, 'status': 'launched'})
    import psutil
    started = time.monotonic()
    env = dict(os.environ, SCALAR_CALIBRATION_WORKER=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8',
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
