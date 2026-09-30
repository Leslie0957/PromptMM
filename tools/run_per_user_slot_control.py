"""CPU cached-list control; no model loading, training or score computation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from run_own_anchor_control import ROOT, read, save, sha, require, paired

PROFILE = ROOT/'docs/research/innovation2/PER_USER_SLOT_CONTROL_PROFILE_V1.json'


def guard(event,args,state):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(ROOT/'data'):
            if not state['sealed'] or p not in {ROOT/'data/sports/train_mat',ROOT/'data/sports/val_mat'}:
                state['denied']+=1
                raise PermissionError('Interaction data outside allowed phase')
            state['opens'].append({'file':p.name,'time_ns':time.time_ns(),'sealed':True})


def construct(ids,scores,r,mask):
    n,k=r.shape
    require(ids.shape==(n,2,k) and scores.shape==ids.shape,'Cache shape')
    require(np.issubdtype(ids.dtype,np.integer) and np.isfinite(scores).all(),'Cache dtype/finite')
    require(r.min()>=0 and r.max()<len(mask),'R range')
    for row in range(n):
        for g in range(2):
            a=ids[row,g];valid=a[a>=0];v=scores[row,g,:len(valid)]
            require(np.array_equal(a[:len(valid)],valid) and (a[len(valid):]==-1).all(),'Cache padding')
            require(len(set(valid))==len(valid) and (valid<len(mask)).all(),'Cache IDs')
            require((mask[valid] == (g==0)).all(),'Cache group')
            require(np.all(v[:-1]>=v[1:]) and np.all(valid[:-1][v[:-1]==v[1:]]<valid[1:][v[:-1]==v[1:]]),'Cache order')
    pattern=mask[r];h=np.empty_like(r);pos=np.zeros((n,2),dtype=int);rows=np.arange(n)
    for j in range(k):
        g=(~pattern[:,j]).astype(int)
        chosen=ids[rows,g,pos[rows,g]]
        require((chosen>=0).all(),'Required prefix unavailable')
        h[:,j]=chosen;pos[rows,g]+=1
    require(np.array_equal(mask[h],pattern) and all(len(set(x))==k for x in h),'H pattern/uniqueness')
    return h,pattern.sum(axis=1)


def metrics(top,users,val,groups,mask):
    den=np.zeros(3,dtype=np.int64);hits=den.copy();con=np.zeros(3)
    # S, nonS, and low-group nonS subset (overlaps nonS).
    subhit=np.zeros(3,dtype=np.int64);subexp=np.zeros(3,dtype=np.int64)
    for row,u in enumerate(users):
        pos=set(val.indices[val.indptr[u]:val.indptr[u+1]])
        require(bool(pos),'Empty Validation user')
        for i in pos:den[groups[i]]+=1
        for i in top[row]:
            flags=np.array([mask[i],not mask[i],groups[i]==0 and not mask[i]])
            subexp+=flags
            if i in pos:
                hits[groups[i]]+=1;con[groups[i]]+=1/len(pos)/len(users);subhit+=flags
    require((den>0).all(),'Empty group')
    return {'denominator':den.tolist(),'hits':hits.tolist(),'micro':(hits/den).tolist(),
            'contribution':con.tolist(),'recall20':float(con.sum()),
            'exposure':np.bincount(groups[top.ravel()],minlength=3).tolist(),
            'subgroup_order':['S','nonS','low_nonS'],'subgroup_hits':subhit.tolist(),'subgroup_exposure':subexp.tolist()}


def decide(rows):
    if all(x['low_H_minus_R']>=-3 and x['recall_H_minus_R']>=-.0002 for x in rows):
        return 'B_within_group_sufficient'
    if all(x['low_H_minus_R']<=-4 and x['recall_H_minus_R']<=.0002 for x in rows):
        return 'B_within_group_insufficient'
    return 'unresolved'


def worker(p,out):
    import psutil
    from initialization_kd_adapter import load_train_val
    access={'sealed':False,'denied':0,'opens':[]}
    sys.addaudithook(lambda e,a:guard(e,a,access))
    started=time.monotonic();samples=[]
    result={'status':'running','states':{},'comparisons':{},'data_access':access,'test_policy':'Test0','model_loads':0,'parameter_updates':0}
    def check():
        row={'seconds':time.monotonic()-started,'rss_bytes':psutil.Process().memory_info().rss}
        samples.append(row)
        require(row['seconds']<p['caps']['wall_seconds'] and row['rss_bytes']<p['caps']['rss_bytes'],'Worker resource cap')
    def inp(name):return ROOT/p['inputs'][name]['path']
    try:
        for name,sp in p['inputs'].items():
            require(sha(ROOT/sp['path'])==sp['sha256'],'Input SHA '+name);check()
        source=read(inp('manifest.json'))
        require(source['source_commit']=='6dbdc400cb029340d2c40bdc65e8f4a442941108','Exposure source')
        target=None;degree=None
        for modality in ['image','text']:
            sp=p['assets'][modality+'_graph'];require(sha(ROOT/sp['path'])==sp['sha256'],'Graph SHA')
            with np.load(ROOT/sp['path'],allow_pickle=False) as z:t,d=z['queries'].copy(),z['degree'].copy()
            if target is not None:require(np.array_equal(t,target) and np.array_equal(d,degree),'Modality identity')
            target,degree=t,d
        require(len(target)==6114 and len(np.unique(target))==6114,'S identity')
        mask=np.zeros(18357,dtype=bool);mask[target]=True
        sealed={};users0=None
        for seed in p['seeds']:
            with np.load(inp(f'seed{seed}_candidates.npz'),allow_pickle=False) as z:
                users,ids,scores=z['users'].copy(),z['ids'].copy(),z['scores'].copy()
                require(np.array_equal(target,z['targets']),'Cache targets')
            with np.load(inp(f'seed{seed}_lists.npz'),allow_pickle=False) as z:
                require(np.array_equal(users,z['users']),'Cache/list users');r=z['R'].copy()
            require(np.array_equal(users,np.arange(35598)),'Users')
            if users0 is not None:require(np.array_equal(users,users0),'Cross-seed users')
            users0=users
            h,k=construct(ids,scores,r,mask)
            np.savez_compressed(out/f'seed{seed}_H.npz',users=users,H=h,k=k,targets=target)
            sealed[str(seed)]=sha(out/f'seed{seed}_H.npz');check()
        seal={'time_ns':time.time_ns(),'H_sha256':sealed,'input_profile':p}
        save(out/'construction_seal.json',seal);access['sealed']=True
        train,val=load_train_val(ROOT/'data/sports',p['assets']['train']['sha256'],p['assets']['validation']['sha256'],35598,18357)
        require(np.array_equal(np.bincount(train.indices,minlength=18357),degree),'Train degree')
        require(np.array_equal(np.flatnonzero(np.diff(val.indptr)),users0),'Val users')
        order=np.lexsort((np.arange(18357),degree));groups=np.empty(18357,dtype=int)
        for g,part in enumerate(np.array_split(order,3)):groups[part]=g
        require(np.isin(target,order[:6119]).all(),'S low membership')
        reference=read(inp('report.json'))
        for seed in p['seeds']:
            with np.load(inp(f'seed{seed}_lists.npz'),allow_pickle=False) as z:tops={s:z[s].copy() for s in ['B','Q','R']}
            with np.load(out/f'seed{seed}_H.npz',allow_pickle=False) as z:tops['H']=z['H'].copy()
            ms={}
            for arm,top in tops.items():
                require(top.shape==(35598,20) and top.min()>=0 and top.max()<18357 and all(len(set(x))==20 for x in top),'List IDs')
                for j,u in enumerate(users0):
                    require(not np.isin(top[j],train.indices[train.indptr[u]:train.indptr[u+1]]).any(),'Seen item')
                m=metrics(top,users0,val,groups,mask);require(m['denominator']==[6347,6389,25163] and np.isfinite(m['recall20']),'Metrics')
                if arm!='H':
                    old=reference['states'][f'{seed}_{arm}']
                    require(abs(m['recall20']-old['recall20'])<=1e-6 and m['hits']==old['hits'] and m['exposure']==old['exposure'],'Original metrics')
                ms[arm]=m;result['states'][f'{seed}_{arm}']=m;check()
            for a,b in [('R','H'),('H','Q'),('H','B')]:
                pu,pi=paired(tops[b],tops[a],users0,val,18357)
                require(abs(float(np.mean((pu[:,0]-pu[:,1])/pu[:,3]))-(ms[a]['recall20']-ms[b]['recall20']))<1e-10,'Paired Recall')
                require(np.array_equal([(pi[:,0]-pi[:,1])[groups==g].sum() for g in range(3)],np.array(ms[a]['hits'])-ms[b]['hits']),'Paired hits')
                np.savez_compressed(out/f'seed{seed}_{a}_minus_{b}.npz',users=users0,per_user=pu,per_item=pi)
            result['comparisons'][str(seed)]={'low_H_minus_R':ms['H']['hits'][0]-ms['R']['hits'][0],
                'recall_H_minus_R':ms['H']['recall20']-ms['R']['recall20'],
                'low_H_minus_Q':ms['H']['hits'][0]-ms['Q']['hits'][0],
                'recall_H_minus_Q':ms['H']['recall20']-ms['Q']['recall20'],
                'low_H_minus_B':ms['H']['hits'][0]-ms['B']['hits'][0],
                'recall_H_minus_B':ms['H']['recall20']-ms['B']['recall20']}
            require(abs((ms['R']['recall20']-ms['Q']['recall20'])-((ms['R']['recall20']-ms['H']['recall20'])+(ms['H']['recall20']-ms['Q']['recall20'])))<1e-12,'Decomposition')
            save(out/'report.json',result);check()
        require(all(sha(out/f'seed{s}_H.npz')==h for s,h in sealed.items()),'H seal changed')
        require(access['denied']==0 and all(x['time_ns']>=seal['time_ns'] for x in access['opens']),'Label chronology')
        result['decision']=decide(list(result['comparisons'].values()));result['status']='completed'
        save(out/'acceptance.json',{'hard_acceptance':True,'new_lists':3,'metric_states':12,'Test0':True})
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
        require(os.environ.get('PER_USER_SLOT_WORKER') == str(os.getppid()), 'Worker must be supervised')
        worker(p, out)
        return
    from run_minimal_ranking_supervision import source_gate
    source = source_gate(dict(p, launch_source_rule='committed_head'))
    require(p['seeds']==[2022,2023,2024] and p['test_policy']=='Test0' and p['screen']=={'low_hit_tolerance':3,'recall_tolerance':.0002}, 'Frozen protocol')
    out.mkdir(parents=True, exist_ok=False)  # consumed once, even if execution fails
    save(out / 'manifest.json', {'source_commit': source, 'profile_sha256': sha(PROFILE),
                                'profile': p, 'command': sys.argv, 'status': 'launched'})
    import psutil
    started = time.monotonic()
    env = dict(os.environ, PER_USER_SLOT_WORKER=str(os.getpid()), CUBLAS_WORKSPACE_CONFIG=':4096:8',
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
