"""CPU-only saved-artifact audit; never scores or loads interaction datasets."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/innovation2/neighbor_shared_residual_three_seed_v1'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(4)
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    def finite(x):
        if torch.is_tensor(x): return bool(torch.isfinite(x).all())
        if isinstance(x, dict): return all(finite(v) for v in x.values())
        if isinstance(x, (list, tuple)): return all(finite(v) for v in x)
        if isinstance(x, float): return math.isfinite(x)
        return True
    profile_path = ROOT/'docs/research/innovation2/NEIGHBOR_SHARED_RESIDUAL_PROFILE_V1.json'
    p, m, summary = read(profile_path), read(OUT/'manifest.json'), read(OUT/'summary.json')
    source = m['source_commit']
    check('source_exists', subprocess.check_output(['git','rev-parse',source],cwd=ROOT,text=True).strip()==source)
    check('profile_hash', sha(profile_path)==m['profile_sha256'])
    for name,h in m['source_files_sha256'].items(): check('source_hash:'+name,sha(ROOT/name)==h)
    for name in list(m['source_files_sha256'])+['tools/run_minimal_ranking_supervision.py','codes/initialization_kd_adapter.py']:
        check('source_unchanged:'+name,not subprocess.check_output(['git','diff',source,'--',name],cwd=ROOT))
    check('profile_commit',json.loads(subprocess.check_output(['git','show',source+':'+str(profile_path.relative_to(ROOT)).replace('\\','/')],cwd=ROOT))==p)
    check('asset_references',m['assets']==p['assets'] and m['seeds']==p['seeds'])
    hashes=read(OUT/'completeness.json')['files_sha256']
    for name,h in hashes.items():check('complete_sha:'+name,sha(OUT/name)==h)
    check('manifest_complete',m['status']=='completed' and all(x=='completed' for v in m['arm_status'].values() for x in v.values()))
    check('test_record',m['test_access']==0)
    resource,exit_record=read(OUT/'resource.json'),read(OUT/'exit.json')
    check('exit',exit_record['exit_code']==0 and exit_record['status']=='completed')
    for key in ['rss_bytes','cuda_allocator_bytes','output_bytes']:
        check('resource:'+key,max(s[key] for s in resource['samples'])<=p['caps'][key])
    check('disk_floor',min(s['free_disk_bytes'] for s in resource['samples'])>=p['caps']['minimum_free_disk_bytes'])
    check('wall',exit_record['seconds']<=p['caps']['wall_seconds'])
    rows={};details={};paired={};screens={}
    for seed in [2022,2023,2024]:
        seed=str(seed);rows[seed]={};details[seed]={}
        epoch0_top=None
        for arm in ['B','N','F','R']:
            folder=OUT/('seed'+seed)/arm;s=read(folder/'status.json');a=read(folder/'acceptance.json')
            tag=seed+arm;curve=s['curve']
            check(tag+':execution',s['status']=='completed' and s['steps']==64200 and a['steps']==64200 and a['status']=='passed')
            check(tag+':epochs',[r['epoch'] for r in curve]==list(range(1,301)))
            check(tag+':finite',finite(s))
            check(tag+':arm_time',s['arm_seconds']<28800)
            check(tag+':initial',abs(s['epoch0']['recall20']-p['evaluation']['B_epoch0_expected'])<1e-6)
            for name,h in a['files_sha256'].items():check(tag+':acceptance:'+name,sha(folder/name)==h)
            best=max(curve,key=lambda r:r['recall20'])
            for label,expected in [('best',best),('final',curve[-1])]:
                ck=torch.load(folder/(label+'.pt'),map_location='cpu',weights_only=False)
                check(tag+label+':identity',ck['source']==source and ck['seed']==int(seed) and ck['arm']==arm and ck['epoch']==expected['epoch'])
                # Runner stores evaluate() tuple (metric dict, optional Top20) in checkpoints.
                metric=ck['metric'][0] if isinstance(ck['metric'],(tuple,list)) else ck['metric']
                check(tag+label+':metric',abs(metric['recall20']-expected['recall20'])<1e-12)
                check(tag+label+':finite',finite(ck))
                check(tag+label+':shape',tuple(ck['user'].shape)==(35598,64) and tuple(ck['item'].shape)==(18357,64))
                check(tag+label+':extra',ck['extra'] is None if arm=='B' else tuple(ck['extra'].shape)==((2,) if arm=='F' else (64,128)))
                opt=ck['optimizer'];states=opt['state']
                check(tag+label+':adam_steps',len(states)==(2 if arm=='B' else 3) and all(float(v['step'])==expected['epoch']*214 for v in states.values()))
                check(tag+label+':adam_config',all(g['lr']==6e-5 and g['weight_decay']==.01 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 for g in opt['param_groups']))
                check(tag+label+':checkpoint_binding',sha(folder/(label+'.pt'))==s['checkpoints'][label]['sha256'])
                if label=='final':
                    export=torch.load(folder/'export.pt',map_location='cpu',weights_only=False)
                    check(tag+':export_finite',finite(export))
                    check(tag+':export_user',torch.equal(export['user'],ck['user']))
                    check(tag+':export_shape',export['item'].shape==ck['item'].shape)
                    if arm=='B':check(tag+':export_B',torch.equal(export['item'],ck['item']))
            for epoch in [0,50,100,200,300]:
                r=s['epoch0'] if epoch==0 else curve[epoch-1];g=r['groups']
                check(tag+str(epoch)+':den',g['denominator']==[6347,6389,25163])
                check(tag+str(epoch)+':micro',np.allclose(np.array(g['hits'])/g['denominator'],g['micro'],rtol=0,atol=1e-14))
                check(tag+str(epoch)+':contribution',abs(sum(g['contribution'])-r['recall20'])<1e-12)
                with np.load(folder/f'top20_epoch{epoch}.npz',allow_pickle=False) as z:
                    top=z['top20'];users=z['users']
                    check(tag+str(epoch)+':top_valid',top.shape==(35598,20) and np.all((top>=0)&(top<18357)) and np.all(np.diff(np.sort(top,axis=1),axis=1)!=0) and np.array_equal(users,np.arange(35598)))
                    if epoch==0:
                        if epoch0_top is None:epoch0_top=top.copy()
                        check(tag+':same_start_top',np.array_equal(top,epoch0_top))
            if arm=='B':check(tag+':regression',abs(curve[-1]['recall20']-p['seeds'][seed]['B_epoch300_expected'])<1e-6)
            rows[seed][arm]=curve[-1]
            check(tag+':summary_row',curve[-1]==summary['rows'][seed][arm])
            details[seed][arm]={'best_epoch':best['epoch'],'best_recall':best['recall20'],'seconds':s['arm_seconds'],'exposure_low':curve[-1]['groups']['exposure'][0]}
        paired[seed]={}
        for name,left,right in [('N_minus_B','N','B'),('F_minus_B','F','B'),('R_minus_B','R','B'),('N_minus_F','N','F'),('N_minus_R','N','R')]:
            l,r=rows[seed][left],rows[seed][right]
            d={'recall20':l['recall20']-r['recall20']}
            for key in ['micro','hits','contribution','exposure']:d[key]=[a-b for a,b in zip(l['groups'][key],r['groups'][key])]
            check(seed+name+':paired',d==summary['paired'][seed][name]);paired[seed][name]=d
        q=paired[seed];screens[seed]='candidate_signal' if q['N_minus_B']['micro'][0]>=.001 and q['N_minus_B']['recall20']>=-.0002 and q['N_minus_F']['micro'][0]>=.0005 and q['N_minus_R']['micro'][0]>=.0005 else 'screen_stop'
        check(seed+':screen',screens[seed]==summary['screens'][seed])
    result={'status':'passed','checks':len(checks),'source':source,'files_hashed':len(hashes),'rows':rows,'paired':paired,'details':details,'screens':screens,'resources':{k:v for k,v in resource.items() if k!='samples'},'exit':exit_record,'scope':'Saved artifacts and source only, no data/Test loading or score evaluation; N/F/R export arithmetic not independently reconstructed'}
    target=ROOT/'docs/research/innovation2/NEIGHBOR_SHARED_RESIDUAL_VERIFICATION_2026-09-30.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','checks','source','files_hashed','screens','resources']},indent=2))


if __name__=='__main__':main()
