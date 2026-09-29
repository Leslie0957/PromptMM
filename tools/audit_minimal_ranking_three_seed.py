"""CPU artifact audit only; no data split, ranking, training or runner imports."""
from pathlib import Path
import sys,json,hashlib,subprocess,copy,math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
def guard(event,args):
 if event=='open' and args and isinstance(args[0],(str,bytes)):
  if '/data/' in str(args[0]).replace('\\','/').lower():raise RuntimeError('Audit denies data split access')
sys.addaudithook(guard)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
checks=[]
def ck(n,b):
 if not b:raise AssertionError(n)
 checks.append(n)
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 if isinstance(x,float):return math.isfinite(x)
 return True
out=ROOT/'exp/innovation2/minimal_ranking_three_seed_v1';cm=read(out/'manifest.json');ce=read(out/'exit.json');src=cm['source_commit']
co= json.loads(subprocess.check_output(['git','show',src+':'+cm['profile']],cwd=ROOT));base=json.loads(subprocess.check_output(['git','show',src+':'+co['base_profile']],cwd=ROOT))
ck('cohort profile SHA',sha(ROOT/cm['profile'])==cm['profile_sha256']);ck('base SHA',sha(ROOT/co['base_profile'])==co['base_profile_sha256'])
ck('cohort completed',ce['status']=='completed' and ce['exit_code']==0 and all(v=='completed' for v in cm['seed_status'].values()))
summary={};fingerprints={};total_hashes=0
for seed in [2022,2023,2024]:
 folder=out/f'seed{seed}';m=read(folder/'manifest.json');r=read(folder/'report.json');ac=read(folder/'acceptance.json');ex=read(folder/'exit.json');res=read(folder/'resource.json');samples=read(folder/'resource_samples.json')
 p=copy.deepcopy(base);delta=co['seeds'][str(seed)]
 for k in ['anchor_manifest','anchor_manifest_sha256']:p[k]=delta[k]
 p['seed']=seed;p['assets']['triplet_tape']=delta['triplet_tape'];p['evaluation']['B_epoch300_expected']=delta['B_epoch300_expected'];p['branch']=co['branch'];p['launch_source_rule']=co['launch_source_rule'];p['command']=co['command'];p['output']=co['output']+f'/seed{seed}'
 ck(f'{seed} effective profile',p==m['effective_profile']);ck(f'{seed} source',m['source_commit']==r['source_commit']==src and m['profile_sha256']==cm['profile_sha256'])
 ck(f'{seed} anchor',sha(ROOT/p['anchor_manifest'])==p['anchor_manifest_sha256'])
 for file,h in m['source_files_sha256'].items():
  ck(f'{seed} source hash {file}',sha(ROOT/file)==h)
  blob=subprocess.check_output(['git','show',src+':'+file],cwd=ROOT)
  ck(f'{seed} source blob {file}',blob.replace(b'\r\n',b'\n')==(ROOT/file).read_bytes().replace(b'\r\n',b'\n'))
 for name,h in ac['artifact_sha256'].items():ck(f'{seed} artifact {name}',sha(folder/name)==h);total_hashes+=1
 ck(f'{seed} candidate copy',sha(folder/'candidates.npz')==p['prepared_candidates']['sha256'])
 ck(f'{seed} exit',ex['exit_code']==0 and ex['status']==r['status']=='completed' and ac['status']=='passed')
 ck(f'{seed} finite report',finite(r));ck(f'{seed} Test0/Val',r['test_denied_open_attempts']==ac['test_denied_open_attempts']==0 and r['validation_calls']==ac['validation_calls']==904 and m['test_policy']=='Test0')
 rows={}
 for arm in ['B','R','A']:
  a=r['arms'][arm];curve=read(folder/f'{arm}_curve.json');ck(f'{seed}{arm} curves',curve==a['curve'] and [v['epoch'] for v in curve]==list(range(1,301)))
  ck(f'{seed}{arm} steps/diagnostics',a['optimizer_steps']==64200 and len(a['epoch_diagnostics'])==300)
  ck(f'{seed}{arm} epoch0',abs(a['epoch0']['recall20']-p['evaluation']['B_epoch0_expected'])<=1e-6)
  best=max(curve,key=lambda z:z['recall20']);final=curve[-1]
  ck(f'{seed}{arm} metrics bounds',all(0<=z[k]<=1 for z in curve for k in ['recall20','ndcg20']))
  for label,chosen in [('best',best),('final',final)]:
   z=torch.load(folder/f'{arm}_{label}.pt',map_location='cpu',weights_only=True)
   ck(f'{seed}{arm}{label} binding',z['arm']==arm and z['epoch']==chosen['epoch'] and z['source_commit']==src and z['profile_sha256']==cm['profile_sha256'] and z['metric']=={k:chosen[k] for k in ['recall20','ndcg20']})
   ck(f'{seed}{arm}{label} model',set(z['model'])=={'user_id_embedding.weight','item_id_embedding.weight'} and all(t.dtype==torch.float32 and torch.isfinite(t).all() for t in z['model'].values()))
   ck(f'{seed}{arm}{label} shapes',[tuple(t.shape) for t in z['model'].values()]==[(35598,64),(18357,64)])
   opt=z['optimizer'];g=opt['param_groups'][0]
   ck(f'{seed}{arm}{label} AdamW',g['lr']==6e-5 and g['weight_decay']==.01 and tuple(g['betas'])==(.9,.999) and g['eps']==1e-8 and len(opt['state'])==2)
   for state,t in zip(opt['state'].values(),z['model'].values()):
    ck(f'{seed}{arm}{label} moment {tuple(t.shape)}',float(state['step'])==chosen['epoch']*214 and state['exp_avg'].shape==t.shape and state['exp_avg_sq'].shape==t.shape and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all() and (state['exp_avg_sq']>=0).all())
  rows[arm]={'recall20':final['recall20'],'recorded_ndcg20':final['ndcg20'],'best_recall20':best['recall20'],'best_epoch':best['epoch']}
 ck(f'{seed} B regression',abs(rows['B']['recall20']-p['evaluation']['B_epoch300_expected'])<=1e-6)
 rows['M']={'recall20':r['mixture']['recall20'],'recorded_ndcg20':r['mixture']['ndcg20']}
 for arm in ['B','R','A','M']:
  with np.load(folder/f'{arm}_final_top20.npz',allow_pickle=False) as z:
   ck(f'{seed}{arm} saved top IDs',z['top20'].shape==(35598,20) and np.all((z['top20']>=0)&(z['top20']<18357)) and all(len(set(row))==20 for row in z['top20']))
 for arm in ['R','A','M']:ck(f'{seed} delta {arm}',abs(rows[arm]['recall20']-rows['B']['recall20']-r['deltas'][arm+'_minus_B'])<1e-14)
 dr=r['deltas']['R_minus_B'];screen='screen_stop' if dr<.001 else 'simple_alternative_or_unresolved' if rows['R']['recall20']-max(rows['A']['recall20'],rows['M']['recall20'])<.0002 else 'candidate_for_replication'
 ck(f'{seed} screen',screen==r['screen'])
 caps=p['caps'];ck(f'{seed} resources',ex['elapsed_seconds']<=caps['wall_seconds'] and r['peak_cuda_allocator_bytes']<=caps['cuda_allocator_bytes'] and all(s['rss_bytes']<=caps['rss_bytes'] and s['free_disk_bytes']>=caps['minimum_free_disk_bytes'] and s['output_bytes']<=caps['output_bytes'] for s in samples))
 size=sum(x.stat().st_size for x in folder.rglob('*') if x.is_file());ck(f'{seed} final bytes',size<=caps['output_bytes'])
 summary[str(seed)]={'rows':rows,'deltas':r['deltas'],'screen':screen,'seconds':ex['elapsed_seconds'],'peak_cuda':r['peak_cuda_allocator_bytes'],'peak_rss':r['peak_sampled_rss_bytes'],'bytes':size}
 fingerprints[str(seed)]={x.name:sha(x) for x in folder.iterdir() if x.is_file()}
ck('cohort time/space',ce['elapsed_seconds']<=86400 and sum(x.stat().st_size for x in out.rglob('*') if x.is_file())<=6442450944)
print(json.dumps({'checks':checks,'passed_checks':len(checks),'acceptance_hash_count':total_hashes,'source':src,'summary':summary,'cohort_exit':ce,'fingerprints':fingerprints,'scope':'CPU checkpoints and saved records only; denies data access; no ranking/training/Test'},indent=2))
