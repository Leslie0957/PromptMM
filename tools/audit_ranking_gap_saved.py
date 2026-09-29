"""Independent CPU audit of saved rankings; never scores or imports project runners."""
from pathlib import Path
import hashlib,json,pickle,re,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def guard(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes)):
        s=str(args[0]).lower().replace('\\','/')
        if '/data/' in s and 'test' in s: raise RuntimeError('Test denied')
sys.addaudithook(guard)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
checks=[]
def ck(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)
def near(a,b):return bool(np.allclose(a,b,atol=1e-12,rtol=0))
out=ROOT/'exp/innovation2/ranking_gap_v1'
m=read(out/'manifest.json'); r=read(out/'report.json'); profile=read(ROOT/m['profile'])
ck('profile hash',sha(ROOT/m['profile'])==m['profile_sha256'])
ck('committed profile',json.loads(subprocess.check_output(['git','show',m['source_commit']+':'+m['profile']],cwd=ROOT))==profile)
for path in ['tools/run_ranking_gap_diagnostic.py','codes/ranking_gap_diagnostic.py','codes/initialization_kd_fast_eval.py','codes/initialization_kd_adapter.py','codes/initialization_kd_interaction.py']:
    blob=subprocess.check_output(['git','show',m['source_commit']+':'+path],cwd=ROOT)
    ck('source '+path,blob.replace(b'\r\n',b'\n')==(ROOT/path).read_bytes().replace(b'\r\n',b'\n'))
handoff=(ROOT/'docs/research/innovation2/RANKING_GAP_DIAGNOSTIC_HANDOFF.md').read_text(encoding='utf-8')
for name,h in re.findall(r'\| `([^`]+)` \| `([0-9a-f]{64})` \|',handoff):ck('artifact '+name,sha(out/name)==h)
for a in m['assets']:ck('asset '+a['path'],sha(ROOT/a['path'])==a['sha256'])
for s in profile['states']:
    original=read((ROOT/s['path']).parents[1]/'launch_manifest.json')
    name=str(s['seed'])+'_'+s['arm']; provenance=m['state_provenance'][name]
    ck('provenance '+name,original['source_commit']==provenance['original_source_commit'] and original['config_digest']==provenance['original_config_digest'] and original['config']['seed']==s['seed'])
mat=[]
for a in [profile['train'],profile['validation']]:
    with (ROOT/a['path']).open('rb') as f:x=pickle.load(f).tocsr(copy=True)
    x.sum_duplicates();x.sort_indices();mat.append(x)
train,val=mat; users=np.flatnonzero(np.diff(val.indptr));n=len(users)
def groups(counts):
    order=sorted(range(len(counts)),key=lambda i:(int(counts[i]),i));g=np.zeros(len(counts),dtype=int)
    for k,part in enumerate(np.array_split(order,3)):g[part]=k
    return g
ug=groups(np.diff(train.indptr));ig=groups(np.bincount(train.indices,minlength=train.shape[1]))
positives=[set(val.indices[val.indptr[u]:val.indptr[u+1]]) for u in users]
known=[set(train.indices[train.indptr[u]:train.indptr[u+1]]) for u in users]
sets={}; tops={}; recall={}
for name in r['states']:
    with np.load(out/(name+'_per_user.npz'),allow_pickle=False) as z:
        ck('users '+name,np.array_equal(z['users'],users));top=z['top20'];hits=z['hit20'];den=z['validation_denominator']
        ck('shape '+name,top.shape==(n,20) and hits.shape==(n,20))
        ck('denominator '+name,np.array_equal(den,np.diff(val.indptr)[users]))
        rebuilt=np.array([[i in positives[u] for i in row] for u,row in enumerate(top)])
        ck('hit labels '+name,np.array_equal(rebuilt,hits))
        ck('valid/filtered '+name,all(len(set(row))==20 and not(set(row)&known[u]) and min(row)>=0 and max(row)<train.shape[1] for u,row in enumerate(top)))
        sets[name]=[set(row[mask]) for row,mask in zip(top,hits)];tops[name]=[set(row) for row in top]
        recall[name]=float(np.mean(hits.sum(1)/den));ck('recall '+name,near(recall[name],r['states'][name]['recall20']))
expected={'teacher':profile['teacher']['expected_recall20'],**{str(s['seed'])+'_'+s['arm']:s['expected_recall20'] for s in profile['states']}}
ck('original numeric regression',all(abs(recall[k]-v)<=1e-6 for k,v in expected.items()))
arrays={};summary={}
for name,reported in r['comparisons'].items():
    rows=[];item=np.zeros((3,3));user=np.zeros((3,4))
    for u,(t,s,pos) in enumerate(zip(sets['teacher'],sets[name],positives)):
        a,b=t-s,s-t;d=len(pos);row=[len(a)/d,len(b)/d,len(t&s)/d,len(pos-t-s)/d,len(tops['teacher'][u]&tops[name][u])/20,int(bool(a))];rows.append(row)
        user[ug[users[u]]]+=[1,d,row[0],row[1]]
        for i in pos:item[ig[i],2]+=1
        for i in a:item[ig[i],0]+=1/d/n
        for i in b:item[ig[i],1]+=1/d/n
    ar=np.array(rows);arrays[name]=ar[:,:2];means=ar.mean(0)
    for k,v in zip(['g','l','common','neither','overlap','teacher_only_user'],means):ck(name+' '+k,near(v,reported['metrics'][k]))
    ck(name+' identity',near(means[0]-means[1],recall['teacher']-recall[name]))
    for j in range(3):
        z=reported['user_groups'][j];ck(name+' usergroup '+str(j),near(user[j],[z['users'],z['positives'],z['G_sum'],z['L_sum']]))
        z=reported['item_groups'][j];ck(name+' itemgroup '+str(j),near(item[j],[z['G_contribution'],z['L_contribution'],z['positives']]))
    summary[name]={'G':means[0],'L':means[1],'N':means[0]-means[1],'teacher_only_users':int(ar[:,5].sum()),'high_item_G_fraction':float(item[2,0]/means[0])}
rng=np.random.default_rng(20260929);boot={k:[] for k in arrays}
for _ in range(2000):
    ix=rng.integers(0,n,size=n)
    for k,a in arrays.items():
        g,l=a[ix].mean(0);boot[k].append([g,l,g-l])
for k,b in boot.items():
    q=np.quantile(b,[.025,.975],axis=0)
    ck(k+' bootstrap',near(q,np.array([r['comparisons'][k]['bootstrap95'][j] for j in ['G','L','N']]).T))
ck('screen',all(summary[str(s)+'_T0']['G']>=.001 for s in [2022,2023,2024]) and r['screen']=='opportunity_present')
ck('exit',read(out/'exit.json')['exit_code']==0)
ck('acceptance',read(out/'acceptance.json')['status']=='passed')
res=read(out/'resource.json')['samples'];caps=profile['caps']
ck('sampled resources',all(x['rss_bytes']<=caps['rss_bytes'] and x['cuda_allocator_bytes']<=caps['cuda_allocator_bytes'] and x['free_disk_bytes']>=caps['minimum_free_disk_bytes'] and x['output_bytes']<=caps['output_bytes'] for x in res))
ck('elapsed',read(out/'exit.json')['elapsed_seconds']<=caps['wall_seconds'])
print(json.dumps({'passed_checks':len(checks),'checks':checks,'summary':summary,'source':m['source_commit'],'audit_mode':'CPU saved rankings + Train/Val labels; no scoring, torch, training or Test','limitations':['Test counters are literals, not instrumented counts','resource.json final monitor overwrites summary fields; samples and report remain','No saved scores: finite scores and tie boundary rely on pinned source/runtime'],'artifact_hashes':{p.name:sha(p) for p in out.iterdir() if p.is_file()}},indent=2))
