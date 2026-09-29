"""One CPU aggregation of saved lists; no model/scoring imports."""
from pathlib import Path
import json,hashlib,sys,time,pickle,subprocess,traceback
import numpy as np
import psutil
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(hits,den,exposure,covered,items):
 total=den.sum(1);rows=[]
 for g in range(3):
  good=den[:,g]>0;n=int(den[:,g].sum())
  rows.append({'hits':int(hits[:,g].sum()),'micro':float(hits[:,g].sum()/n) if n else None,'macro':float(np.mean(hits[good,g]/den[good,g])) if good.any() else None,'contribution':float(np.mean(hits[:,g]/total)),'exposure':float(exposure[g]/(20*len(den))),'coverage':float(covered[g]/items[g])})
 return rows

def main():
 if '--self-test' in sys.argv:
  h=np.array([[1,0,0],[0,1,0]]);den=np.array([[2,0,0],[1,1,0]])
  r=stats(h,den,[20,20,0],[1,1,0],[2,2,2])
  assert r[0]['micro']==1/3 and r[0]['macro']==.25 and r[0]['contribution']==.25 and r[2]['micro'] is None
  assert sum(x['contribution'] for x in r)==.5 and sum(x['exposure'] for x in r)==1
  print('Synthetic denominator/contribution/exposure/empty-group checks passed');return
 profile=read(ROOT/'docs/research/innovation2/SPARSE_ITEM_ASSESSMENT_PROFILE_V1.json')
 dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines()
 assert all(x.startswith('?? archive/reviews/') for x in dirty),'Dirty source'
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()==profile['branch']
 out=ROOT/profile['output'];out.mkdir(parents=True,exist_ok=False);start=time.monotonic();samples=[];denied=[]
 def guard(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes)):
   s=str(args[0]).lower().replace('\\','/')
   if '/data/' in s and 'test' in s:denied.append(s);raise RuntimeError('Test denied')
 sys.addaudithook(guard)
 def check():
  row={'seconds':time.monotonic()-start,'rss':psutil.Process().memory_info().rss,'bytes':sum(p.stat().st_size for p in out.iterdir() if p.is_file())};samples.append(row)
  assert row['seconds']<600 and row['rss']<2147483648 and row['bytes']<104857600,'Resource cap'
 manifest={'source':head,'profile':profile,'inputs':{},'mode':'saved-list CPU aggregation; no scoring'};write(out/'manifest.json',manifest)
 try:
  anchor=read(ROOT/profile['anchor_profile']);ver=read(ROOT/profile['verification']);src=ROOT/'exp/innovation2/ranking_gap_v1';old=read(src/'report.json')
  matrices=[]
  for key in ['train','validation']:
   a=anchor[key];p=ROOT/a['path'];assert sha(p)==a['sha256'];manifest['inputs'][a['path']]=a['sha256']
   with p.open('rb') as f:m=pickle.load(f).tocsr(copy=True)
   m.sum_duplicates();m.sort_indices();matrices.append(m)
  train,val=matrices;users=np.flatnonzero(np.diff(val.indptr));n=len(users);freq=np.bincount(train.indices,minlength=train.shape[1]);order=np.lexsort((np.arange(len(freq)),freq));groups=np.empty(len(freq),dtype=int)
  for g,part in enumerate(np.array_split(order,3)):groups[part]=g
  pos=[set(val.indices[val.indptr[u]:val.indptr[u+1]].tolist()) for u in users];known=[set(train.indices[train.indptr[u]:train.indptr[u+1]].tolist()) for u in users]
  den=np.array([np.bincount(groups[list(p)],minlength=3) for p in pos]);support=[]
  for g in range(3):
   mask=groups==g;v=freq[mask];support.append({'items':int(mask.sum()),'min_frequency':int(v.min()),'median_frequency':float(np.median(v)),'max_frequency':int(v.max()),'zero_frequency':int((v==0).sum()),'positive_pairs':int(den[:,g].sum()),'positive_users':int((den[:,g]>0).sum()),'positive_items':len({i for p in pos for i in p if groups[i]==g})})
  boundaries=[]
  for split in [6119,12238]:
   left,right=int(freq[order[split-1]]),int(freq[order[split]]);boundaries.append({'left_frequency':left,'right_frequency':right,'same_frequency':left==right,'tied_items_by_group':[int(((freq==left)&(groups==g)).sum()) for g in range(3)] if left==right else []})
  result={'groups':support,'boundaries':boundaries,'states':{},'differences':{},'users':n};sets={}
  for name in profile['states']:
   file=name+'_per_user.npz';p=src/file;h=sha(p);assert h==ver['artifact_hashes'][file];manifest['inputs'][str(p.relative_to(ROOT))]=h
   with np.load(p,allow_pickle=False) as z:
    top=z['top20'];assert np.array_equal(z['users'],users) and np.array_equal(z['validation_denominator'],den.sum(1))
    assert top.shape==(n,20) and np.all((top>=0)&(top<len(groups)))
    assert all(len(set(t))==20 and not(set(t)&known[j]) for j,t in enumerate(top))
    hitsets=[set(t)&pos[j] for j,t in enumerate(top)];sets[name]=hitsets
    assert np.array_equal(z['hit20'],np.array([[int(i) in pos[j] for i in t] for j,t in enumerate(top)]))
    hits=np.array([np.bincount(groups[list(hs)],minlength=3) if hs else np.zeros(3,dtype=int) for hs in hitsets])
    exposure=np.bincount(groups[top.ravel()],minlength=3);covered=np.bincount(groups[np.unique(top)],minlength=3)
   rows=stats(hits,den,exposure,covered,[x['items'] for x in support]);recall=float(np.mean(hits.sum(1)/den.sum(1)))
   assert abs(recall-old['states'][name]['recall20'])<1e-10 and abs(sum(x['contribution'] for x in rows)-recall)<1e-10 and abs(sum(x['exposure'] for x in rows)-1)<1e-10
   assert all(x[k] is None or 0<=x[k]<=1 for x in rows for k in ['micro','macro','exposure','coverage'])
   result['states'][name]={'recall':recall,'groups':rows};check()
  for name in profile['states'][1:]:
   gl=np.zeros((3,2))
   for j,(t,s) in enumerate(zip(sets['teacher'],sets[name])):
    for col,items in enumerate([t-s,s-t]):
     for i in items:gl[groups[i],col]+=1/len(pos[j])/n
   expected=old['comparisons'][name]['item_groups']
   assert np.allclose(gl,[[x['G_contribution'],x['L_contribution']] for x in expected],atol=1e-10,rtol=0)
  for seed in [2022,2023,2024]:
   for a,b in [('teacher',f'{seed}_R1'),('teacher',f'{seed}_T0'),(f'{seed}_T0',f'{seed}_R1')]:
    result['differences'][a+' minus '+b]=[{k:result['states'][a]['groups'][g][k]-result['states'][b]['groups'][g][k] for k in ['micro','macro','contribution','exposure','coverage']} for g in range(3)]
  write(out/'manifest.json',manifest);write(out/'report.json',result);check();write(out/'resource.json',{'samples':samples,'sampled_peak_rss':max(x['rss'] for x in samples),'seconds':time.monotonic()-start,'cooperative':True})
  write(out/'acceptance.json',{'status':'passed','states':7,'group_rows':21,'identity_checks':'Recall/group contribution/exposure/prior G-L passed','test_denied_attempts':len(denied),'training':0,'scoring':0,'hashes':{x.name:sha(x) for x in out.iterdir() if x.is_file()}});write(out/'exit.json',{'status':'completed','exit_code':0,'seconds':time.monotonic()-start})
 except BaseException:
  (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');write(out/'exit.json',{'status':'failed','exit_code':1,'seconds':time.monotonic()-start});raise
if __name__=='__main__':main()
