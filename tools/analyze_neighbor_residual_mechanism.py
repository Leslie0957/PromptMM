"""Bounded CPU algebra on saved assets; no interaction data or scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/research/innovation2/NEIGHBOR_RESIDUAL_MECHANISM_2026-09-30.json'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def energy(a):return float(np.sum(np.asarray(a,dtype=np.float64)**2))


def decomposition(neighbor, own):
    n,s=np.asarray(neighbor,dtype=np.float64),np.asarray(own,dtype=np.float64)
    y=n+s;total=energy(y)
    cross=float(2*np.sum(n*s))
    assert np.isclose(energy(n)+energy(s)+cross,total,rtol=1e-10,atol=1e-9)
    variation=n-n.mean(axis=0)
    return {'total_energy':total,'neighbor_energy':energy(n),'negative_self_energy':energy(s),
            'cross_term':cross,'neighbor_variation_energy':energy(variation),
            'mean_neighbor_approx_relative_error':float(np.sqrt(energy(variation)/total)),
            'neighbor_centered_fraction':energy(variation)/energy(n),
            'negative_self_fraction_of_component_energy':energy(s)/(energy(n)+energy(s))}


def main():
    if OUT.exists():raise RuntimeError('Output already exists; no overwrite')
    start=time.monotonic();peak=0
    def guard(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(ROOT/'data'):raise PermissionError('All interaction data denied')
    sys.addaudithook(guard)
    torch.set_num_threads(4)
    def check():
        nonlocal peak
        peak=max(peak,psutil.Process().memory_info().rss)
        assert peak<4*2**30 and time.monotonic()-start<600,'Analysis resource cap'
    def read(p):return json.loads(p.read_text(encoding='utf-8'))
    profile=read(ROOT/'docs/research/innovation2/NEIGHBOR_SHARED_RESIDUAL_PROFILE_V1.json')
    run=ROOT/profile['output'];manifest=read(run/'manifest.json')
    bindings=read(run/'completeness.json')['files_sha256'];inputs={}
    def bound(p,h):
        assert sha(p)==h,str(p);inputs[str(p.relative_to(ROOT))]=h
    asset=profile['assets']['teacher'];path=ROOT/asset['path'];bound(path,asset['sha256'])
    q=torch.load(path,map_location='cpu',weights_only=True)['items'].numpy()
    graphs={}
    for m in ['image','text']:
        spec=profile['assets'][m+'_graph'];p=ROOT/spec['path'];bound(p,spec['sha256'])
        with np.load(p,allow_pickle=False) as z:graphs[m]={k:z[k].copy() for k in ['queries','real','random']}
    ids=graphs['image']['queries'];assert np.array_equal(ids,graphs['text']['queries'])
    result={'scope':'posthoc CPU algebra; no data, no scoring, no causal ablation','inputs':inputs,'features':{},'states':{}}
    for variant in ['N','R']:
        nblocks=[];sblocks=[];xblocks=[]
        for m in ['image','text']:
            graph=graphs[m]['real'] if variant=='N' else graphs[m]['random'][0]
            c=q[graph].mean(axis=1);v=c-q[ids]
            scale=float(np.sqrt(np.mean(np.sum(v.astype(np.float64)**2,axis=1))))
            xblocks.append(v/scale/np.sqrt(2.0))
            nblocks.append(c.astype(np.float64)/scale/np.sqrt(2.0))
            sblocks.append(-q[ids].astype(np.float64)/scale/np.sqrt(2.0))
        x=np.zeros((len(q),128),dtype=np.float32);x[ids]=np.concatenate(xblocks,axis=1)
        expected=manifest['graphs']['feature_sha256']['N_F' if variant=='N' else 'R']
        assert hashlib.sha256(x.tobytes()).hexdigest()==expected
        n,s=np.concatenate(nblocks,axis=1),np.concatenate(sblocks,axis=1)
        assert np.allclose(n+s,x[ids],rtol=1e-5,atol=1e-6)
        result['features'][variant]=decomposition(n,s)
        for seed in [2022,2023,2024]:
            for arm in (['N','F'] if variant=='N' else ['R']):
                folder=run/f'seed{seed}'/arm
                for file in ['final.pt','export.pt','status.json']:bound(folder/file,bindings[f'seed{seed}/{arm}/{file}'])
                ck=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
                export=torch.load(folder/'export.pt',map_location='cpu',weights_only=False)
                w=ck['extra'].numpy().astype(np.float64)
                if arm=='F':
                    zn=n[:,:64]*w[0]+n[:,64:]*w[1];zs=s[:,:64]*w[0]+s[:,64:]*w[1]
                    xt=torch.from_numpy(x);res=ck['extra'][0]*xt[:,:64]+ck['extra'][1]*xt[:,64:]
                else:
                    zn=n@w.T;zs=s@w.T;res=torch.nn.functional.linear(torch.from_numpy(x),ck['extra'])
                expected_item=ck['item']+res
                max_error=float((expected_item-export['item']).abs().max())
                assert torch.allclose(expected_item,export['item'],rtol=1e-5,atol=1e-5)
                assert torch.equal(ck['user'],export['user'])
                mask=np.ones(len(q),dtype=bool);mask[ids]=False
                assert torch.equal(expected_item[mask],ck['item'][mask])
                status=read(folder/'status.json');last=status['curve'][-1]['groups']
                row=decomposition(zn,zs)
                row.update(export_max_abs_error=max_error,low_hits=last['hits'][0],
                           low_exposure=last['exposure'][0],
                           hits_per_low_exposure=last['hits'][0]/last['exposure'][0],
                           low_exposure_fraction=last['exposure'][0]/711960)
                if arm=='F':row['scalars']=w.tolist()
                result['states'][f'{seed}_{arm}']=row
                check()
    result['seconds']=time.monotonic()-start;result['sampled_peak_rss']=peak
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))


if __name__=='__main__':main()
