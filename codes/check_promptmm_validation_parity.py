"""Manual one-shot fixed-checkpoint Validation parity; no training or Test."""
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
from datetime import datetime
from promptmm_release_resource import ROOT, sha256, tensor_digest
from promptmm_release_validation import atomic_json, load_validation, restore_best

RUN_ID='sports_promptmm_validation_parity_seed2022_lr6e5_v1'
CHECKPOINT=ROOT/'exp/promptmm_release/sports_promptmm_release_validation300_seed2022_lr6e5_v1/best.pt'
CHECKPOINT_SHA='e32f9907d68874f09f4327a6ccfba8b4c86e1b7c73c7d0a06713f990b6e94e27'


def main():
    if len(sys.argv)!=1:raise ValueError('No overrides permitted.')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Clean committed source required.')
    out=ROOT/'exp/validation_checks'/RUN_ID
    out.mkdir(parents=True,exist_ok=False)
    report=dict(status='started',run_id=RUN_ID,launch_commit=head,launch_dirty=False,
                command=[sys.executable,*sys.argv],started_at=datetime.now().astimezone().isoformat(),
                checkpoint_sha256=CHECKPOINT_SHA,optimizer_steps=0,test_evaluations=0,
                test_split_loaded=False,paper_ready_eligible=False,users_checked=0,validation_passes=0,
                source={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),
                    ROOT/'codes/promptmm_validation_fast.py',ROOT/'codes/promptmm_release.py',
                    ROOT/'codes/promptmm_release_validation.py',ROOT/'codes/promptmm_release_resource.py',
                    ROOT/'codes/utility/metrics.py']})
    def save():atomic_json(out/'report.json',report)
    save()
    try:
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        import numpy as np
        import torch
        from promptmm_release import ReleaseStudent,release_graphs,sparse_tensor
        from promptmm_validation_fast import rank_fast,rank_legacy,accumulate,evaluate_fast
        from promptmm_release_validation import evaluate_validation
        if not torch.cuda.is_available():raise RuntimeError('CUDA required.')
        if sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('Checkpoint identity mismatch.')
        c=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        data=ROOT/'data/sports'
        if sha256(data/'train_mat')!=c['input_sha256']['train_mat']:raise RuntimeError('Train identity mismatch.')
        with (data/'train_mat').open('rb') as f:train=pickle.load(f).tocsr()
        val=load_validation(data,train,c['input_sha256']['val_mat'])
        report['input_sha256']={k:c['input_sha256'][k] for k in ('train_mat','val_mat')}
        device=torch.device('cuda:0')
        model=ReleaseStudent(*train.shape,c['config']['embedding_dim'],c['config']['layers']).to(device)
        state=c['model_state_dict']
        model.init_user_item_embed(state['user_id_embedding.weight'].to(device),state['item_id_embedding.weight'].to(device))
        restore_best(CHECKPOINT,model)
        before=tensor_digest(model)
        adj=sparse_tensor(release_graphs(train)[2],device)
        ks=(10,20,40,50);users=np.flatnonzero(val.getnnz(axis=1)).tolist()
        all_items=set(range(train.shape[1]));model.eval()
        totals=[{k:np.zeros(4,dtype=np.float64) for k in ('recall','precision','ndcg','hit_ratio')} for _ in range(2)]
        seconds=[0.,0.]
        # Compare full ranked IDs from exactly the same score array. Alternate
        # ranker order by user to reduce systematic warm-cache timing bias.
        with torch.no_grad():
            ue,ie=model(adj)
            for offset in range(0,len(users),256):
                batch=users[offset:offset+256];scores=(ue[batch] @ ie.T).cpu().numpy()
                if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite scores.')
                for row,u in enumerate(batch):
                    candidates=list(all_items-set(train[u].indices));truth=set(val[u].indices)
                    ranks={}
                    for which in ((0,1) if u%2==0 else (1,0)):
                        start=time.perf_counter();ranks[which]=(rank_legacy if which==0 else rank_fast)(scores[row],candidates,50)
                        seconds[which]+=time.perf_counter()-start
                        accumulate(totals[which],ranks[which],truth,ks,len(users))
                    if ranks[0]!=ranks[1]:raise RuntimeError(f'Rank mismatch at user{u}')
                    report['users_checked']+=1
                if offset%4096==0:print('Rank parity users:',report['users_checked'],flush=True)
        expected={k:v.tolist() for k,v in totals[0].items()}
        if expected!={k:v.tolist() for k,v in totals[1].items()}:raise RuntimeError('Metric mismatch.')
        if report['users_checked']!=len(users):raise RuntimeError('Incomplete user coverage.')
        report.update(rank_ids_exact=True,paired_metrics=expected,ranking_seconds=seconds,validation_passes=1)
        save()
        # Full function checks, once each, same frozen model and batch shape.
        times={};outputs={}
        for name,fn in [('legacy',evaluate_validation),('fast',evaluate_fast)]:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            outputs[name]=fn(model,adj,train,val)
            report['validation_passes']+=1
            torch.cuda.synchronize();times[name]=time.perf_counter()-start
            report[name+'_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            print(name,'seconds:',times[name],flush=True)
        if outputs['legacy']!=expected or outputs['fast']!=expected:raise RuntimeError('End-to-end metric mismatch.')
        if tensor_digest(model)!=before or sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise RuntimeError('Model/checkpoint changed.')
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=head or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise RuntimeError('Source changed.')
        for name,h in report['source'].items():
            if sha256(ROOT/name)!=h:raise RuntimeError('Source hash changed.')
        report.update(status='completed',metrics_exact=True,checkpoint_unchanged=True,
                      validation_passes=3,end_to_end_seconds=times,environment=dict(python=sys.version,
                      torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0)),
                      historical_metric_max_abs_diff=max(abs(outputs['legacy'][k][i]-c['validation_metrics'][k][i]) for k in expected for i in range(4)))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        report['finished_at']=datetime.now().astimezone().isoformat();save();print('Report:',out/'report.json',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
