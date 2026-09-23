"""Manual eight-batch zero-update gradient diagnostic at shared initialization."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codes'))
OUT=ROOT/'exp/gradient_checks/sports_sharedinit_seed2022_v2'
TRAIN_SHA='5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8'

def native_batch_ids(batch):
    """Preserve sampled IDs/order while removing NumPy scalar wrappers."""
    return [[int(value) for value in side] for side in batch]

def source():
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Clean committed source required.')
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--describe',action='store_true');args=parser.parse_args()
    if args.describe:
        print('seed2022; 8 batches x1024; shared initialization; zero updates; train split only; output '+str(OUT));return
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    head=source();OUT.mkdir(parents=True,exist_ok=False)
    from promptmm_release_validation import atomic_json
    report=dict(status='started',launch_commit=head,launch_dirty=False,seed=2022,batches=8,batch_size=1024,
                optimizer_steps=0,validation_evaluations=0,test_evaluations=0,test_split_loaded=False,
                started_at=datetime.now().astimezone().isoformat(),rows=[])
    def save():atomic_json(OUT/'report.json',report)
    save()
    try:
        import numpy as np
        import torch
        from promptmm_release_resource import sha256
        from shared_initialization import load_shared,ASSET
        from initialization_audit import describe_tensor
        from td_distill_model_no_projection import TDDistillNoProjectionModel
        from gradient_diagnostic import inspect_batch,COEFFICIENTS
        # Bypass Data.__init__: it loads Validation/Test. Reuse only train reader and sampler.
        sys.argv=['diagnose_sports_gradients.py']
        from utility.load_data import Data
        if not torch.cuda.is_available():raise RuntimeError('CUDA required.')
        train=ROOT/'data/sports/train_mat'
        if sha256(train)!=TRAIN_SHA:raise RuntimeError('Training data fingerprint mismatch.')
        data=Data.__new__(Data);matrix=data._load_sparse_matrix(train)
        if matrix.shape!=(35598,18357):raise RuntimeError('Training shape mismatch.')
        data.n_users,data.n_items=matrix.shape;data.batch_size=1024
        data.exist_users=np.unique(matrix.nonzero()[0]).astype(int).tolist()
        data.train_items=data._build_interaction_dict(matrix)
        tensors,identity=load_shared(ROOT,'cuda:0',35598,18357)
        model=TDDistillNoProjectionModel(35598,18357,64,64,user_head_names=()).to('cuda:0')
        model.init_user_item_embed(tensors['users'],tensors['items']);model.train()
        def state():return {k:describe_tensor(v) for k,v in model.state_dict().items()}
        before=state()
        assert before['user_id_embedding.weight']==identity['tensors']['users']
        assert before['item_id_embedding.weight']==identity['tensors']['items']
        files=['tools/diagnose_sports_gradients.py','codes/gradient_diagnostic.py','codes/td_distill_model.py',
               'codes/td_distill_model_no_projection.py','codes/utility/load_data.py','codes/shared_initialization.py','codes/main_mmlight.py']
        report.update(shared_asset=identity,initial_state=before,train_sha256=TRAIN_SHA,coefficients=COEFFICIENTS,
                      source_fingerprints={f:sha256(ROOT/f) for f in files},
                      environment=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,gpu=torch.cuda.get_device_name(0)),
                      sampling='Data.sample; seed reset after construction; diagnostic batches, not historical training replay',
                      limitation='Initial-state raw gradients only; not actual AdamW steps or whole-trajectory causal evidence.')
        random.seed(2022);np.random.seed(2022);torch.manual_seed(2022)
        batches=[native_batch_ids(data.sample()) for _ in range(8)]
        atomic_json(OUT/'batches.json',batches);report['batches_sha256']=sha256(OUT/'batches.json');save()
        for index,batch in enumerate(batches):
            row=inspect_batch(model,tensors,tuple(torch.tensor(v,device='cuda:0',dtype=torch.long) for v in batch))
            row['batch']=index;report['rows'].append(row);save()
            print('Gradient batch',index+1,'/8; weighted distill/BPR norm:',row['gradients']['all']['weighted_distill']['relative_to_bpr'],flush=True)
        if state()!=before or any(p.grad is not None for p in model.parameters()):raise RuntimeError('Student state/grad buffers changed.')
        if source()!=head or any(sha256(ROOT/f)!=v for f,v in report['source_fingerprints'].items()):raise RuntimeError('Source changed.')
        if sha256(ROOT/ASSET)!=identity['sha256'] or sha256(train)!=TRAIN_SHA:raise RuntimeError('Input changed.')
        report.update(status='completed',parameters_unchanged=True,final_state=state(),gradient_linearity_checked=True)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        report['finished_at']=datetime.now().astimezone().isoformat();save();print('Report:',OUT/'report.json')

if __name__=='__main__':main()
