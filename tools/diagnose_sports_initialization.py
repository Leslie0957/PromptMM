"""Manual zero-update repeatability diagnostic through the real TD setup path."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codes'))
from promptmm_release_validation import atomic_json
from promptmm_release_resource import sha256
from initialization_audit import describe_tensor, EXPECTED_TD_TEACHER_INITIAL
from utility.dataset_profiles import SPORTS_BPR_TEACHER_INIT_PROFILE

RUN_ID='sports_initialization_repeatability_seed2022_v1'
OUT=ROOT/'exp/initialization_checks'/RUN_ID
NAMES=('users','items','image_items','text_items','image_users','text_users')


def compare(left,right):
    import torch
    result={}
    for name in NAMES:
        a,b=left[name],right[name]
        if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Tensor structure mismatch: '+name)
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite output.')
        delta=(a.double()-b.double()).abs()
        result[name]=dict(exact=torch.equal(a,b),max_abs=float(delta.max()),
                          mean_abs=float(delta.mean()),different_elements=int((a!=b).sum()))
    return result


def source_state():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Clean committed source required.')
    return head


def worker(index):
    import torch
    folder=OUT/f'process{index}';folder.mkdir(exist_ok=False)
    report=dict(status='started',process=index,launch_commit=source_state(),
                started_at=datetime.now().astimezone().isoformat(),optimizer_steps=0,
                validation_evaluations=0,test_evaluations=0,teacher_forwards=0,
                historical_tensor_values_available=False,test_structural_read=True)
    def save():atomic_json(folder/'report.json',report)
    save();trainer=None
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA required.')
        sys.argv=['codes/main_mmlight.py','--dataset','sports','--student_profile',SPORTS_BPR_TEACHER_INIT_PROFILE,'--gpu_id','0']
        import main_mmlight as entry
        entry.select_dataset();entry.set_seed(entry.args.seed)
        report['environment']=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),numpy=entry.np.__version__,dgl=entry.dgl.__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
        def forbidden(*args,**kwargs):raise RuntimeError('Diagnostic forbids optimizer steps or ranking.')
        entry.Trainer.test=forbidden
        entry.Trainer.train_td_distill=forbidden
        torch.optim.AdamW.step=forbidden
        trainer=entry.Trainer.__new__(entry.Trainer)
        trainer.__init__(data_config=dict(n_users=entry.data_generator.n_users,n_items=entry.data_generator.n_items))
        trainer._update_run_manifest(status='initialization_diagnostic_started',diagnostic_only=True)
        report['manifest']=trainer.run_manifest_path
        report['source']=trainer.run_manifest['code_fingerprints']
        report['diagnostic_source_sha256']=sha256(Path(__file__))
        save()
        def capture(t,initial):
            from promptmm_release_resource import tensor_digest
            before=(tensor_digest(t.teacher_model),tensor_digest(t.prompt_module))
            report['teacher_training']=t.teacher_model.training
            report['prompt_training']=t.prompt_module.training
            if t.teacher_model.training or t.prompt_module.training:raise RuntimeError('Expected eval mode.')
            baseline=None;report['passes']=[]
            for attempt in range(3):
                if attempt==0:values=initial
                else:
                    with torch.no_grad():values=t.teacher_model(t.ui_graph,t.iu_graph,t.prompt_module)[:6]
                tensors={k:v.detach().cpu().clone() for k,v in zip(NAMES,values)}
                if len(tensors)!=6:raise RuntimeError('Missing teacher outputs.')
                identities={k:describe_tensor(v) for k,v in tensors.items()}
                if baseline is None:
                    baseline=tensors
                    torch.save(tensors,folder/'initial_tensors.pt')
                    report['tensor_file_sha256']=sha256(folder/'initial_tensors.pt')
                row=dict(index=attempt,identities=identities,difference_from_first=compare(baseline,tensors),
                         historical_hash_matches={k:identities[k]==v for k,v in EXPECTED_TD_TEACHER_INITIAL.items()})
                report['passes'].append(row);report['teacher_forwards']+=1;save()
                print('Process',index,'forward',attempt+1,'historical matches:',row['historical_hash_matches'],flush=True)
            if before!=(tensor_digest(t.teacher_model),tensor_digest(t.prompt_module)):raise RuntimeError('Teacher/prompt changed.')
            report['teacher_prompt_unchanged']=True
            report['dataset_identity']=t.run_manifest['dataset_identity']
            report['teacher_identity']=t.run_manifest['teacher_checkpoint_fingerprint']
            t._update_run_manifest(status='initialization_diagnostic_completed',diagnostic_report=str(folder/'report.json'),
                                   optimizer_steps=0,validation_evaluations=0,test_evaluations=0)
        trainer.train(initialization_diagnostic=capture)
        if report['teacher_forwards']!=3:raise RuntimeError('Diagnostic callback not completed.')
        if source_state()!=report['launch_commit']:raise RuntimeError('Source changed.')
        for fp in report['source'].values():
            if sha256(fp['path'])!=fp['sha256']:raise RuntimeError('Source hash changed.')
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        if trainer is not None and hasattr(trainer,'run_manifest'):
            trainer._update_run_manifest(status='initialization_diagnostic_failed',failure_message=str(exc))
        raise
    finally:
        report['finished_at']=datetime.now().astimezone().isoformat();save()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',type=int,choices=[0,1]);p.add_argument('--describe',action='store_true');args=p.parse_args()
    if args.describe:
        print(json.dumps(dict(run_id=RUN_ID,processes=2,forwards_per_process=3,optimizer_steps=0,ranking_evaluations=0,output=str(OUT)),indent=2));return
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    if args.worker is not None:return worker(args.worker)
    head=source_state();OUT.mkdir(parents=True,exist_ok=False)
    report=dict(status='started',launch_commit=head,started_at=datetime.now().astimezone().isoformat(),
                optimizer_steps=0,validation_evaluations=0,test_evaluations=0,processes=[])
    def save():atomic_json(OUT/'report.json',report)
    save()
    try:
        for index in (0,1):
            if source_state()!=head:raise RuntimeError('Source changed between processes.')
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(index)]
            subprocess.run(command,cwd=ROOT,check=True)
            r=json.loads((OUT/f'process{index}/report.json').read_text())
            if r['status']!='completed' or r['teacher_forwards']!=3:raise RuntimeError('Incomplete worker.')
            report['processes'].append(r);save()
        import torch
        first=torch.load(OUT/'process0/initial_tensors.pt',map_location='cpu',weights_only=False)
        second=torch.load(OUT/'process1/initial_tensors.pt',map_location='cpu',weights_only=False)
        report['cross_process_difference']=compare(first,second)
        if source_state()!=head:raise RuntimeError('Source changed.')
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        report['finished_at']=datetime.now().astimezone().isoformat();save();print('Diagnostic report:',OUT/'report.json',flush=True)


if __name__=='__main__':main()
