"""Serial M1 same-checkpoint deployment measurement; manual once-only entry."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'tools'))
import cached_deployment_benchmark as bench
import innovation1_cost_m0 as m0

OUT = ROOT / 'exp/efficiency/innovation1_cost_v1/m1/serial_v2'
FAILED_V1 = ROOT / 'exp/efficiency/innovation1_cost_v1/m1/serial_v1/report.json'
PYTHON = Path('D:/miniconda/envs/run_5060/python.exe')
M0 = ROOT / 'exp/efficiency/innovation1_cost_v1/m0/smoke_v1'
LIMITS = dict(wall_seconds=3600, cuda_allocated_bytes=4*1024**3,
              rss_bytes=8*1024**3, output_bytes=2*1024**3,
              free_disk_bytes=10*1024**3)
SOURCE = ('tools/run_innovation1_cost_m1.py', 'tools/innovation1_cost_m0.py',
          'codes/cached_deployment_benchmark.py', 'codes/promptmm_release.py',
          'codes/Models_mmlight.py', 'codes/utility/norm.py',
          'codes/utility/hard_token_cache.py', 'codes/utility/experiment_protocol.py')


def now():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def schedule():
    slots = [f'{prefix}_{seed}' for prefix in ('s1','s2','s3') for seed in (2022,2023,2024)]
    identities = slots + ['teacher']
    return [dict(round=round_id, identity=identity) for round_id in range(3)
            for identity in identities[round_id*3:] + identities[:round_id*3]]


def compare_selected_tables(actual, reference_cpu):
    if any(t.device.type!='cpu' for t in reference_cpu):
        raise RuntimeError('Selected table parity reference must stay on CPU')
    return bench.compare_reference(actual,reference_cpu)


def self_test():
    import torch
    conditions=schedule()
    if len(conditions)!=30 or len({(c['round'],c['identity']) for c in conditions})!=30:
        raise AssertionError('M1 schedule incomplete')
    for identity in {c['identity'] for c in conditions}:
        if sum(c['identity']==identity for c in conditions)!=3:
            raise AssertionError('Identity not measured thrice')
    users=torch.arange(8*4,dtype=torch.float32).reshape(8,4)/100
    items=torch.arange(25*4,dtype=torch.float32).reshape(25,4)/100
    tables=(users,items)
    request=(torch.tensor([0,2],dtype=torch.int64),
             torch.tensor([0,1],dtype=torch.int64),
             torch.tensor([24,0],dtype=torch.int64))
    values,indices=bench.score(tables,request,k=20)
    if values.shape!=(2,20) or 24 in indices[0].tolist() or 0 in indices[1].tolist():
        raise AssertionError('Train-only masking/top-k synthetic check failed')
    copied=tuple(t.clone() for t in tables)
    if not all(x['allclose'] for x in compare_selected_tables(copied,tables)):
        raise AssertionError('Table parity synthetic check failed')
    try:
        compare_selected_tables(copied,(torch.empty(1,device='meta'),))
    except RuntimeError:
        pass
    else:
        raise AssertionError('Non-CPU reference was accepted')
    if not m0.unexpected_source_status(['?? check/','?? unexpected.py']):
        raise AssertionError('Source guard synthetic check failed')
    print('M1 synthetic schedule, mask/top-k, table parity and source guard passed; no real assets loaded')


def guard(start, out, torch=None):
    import psutil
    if time.monotonic()-start >= LIMITS['wall_seconds']:
        raise RuntimeError('M1 wall limit')
    free = shutil.disk_usage(out).free
    if free < LIMITS['free_disk_bytes']:
        raise RuntimeError('M1 free disk limit')
    output = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if output > LIMITS['output_bytes']:
        raise RuntimeError('M1 output limit')
    proc = psutil.Process()
    rss = proc.memory_info().rss + sum(p.memory_info().rss for p in proc.children(recursive=True) if p.is_running())
    if rss > LIMITS['rss_bytes']:
        raise RuntimeError('M1 RSS limit')
    allocated = torch.cuda.memory_allocated() if torch is not None else None
    if allocated is not None and allocated > LIMITS['cuda_allocated_bytes']:
        raise RuntimeError('M1 CUDA allocated limit')
    return dict(free_disk_bytes=free, output_bytes=output, process_tree_rss_bytes=rss,
                cuda_allocated_bytes=allocated)


def environment(torch):
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                total_cuda_bytes=torch.cuda.get_device_properties(0).total_memory,
                threads=torch.get_num_threads(), tf32=torch.backends.cuda.matmul.allow_tf32,
                nvidia_smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.sm,utilization.gpu,memory.used',
                                           '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                          timeout=5).stdout.strip())


def telemetry(stop):
    path=OUT/'telemetry.jsonl'
    with path.open('x',encoding='utf-8') as stream:
        while not stop.is_set():
            try:
                sample=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,clocks.sm,utilization.gpu,memory.used',
                                       '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                value=dict(at=now(),exit_code=sample.returncode,values=sample.stdout.strip(),error=sample.stderr.strip())
            except (OSError,subprocess.TimeoutExpired) as exc:
                value=dict(at=now(),unavailable=repr(exc))
            stream.write(json.dumps(value)+'\n')
            stream.flush()
            stop.wait(2)


def load_original(row, train, torch):
    """Return original-model output generator, retained source storage and setup notes."""
    from promptmm_release import ReleaseStudent, release_graphs, sparse_tensor
    if row is None:
        teacher, prompt, ui, iu, phases = bench.load_teacher('cuda:0')
        def generate():
            return tuple(teacher(ui, iu, prompt)[:2])
        return generate, bench.module_tensors(teacher,'teacher') + bench.module_tensors(prompt,'prompt') + [('ui',ui),('iu',iu)], phases
    checkpoint = ROOT / row['checkpoint_path']
    bench.check_hash(checkpoint, row['checkpoint_sha256'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if row['method'] == 'promptmm_release':
        _, _, graph = release_graphs(train)
        adj = sparse_tensor(graph, 'cuda:0').coalesce()
        state = saved['model_state_dict']
        for side in ('user','item'):
            if not torch.equal(state[f'{side}_id_embedding.weight'], state[f'{side}_id_embedding_pre.weight']):
                raise RuntimeError('Release alias mismatch')
        model = ReleaseStudent(*train.shape, 64, 1).to('cuda:0')
        model.init_user_item_embed(state['user_id_embedding.weight'].to('cuda:0').clone(),
                                   state['item_id_embedding.weight'].to('cuda:0').clone())
        model.load_state_dict(state, strict=True)
        model.assert_aliases()
        model.eval()
        def generate():
            return tuple(model(adj))
        return generate, bench.module_tensors(model,'release') + [('adj',adj)], dict(graph='native release_graphs')
    infer = ROOT / row['td_infer_path']
    bench.check_hash(infer, row['td_infer_sha256'])
    thin = torch.load(infer, map_location='cpu', weights_only=False)
    for side in ('user','item'):
        key=f'{side}_id_embedding.weight'
        if not torch.equal(saved['model_state_dict'][key], thin[key]):
            raise RuntimeError('TD selected/inference mismatch')
    base = (thin['user_id_embedding.weight'].to('cuda:0'), thin['item_id_embedding.weight'].to('cuda:0'))
    def generate():
        return tuple(t.clone() for t in base)
    return generate, [('users',base[0]),('items',base[1])], dict(graph='native TD ID table export')


def worker(index, head):
    import numpy as np
    import torch
    m0.require_clean_source()
    if subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()!=head:
        raise RuntimeError('Worker launch source drift')
    m0.install_split_guard()
    batch=m0.read_json(OUT/'batch.json')
    condition=schedule()[index]
    if batch['source_commit']!=head or batch['active_index']!=index:
        raise RuntimeError('Undeclared worker')
    for path, expected in batch['source_sha256'].items():
        bench.check_hash(ROOT/path, expected)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, LIMITS['cuda_allocated_bytes']/torch.cuda.get_device_properties(0).total_memory),0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    started=time.monotonic()
    folder=OUT/f"round{condition['round']}_{condition['identity']}"
    folder.mkdir(exist_ok=False)
    path=folder/'report.json'
    report=dict(status='started', condition=condition, source_commit=head,
                validation_accesses=0,test_accesses=0,optimizer_steps=0,
                environment=environment(torch), started_at=now())
    write(path,report)
    try:
        bindings=m0.read_json(m0.MAP)
        if bench.sha256(m0.MAP)!=batch['bindings_sha256']:
            raise RuntimeError('Binding map drift')
        identity=condition['identity']
        row=next((x for x in bindings['rows'] if x['slot']==identity),None)
        if identity!='teacher' and row is None:
            raise RuntimeError('Unknown identity')
        old=m0.read_json(M0/'worker.json')
        export=old['teacher'] if row is None else next(x for x in old['rows'] if x['slot']==identity)
        bench.check_hash(export['export_path'],export['export_sha256'])
        saved=torch.load(export['export_path'],map_location='cpu',weights_only=True)
        reference_cpu=(saved['users'], saved['items'])
        reference=(reference_cpu[0].to('cuda:0'), reference_cpu[1].to('cuda:0'))
        train_path=ROOT/'data/sports/train_mat'
        bench.check_hash(train_path,bindings['train_sha256'])
        start=time.perf_counter()
        with train_path.open('rb') as stream:
            train=pickle.load(stream).tocsr()
        report['load_train_seconds']=time.perf_counter()-start
        order=bench.request_order(train)
        report['request_order_sha256']=bench.tensor_digest((torch.from_numpy(order),))
        start=time.perf_counter()
        with torch.inference_mode():
            generate, source_tensors, semantics=load_original(row,train,torch)
        torch.cuda.synchronize()
        report['original_setup_seconds']=time.perf_counter()-start
        report['original_semantics']=semantics
        report['resident_storage']=bench.storage_inventory(source_tensors)
        with torch.inference_mode():
            tick=time.perf_counter()
            actual=generate()
            torch.cuda.synchronize()
            report['generation_once_seconds']=time.perf_counter()-tick
            bench.tables_ok(actual)
            parity=compare_selected_tables(actual,reference_cpu)
            if not all(x['allclose'] for x in parity):
                raise RuntimeError('Original/M0 selected table mismatch')
            report['original_m0_parity']=parity
            for _ in range(3):
                generate()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            samples=[]
            for _ in range(10):
                torch.cuda.synchronize()
                tick=time.perf_counter()
                output=generate()
                torch.cuda.synchronize()
                samples.append((time.perf_counter()-tick)*1000)
                bench.tables_ok(output)
                guard(started,OUT,torch)
            report['generation_wall_ms']=samples
            report['generation_summary']=bench.summary(samples)
            report['generation_memory']=bench.gpu_memory()
            tick=time.perf_counter()
            output=generate()
            torch.cuda.synchronize()
            report['export_generation_seconds']=time.perf_counter()-tick
            tick=time.perf_counter()
            cpu=tuple(t.cpu() for t in output)
            torch.cuda.synchronize()
            report['device_to_host_seconds']=time.perf_counter()-tick
            target=folder/'tables.pt'
            tick=time.perf_counter()
            with target.open('xb') as stream:
                torch.save(dict(users=cpu[0],items=cpu[1],embedding_dim=64),stream)
            report['write_seconds']=time.perf_counter()-tick
            reloaded=torch.load(target,map_location='cpu',weights_only=True)
            if not all(torch.allclose(a,b,atol=1e-4,rtol=1e-4) for a,b in
                       zip(cpu,(reloaded['users'],reloaded['items']))):
                raise RuntimeError('New export roundtrip mismatch')
            report['export']=dict(path=str(target),sha256=bench.sha256(target),bytes=target.stat().st_size,
                                  logical_bytes=sum(t.numel()*t.element_size() for t in cpu))
            report['offline_total_seconds']=(report['load_train_seconds']+report['original_setup_seconds']+
                                              report['export_generation_seconds']+
                                              report['device_to_host_seconds']+report['write_seconds'])
            report['offline_total_excludes_m0_reference_load_and_repeated_generation']=True
            for size in bench.BATCHES:
                section={}
                bench.measure_online(reference,train,order,size,section,folder/f'online_b{size}.json')
                report[f'online_b{size}']=section
                guard(started,OUT,torch)
        report['resource_end']=guard(started,OUT,torch)
        report['status']='completed'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report['finished_at']=now()
        write(path,report)


def parent():
    import psutil
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError('Declared Python required')
    m0.require_clean_source()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    failed=m0.read_json(FAILED_V1)
    if (failed['status']!='failed' or failed['source_commit']!='978254941ae06255224a69a1c0328a77d99ecf26'
            or failed['completed']):
        raise RuntimeError('Consumed v1 failure anchor differs')
    m0_report=m0.read_json(M0/'report.json')
    if m0_report['status']!='completed' or m0_report['source_commit']!='aa7eddd9cc707e6cac8781316c818fb566a6e2db':
        raise RuntimeError('M0 audit anchor incomplete')
    if bench.sha256(m0.MAP)!=m0_report['binding_sha256']:
        raise RuntimeError('M0 binding anchor drift')
    if bench.sha256(M0/'worker.json')!=m0_report['worker_report_sha256']:
        raise RuntimeError('M0 worker anchor drift')
    OUT.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report=dict(status='started',source_commit=head,started_at=now(),schedule=schedule(),
                source_sha256={p:bench.sha256(ROOT/p) for p in SOURCE},
                bindings_sha256=bench.sha256(m0.MAP),m0_worker_sha256=bench.sha256(M0/'worker.json'),
                limits=LIMITS,active_index=None,completed=[])
    write(OUT/'batch.json',report)
    process=None
    request_fingerprints={}
    stop=threading.Event()
    monitor=threading.Thread(target=telemetry,args=(stop,),daemon=True)
    monitor.start()
    try:
        for index,condition in enumerate(schedule()):
            guard(start,OUT)
            m0.require_clean_source()
            report['active_index']=index
            command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(index),'--head',head]
            report['active_command']=command
            write(OUT/'batch.json',report)
            label=f"round{condition['round']}_{condition['identity']}"
            print(f'[{index+1}/30] {label}',flush=True)
            with (OUT/f'{label}.stdout.txt').open('x',encoding='utf-8') as stdout, (OUT/f'{label}.stderr.txt').open('x',encoding='utf-8') as stderr:
                process=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                while process.poll() is None:
                    guard(start,OUT)
                    time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f'M1 worker {label} exited {process.returncode}')
            result=OUT/label/'report.json'
            value=m0.read_json(result)
            if value['status']!='completed' or value['condition']!=condition:
                raise RuntimeError('M1 worker acceptance failed')
            if any(value[k] != 0 for k in ('validation_accesses','test_accesses','optimizer_steps')):
                raise RuntimeError('Prohibited access/update reported')
            if not all(x['allclose'] for x in value['original_m0_parity']):
                raise RuntimeError('Original/M0 parity failed')
            if len(value['generation_wall_ms'])!=10:
                raise RuntimeError('Incomplete generation samples')
            for size in bench.BATCHES:
                section=value[f'online_b{size}']
                if len(section['raw_wall_ms'])!=100 or len(section['gemm_only_ms'])!=100:
                    raise RuntimeError('Incomplete online samples')
                fingerprint=section['request_tensor_sha256']
                if size in request_fingerprints and request_fingerprints[size]!=fingerprint:
                    raise RuntimeError('Train-only request identity differs across conditions')
                request_fingerprints[size]=fingerprint
            export=value['export']
            bench.check_hash(export['path'],export['sha256'])
            report['completed'].append(dict(condition=condition,report=str(result),sha256=bench.sha256(result)))
            write(OUT/'batch.json',report)
        report.update(status='completed',active_index=None,wall_seconds=time.monotonic()-start)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            tree=psutil.Process(process.pid)
            for child in tree.children(recursive=True):
                child.kill()
            tree.kill()
            process.wait(timeout=10)
        report.update(status='failed',error=repr(exc),traceback=traceback.format_exc(),
                      wall_seconds=time.monotonic()-start)
        raise
    finally:
        stop.set()
        monitor.join(timeout=8)
        report['telemetry']=dict(path=str(OUT/'telemetry.jsonl'),
                                 sha256=bench.sha256(OUT/'telemetry.jsonl') if (OUT/'telemetry.jsonl').exists() and not monitor.is_alive() else None)
        report['finished_at']=now()
        write(OUT/'report.json',report)
        print('Preserved M1 report:',OUT/'report.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--worker',type=int,choices=range(30),help=argparse.SUPPRESS)
    parser.add_argument('--head',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    elif args.dry_run:
        assert len(schedule())==30 and len(set((c['round'],c['identity']) for c in schedule()))==30
        for index,condition in enumerate(schedule()):
            print(index,condition)
    elif args.worker is not None:
        worker(args.worker,args.head)
    else:
        parent()


if __name__=='__main__':
    main()
