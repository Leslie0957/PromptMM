"""One manual, serial M2 short-window update-cost cohort. No evaluation route."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
sys.path.insert(0, str(ROOT / 'tools'))
import cached_deployment_benchmark as bench
import innovation1_cost_m0 as m0

OUT = ROOT / 'exp/efficiency/innovation1_cost_v1/m2/serial_v1'
M0 = ROOT / 'exp/efficiency/innovation1_cost_v1/m0/smoke_v1'
M1 = ROOT / 'exp/efficiency/innovation1_cost_v1/m1/serial_v2'
M1_REPORT_SHA = 'c17b0a3b9b96ac0f35526dd886e89a02b12444e3fdf6fcfc02f84a282d7d3539'
PYTHON = Path('D:/miniconda/envs/run_5060/python.exe')
WARMUP, MEASURE, BATCH_SIZE = 20, 100, 1024
LIMITS = dict(wall_seconds=3600, cuda_allocated_bytes=4*1024**3,
              rss_bytes=8*1024**3, output_bytes=2*1024**3,
              free_disk_bytes=10*1024**3)
SOURCES = ('tools/run_innovation1_cost_m2.py', 'tools/innovation1_cost_m0.py',
           'codes/cached_deployment_benchmark.py', 'codes/promptmm_release.py',
           'codes/promptmm_release_resource.py',
           'codes/td_distill_model_no_projection.py', 'codes/initialization_audit.py',
           'codes/Models_mmlight.py', 'codes/utility/hard_token_cache.py',
           'codes/utility/dataset_profiles.py', 'codes/utility/norm.py')


def now():
    return datetime.now().astimezone().isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def schedule():
    methods = ('s1', 's2', 's3')
    seeds = (2022, 2023, 2024)
    return [dict(round=r, slot=f'{methods[(j+r)%3]}_{seeds[(i+r)%3]}')
            for r in range(3) for i in range(3) for j in range(3)]


def tape_seed(round_id, seed):
    return 9026200 + round_id*100 + seed - 2022


def self_test():
    import numpy as np
    import scipy.sparse as sp
    import torch
    from promptmm_release import (ResourceConfig, ReleaseStudent, release_graphs,
                                  release_objective, sample_triplets, sparse_tensor)
    from td_distill_model_no_projection import (TDDistillNoProjectionModel, bpr_loss,
                                                 directional_distillation_loss)
    conditions = schedule()
    assert len(conditions) == 27
    assert len({(c['round'], c['slot']) for c in conditions}) == 27
    assert all(sum(c['slot'] == slot for c in conditions) == 3
               for slot in {c['slot'] for c in conditions})
    for r in range(3):
        block = [c['slot'].split('_')[0] for c in conditions if c['round'] == r]
        assert all(block[j] != block[j+1] for j in range(8))
    # Same seeded main stream for each arm; release extras have a separate RNG phase.
    train = sp.csr_matrix(np.tile(np.eye(5, dtype=np.float32), (4, 1)))
    def sample():
        random.seed(7); np.random.seed(7)
        return sample_triplets(train, list(range(20)), 4)
    assert all(np.array_equal(a, b) for a, b in zip(sample(), sample()))
    td = TDDistillNoProjectionModel(20, 5, 4, 4, item_head_names=('image', 'text'),
                                    user_head_names=())
    users = torch.tensor([0, 1]); pos = torch.tensor([0, 1]); neg = torch.tensor([2, 3])
    initial = td.user_id_embedding.weight.detach().clone()
    user_vec = td.user_id_embedding(users); pos_vec = td.item_id_embedding(pos)
    td_loss = bpr_loss(user_vec, pos_vec, td.item_id_embedding(neg))
    td_loss = td_loss + .3 * (directional_distillation_loss(pos_vec, torch.ones_like(pos_vec)) +
                             .3 * directional_distillation_loss(pos_vec, torch.zeros_like(pos_vec))) / 1.3
    td_opt = torch.optim.AdamW(td.parameters(), lr=6e-5, weight_decay=.01)
    td_loss.backward(); td_opt.step()
    assert torch.isfinite(td_loss) and not torch.equal(initial, td.user_id_embedding.weight)
    release = ReleaseStudent(20, 5, 4, 1)
    release.init_user_item_embed(torch.randn(20, 4), torch.randn(5, 4))
    release.assert_aliases()
    adj = sparse_tensor(release_graphs(train)[2], 'cpu')
    output = release(adj)
    teacher = tuple(torch.randn_like(output[0] if side == 'u' else output[1])
                    for side in ('u', 'i', 'i', 'i', 'u', 'u'))
    candidates = torch.tensor([[0, 2, 3], [1, 3, 4]])
    cfg = ResourceConfig(batch_size=2, embedding_dim=4, negative_count=2)
    rel_loss = release_objective(output, teacher, users, pos, neg, candidates, cfg)['total']
    assert torch.isfinite(rel_loss)
    rel_loss.backward()
    assert any(p.grad is not None for p in release.parameters())
    assert WARMUP + MEASURE == 120 and BATCH_SIZE == 1024
    assert m0.unexpected_source_status(['?? check/', '?? rogue.py'])
    print('M2 synthetic schedule/tape/native objective/source gates passed; no real assets opened')


def guard(start, torch=None):
    import psutil
    if time.monotonic() - start >= LIMITS['wall_seconds']:
        raise RuntimeError('M2 wall limit')
    free = shutil.disk_usage(OUT.parent).free
    if free < LIMITS['free_disk_bytes']:
        raise RuntimeError('M2 free-disk limit')
    output = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if output > LIMITS['output_bytes']:
        raise RuntimeError('M2 output limit')
    proc = psutil.Process()
    rss = proc.memory_info().rss + sum(p.memory_info().rss for p in proc.children(recursive=True)
                                       if p.is_running())
    if rss > LIMITS['rss_bytes']:
        raise RuntimeError('M2 RSS limit')
    allocated = torch.cuda.memory_allocated() if torch is not None else None
    if allocated is not None and allocated > LIMITS['cuda_allocated_bytes']:
        raise RuntimeError('M2 CUDA allocated limit')
    return dict(free_disk_bytes=free, output_bytes=output,
                process_tree_rss_bytes=rss, cuda_allocated_bytes=allocated)


def environment(torch):
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                total_cuda_bytes=torch.cuda.get_device_properties(0).total_memory,
                threads=torch.get_num_threads(), tf32=torch.backends.cuda.matmul.allow_tf32,
                nvidia_smi=subprocess.run(
                    ['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,clocks.sm,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5).stdout.strip())


def telemetry(stop):
    with (OUT / 'telemetry.jsonl').open('x', encoding='utf-8') as stream:
        while not stop.is_set():
            try:
                proc = subprocess.run(
                    ['nvidia-smi', '--query-gpu=temperature.gpu,clocks.sm,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
                value = dict(at=now(), exit_code=proc.returncode,
                             values=proc.stdout.strip(), error=proc.stderr.strip())
            except (OSError, subprocess.TimeoutExpired) as exc:
                value = dict(at=now(), unavailable=repr(exc))
            stream.write(json.dumps(value) + '\n')
            stream.flush()
            stop.wait(2)


def make_tape(train, seed, round_id):
    import numpy as np
    from promptmm_release import sample_triplets
    random.seed(tape_seed(round_id, seed))
    np.random.seed(tape_seed(round_id, seed))
    existing = np.flatnonzero(np.diff(train.indptr)).astype(int).tolist()
    rows = [sample_triplets(train, existing, BATCH_SIZE) for _ in range(WARMUP + MEASURE)]
    tape = np.asarray(rows, dtype=np.int64)
    if tape.shape != (120, 3, BATCH_SIZE):
        raise RuntimeError('Main triplet tape shape')
    digest = hashlib.sha256(tape.tobytes()).hexdigest()
    return tape, digest


def tensor_digest(model):
    return bench.tensor_digest(tuple(p.detach() for p in model.parameters()))


def update_summary(samples):
    result = bench.summary(samples, BATCH_SIZE)
    result['triplets_per_second'] = result.pop('users_per_second')
    return result


def finite_state(model, optimizer, torch):
    for p in model.parameters():
        if not torch.isfinite(p).all():
            raise FloatingPointError('Nonfinite student state')
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError('Nonfinite AdamW state')


def setup_td(row, teacher_out, shape, torch):
    from td_distill_model_no_projection import TDDistillNoProjectionModel
    from initialization_audit import describe_tensor
    nu, ni = shape
    cfg = row['config']
    targets = dict(item_image=teacher_out[2].detach(), item_text=teacher_out[3].detach(),
                   user_image=teacher_out[4].detach(), user_text=teacher_out[5].detach())
    heads_i = tuple(k for k, c in (('image', 'td_item_image_rate'), ('text', 'td_item_text_rate'))
                    if cfg[c] > 0)
    heads_u = tuple(k for k, c in (('image', 'td_user_image_rate'), ('text', 'td_user_text_rate'))
                    if cfg[c] > 0)
    model = TDDistillNoProjectionModel(
        nu, ni, cfg['embed_size'], targets['item_image'].shape[-1],
        targets['user_image'].shape[-1], item_head_names=heads_i,
        user_head_names=heads_u).to('cuda:0')
    # Paired cold ID state from the immutable formal Full seed asset, reused for BPR.
    peer = next(x for x in m0.read_json(m0.MAP)['rows']
                if x['slot'] == f"s2_{row['seed']}")
    initial = peer['initialization']['record']
    bench.check_hash(initial['path'], initial['sha256'])
    state = torch.load(initial['path'], map_location='cpu', weights_only=True)
    model.init_user_item_embed(state['user'], state['item'])
    identity = dict(rule='paired formal cold ID table for BPR/Full cost window',
                    path=initial['path'], sha256=initial['sha256'],
                    users=describe_tensor(model.user_id_embedding.weight),
                    items=describe_tensor(model.item_id_embedding.weight))
    optimizer = torch.optim.AdamW([{'params': model.parameters()}],
                                  lr=cfg['student_lr'], weight_decay=cfg['student_weight_decay'])
    return model, optimizer, targets, identity


def setup_release(row, teacher_out, train, prompt, torch):
    from promptmm_release import ReleaseStudent, release_graphs, sparse_tensor
    from initialization_audit import initialize_release
    cfg = row['config']
    model = ReleaseStudent(train.shape[0], train.shape[1],
                           cfg['embedding_dim'], cfg['layers']).to('cuda:0')
    identity = initialize_release(model, teacher_out[0], teacher_out[1], 'teacher')
    _, _, adj = release_graphs(train)
    adj = sparse_tensor(adj, 'cuda:0')
    optimizer = torch.optim.AdamW([{'params': model.parameters()},
                                   {'params': prompt.parameters()}],
                                  lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'],
                                  foreach=False)
    return model, optimizer, adj, identity


def worker(index, head):
    import numpy as np
    import torch
    m0.require_clean_source()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head:
        raise RuntimeError('M2 launch source drift')
    m0.install_split_guard()
    batch = m0.read_json(OUT / 'batch.json')
    condition = schedule()[index]
    if batch['source_commit'] != head or batch['active_index'] != index:
        raise RuntimeError('Undeclared M2 worker')
    for name, expected in batch['source_sha256'].items():
        bench.check_hash(ROOT / name, expected)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, LIMITS['cuda_allocated_bytes'] /
            torch.cuda.get_device_properties(0).total_memory), 0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    folder = OUT / f"round{condition['round']}_{condition['slot']}"
    folder.mkdir(exist_ok=False)
    path = folder / 'report.json'
    started = time.monotonic()
    report = dict(status='started', condition=condition, source_commit=head,
                  started_at=now(), environment=environment(torch),
                  validation_accesses=0, test_accesses=0, optimizer_steps=0,
                  warmup_steps=0, measured_steps=0)
    save(path, report)
    try:
        bindings = m0.read_json(m0.MAP)
        if bench.sha256(m0.MAP) != batch['binding_sha256']:
            raise RuntimeError('M0 binding map drift')
        row = next(x for x in bindings['rows'] if x['slot'] == condition['slot'])
        if row['teacher_sha256'] != bindings['teacher_sha256']:
            raise RuntimeError('Teacher binding drift')
        train_path = ROOT / 'data/sports/train_mat'
        bench.check_hash(train_path, bindings['train_sha256'])
        with train_path.open('rb') as stream:
            train = pickle.load(stream).tocsr()
        if train.shape != (35598, 18357):
            raise RuntimeError('Train shape drift')
        tape_started = time.perf_counter()
        tape, tape_sha = make_tape(train, row['seed'], condition['round'])
        report['tape_generation_seconds'] = time.perf_counter() - tape_started
        report['triplet_tape'] = dict(seed=tape_seed(condition['round'], row['seed']),
                                      sha256=tape_sha, shape=list(tape.shape))
        report['main_triplets'] = BATCH_SIZE * len(tape)
        teacher, prompt, ui, iu, teacher_phases = bench.load_teacher('cuda:0')
        report['teacher_setup_phases'] = teacher_phases
        teacher.requires_grad_(False)
        prompt.requires_grad_(False)
        with torch.no_grad():
            teacher_out = teacher(ui, iu, prompt)
        method = row['method']
        if method in ('bpr', 'full'):
            model, optimizer, targets, initial = setup_td(row, teacher_out, train.shape, torch)
            adj = None
        elif method == 'promptmm_release':
            model, optimizer, adj, initial = setup_release(row, teacher_out, train, prompt, torch)
            targets = None
        else:
            raise RuntimeError('Unexpected method')
        report['method'] = method
        report['config'] = row['config']
        report['initialization'] = initial
        report['optimizer'] = dict(kind='AdamW', groups=[
            dict(lr=g['lr'], weight_decay=g['weight_decay'],
                 parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups])
        report['student_parameter_count'] = sum(p.numel() for p in model.parameters())
        report['initial_student_digest'] = tensor_digest(model)
        report['teacher_digest_before'] = tensor_digest(teacher)
        report['prompt_digest_before'] = tensor_digest(prompt)
        report['setup_resource'] = guard(started, torch)
        report['setup_memory'] = bench.gpu_memory()
        save(path, report)
        from td_distill_model_no_projection import bpr_loss, directional_distillation_loss
        from promptmm_release import release_candidates, release_objective
        if method == 'promptmm_release':
            from promptmm_release_resource import load_dgl
            dgl = load_dgl()
            dgl.seed(tape_seed(condition['round'], row['seed']) + 9000)
        raw_ms, inclusive_ms, diagnostic = [], [], []
        loss_first = loss_last = None
        finite_objective_checks = 0
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for step in range(WARMUP + MEASURE):
            if step % 5 == 0:
                guard(started, torch)
            model.train()
            if method == 'promptmm_release':
                teacher.train()
            inclusive_start = time.perf_counter()
            users, pos, neg = (torch.as_tensor(tape[step, j], dtype=torch.long, device='cuda:0')
                               for j in range(3))
            torch.cuda.synchronize()
            core_start = time.perf_counter()
            segmented = step in (WARMUP-3, WARMUP-2, WARMUP-1)
            marks = [core_start] if segmented else None
            def mark():
                if segmented:
                    torch.cuda.synchronize()
                    marks.append(time.perf_counter())
            candidates = release_candidates(train, tape[step, 0].tolist(),
                                            tape[step, 1].tolist(),
                                            row['config']['negative_count'], 'cuda:0', dgl) \
                if method == 'promptmm_release' else None
            mark()  # release candidate construction; zero for TD
            if method == 'promptmm_release':
                student_out = model(adj)
                mark()
                with torch.no_grad():
                    current_teacher = teacher(ui, iu, prompt)
                mark()
                losses = release_objective(student_out, current_teacher,
                                           users, pos, neg, candidates,
                                           SimpleNamespace(**row['config']))
                objective = losses['total']
            else:
                u = model.user_id_embedding(users)
                p = model.item_id_embedding(pos)
                n = model.item_id_embedding(neg)
                base = bpr_loss(u, p, n)
                if method == 'full':
                    cfg = row['config']
                    components = []
                    rates = []
                    for name, tensor, selected in (
                        ('item_image', targets['item_image'], p),
                        ('item_text', targets['item_text'], p),
                        ('user_image', targets['user_image'], u),
                        ('user_text', targets['user_text'], u)):
                        rate = cfg['td_' + name + '_rate']
                        if rate > 0:
                            components.append(rate * directional_distillation_loss(
                                selected, tensor[pos] if name.startswith('item') else tensor[users]))
                            rates.append(rate)
                    objective = base + cfg['td_distill_alpha'] * sum(components) / sum(rates)
                else:
                    objective = base
                mark()  # TD student and loss, teacher semantics already cached
                mark()  # no per-step teacher forward
            mark()
            optimizer.zero_grad(set_to_none=(method == 'promptmm_release'))
            objective.backward()
            mark()
            optimizer.step()
            torch.cuda.synchronize()
            mark()
            core_ms = (time.perf_counter() - core_start) * 1000
            inclusive = (time.perf_counter() - inclusive_start) * 1000
            if segmented:
                diagnostic.append(dict(step=step+1, candidate_ms=(marks[1]-marks[0])*1000,
                                       student_ms=(marks[2]-marks[1])*1000,
                                       teacher_ms=(marks[3]-marks[2])*1000,
                                       loss_ms=(marks[4]-marks[3])*1000,
                                       backward_ms=(marks[5]-marks[4])*1000,
                                       optimizer_ms=(marks[6]-marks[5])*1000,
                                       synchronized_diagnostic=True))
            if not torch.isfinite(objective.detach()).all().item():
                raise FloatingPointError('Nonfinite objective')
            finite_objective_checks += 1
            if step in (0, WARMUP - 1, WARMUP + MEASURE - 1):
                value = float(objective.detach())
                if not math.isfinite(value):
                    raise FloatingPointError('Nonfinite objective')
                if loss_first is None:
                    loss_first = value
                loss_last = value
            report['optimizer_steps'] += 1
            if step < WARMUP:
                report['warmup_steps'] += 1
            else:
                report['measured_steps'] += 1
                raw_ms.append(core_ms)
                inclusive_ms.append(inclusive)
            del users, pos, neg, candidates, objective
        report['loss_first_last'] = [loss_first, loss_last]
        report['finite_objective_checks'] = finite_objective_checks
        report['raw_core_ms'] = raw_ms
        report['raw_including_batch_ms'] = inclusive_ms
        report['including_batch_boundary'] = 'precomputed tape row to CUDA plus method-specific candidates and optimizer; excludes initial CPU tape generation'
        report['core_boundary'] = 'post main-triplet CUDA copy through native candidate construction, student/teacher/loss/backward/step; CUDA synchronized per update'
        report['synchronized_warmup_diagnostics'] = diagnostic
        report['candidate_count_per_release_update'] = BATCH_SIZE * (1 + row['config']['negative_count']) \
            if method == 'promptmm_release' else 0
        report['core_summary'] = update_summary(raw_ms)
        report['including_batch_summary'] = update_summary(inclusive_ms)
        report['step_memory'] = bench.gpu_memory()
        report['final_student_digest'] = tensor_digest(model)
        if report['initial_student_digest'] == report['final_student_digest']:
            raise RuntimeError('Student did not update')
        if tensor_digest(teacher) != report['teacher_digest_before'] or tensor_digest(prompt) != report['prompt_digest_before']:
            raise RuntimeError('Frozen teacher/prompt mutated')
        if method == 'promptmm_release':
            model.assert_aliases()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('Missing/nonfinite final student gradient')
        finite_state(model, optimizer, torch)
        report['final_resource'] = guard(started, torch)
        report['status'] = 'completed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_at'] = now()
        save(path, report)


def parent():
    import psutil
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError('Declared Python required')
    m0.require_clean_source()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    m0_report = m0.read_json(M0 / 'report.json')
    m1_report = m0.read_json(M1 / 'report.json')
    if (m0_report['status'] != 'completed' or
            m0_report['source_commit'] != 'aa7eddd9cc707e6cac8781316c818fb566a6e2db' or
            m1_report['status'] != 'completed' or len(m1_report['completed']) != 30):
        raise RuntimeError('M0/M1 completion anchors absent')
    if m1_report['source_commit'] != '6806b1c8ff74198feeee355f8bcabfe4127555b6':
        raise RuntimeError('M1 source anchor drift')
    if bench.sha256(m0.MAP) != m1_report['bindings_sha256']:
        raise RuntimeError('M0/M1 binding anchor drift')
    if bench.sha256(M1 / 'report.json') != M1_REPORT_SHA:
        raise RuntimeError('Audited M1 report drift')
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='started', source_commit=head, started_at=now(),
                  schedule=schedule(), limits=LIMITS, active_index=None,
                  completed=[], binding_sha256=bench.sha256(m0.MAP),
                  m1_report_sha256=bench.sha256(M1 / 'report.json'),
                  source_sha256={p: bench.sha256(ROOT / p) for p in SOURCES})
    save(OUT / 'batch.json', report)
    stop = threading.Event()
    monitor = threading.Thread(target=telemetry, args=(stop,), daemon=True)
    monitor.start()
    process = None
    try:
        tapes = {}
        for index, condition in enumerate(schedule()):
            guard(started)
            m0.require_clean_source()
            report['active_index'] = index
            command = [sys.executable, '-B', str(Path(__file__).resolve()),
                       '--worker', str(index), '--head', head]
            report['active_command'] = command
            save(OUT / 'batch.json', report)
            label = f"round{condition['round']}_{condition['slot']}"
            print(f'[{index+1}/27] {label}', flush=True)
            with (OUT / f'{label}.stdout.txt').open('x', encoding='utf-8') as stdout, \
                 (OUT / f'{label}.stderr.txt').open('x', encoding='utf-8') as stderr:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                while process.poll() is None:
                    guard(started)
                    time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f'M2 worker {label} exited {process.returncode}')
            path = OUT / label / 'report.json'
            value = m0.read_json(path)
            if value['status'] != 'completed' or value['condition'] != condition:
                raise RuntimeError('M2 worker acceptance failed')
            if (value['test_accesses'], value['validation_accesses'], value['optimizer_steps'],
                    value['warmup_steps'], value['measured_steps']) != (0, 0, 120, 20, 100):
                raise RuntimeError('M2 access/step count failure')
            if len(value['raw_core_ms']) != 100 or len(value['raw_including_batch_ms']) != 100:
                raise RuntimeError('M2 raw sample count failure')
            key = (condition['round'], int(condition['slot'].split('_')[1]))
            previous = tapes.setdefault(key, value['triplet_tape']['sha256'])
            if previous != value['triplet_tape']['sha256']:
                raise RuntimeError('Paired main triplet stream differs')
            report['completed'].append(dict(condition=condition, report=str(path),
                                            sha256=bench.sha256(path)))
            save(OUT / 'batch.json', report)
        report.update(status='completed', active_index=None, wall_seconds=time.monotonic()-started)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            tree = psutil.Process(process.pid)
            for child in tree.children(recursive=True):
                child.kill()
            tree.kill()
            process.wait(timeout=10)
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                      wall_seconds=time.monotonic()-started)
        raise
    finally:
        stop.set()
        monitor.join(timeout=8)
        report['telemetry'] = dict(path=str(OUT / 'telemetry.jsonl'),
                                   sha256=bench.sha256(OUT / 'telemetry.jsonl')
                                   if (OUT / 'telemetry.jsonl').exists() and not monitor.is_alive() else None)
        report['finished_at'] = now()
        save(OUT / 'report.json', report)
        print('Preserved M2 report:', OUT / 'report.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(27), help=argparse.SUPPRESS)
    parser.add_argument('--head', help=argparse.SUPPRESS)
    cli = parser.parse_args()
    if cli.self_test:
        self_test()
    elif cli.dry_run:
        assert len(schedule()) == 27
        print(json.dumps(dict(output=str(OUT), conditions=schedule(), limits=LIMITS,
                              warmup=WARMUP, measured=MEASURE, batch_size=BATCH_SIZE), indent=2))
    elif cli.worker is not None:
        if not cli.head:
            raise RuntimeError('Worker requires parent HEAD')
        worker(cli.worker, cli.head)
    else:
        parent()


if __name__ == '__main__':
    main()
