"""M0 identity audit and bounded, train-only model/export parity smoke.

The default command is metadata-only. --smoke-real is one exclusive attempt;
it never opens Validation/Test or updates a model. M1/M2 have no route here.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / 'docs/research/INNOVATION1_COST_ASSET_INVENTORY_2026-09-28.json'
BATCH = ROOT / 'exp/formal_closeout_cohort/innovation1_eval_recovery_v1/batch.json'
MAP = ROOT / 'docs/research/INNOVATION1_COST_M0_BINDINGS.json'
OUT = ROOT / 'exp/efficiency/innovation1_cost_v1/m0/smoke_v1'
TEACHER_SHA = '57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea'
TRAIN_SHA = '5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8'
LIMITS = dict(wall_seconds=600, cuda_allocated_bytes=4 * 1024**3,
              process_tree_rss_bytes=8 * 1024**3, output_bytes=2 * 1024**3,
              free_disk_bytes=10 * 1024**3, updates_per_method=0,
              requests_per_method=3)
SLOT_METHOD = {'s1': 'bpr', 's2': 'full', 's3': 'promptmm_release'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def unexpected_source_status(lines):
    return [line for line in lines if line != '?? check/']


def require_clean_source():
    lines = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'],
                                    cwd=ROOT, text=True).splitlines()
    unexpected = unexpected_source_status(lines)
    if unexpected:
        raise RuntimeError('Clean committed source required; unexpected status: ' + repr(unexpected))


def assert_inside(path):
    p = Path(path).resolve()
    if not p.is_relative_to(ROOT.resolve()):
        raise ValueError('Asset outside repository: ' + str(p))
    return p


def rel(path):
    return str(assert_inside(path).relative_to(ROOT.resolve())).replace('\\', '/')


def build_bindings():
    inventory = read_json(INVENTORY)
    batch = read_json(BATCH)
    rows = {(r['method'], r['seed']): r for r in inventory['rows']}
    steps = {s['name']: s for s in batch['steps']}
    if len(rows) != 9 or len(steps) != len(batch['steps']):
        raise ValueError('Unexpected inventory or duplicate batch step')
    bindings = []
    teachers = set()
    for prefix, method in SLOT_METHOD.items():
        for seed in (2022, 2023, 2024):
            slot = f'{prefix}_{seed}'
            item = rows[(method, seed)]
            step = steps['final_test_' + slot]
            if step['status'] != 'completed' or step['phase'] != 'final_test_once':
                raise ValueError('Final quality step incomplete: ' + slot)
            quality_path = assert_inside(step['verified_report_path'])
            q = read_json(quality_path)
            source_path = assert_inside(q['source_path'])
            selected_path = assert_inside(q['selected_checkpoint_path'])
            anchors = {rel(ROOT / a['path']): a for a in item['assets']}
            for path, expected in ((source_path, q['source_sha256']),
                                   (selected_path, q['selected_checkpoint_sha256'])):
                a = anchors.get(rel(path))
                if a is None or a['anchor_sha256'] != expected or path.stat().st_size != a['anchor_bytes']:
                    raise ValueError('Asset anchor mismatch: ' + slot + ' ' + str(path))
            if (q['slot'] != slot or q['dataset'] != 'sports' or q['method'] != method
                    or q['seed'] != seed or q['status'] != 'completed'
                    or q['student_test_evaluations'] != 1 or q['teacher_test_evaluations'] != 0
                    or q['evaluated_checkpoint_sha256'] != q['selected_checkpoint_sha256']):
                raise ValueError('Quality identity/protocol mismatch: ' + slot)
            source = read_json(source_path)
            if method == 'promptmm_release':
                epoch, recall = source['best_epoch'], source['best_validation']['recall'][1]
                teacher = source['shared_teacher']
                init = {'rule': 'teacher table initialization', 'checkpoint_sha256': teacher['sha256']}
                config = source['config']
                source_commit = source['launch_commit']
                if source['best_checkpoint_sha256'] != q['selected_checkpoint_sha256']:
                    raise ValueError('Release checkpoint mismatch: ' + slot)
            else:
                epoch, recall = source['best_selection_epoch'], source['best_selection_recall']
                teacher = source['teacher_checkpoint_fingerprint']
                init = {'rule': 'paired cold initialization' if method == 'full' else 'native random initialization',
                        'record': source.get('paired_cold_initial')}
                config = item['parameters']
                source_commit = None  # Original TD manifest has hashes, but no Git commit field.
                if rel(source['td_full_checkpoint']) != rel(selected_path):
                    raise ValueError('TD selected path mismatch: ' + slot)
            if epoch != q['source_selected_epoch'] or abs(recall - q['source_selection_recall']) > 1e-12:
                raise ValueError('Selection mismatch: ' + slot)
            if teacher['sha256'] != TEACHER_SHA:
                raise ValueError('Shared teacher mismatch: ' + slot)
            teachers.add(teacher['sha256'])
            bindings.append(dict(slot=slot, dataset='sports', method=method, seed=seed,
                                 source_commit=source_commit,
                                 source_code_fingerprints=source.get('code_fingerprints', source.get('source')),
                                 config=config, teacher_path=rel(teacher['path']),
                                 teacher_sha256=teacher['sha256'], initialization=init,
                                 source_path=rel(source_path), source_sha256=q['source_sha256'],
                                 checkpoint_path=rel(selected_path), checkpoint_sha256=q['selected_checkpoint_sha256'],
                                 selection='Validation Recall@20; strict improvement, earliest tie',
                                 selected_epoch=epoch, selection_recall20=recall,
                                 quality_report_path=rel(quality_path), quality_report_sha256=digest(quality_path),
                                 final_test_metrics=q['test_result'],
                                 td_infer_path=rel(source['td_infer_checkpoint']) if method != 'promptmm_release' else None,
                                 td_infer_sha256=anchors[rel(source['td_infer_checkpoint'])]['anchor_sha256']
                                 if method != 'promptmm_release' else None))
    if teachers != {TEACHER_SHA}:
        raise ValueError('Not one common teacher')
    return dict(scope='M0 static binding; source/checkpoint payload SHA inherited until real smoke',
                quality_batch_source_commit=batch['source_commit'],
                teacher_sha256=TEACHER_SHA, train_sha256=TRAIN_SHA,
                rows=bindings)


def check_limits(start, out, torch=None):
    if time.monotonic() - start >= LIMITS['wall_seconds']:
        raise RuntimeError('M0 wall time exceeded')
    if shutil.disk_usage(out).free < LIMITS['free_disk_bytes']:
        raise RuntimeError('M0 free disk below cap')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > LIMITS['output_bytes']:
        raise RuntimeError('M0 output cap exceeded')
    if torch is not None and torch.cuda.memory_allocated() > LIMITS['cuda_allocated_bytes']:
        raise RuntimeError('M0 CUDA allocated cap exceeded')
    import psutil
    proc = psutil.Process()
    rss = proc.memory_info().rss + sum(p.memory_info().rss for p in proc.children(recursive=True) if p.is_running())
    if rss > LIMITS['process_tree_rss_bytes']:
        raise RuntimeError('M0 process-tree RSS cap exceeded')
    return dict(cuda_allocated_bytes=torch.cuda.memory_allocated() if torch else 0,
                process_tree_rss_bytes=rss, free_disk_bytes=shutil.disk_usage(out).free)


def install_split_guard():
    def audit(event, args):
        if event != 'open' or not args:
            return
        try:
            name = Path(args[0]).name.lower()
        except (TypeError, ValueError):
            return
        if name in {'test_mat', 'val_mat', 'validation_mat', 'test.txt', 'val.txt'}:
            raise RuntimeError('M0 forbids Validation/Test file I/O: ' + name)
    sys.addaudithook(audit)


def synthetic_check():
    import numpy as np
    import scipy.sparse as sp
    import torch
    sys.path.insert(0, str(ROOT / 'codes'))
    from promptmm_release import ReleaseStudent, release_graphs, sparse_tensor
    train = sp.csr_matrix(np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32))
    _, _, adj = release_graphs(train)
    torch.manual_seed(7)
    student = ReleaseStudent(3, 4, 2, 1)
    initial_u = torch.randn(3, 2)
    initial_i = torch.randn(4, 2)
    student.init_user_item_embed(initial_u, initial_i)
    student.assert_aliases()
    with torch.inference_mode():
        original = student(sparse_tensor(adj, 'cpu'))
    exported = tuple(t.detach().clone() for t in original)
    if not all(torch.equal(a, b) for a, b in zip(original, exported)):
        raise AssertionError('Synthetic export failed')
    if torch.equal(original[0], student.user_id_embedding.weight):
        raise AssertionError('Synthetic graph output incorrectly collapsed to raw IDs')
    for user in range(3):
        a = original[0][user] @ original[1].T
        b = exported[0][user] @ exported[1].T
        mask = train.indices[train.indptr[user]:train.indptr[user + 1]]
        a[mask] = -torch.inf
        b[mask] = -torch.inf
        if not torch.equal(a, b):
            raise AssertionError('Synthetic masked request differs')
    if unexpected_source_status(['?? check/']):
        raise AssertionError('Existing unrelated check/ should be allowed')
    if not unexpected_source_status(['?? check/', '?? tools/new_untracked.py']):
        raise AssertionError('New untracked code should be rejected')
    if not unexpected_source_status([' M tools/innovation1_cost_m0.py']):
        raise AssertionError('Tracked changes should be rejected')
    print('M0 synthetic release graph/export/masked scores: passed; no real assets loaded')


def real_smoke(bindings, out, start):
    import numpy as np
    import torch
    import scipy.sparse as sp
    sys.path.insert(0, str(ROOT / 'codes'))
    from promptmm_release import ReleaseStudent, release_graphs, sparse_tensor
    if not torch.cuda.is_available():
        raise RuntimeError('M0 requires CUDA; no CPU substitution')
    torch.cuda.set_device(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, LIMITS['cuda_allocated_bytes'] / total_memory), 0)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if digest(ROOT / bindings['rows'][0]['teacher_path']) != TEACHER_SHA:
        raise RuntimeError('Shared teacher SHA mismatch')
    train_path = ROOT / 'data/sports/train_mat'
    if digest(train_path) != TRAIN_SHA:
        raise RuntimeError('Train SHA mismatch')
    with train_path.open('rb') as stream:
        train = pickle.load(stream).tocsr()
    train.sum_duplicates()
    train.eliminate_zeros()
    if train.shape != (35598, 18357):
        raise RuntimeError('Train shape mismatch')
    eligible = np.flatnonzero((np.diff(train.indptr) > 0) & (np.diff(train.indptr) <= train.shape[1] - 20))
    users = np.random.default_rng(9026026).choice(eligible, size=3, replace=False).tolist()
    ui, iu, adj = release_graphs(train)
    if adj.shape != (sum(train.shape),) * 2:
        raise RuntimeError('Release adjacency shape mismatch')
    adj_gpu = sparse_tensor(adj, 'cuda:0').coalesce()

    def verify_tables(pair):
        if len(pair) != 2:
            raise RuntimeError('Expected two representation tables')
        for tensor, shape in zip(pair, ((35598, 64), (18357, 64))):
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                raise RuntimeError('Representation shape, dtype or finite check failed')

    results = []
    for row in bindings['rows']:
        check_limits(start, out, torch)
        if digest(ROOT / row['source_path']) != row['source_sha256']:
            raise RuntimeError('Source report/manifest SHA mismatch: ' + row['slot'])
        checkpoint = ROOT / row['checkpoint_path']
        if digest(checkpoint) != row['checkpoint_sha256']:
            raise RuntimeError('Checkpoint SHA mismatch: ' + row['slot'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if row['method'] == 'promptmm_release':
            model = ReleaseStudent(*train.shape, 64, 1).to('cuda:0')
            # Restore the original alias registration before the selected state.
            state = saved['model_state_dict']
            for side in ('user', 'item'):
                if not torch.equal(state[f'{side}_id_embedding.weight'], state[f'{side}_id_embedding_pre.weight']):
                    raise RuntimeError('Release alias values disagree')
            model.init_user_item_embed(state['user_id_embedding.weight'].to('cuda:0').clone(),
                                       state['item_id_embedding.weight'].to('cuda:0').clone())
            model.load_state_dict(state, strict=True)
            model.assert_aliases()
            model.eval()
            with torch.inference_mode():
                original = model(adj_gpu)
        else:
            state = saved['model_state_dict']
            infer_path = ROOT / row['td_infer_path']
            if digest(infer_path) != row['td_infer_sha256']:
                raise RuntimeError('TD inference file SHA mismatch')
            infer = torch.load(infer_path, map_location='cpu', weights_only=False)
            for side in ('user', 'item'):
                key = f'{side}_id_embedding.weight'
                if not torch.equal(state[key], infer[key]):
                    raise RuntimeError('TD selected/infer table mismatch')
            original = (state['user_id_embedding.weight'].to('cuda:0'),
                        state['item_id_embedding.weight'].to('cuda:0'))
            del infer
        tables = tuple(t.detach().contiguous() for t in original)
        verify_tables(tables)
        export_path = out / (row['slot'] + '_tables.pt')
        with export_path.open('xb') as stream:
            torch.save(dict(users=tables[0].cpu(), items=tables[1].cpu(), embedding_dim=64), stream)
        loaded = torch.load(export_path, map_location='cpu', weights_only=True)
        exported = (loaded['users'].to('cuda:0'), loaded['items'].to('cuda:0'))
        if any(not torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(original, exported)):
            raise RuntimeError('Serialized table differs from original output')
        checks = []
        for u in (users[row['seed'] - 2022],):
            # One fixed request per seed: three requests total per method.
            score_original = original[0][u] @ original[1].T
            score_export = exported[0][u] @ exported[1].T
            mask = train.indices[train.indptr[u]:train.indptr[u + 1]]
            score_original[mask] = -torch.inf
            score_export[mask] = -torch.inf
            finite = torch.isfinite(score_original)
            delta = (score_original[finite] - score_export[finite]).abs().max().item()
            if not torch.equal(torch.isinf(score_original), torch.isinf(score_export)) or delta > 1e-4:
                raise RuntimeError('Fixed request score mismatch')
            checks.append(dict(user=int(u), max_abs_score_delta=delta, excluded_train_items=len(mask)))
        logical_bytes = sum(t.numel() * t.element_size() for t in tables)
        results.append(dict(slot=row['slot'], dtype=str(tables[0].dtype), device=str(tables[0].device),
                            table_shapes=[list(t.shape) for t in tables], logical_bytes=logical_bytes,
                            export_path=str(export_path), export_sha256=digest(export_path),
                            export_file_bytes=export_path.stat().st_size,
                            requests=checks, memory=check_limits(start, out, torch)))
        if row['method'] == 'promptmm_release':
            del model
        del original, exported, tables, loaded, saved
        torch.cuda.empty_cache()
    import cached_deployment_benchmark as legacy_deploy
    with torch.inference_mode():
        teacher, prompt, teacher_ui, teacher_iu, teacher_phases = legacy_deploy.load_teacher('cuda:0')
        teacher_tables = tuple(teacher(teacher_ui, teacher_iu, prompt)[:2])
    verify_tables(teacher_tables)
    teacher_export = out / 'teacher_tables.pt'
    with teacher_export.open('xb') as stream:
        torch.save(dict(users=teacher_tables[0].cpu(), items=teacher_tables[1].cpu(), embedding_dim=64), stream)
    teacher_saved = torch.load(teacher_export, map_location='cpu', weights_only=True)
    teacher_restored = (teacher_saved['users'].to('cuda:0'), teacher_saved['items'].to('cuda:0'))
    if any(not torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(teacher_tables, teacher_restored)):
        raise RuntimeError('Teacher export differs from original model output')
    teacher_checks = []
    for u in users:
        original_score = teacher_tables[0][u] @ teacher_tables[1].T
        export_score = teacher_restored[0][u] @ teacher_restored[1].T
        mask = train.indices[train.indptr[u]:train.indptr[u + 1]]
        original_score[mask] = -torch.inf
        export_score[mask] = -torch.inf
        finite = torch.isfinite(original_score)
        delta = (original_score[finite] - export_score[finite]).abs().max().item()
        if not torch.equal(torch.isinf(original_score), torch.isinf(export_score)) or delta > 1e-4:
            raise RuntimeError('Teacher request score mismatch')
        teacher_checks.append(dict(user=int(u), max_abs_score_delta=delta))
    teacher_result = dict(dtype=str(teacher_tables[0].dtype),
                          table_shapes=[list(t.shape) for t in teacher_tables],
                          export_path=str(teacher_export), export_sha256=digest(teacher_export),
                          export_file_bytes=teacher_export.stat().st_size,
                          setup_phases=teacher_phases, requests=teacher_checks,
                          memory=check_limits(start, out, torch))
    return dict(rows=results, teacher=teacher_result, train_users=users, train_accesses=2,
                validation_accesses=0, test_accesses=0, optimizer_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-bindings', action='store_true')
    parser.add_argument('--smoke-real', action='store_true')
    parser.add_argument('--synthetic-check', action='store_true')
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--head', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.synthetic_check:
        synthetic_check()
        return
    if args.write_bindings and args.smoke_real:
        parser.error('Choose one mode')
    bindings = build_bindings()
    if args.write_bindings:
        MAP.write_text(json.dumps(bindings, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        print('Wrote', MAP, 'with', len(bindings['rows']), 'static bindings')
        return
    if not args.smoke_real and not args.internal_worker:
        if not MAP.exists() or read_json(MAP) != bindings:
            raise RuntimeError('Committed static bindings differ')
        print('M0 static bindings verified: nine students, one teacher; no model/split loaded')
        return
    if read_json(MAP) != bindings:
        raise RuntimeError('Static binding drift')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if args.internal_worker:
        require_clean_source()
        if args.head != head or not OUT.is_dir() or not (OUT / 'batch.json').exists():
            raise RuntimeError('Undeclared M0 worker')
        if read_json(OUT / 'batch.json')['source_commit'] != head:
            raise RuntimeError('M0 parent source mismatch')
        install_split_guard()
        started = time.monotonic()
        report = dict(status='started', source_commit=head, test_accesses=0,
                      validation_accesses=0, optimizer_steps=0)
        try:
            report.update(real_smoke(bindings, OUT, started))
            check_limits(started, OUT)
            report['status'] = 'completed'
        except BaseException as exc:
            report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            (OUT / 'worker.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return
    if not args.smoke_real:
        parser.error('Choose a mode')
    if shutil.disk_usage(ROOT).free < LIMITS['free_disk_bytes']:
        raise RuntimeError('Insufficient free disk before attempt')
    require_clean_source()
    OUT.mkdir(parents=True, exist_ok=False)
    report = dict(status='started', stage='M0 non-formal real smoke', attempt=1,
                  started_at=datetime.now().astimezone().isoformat(), source_commit=head,
                  limits=LIMITS, binding_sha256=digest(MAP), test_accesses=0,
                  validation_accesses=0, optimizer_steps=0)
    (OUT / 'batch.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--internal-worker', '--head', head]
    report['worker_command'] = command
    start = time.monotonic()
    import psutil
    process = None
    try:
        with (OUT / 'stdout.txt').open('x', encoding='utf-8') as stdout, (OUT / 'stderr.txt').open('x', encoding='utf-8') as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            while process.poll() is None:
                if time.monotonic() - start >= LIMITS['wall_seconds']:
                    raise RuntimeError('M0 supervisor wall limit')
                if shutil.disk_usage(ROOT).free < LIMITS['free_disk_bytes']:
                    raise RuntimeError('M0 supervisor free disk limit')
                if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) > LIMITS['output_bytes']:
                    raise RuntimeError('M0 supervisor output limit')
                parent = psutil.Process(process.pid)
                rss = parent.memory_info().rss + sum(p.memory_info().rss for p in parent.children(recursive=True) if p.is_running())
                report['peak_sampled_process_tree_rss_bytes'] = max(report.get('peak_sampled_process_tree_rss_bytes', 0), rss)
                if rss > LIMITS['process_tree_rss_bytes']:
                    raise RuntimeError('M0 supervisor process-tree RSS limit')
                time.sleep(0.25)
        report['worker_exit_code'] = process.returncode
        worker_path = OUT / 'worker.json'
        if process.returncode != 0 or not worker_path.exists():
            raise RuntimeError('M0 worker failed; inspect retained stderr/worker report')
        worker = read_json(worker_path)
        if (worker['status'] != 'completed' or len(worker['rows']) != 9
                or any(len(r['requests']) != 1 for r in worker['rows'])
                or len(worker['teacher']['requests']) != 3
                or worker['test_accesses'] or worker['validation_accesses'] or worker['optimizer_steps']):
            raise RuntimeError('M0 worker acceptance failed')
        report.update(status='completed', worker_report_sha256=digest(worker_path),
                      wall_seconds=time.monotonic() - start)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            for child in psutil.Process(process.pid).children(recursive=True):
                child.kill()
            process.kill()
            process.wait(timeout=10)
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc(),
                      wall_seconds=time.monotonic() - start)
        raise
    finally:
        report['finished_at'] = datetime.now().astimezone().isoformat()
        (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print('Preserved M0 report:', OUT / 'report.json', flush=True)


if __name__ == '__main__':
    main()
