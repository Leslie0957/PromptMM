"""Train-only, zero-update deployment measurements; no training-entry imports.

Real assets are opened only by the user-launched runner. Import and unit tests
are safe on CPU. Timing results are workload measurements, not quality metrics.
"""
import ast
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import types

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/efficiency/sports_cached_deployment_seed2022_v2'
PYTHON = Path('D:/miniconda/envs/run_5060/python.exe')
TRAIN = ROOT / 'data/sports/train_mat'
TRAIN_SHA = '5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8'
SHARED = ROOT / 'exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/process0/initial_tensors.pt'
SHARED_SHA = 'e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20'
TEACHER_SHA = '57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea'
MODEL_SHA = 'ec993272ccf07d24a70b41bb9fb87273d95e01d2de6eb1dfb1950c10172933dc'
MANIFESTS = {
    'F': ('2026-09-23 11_07_51.352217_sports_light_init_pid2784',
          '825029a48243ac8780b366f088a754d0f9f3f0738e6e69449aaaaf3cc05530fa',
          'bd651d78ab8853ba987c407a23cd485581ccf319a1b6c9202c5413091fc6879c'),
    'B': ('2026-09-23 16_11_54.129842_sports_light_init_pid36040',
          '300b02c678b9e6e2c5988d79aaa7baa7cd2e2e517b17b9ccdd9b68751edd66a9',
          '2c35884bb4d1552fb346689732e3dcda9cbb6f59d50e8113495d62ae1e396e18'),
}
REQUEST_SEED = 9026026
BATCHES = (1, 128, 1024)
ONLINE_WARMUP, ONLINE_RUNS = 20, 100
OFFLINE_WARMUP, OFFLINE_RUNS = 3, 10
SHAPES = ((35598, 64), (18357, 64))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_hash(path, expected):
    if sha256(path) != expected:
        raise RuntimeError('Asset fingerprint mismatch: ' + str(path))


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def tensor_digest(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        if tensor.layout != torch.strided:
            sparse = tensor.detach().coalesce()
            h.update(str((tuple(sparse.shape), sparse.layout)).encode())
            h.update(tensor_digest((sparse.indices(), sparse.values())).encode())
            continue
        t = tensor.detach().cpu().contiguous()
        h.update(str((tuple(t.shape), t.dtype)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def tables_ok(tables, shapes=SHAPES):
    if len(tables) != 2:
        raise ValueError('Expected user and item tables')
    for t, shape in zip(tables, shapes):
        if tuple(t.shape) != tuple(shape) or t.dtype != torch.float32:
            raise ValueError('Unexpected table shape/dtype')
        if not t.is_contiguous() or not torch.isfinite(t).all():
            raise ValueError('Nonfinite or noncontiguous table')


def graph_normalize(matrix):
    """Exact Trainer.csr_norm(mean_flag=True), including dtype and epsilon."""
    degree = np.array(matrix.sum(1))
    scale = np.power(degree + 1e-8, -0.5).flatten()
    scale[np.isinf(scale)] = 0.
    return sp.diags(scale) * matrix


def sparse_tensor(matrix, device):
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data).to(torch.float32)
    return torch.sparse_coo_tensor(indices, values, coo.shape,
                                   dtype=torch.float32, device=device,
                                   check_invariants=True).coalesce()


def request_order(train, largest_batch=1024):
    train = train.tocsr(copy=True)
    train.sum_duplicates()
    train.eliminate_zeros()
    counts = np.diff(train.indptr)
    eligible = np.flatnonzero((counts > 0) & (counts <= train.shape[1] - 20))
    if len(eligible) < largest_batch:
        raise ValueError('Insufficient eligible train-only users')
    return np.random.default_rng(REQUEST_SEED).permutation(eligible).astype(np.int64)


def make_requests(train, order, batch_size, count=120, device='cpu'):
    """Preload only sparse exclusion indices, never dense all-user score masks."""
    requests = []
    for index in range(count):
        ids = order[(np.arange(batch_size) + index * batch_size) % len(order)]
        mask = train[ids].tocoo()
        requests.append((torch.as_tensor(ids, device=device),
                         torch.as_tensor(mask.row.astype(np.int64), device=device),
                         torch.as_tensor(mask.col.astype(np.int64), device=device)))
    return requests


def score(tables, request, k=20):
    """Ties use torch.topk's native order; no artificial score perturbation."""
    users, items = tables
    ids, rows, columns = request
    scores = users[ids] @ items.T
    scores[rows, columns] = -torch.inf
    values, indices = torch.topk(scores, k, dim=1, sorted=True)
    return values.cpu(), indices.cpu()


def storage_inventory(named_tensors):
    """Count actual backing storage once, including sparse indices and values."""
    seen, entries = set(), []
    for name, tensor in named_tensors:
        components = [(name, tensor)] if tensor.layout == torch.strided else [
            (name + '.indices', tensor.coalesce().indices()),
            (name + '.values', tensor.coalesce().values())]
        for label, t in components:
            storage = t.untyped_storage()
            key = (str(t.device), storage.data_ptr(), storage.nbytes())
            shared = key in seen
            entries.append(dict(name=label, shape=list(t.shape), dtype=str(t.dtype),
                                device=str(t.device), storage_bytes=storage.nbytes(),
                                counted_bytes=0 if shared else storage.nbytes()))
            seen.add(key)
    return dict(unique_bytes=sum(e['counted_bytes'] for e in entries), tensors=entries)


def module_tensors(module, prefix):
    values = list(module.named_parameters()) + list(module.named_buffers())
    # Teacher image_feats/text_feats and prompt ui_graph are plain attributes.
    for name, value in vars(module).items():
        if isinstance(value, torch.Tensor):
            values.append((name, value))
    return [(prefix + '.' + name, value) for name, value in values]


def manifest(arm):
    name, expected, _ = MANIFESTS[arm]
    path = ROOT / 'exp/runs/sports' / ('run_manifest__' + name + '.json')
    check_hash(path, expected)
    m = json.loads(path.read_text(encoding='utf-8'))
    if (m['status'] != 'validation_completed' or m['final_test_performed']
            or m['teacher_final_test_performed'] or m['run_final_test']):
        raise ValueError('Unexpected source protocol')
    if m['teacher_checkpoint_fingerprint']['sha256'] != TEACHER_SHA:
        raise ValueError('Unexpected teacher source')
    return m, path


def preflight():
    """User runtime only: verify assets and prepare identical deployment formats."""
    check_hash(TRAIN, TRAIN_SHA)
    check_hash(SHARED, SHARED_SHA)
    check_hash(ROOT / 'codes/Models_mmlight.py', MODEL_SHA)
    with TRAIN.open('rb') as f:
        train = pickle.load(f).tocsr()
    order = request_order(train)
    np.save(OUT / 'request_order.npy', order, allow_pickle=False)
    assets = {str(TRAIN): TRAIN_SHA, str(SHARED): SHARED_SHA}
    tables = torch.load(SHARED, map_location='cpu', weights_only=True)
    teacher = (tables['users'], tables['items'])
    tables_ok(teacher)
    exports = {}
    for arm in ('T', 'F', 'B'):
        if arm == 'T':
            pair = teacher
        else:
            m, path = manifest(arm)
            assets[str(path)] = MANIFESTS[arm][1]
            check_hash(m['td_full_checkpoint'], MANIFESTS[arm][2])
            assets[m['td_full_checkpoint']] = MANIFESTS[arm][2]
            full = torch.load(m['td_full_checkpoint'], map_location='cpu', weights_only=False)
            infer = torch.load(m['td_infer_checkpoint'], map_location='cpu', weights_only=False)
            keys = ('user_id_embedding.weight', 'item_id_embedding.weight')
            if not all(torch.equal(full['model_state_dict'][k], infer[k]) for k in keys):
                raise ValueError('Inference tables differ from selected checkpoint')
            pair = tuple(infer[k] for k in keys)
            assets[m['td_infer_checkpoint']] = sha256(m['td_infer_checkpoint'])
            del full, infer
        tables_ok(pair)
        path = OUT / (arm + '_tables.pt')
        with path.open('xb') as f:
            torch.save(dict(users=pair[0], items=pair[1], embedding_dim=64), f)
        exports[arm] = dict(path=str(path), sha256=sha256(path),
                            tensor_sha256=tensor_digest(pair), file_bytes=path.stat().st_size,
                            logical_bytes=sum(t.numel() * t.element_size() for t in pair))
    m, _ = manifest('F')
    teacher_path = str(ROOT / m['resolved_arguments']['teacher_checkpoint'])
    assets[teacher_path] = TEACHER_SHA
    for modality in ('image', 'text'):
        record = m['active_teacher_hard_token_cache'][modality]
        for item in ('cache_file', 'source_feature'):
            assets[record[item]['path']] = record[item]['sha256']
    for path, expected in assets.items():
        check_hash(path, expected)
    return dict(status='passed', assets=assets,
                asset_file_bytes={p: Path(p).stat().st_size for p in assets}, exports=exports,
                request_sha256=sha256(OUT / 'request_order.npy'),
                request_seed=REQUEST_SEED, train_shape=list(train.shape),
                eligible_users=len(order), validation_accesses=0, test_accesses=0,
                optimizer_steps=0)


def load_original_models(arguments):
    """Isolated worker: original classes with pinned args, no parser side effects."""
    if 'Models_mmlight' in sys.modules or 'utility.parser' in sys.modules:
        raise RuntimeError('Teacher adapter requires an isolated process')
    check_hash(ROOT / 'codes/Models_mmlight.py', MODEL_SHA)
    proxy = types.ModuleType('utility.parser')
    proxy.args = types.SimpleNamespace(**arguments)
    sys.modules['utility.parser'] = proxy
    return importlib.import_module('Models_mmlight')


def load_teacher(device='cuda:0'):
    """Original teacher + PromptLearner; caches read only; no Data or Trainer."""
    m, _ = manifest('F')
    arguments = dict(m['resolved_arguments'])
    arguments['data_path'] = str(ROOT / 'data') + os.sep
    args = types.SimpleNamespace(**arguments)
    phases = {}
    start = time.perf_counter()
    checkpoint_path = ROOT / args.teacher_checkpoint
    check_hash(checkpoint_path, TEACHER_SHA)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if state['teacher_inference_config'] != m['teacher_checkpoint_metadata']['teacher_inference_config']:
        raise ValueError('Teacher configuration mismatch')
    for key, value in state['teacher_inference_config'].items():
        actual = ast.literal_eval(arguments[key]) if key == 'weight_size' else arguments[key]
        if actual != value:
            raise ValueError('Resolved teacher argument differs: ' + key)
    features, cache = {}, {}
    for modality in ('image', 'text'):
        r = m['active_teacher_hard_token_cache'][modality]
        check_hash(r['source_feature']['path'], r['source_feature']['sha256'])
        check_hash(r['cache_file']['path'], r['cache_file']['sha256'])
        features[modality] = np.load(r['source_feature']['path'], allow_pickle=False)
        with Path(r['cache_file']['path']).open('rb') as f:
            payload = pickle.load(f)
        if payload['metadata']['cache_identity'] != r['cache_identity']:
            raise ValueError('Hard-token cache metadata mismatch')
        cache[modality] = (payload['values'], r)
    check_hash(TRAIN, TRAIN_SHA)
    with TRAIN.open('rb') as f:
        train = pickle.load(f).tocsr()
    phases['read_and_hash_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    ui_cpu, iu_cpu = graph_normalize(train), graph_normalize(train.T)
    phases['graph_normalization_cpu_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    models = load_original_models(arguments)
    phases['original_model_import_seconds'] = time.perf_counter() - start

    def cache_only(feature_array, source_feature_path, dataset_dir, modality,
                   method, n_components, random_state):
        if method != args.hard_token_type or n_components != args.embed_size or random_state != args.hard_token_seed:
            raise ValueError('Unexpected hard-token construction')
        values, record = cache[modality]
        if values.shape != (train.shape[1], 64) or not np.isfinite(values).all():
            raise ValueError('Invalid hard-token values')
        return values, record

    original_cache_loader = models.load_or_create_hard_token_cache
    models.load_or_create_hard_token_cache = cache_only
    if args.hard_token_type != 'pca' or args.eval_protocol != 'val_test_once_v1':
        raise ValueError('Unsupported pinned teacher configuration')
    start = time.perf_counter()
    ui, iu = sparse_tensor(ui_cpu, device), sparse_tensor(iu_cpu, device)
    torch.cuda.synchronize()
    phases['graph_transfer_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    teacher = models.Teacher_Model(train.shape[0], train.shape[1], args.embed_size,
                                   ast.literal_eval(args.weight_size),
                                   ast.literal_eval(args.mess_dropout),
                                   features['image'], features['text']).to(device)
    try:
        prompt = models.PromptLearner(features['image'], features['text'], ui).to(device)
    finally:
        models.load_or_create_hard_token_cache = original_cache_loader
    teacher.load_state_dict(state['teacher_model'], strict=True)
    prompt.load_state_dict(state['prompt_module'], strict=True)
    teacher.eval().requires_grad_(False)
    prompt.eval().requires_grad_(False)
    torch.cuda.synchronize()
    # Constructors themselves call .cuda(); report this mixed stage honestly.
    phases['construction_modal_transfer_restore_seconds'] = time.perf_counter() - start
    return teacher, prompt, ui, iu, phases


def summary(milliseconds, batch_size=1):
    if not milliseconds or not all(math.isfinite(x) and x > 0 for x in milliseconds):
        raise ValueError('Invalid timing samples')
    return dict(median_ms=float(np.median(milliseconds)),
                p95_ms=float(np.percentile(milliseconds, 95)),
                users_per_second=batch_size * len(milliseconds) * 1000 / sum(milliseconds))


def gpu_memory():
    return dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved())


def working_set():
    try:
        import psutil
        return dict(bytes=psutil.Process().memory_info().rss, source='psutil rss/Windows working set')
    except (ImportError, OSError) as error:
        return dict(bytes=None, unavailable=repr(error))


def measure_online(tables, train, order, batch_size, report, path):
    start = time.perf_counter()
    requests = make_requests(train, order, batch_size, device='cuda:0')
    torch.cuda.synchronize()
    report['request_preload_seconds'] = time.perf_counter() - start
    report['request_tensor_sha256'] = tensor_digest([t for r in requests for t in r])
    for request in requests[:ONLINE_WARMUP]:
        score(tables, request)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    report['memory_baseline'] = gpu_memory()
    raw, gemm, digests = [], [], []
    for index, request in enumerate(requests[ONLINE_WARMUP:]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        values, ids = score(tables, request)
        torch.cuda.synchronize()
        raw.append((time.perf_counter() - start) * 1000)
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite top-k result')
        digests.append(tensor_digest((values, ids)))
        report.update(raw_wall_ms=raw, result_digests=digests)
        # Checkpoint after each completed measurement; disk I/O is outside timer.
        save_json(path, report)
    report['memory_online'] = gpu_memory()
    report['memory_online']['incremental_peak_allocated'] = (
        report['memory_online']['peak_allocated'] - report['memory_baseline']['allocated'])
    report['working_set_online'] = working_set()
    # Separate diagnostic loop: CUDA events do not instrument service wall time.
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for request in requests[ONLINE_WARMUP:]:
        users = tables[0][request[0]]
        torch.cuda.synchronize()
        begin.record()
        result = users @ tables[1].T
        end.record()
        end.synchronize()
        gemm.append(begin.elapsed_time(end))
        del result
    report.update(gemm_only_ms=gemm, summary=summary(raw, batch_size),
                  gemm_summary=summary(gemm, batch_size))


def compare_reference(actual, reference):
    result = []
    for a, r in zip(actual, reference):
        delta = (a.cpu() - r).abs()
        result.append(dict(max_abs=float(delta.max()),
                           max_relative=float((delta / r.abs().clamp_min(1e-12)).max()),
                           allclose=bool(torch.allclose(a.cpu(), r, atol=1e-4, rtol=1e-4))))
    return result


def measure_offline(arm, tables, report, path, folder):
    if arm == 'T':
        teacher, prompt, ui, iu, phases = load_teacher()
        report['teacher_setup'] = phases
        named = module_tensors(teacher, 'teacher') + module_tensors(prompt, 'prompt') + [('ui', ui), ('iu', iu)]
        before = tensor_digest([t for _, t in named])
        def generate():
            return tuple(teacher(ui, iu, prompt)[:2])
    else:
        named = [('users', tables[0]), ('items', tables[1])]
        def generate():
            return tuple(t.clone() for t in tables)
    report['resident_storage'] = storage_inventory(named)
    actual = generate()
    tables_ok(actual)
    reference = tuple(t.cpu() for t in tables)
    report['reference_comparison'] = compare_reference(actual, reference)
    save_json(path, report)
    if not all(x['allclose'] for x in report['reference_comparison']):
        raise RuntimeError('Fixed reference tolerance failed; no retry')
    del actual, reference
    for _ in range(OFFLINE_WARMUP):
        output = generate()
        del output
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    report['memory_baseline'] = gpu_memory()
    samples = []
    for _ in range(OFFLINE_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = generate()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        tables_ok(output)
        report['raw_wall_ms'] = samples
        save_json(path, report)
        del output
    report['memory_generation'] = gpu_memory()
    report['memory_generation']['incremental_peak_allocated'] = (
        report['memory_generation']['peak_allocated'] - report['memory_baseline']['allocated'])
    report['working_set_generation'] = working_set()
    output = generate()
    start = time.perf_counter()
    cpu = tuple(t.cpu() for t in output)
    torch.cuda.synchronize()
    report['export_device_to_host_seconds'] = time.perf_counter() - start
    start = time.perf_counter()
    destination = folder / 'tables.pt'
    with destination.open('xb') as f:
        torch.save(dict(users=cpu[0], items=cpu[1], embedding_dim=64), f)
    report['serialization_seconds'] = time.perf_counter() - start
    report['export'] = dict(path=str(destination), file_bytes=destination.stat().st_size,
                            sha256=sha256(destination), logical_bytes=sum(t.numel()*t.element_size() for t in cpu))
    report['summary'] = summary(samples)
    report['summary'].pop('users_per_second')
    if arm == 'T':
        after = tensor_digest([t for _, t in named])
        if before != after:
            raise RuntimeError('Teacher/prompt state mutated')
        report['teacher_state_unchanged'] = True
        report['teacher_forwards'] = 1 + OFFLINE_WARMUP + OFFLINE_RUNS + 1
    else:
        report['teacher_forwards'] = 0


def install_io_guard(output):
    """Fail closed on split reads or writes outside this exclusive result family."""
    output = Path(output).resolve()
    null_device = Path(os.devnull).resolve()
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        # subprocess/platform metadata legitimately opens the OS null sink
        # read-write. Permit only this exact device, never a basename/prefix.
        if path == null_device:
            return
        if path.name.lower() in ('test_mat', 'val_mat', 'validation_mat', 'test.txt', 'val.txt'):
            raise RuntimeError('Held-out split access forbidden: ' + str(path))
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing and not path.is_relative_to(output):
            raise RuntimeError('Write outside benchmark output forbidden: ' + str(path))
    sys.addaudithook(audit)
