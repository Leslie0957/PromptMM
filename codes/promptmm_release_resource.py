"""Train-only, non-formal resource probe. Invoked by main_mmlight early dispatch.

No validation/test loader, evaluation import, checkpoint publication or retries.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback
import types

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'sports_promptmm_release_resource_seed2022_v1'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--promptmm_release_resource_check', action='store_true', required=True)
    p.add_argument('--dataset', choices=['sports'], default='sports')
    p.add_argument('--steps', type=int, choices=[3], default=3)
    p.add_argument('--gpu_id', type=int, choices=[0], default=0)
    p.add_argument('--describe', action='store_true', help='Print specification only; no data/GPU/run.')
    return p.parse_args(argv)


def claim_run_directory(root):
    path = Path(root) / 'exp/resource_checks' / RUN_ID
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_training_inputs(data_dir):
    import numpy as np
    with (Path(data_dir) / 'train_mat').open('rb') as f:
        train = pickle.load(f).tocsr()
    return (train, np.load(Path(data_dir) / 'image_feat.npy'),
            np.load(Path(data_dir) / 'text_feat.npy'))


def load_dgl():
    # Same Windows distributed/GraphBolt workaround as the existing active entry.
    if os.name == 'nt':
        stub = types.ModuleType('dgl.distributed')
        stub.__file__ = __file__
        def unavailable(*a, **kw):
            raise RuntimeError('Distributed DGL is not used by this resource probe.')
        class UnavailableDistributed:
            def __init__(self, *a, **kw):
                unavailable()
        def lookup(name):
            if name.startswith('__'):
                raise AttributeError(name)
            if name.startswith('Dist'):
                return UnavailableDistributed
            return unavailable
        stub.__getattr__ = lookup
        sys.modules.setdefault('dgl.distributed', stub)
        bolt = types.ModuleType('dgl.graphbolt')
        bolt.__file__ = __file__
        sys.modules.setdefault('dgl.graphbolt', bolt)
    import dgl
    return dgl


def load_teacher_class(config):
    # Load only the shared model, avoiding utility.parser's CLI/global side effects.
    # Its module-global args retain this explicit namespace after the shim is removed.
    shim = types.ModuleType('utility.parser')
    shim.args = types.SimpleNamespace(**config)
    previous = sys.modules.get('utility.parser')
    sys.modules['utility.parser'] = shim
    try:
        spec = importlib.util.spec_from_file_location('_release_shared_teacher', ROOT / 'codes/Models_mmlight.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Teacher_Model
    finally:
        if previous is None:
            sys.modules.pop('utility.parser', None)
        else:
            sys.modules['utility.parser'] = previous


def tensor_digest(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main(argv=None):
    cli = parse_args(argv)
    from promptmm_release import IDENTITY, UPSTREAM, ResourceConfig
    cfg = ResourceConfig()
    specification = dict(identity=IDENTITY, upstream=UPSTREAM, run_id=RUN_ID,
                         config=asdict(cfg), non_formal=True, validation_evaluations=0,
                         test_evaluations=0, held_out_splits_loaded=False,
                         optimizer_steps_limit=3, warmup_steps=1,
                         formal_training_supported=False, paper_ready_eligible=False)
    if cli.describe:
        print(json.dumps(specification, indent=2))
        return 0
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Clean committed source required; do not start from a dirty tree.')
    # Exclusive directory is the one-launch guard, including failed attempts.
    run_dir = claim_run_directory(ROOT)
    report_path = run_dir / 'report.json'
    report = dict(specification, status='started', launch_commit=head, launch_dirty=False,
                  started_at=datetime.now().astimezone().isoformat(), steps=[],
                  command=[sys.executable, *sys.argv], python=sys.version,
                  source={str(p.relative_to(ROOT)): sha256(p) for p in [
                      ROOT/'codes/main_mmlight.py', Path(__file__), ROOT/'codes/promptmm_release.py',
                      ROOT/'codes/Models_mmlight.py']})
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save()
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cli.gpu_id)
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
        from promptmm_release import (ReleaseStudent, release_graphs, sparse_tensor,
                                      sample_triplets, release_candidates, release_objective)
        from utility.dataset_profiles import SPORTS_TEACHER_CHECKPOINT
        from utility.sports_validation_reuse import SPORTS_TEACHER_SHA256
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required; no implicit CPU fallback.')
        dgl = load_dgl()
        random.seed(cfg.seed); np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed); torch.cuda.manual_seed_all(cfg.seed); dgl.seed(cfg.seed)
        device = torch.device('cuda:0')
        report['environment'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                     dgl=dgl.__version__, gpu=torch.cuda.get_device_name(0),
                                     total_memory=torch.cuda.get_device_properties(0).total_memory)
        teacher_path = ROOT / SPORTS_TEACHER_CHECKPOINT
        if sha256(teacher_path) != SPORTS_TEACHER_SHA256:
            raise RuntimeError('Pinned teacher fingerprint mismatch.')
        checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=False)
        if checkpoint['dataset'] != 'sports' or checkpoint['best_epoch'] != 37:
            raise RuntimeError('Unexpected teacher identity.')
        if checkpoint['paper_ready_eligible'] is not False:
            raise RuntimeError('Expected original Validation-only eligibility record.')
        identity = checkpoint['dataset_identity']
        report['shared_teacher'] = dict(path=str(teacher_path), sha256=SPORTS_TEACHER_SHA256,
                                       epoch=37, original_paper_ready_eligible=False,
                                       conversion_identity=identity['conversion_manifest'])
        data_dir = ROOT / 'data/sports'
        expected = {'train_mat': identity['matrices']['train']['sha256'],
                    'image_feat.npy': identity['features']['image']['sha256'],
                    'text_feat.npy': identity['features']['text']['sha256']}
        report['input_sha256'] = {'teacher': SPORTS_TEACHER_SHA256}
        for name, digest in expected.items():
            actual = sha256(data_dir / name)
            if actual != digest:
                raise RuntimeError('Input fingerprint mismatch: ' + name)
            report['input_sha256'][name] = actual
        # Explicit allowlist: never invoke Data, batch_test, dataset_preflight or Trainer.
        train, image, text = load_training_inputs(data_dir)
        nu, ni = train.shape
        existing = np.unique(train.nonzero()[0]).astype(int).tolist()
        report['train_shape'] = [nu, ni]
        report['train_interactions'] = int(train.nnz)
        ui, iu, adj = [sparse_tensor(g, device) for g in release_graphs(train)]
        settings = dict(checkpoint['teacher_inference_config'], drop_rate=0.2, prompt_dropout=0.)
        teacher_class = load_teacher_class(settings)
        torch.cuda.reset_peak_memory_stats()
        teacher = teacher_class(nu, ni, 64, list(settings['weight_size']), [0.1, 0.1], image, text).to(device)
        teacher.load_state_dict(checkpoint['teacher_model'], strict=True)
        teacher.requires_grad_(False)

        class StoredPrompt(nn.Module):
            def __init__(self, state):
                super().__init__()
                self.register_buffer('item_hard_token', state['item_hard_token'].to(device))
                self.register_buffer('user_hard_token', state['user_hard_token'].to(device))
                self.trans_user = nn.Linear(64, 64, device=device)
                self.trans_item = nn.Linear(64, 64, device=device)
                self.load_state_dict(state, strict=True)
            def forward(self):
                # Pinned p=0, retaining original functional-dropout call semantics.
                return (F.dropout(self.trans_user(self.user_hard_token), 0.),
                        F.dropout(self.trans_item(self.item_hard_token), 0.))

        prompt = StoredPrompt(checkpoint['prompt_module'])
        del checkpoint, image, text
        teacher_before, prompt_before = tensor_digest(teacher), tensor_digest(prompt)
        teacher.eval()
        with torch.no_grad():
            initial = teacher(ui, iu, prompt)
        student = ReleaseStudent(nu, ni, 64, cfg.layers).to(device)
        # Initialize after placement: .to after alias creation may break storage sharing.
        student.init_user_item_embed(initial[0], initial[1])
        student.assert_aliases()
        del initial
        optimizer = torch.optim.AdamW([{'params': student.parameters()}, {'params': prompt.parameters()}],
                                      lr=cfg.learning_rate, weight_decay=cfg.weight_decay, foreach=False)
        report['registered_student_parameters'] = sum(p.numel() for p in student.parameters())
        report['distinct_student_storage_elements'] = sum(
            p.numel() for p in {p.data_ptr(): p for p in student.parameters()}.values())
        report['teacher_inference_config'] = settings
        report['setup_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['setup_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report['adaptations'] = ['shared local teacher/checkpoint', 'CPU DGL sampler; same call/ignored row semantics',
                                 'foreach=False for explicit sequential alias updates',
                                 'unused upstream diagnostic loss computations omitted; not an efficiency baseline']
        save()
        student_before = tensor_digest(student)
        torch.cuda.synchronize()
        for step in range(3):
            report['active_step'] = step + 1
            save()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            teacher.train(); student.train()
            users, pos, neg = sample_triplets(train, existing, cfg.batch_size)
            student_out = student(adj)
            with torch.no_grad():
                teacher_out = teacher(ui, iu, prompt)
            candidates = release_candidates(train, users, pos, cfg.negative_count, device, dgl)
            losses = release_objective(student_out, teacher_out, users, pos, neg, candidates, cfg)
            if not all(torch.isfinite(v).all().item() for v in losses.values()):
                raise FloatingPointError('Nonfinite release objective; no correction or retry permitted.')
            optimizer.zero_grad(set_to_none=True)
            losses['total'].backward()
            for parameter in student.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Missing/nonfinite student gradient.')
            if any(p.grad is not None for p in teacher.parameters()) or any(p.grad is not None for p in prompt.parameters()):
                raise RuntimeError('Release teacher/prompt should have no gradient.')
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in student.parameters()):
                raise FloatingPointError('Nonfinite updated student weight.')
            for state in optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        raise FloatingPointError('Nonfinite optimizer state.')
            student.assert_aliases()
            torch.cuda.synchronize()
            report['steps'].append(dict(step=step+1, warmup=step==0,
                                       seconds=time.perf_counter()-start,
                                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                       peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                       losses={k: float(v.detach()) for k,v in losses.items()}))
            del student_out, teacher_out, candidates, losses
            save()
        if tensor_digest(teacher) != teacher_before or tensor_digest(prompt) != prompt_before:
            raise RuntimeError('Frozen teacher/prompt state changed.')
        if tensor_digest(student) == student_before:
            raise RuntimeError('Student did not update.')
        if sha256(teacher_path) != SPORTS_TEACHER_SHA256:
            raise RuntimeError('Original teacher file changed.')
        report.update(status='resource_completed', teacher_unchanged=True, prompt_unchanged=True,
                      student_updated=True, alias_preserved=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        try:
            if torch.cuda.is_initialized():
                report['failure_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                report['failure_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        except Exception:
            pass  # Never mask the original failure (including an import failure).
        raise
    finally:
        report['finished_at'] = datetime.now().astimezone().isoformat()
        save()
        print('Resource report:', report_path, flush=True)
    return 0
