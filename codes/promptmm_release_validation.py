"""Isolated fixed-budget release diagnostic; never opens Test data."""
import argparse
from dataclasses import asdict
from datetime import datetime
import heapq
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback

from promptmm_release_resource import (ROOT, sha256, load_training_inputs, load_dgl,
                                       load_teacher_class, tensor_digest)

RUN_ID = 'sports_promptmm_release_validation30_seed2022_v1'
EPOCHS = 30
KS = (10, 20, 40, 50)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--promptmm_release_validation', action='store_true', required=True)
    p.add_argument('--dataset', choices=['sports'], default='sports')
    p.add_argument('--seed', type=int, choices=[2022, 2023, 2024], default=2022)
    p.add_argument('--epochs', type=int, choices=[EPOCHS, 120, 300], default=EPOCHS)
    p.add_argument('--student_lr', type=float, choices=[2e-5, 6e-5], default=2e-5)
    p.add_argument('--gpu_id', type=int, choices=[0], default=0)
    p.add_argument('--describe', action='store_true')
    cli = p.parse_args(argv)
    if cli.epochs not in (120, 300) and cli.seed != 2022:
        p.error('Only the existing seed2022 short diagnostic or three-seed120/300 batches are declared.')
    if cli.student_lr != 2e-5 and (cli.seed != 2022 or cli.epochs != 300):
        p.error('lr6e-5 is declared only for seed2022 with300 epochs.')
    return cli


def run_identity(seed, epochs, learning_rate=2e-5):
    if learning_rate == 6e-5:
        return f'sports_promptmm_release_validation{epochs}_seed{seed}_lr6e5_v1'
    if learning_rate != 2e-5:
        raise ValueError('Undeclared learning rate.')
    return f'sports_promptmm_release_validation{epochs}_seed{seed}_v1'


def claim_run_directory(root, run_id=RUN_ID):
    path = Path(root) / 'exp/promptmm_release' / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_validation(data_dir, train, expected_hash):
    path = Path(data_dir) / 'val_mat'
    if sha256(path) != expected_hash:
        raise RuntimeError('Validation fingerprint mismatch.')
    with path.open('rb') as f:
        val = pickle.load(f).tocsr()
    if val.shape != train.shape or val.nnz == 0 or train.multiply(val).nnz:
        raise ValueError('Invalid Validation shape, empty split or train overlap.')
    return val


def evaluate_validation(student, adj, train, val, batch_size=256, ks=KS):
    # Same candidate set/order, heapq ranking and legacy metric definitions as
    # utility.batch_test, without importing its global Test-loading Data object.
    import numpy as np
    import torch
    from utility import metrics
    student.eval()
    totals = {k: np.zeros(len(ks), dtype=np.float64)
              for k in ('recall', 'precision', 'ndcg', 'hit_ratio')}
    users = np.flatnonzero(val.getnnz(axis=1)).tolist()
    if not users:
        raise ValueError('No Validation users.')
    with torch.no_grad():
        ue, ie = student(adj)
        all_items = set(range(train.shape[1]))
        for offset in range(0, len(users), batch_size):
            batch = users[offset:offset + batch_size]
            scores = (ue[batch] @ ie.T).cpu().numpy()
            if not np.isfinite(scores).all():
                raise FloatingPointError('Nonfinite Validation score.')
            for row, u in enumerate(batch):
                excluded = set(train.indices[train.indptr[u]:train.indptr[u+1]])
                candidates = list(all_items - excluded)
                if len(candidates) < max(ks):
                    raise ValueError('Too few eligible ranking candidates.')
                truth = set(val.indices[val.indptr[u]:val.indptr[u+1]])
                ranked = heapq.nlargest(max(ks), candidates, key=lambda i: scores[row, i])
                relevance = [int(i in truth) for i in ranked]
                for j, k in enumerate(ks):
                    totals['recall'][j] += metrics.recall_at_k(relevance, k, len(truth)) / len(users)
                    totals['precision'][j] += metrics.precision_at_k(relevance, k) / len(users)
                    totals['ndcg'][j] += metrics.ndcg_at_k(relevance, k) / len(users)
                    totals['hit_ratio'][j] += metrics.hit_at_k(relevance, k) / len(users)
    if not all(np.isfinite(v).all() for v in totals.values()):
        raise FloatingPointError('Nonfinite Validation metric.')
    return {k: v.tolist() for k, v in totals.items()}


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def save_best(path, student, epoch, metrics, metadata, best_score):
    import torch
    score = metrics['recall'][1]
    if not __import__('math').isfinite(score):
        raise FloatingPointError('Nonfinite selection metric.')
    if score <= best_score:
        return best_score, False
    student.assert_aliases()
    # CPU transfer may break aliasing in serialized entries. Restore into an
    # explicitly aliased model and verify both saved copies agree.
    state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    payload = dict(metadata, model_state_dict=state, best_epoch=epoch,
                   validation_metrics=metrics, best_selection_recall=score,
                   student_digest=tensor_digest(student), paper_ready_eligible=False,
                   resumable=False)
    path = Path(path)
    tmp = path.with_suffix('.pt.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return score, True


def restore_best(path, student):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=False)
    state = saved['model_state_dict']
    for side in ('user', 'item'):
        if not torch.equal(state[f'{side}_id_embedding.weight'],
                           state[f'{side}_id_embedding_pre.weight']):
            raise RuntimeError('Inconsistent saved alias values.')
    student.assert_aliases()
    student.load_state_dict(state, strict=True)
    student.assert_aliases()
    if tensor_digest(student) != saved['student_digest']:
        raise RuntimeError('Best checkpoint roundtrip mismatch.')
    return saved


def main(argv=None):
    cli = parse_args(argv)
    from promptmm_release import IDENTITY, UPSTREAM, ResourceConfig
    cfg = ResourceConfig(seed=cli.seed, learning_rate=cli.student_lr)
    epochs = cli.epochs
    run_id = run_identity(cfg.seed, epochs, cfg.learning_rate)
    config = asdict(cfg)
    del config['steps']
    spec = dict(identity=IDENTITY, upstream=UPSTREAM, run_id=run_id, config=config,
                epochs=epochs, early_stopping=False, validation_every=1, ks=list(KS),
                primary_metric='Recall@20', selection='strict improvement, earliest tie',
                validation_batch_size=256, diagnostic_only=True, paper_ready_eligible=False,
                teacher_test_evaluations=0, test_evaluations=0, test_split_loaded=False,
                validation_evaluations=0, evaluation_protocol='validation_only_release_v1',
                shared_anchor='SPORTS_CONVERTED_20260916')
    if cli.describe:
        print(json.dumps(spec, indent=2))
        return 0
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Clean committed launch source required.')
    run_dir = claim_run_directory(ROOT, run_id)
    report_path, best_path = run_dir/'report.json', run_dir/'best.pt'
    report = dict(spec, status='started', launch_commit=head, launch_dirty=False,
                  command=[sys.executable, *sys.argv], python=sys.version,
                  started_at=datetime.now().astimezone().isoformat(), curve=[],
                  source={str(p.relative_to(ROOT)): sha256(p) for p in (
                      ROOT/'codes/main_mmlight.py', Path(__file__),
                      ROOT/'codes/promptmm_release.py', ROOT/'codes/promptmm_release_resource.py',
                      ROOT/'codes/Models_mmlight.py', ROOT/'codes/utility/metrics.py',
                      ROOT/'codes/utility/dataset_profiles.py', ROOT/'codes/utility/sports_validation_reuse.py')})
    def save():
        atomic_json(report_path, report)
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
        # Read train/features and the pinned Validation matrix only.
        train, image, text = load_training_inputs(data_dir)
        validation = load_validation(data_dir, train, identity['matrices']['validation']['sha256'])
        report['input_sha256']['val_mat'] = identity['matrices']['validation']['sha256']
        report['validation_users'] = int((validation.getnnz(axis=1) > 0).sum())
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
        batches = train.nnz // cfg.batch_size + 1
        report['batches_per_epoch'] = batches
        report['optimizer_steps_limit'] = epochs * batches
        report['optimizer_steps_completed'] = 0
        report['finite_updates_checked'] = True
        report['metric_definition'] = 'legacy batch_test: NDCG ideal sorts top-max(Ks) relevance'
        metadata = dict(identity=IDENTITY, upstream=UPSTREAM, config=config,
                        launch_commit=head, input_sha256=report['input_sha256'],
                        shared_teacher=report['shared_teacher'],
                        evaluation_protocol=report['evaluation_protocol'],
                        graph_semantics='release column order', layers=cfg.layers)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        report['initial_validation'] = evaluate_validation(student, adj, train, validation)
        report['initial_validation_seconds'] = time.perf_counter() - t
        report['initial_validation_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['validation_evaluations'] += 1
        report['initial_validation_selected'] = False
        save()
        print('Initial Validation Recall@20:', report['initial_validation']['recall'][1], flush=True)
        best_score = -1.
        for epoch in range(1, epochs + 1):
            report['active_epoch'] = epoch
            save()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            sums = {}
            teacher.train(); student.train()
            for step in range(batches):
                report['active_batch'] = step + 1
                users, pos, neg = sample_triplets(train, existing, cfg.batch_size)
                student_out = student(adj)
                with torch.no_grad():
                    teacher_out = teacher(ui, iu, prompt)
                candidates = release_candidates(train, users, pos, cfg.negative_count, device, dgl)
                losses = release_objective(student_out, teacher_out, users, pos, neg, candidates, cfg)
                if not all(torch.isfinite(v).all().item() for v in losses.values()):
                    raise FloatingPointError('Nonfinite release objective.')
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
                report['optimizer_steps_completed'] += 1
                for name, value in losses.items():
                    sums[name] = sums.get(name, 0.) + float(value.detach()) / batches
                del student_out, teacher_out, candidates, losses
            torch.cuda.synchronize()
            train_seconds = time.perf_counter() - start
            train_memory = torch.cuda.max_memory_allocated()
            train_reserved = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = evaluate_validation(student, adj, train, validation)
            torch.cuda.synchronize()
            validation_seconds = time.perf_counter() - start
            report['validation_evaluations'] += 1
            entry = dict(epoch=epoch, losses_mean=sums, validation=result,
                         train_seconds=train_seconds, validation_seconds=validation_seconds,
                         train_peak_allocated_bytes=train_memory, train_peak_reserved_bytes=train_reserved,
                         validation_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         validation_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            best_score, improved = save_best(best_path, student, epoch, result, metadata, best_score)
            entry['selected'] = improved
            report['curve'].append(entry)
            if improved:
                report['best_epoch'] = epoch
                report['best_validation'] = result
                report['best_checkpoint_sha256'] = sha256(best_path)
            save()
            print(f'Epoch {epoch}/{epochs}: train={train_seconds:.1f}s val={validation_seconds:.1f}s '
                  f'Recall@20={result["recall"][1]:.8f} best_epoch={report["best_epoch"]}', flush=True)
        if tensor_digest(teacher) != teacher_before or tensor_digest(prompt) != prompt_before:
            raise RuntimeError('Frozen teacher/prompt changed.')
        if tensor_digest(student) == student_before:
            raise RuntimeError('Student did not update.')
        if sha256(teacher_path) != SPORTS_TEACHER_SHA256:
            raise RuntimeError('Teacher file changed.')
        if report['optimizer_steps_completed'] != epochs * batches or report['validation_evaluations'] != epochs + 1:
            raise RuntimeError('Incomplete diagnostic budget.')
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != head:
            raise RuntimeError('Launch HEAD changed during diagnostic.')
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
            raise RuntimeError('Source tree became dirty during diagnostic.')
        for name, digest in report['source'].items():
            if sha256(ROOT/name) != digest:
                raise RuntimeError('Source changed during diagnostic: ' + name)
        report['completion_source_unchanged'] = True
        restored = restore_best(best_path, student)
        if restored['best_epoch'] != report['best_epoch'] or restored['validation_metrics'] != report['best_validation']:
            raise RuntimeError('Selected checkpoint metadata mismatch.')
        report.update(status='completed', teacher_unchanged=True, prompt_unchanged=True,
                      student_updated=True, alias_preserved=True, best_checkpoint_roundtrip=True,
                      best_checkpoint_path=str(best_path), best_checkpoint_sha256=sha256(best_path),
                      selection_ranking_repeated=False, resumable=False)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_at'] = datetime.now().astimezone().isoformat()
        save()
        print('Validation diagnostic report:', report_path, flush=True)
    return 0
