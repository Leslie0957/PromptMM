"""Isolated Innovation1 selected-student Validation replay and one-time Test."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import traceback

from promptmm_release_resource import ROOT, sha256, tensor_digest

LEDGER = ROOT / 'docs/research/INNOVATION1_REUSE_ASSETS_2026-09-26.json'
KS = (10, 20, 40, 50)
SLOTS = tuple(f'{arm}_{seed}' for arm in ('b3', 's1', 's2', 's3')
              for seed in (2022, 2023, 2024))


def now():
    return datetime.now().astimezone().isoformat()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def clean_head():
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Clean committed source required.')
    return head


def slot_source(slot, ledger, cohort_id=None):
    arm, seed = slot.split('_')
    seed = int(seed)
    if arm == 'b3':
        baby_root = ROOT / 'exp/promptmm_release_baby'
        if cohort_id:
            baby_root = baby_root / cohort_id
        base = baby_root / f'baby_promptmm_release_cap1000_patience7_seed{seed}_lr6e5_v1'
        return dict(dataset='baby', method='promptmm_release', seed=seed,
                    source_path=base/'report.json', checkpoint_path=base/'best.pt',
                    infer_path=None, original=None, cohort_id=cohort_id)
    method = {'s1': 'bpr', 's2': 'full', 's3': 'promptmm_release'}[arm]
    matches = [x for x in ledger['sports'] if x['seed'] == seed and x['method'] == method]
    if len(matches) != 1:
        raise RuntimeError('Source ledger must select exactly one Sports record.')
    selected = matches[0]
    assets = selected['assets']
    if method == 'promptmm_release':
        report_asset, best_asset = assets
        infer_asset = None
    else:
        report_asset, best_asset, infer_asset = assets
    for asset in assets:
        path = ROOT / asset['path']
        if not path.is_file() or sha256(path) != asset['sha256']:
            raise RuntimeError('Selected Sports asset drift: ' + asset['path'])
    return dict(dataset='sports', method=method, seed=seed,
                source_path=ROOT/report_asset['path'], checkpoint_path=ROOT/best_asset['path'],
                infer_path=ROOT/infer_asset['path'] if infer_asset else None, original=selected)


def load_inputs(source, ledger):
    """Train and Validation only. Never refer to test_mat here."""
    dataset = source['dataset']
    identity = (source['original']['data_identity'] if dataset == 'sports'
                else ledger['baby_existing_formal'][0]['data_identity'])
    data_dir = ROOT / 'data' / dataset
    matrices = identity['matrices']
    for filename, key in (('train_mat', 'train'), ('val_mat', 'validation')):
        if sha256(data_dir/filename) != matrices[key]['sha256']:
            raise RuntimeError('Data fingerprint drift: ' + filename)
    with (data_dir/'train_mat').open('rb') as handle:
        train = pickle.load(handle).tocsr()
    with (data_dir/'val_mat').open('rb') as handle:
        val = pickle.load(handle).tocsr()
    if (train.shape != val.shape or train.multiply(val).nnz
            or train.shape != tuple(matrices['train']['shape'])):
        raise RuntimeError('Train/Validation shape or overlap mismatch.')
    return train, val, identity


def check_source(source, identity):
    report = json.loads(source['source_path'].read_text(encoding='utf-8'))
    checkpoint_hash = sha256(source['checkpoint_path'])
    if source['dataset'] == 'baby':
        if (report.get('status') != 'validation_completed'
                or report.get('cohort_id') != source.get('cohort_id')
                or report.get('dataset') != 'baby'
                or report.get('identity') != 'PromptMM-release-Baby-sharedTeacher-v1'
                or report.get('config', {}).get('learning_rate') != 6e-5
                or report.get('early_stopping_patience') != 7
                or report.get('epochs_cap') != 1000
                or report.get('test_evaluations') != 0
                or report.get('test_split_loaded') is not False
                or report.get('best_checkpoint_sha256') != checkpoint_hash):
            raise RuntimeError('Baby release source is incomplete or mismatched.')
        if report['input_sha256']['train_mat'] != identity['matrices']['train']['sha256'] or \
           report['input_sha256']['val_mat'] != identity['matrices']['validation']['sha256']:
            raise RuntimeError('Baby source split identity mismatch.')
        if report['shared_teacher']['sha256'] != identity_teacher_sha('baby'):
            raise RuntimeError('Baby source teacher mismatch.')
    elif source['method'] == 'promptmm_release':
        if (report.get('status') != 'completed' or report.get('test_evaluations') != 0
                or report.get('test_split_loaded') is not False
                or report.get('best_checkpoint_sha256') != checkpoint_hash
                or report.get('config', {}).get('learning_rate') != 6e-5):
            raise RuntimeError('Sports release source is incomplete or mismatched.')
    else:
        if (report.get('status') != 'validation_completed'
                or report.get('final_test_performed') is not False
                or report.get('selection_split') != 'validation'
                or report.get('run_final_test') is not False
                or report.get('best_selection_epoch') != source['original']['best_epoch_zero_based']
                or report.get('student_config_profile') != source['original']['profile']):
            raise RuntimeError('Sports TD source is incomplete or mismatched.')
    return report, checkpoint_hash


def identity_teacher_sha(dataset):
    return ('b1c7eb9bb2af741924868a61b758bc4d1e2a7a92c9cf2906bf60a32db9b69bd4'
            if dataset == 'baby' else
            '57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea')


def selected_embeddings(source, report, train, device):
    import torch
    if source['method'] != 'promptmm_release':
        full = torch.load(source['checkpoint_path'], map_location='cpu', weights_only=False)
        infer = torch.load(source['infer_path'], map_location='cpu', weights_only=False)
        if (full.get('student_model_type') != 'td_distill_no_projection'
                or full.get('best_epoch') != report['best_selection_epoch']
                or abs(float(full['best_selection_recall']) - float(report['best_selection_recall'])) > 1e-12):
            raise RuntimeError('Selected TD full checkpoint metadata mismatch.')
        state = full['model_state_dict']
        for side, size in (('user', train.shape[0]), ('item', train.shape[1])):
            key = f'{side}_id_embedding.weight'
            if (key not in state or key not in infer
                    or state[key].shape != (size, 64)
                    or not torch.equal(state[key], infer[key])
                    or not torch.isfinite(infer[key]).all()):
                raise RuntimeError('TD full/infer selected tensor mismatch: ' + key)
        return (infer['user_id_embedding.weight'].to(device),
                infer['item_id_embedding.weight'].to(device),
                dict(full_checkpoint_sha256=sha256(source['checkpoint_path']),
                     infer_checkpoint_sha256=sha256(source['infer_path']),
                     selected_epoch=full['best_epoch'],
                     selected_recall=float(full['best_selection_recall'])))
    from promptmm_release import ReleaseStudent, release_graphs, sparse_tensor
    from promptmm_release_validation import restore_best
    saved = torch.load(source['checkpoint_path'], map_location='cpu', weights_only=False)
    config = report['config']
    state = saved['model_state_dict']
    users = state['user_id_embedding.weight']
    items = state['item_id_embedding.weight']
    if (users.shape != (train.shape[0], 64) or items.shape != (train.shape[1], 64)
            or saved['best_epoch'] != report['best_epoch']
            or saved['validation_metrics'] != report['best_validation']
            or config['layers'] != 1):
        raise RuntimeError('Selected release checkpoint metadata mismatch.')
    student = ReleaseStudent(train.shape[0], train.shape[1], 64, config['layers']).to(device)
    student.init_user_item_embed(users.to(device), items.to(device))
    restored = restore_best(source['checkpoint_path'], student)
    student.eval()
    adj = sparse_tensor(release_graphs(train)[2], device)
    with torch.no_grad():
        ue, ie = student(adj)
    if (not torch.isfinite(ue).all() or not torch.isfinite(ie).all()
            or restored['student_digest'] != tensor_digest(student)):
        raise RuntimeError('Release restored embeddings are nonfinite or altered.')
    return ue, ie, dict(full_checkpoint_sha256=sha256(source['checkpoint_path']),
                        selected_epoch=saved['best_epoch'],
                        selected_recall=float(saved['best_selection_recall']),
                        restored_student_digest=restored['student_digest'], alias_preserved=True)


def evaluate_embeddings(ue, ie, train, truth, *, ks=KS, batch_size=256):
    """Legacy train-only candidate order and metric definitions; no global Data loader."""
    import numpy as np
    import torch
    from promptmm_validation_fast import rank_fast, accumulate
    users = np.flatnonzero(truth.getnnz(axis=1)).tolist()
    if not users or truth.shape != train.shape or train.multiply(truth).nnz:
        raise RuntimeError('Invalid held-out user set, shape or Train overlap.')
    if ue.shape != (train.shape[0], 64) or ie.shape != (train.shape[1], 64):
        raise RuntimeError('Embedding shape mismatch.')
    totals = {k: np.zeros(len(ks), dtype=np.float64)
              for k in ('recall', 'precision', 'ndcg', 'hit_ratio')}
    all_items = set(range(train.shape[1]))
    with torch.no_grad():
        for offset in range(0, len(users), batch_size):
            batch = users[offset:offset + batch_size]
            scores = (ue[batch] @ ie.T).cpu().numpy()
            if not np.isfinite(scores).all():
                raise FloatingPointError('Nonfinite ranking score.')
            for row, user in enumerate(batch):
                candidates = list(all_items - set(train.indices[train.indptr[user]:train.indptr[user+1]]))
                if len(candidates) < max(ks):
                    raise RuntimeError('Too few ranking candidates.')
                positive = set(truth.indices[truth.indptr[user]:truth.indptr[user+1]])
                accumulate(totals, rank_fast(scores[row], candidates, max(ks)),
                           positive, ks, len(users))
    if not all(np.isfinite(v).all() for v in totals.values()):
        raise FloatingPointError('Nonfinite ranking metric.')
    return {k: v.tolist() for k, v in totals.items()}, len(users)


def expected_validation_recall(source, report):
    if source['method'] == 'promptmm_release':
        return float(report['best_validation']['recall'][1])
    return float(report['best_selection_recall'])


def run(slot, mode, gpu_id=0, cohort_id=None):
    if mode not in ('preflight', 'final-test-once'):
        raise ValueError('Invalid mode.')
    head = clean_head()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    ledger_hash = sha256(LEDGER)
    ledger = json.loads(LEDGER.read_text(encoding='utf-8'))
    source = slot_source(slot, ledger, cohort_id)
    family = ROOT / 'exp' / ('formal_closeout_preflight' if mode == 'preflight'
                           else 'formal_closeout_eval')
    base = family / cohort_id / slot if cohort_id else family / slot
    if mode == 'final-test-once':
        preflight_root = ROOT / 'exp/formal_closeout_preflight'
        preflight_path = ((preflight_root / cohort_id / slot) if cohort_id else
                          (preflight_root / slot)) / 'report.json'
        preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
        if (preflight.get('status') != 'passed' or preflight.get('launch_commit') != head
                or preflight.get('cohort_id') != cohort_id
                or preflight.get('ledger_sha256') != ledger_hash):
            raise RuntimeError('Successful same-source preflight required before Test.')
    base.mkdir(parents=True, exist_ok=False)
    report_path = base / 'report.json'
    result = dict(slot=slot, mode=mode, cohort_id=cohort_id, status='started', started_at=now(),
                  launch_commit=head, ledger_sha256=ledger_hash,
                  command=[sys.executable, *sys.argv], teacher_test_evaluations=0,
                  student_test_evaluations=0, test_split_loaded=False,
                  test_access_started=False, paper_ready_eligible=False,
                  source_files_sha256={str(path.relative_to(ROOT)): sha256(path)
                                       for path in (Path(__file__),
                                                    ROOT/'codes/promptmm_release.py',
                                                    ROOT/'codes/promptmm_release_validation.py',
                                                    ROOT/'codes/promptmm_validation_fast.py',
                                                    ROOT/'codes/utility/metrics.py')})
    def save():
        atomic_json(report_path, result)
    save()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no implicit CPU fallback.')
        device = torch.device('cuda:0')
        result['environment'] = dict(python=sys.version, torch=torch.__version__,
                                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
        train, val, identity = load_inputs(source, ledger)
        source_report, checkpoint_hash = check_source(source, identity)
        result.update(dataset=source['dataset'], method=source['method'], seed=source['seed'],
                      source_path=str(source['source_path']),
                      source_sha256=sha256(source['source_path']),
                      selected_checkpoint_path=str(source['checkpoint_path']),
                      selected_checkpoint_sha256=checkpoint_hash,
                      source_selected_epoch=(source_report.get('best_epoch') if
                                             source['method'] == 'promptmm_release' else
                                             source_report['best_selection_epoch']),
                      source_selection_recall=expected_validation_recall(source, source_report),
                      train_sha256=identity['matrices']['train']['sha256'],
                      validation_sha256=identity['matrices']['validation']['sha256'],
                      test_expected_sha256=identity['matrices']['test']['sha256'],
                      candidate_exclusion='train_only', ks=list(KS),
                      metrics_implementation_sha256=sha256(ROOT/'codes/utility/metrics.py'),
                      ranking_implementation_sha256=sha256(ROOT/'codes/promptmm_validation_fast.py'))
        save()
        ue, ie, selected = selected_embeddings(source, source_report, train, device)
        result['selected_state'] = selected
        validation, count = evaluate_embeddings(ue, ie, train, val)
        result['validation_replay'] = validation
        result['validation_users'] = count
        result['validation_recall_delta'] = abs(validation['recall'][1] - result['source_selection_recall'])
        if result['validation_recall_delta'] > 1e-10:
            raise RuntimeError('Selected checkpoint Validation replay differs from source.')
        if mode == 'preflight':
            result['status'] = 'passed'
        else:
            for key in ('source_sha256', 'selected_checkpoint_sha256', 'source_selected_epoch'):
                if result[key] != preflight[key]:
                    raise RuntimeError('Source or selected checkpoint changed after preflight.')
            if result['validation_recall_delta'] > preflight['validation_recall_delta'] + 1e-12:
                raise RuntimeError('Validation parity worsened after preflight.')
            # Persist the irreversible attempt BEFORE reading or hashing Test bytes.
            result['test_access_started'] = True
            result['test_access_started_at'] = now()
            result['student_test_attempts'] = 1
            save()
            test_path = ROOT / 'data' / source['dataset'] / 'test_mat'
            if sha256(test_path) != identity['matrices']['test']['sha256']:
                raise RuntimeError('Test fingerprint mismatch.')
            with test_path.open('rb') as handle:
                test = pickle.load(handle).tocsr()
            if test.shape != val.shape or val.multiply(test).nnz:
                raise RuntimeError('Test shape or Validation overlap mismatch.')
            result['test_split_loaded'] = True
            result['test_sha256'] = identity['matrices']['test']['sha256']
            save()
            test_result, users = evaluate_embeddings(ue, ie, train, test)
            result['test_result'] = test_result
            result['test_users'] = users
            result['student_test_evaluations'] = 1
            result['evaluated_checkpoint_sha256'] = sha256(source['checkpoint_path'])
            if result['evaluated_checkpoint_sha256'] != checkpoint_hash:
                raise RuntimeError('Selected checkpoint changed during Test.')
            result['status'] = 'completed'
        if clean_head() != head:
            raise RuntimeError('Source commit changed during evaluation.')
        if sha256(LEDGER) != ledger_hash:
            raise RuntimeError('Source ledger changed during evaluation.')
        for name, digest in result['source_files_sha256'].items():
            if sha256(ROOT/name) != digest:
                raise RuntimeError('Evaluator source file changed: ' + name)
        if result['status'] == 'completed':
            result['paper_ready_eligible'] = True
    except BaseException as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        result['traceback'] = traceback.format_exc()
        raise
    finally:
        result['finished_at'] = now()
        save()
        print(f'{mode} report: {report_path}', flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--final-test-once', action='store_true')
    parser.add_argument('--slot', required=True, choices=SLOTS)
    parser.add_argument('--gpu_id', type=int, choices=[0], default=0)
    parser.add_argument('--cohort-id', choices=['innovation1_fixed_v2'])
    args = parser.parse_args(argv)
    return run(args.slot, 'preflight' if args.preflight else 'final-test-once',
               args.gpu_id, args.cohort_id)


if __name__ == '__main__':
    raise SystemExit(main())
