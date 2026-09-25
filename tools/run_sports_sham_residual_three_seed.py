"""One-attempt manual Sports real/sham residual cohort; preflight before training."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
import torch

from sports_paired_cold import sha256
from sports_sham_residual import (
    COHORT, PAIR_SHA, SHAM_SEEDS, TEACHER_ASSET_SHA, geometry_error,
    replay_session, sham_text, teacher_tensors,
)
from utility.dataset_profiles import SPORTS_SHAM_RESIDUAL_PROFILES, SPORTS_STUDENT_PROFILES
from run_sports_validation300 import verify_outcome

OUT = ROOT / COHORT
REPORT = OUT / 'batch.json'
PREFLIGHT = OUT / 'preflight.json'
PRIOR_BATCH = ROOT / 'exp/paired_coldinit/sports_three_seed_v1/batch.json'
PRIOR_BATCH_SHA = 'b254d9e969edd3e001a4a85c2b14b69f8eb68b3d279d667100973c62fe7797b8'
TRAIN_SHA = '5361c5486dddbf50084d011278a11d2f18a37e258f87be218b070ff02f4ab0f8'
TEACHER_SHA = '57673a54603e2680416925d0c11b24d045b0ef14870feee292011032ea33abea'
CONVERSION_SHA = '3772a17c8b70fa4739653534dca8d542e67e0649f1f44ccba4194fec17bc1104'
PROFILES = [f'sports_student_{arm}_textresidual_seed{seed}_val300_v1'
            for seed in (2022, 2023, 2024) for arm in ('real', 'sham')]
RUNS = ROOT / 'exp/runs/sports'


def commands(python):
    return [[str(python), '-B', 'codes/main_mmlight.py', '--dataset', 'sports',
             '--student_profile', name, '--gpu_id', '0'] for name in PROFILES]


def check_source(head):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)
    if actual != head or dirty.strip():
        raise RuntimeError('Committed clean launch source changed')


def gradient_scale(initial, image, text, sham, batches):
    """No optimizer or .grad writes; eight saved train-only batches."""
    from td_distill_model_no_projection import bpr_loss, directional_distillation_loss
    users = initial['user'].detach().clone().requires_grad_(True)
    items = initial['item'].detach().clone().requires_grad_(True)
    real_squares, sham_squares, image_squares, bpr_squares = [], [], [], []
    for batch in batches:
        user_ids, positive_ids, negative_ids = [torch.as_tensor(x, dtype=torch.long) for x in batch]
        image_loss = directional_distillation_loss(items[positive_ids], image[positive_ids])
        text_loss = directional_distillation_loss(items[positive_ids], text[positive_ids])
        sham_loss = directional_distillation_loss(items[positive_ids], sham[positive_ids])
        real = 0.3 * (image_loss + 0.3 * text_loss) / 1.3
        control = 0.3 * (image_loss + 0.3 * sham_loss) / 1.3
        bpr = bpr_loss(users[user_ids], items[positive_ids], items[negative_ids])
        gradients = [torch.autograd.grad(loss, items, retain_graph=True)[0]
                     for loss in (real, control, 0.3 * image_loss / 1.3)]
        gradients.append(torch.autograd.grad(bpr, items)[0])
        if not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Nonfinite zero-update gradient')
        for destination, value in zip((real_squares, sham_squares, image_squares, bpr_squares), gradients):
            destination.append(float(value.double().square().sum()))
    if users.grad is not None or items.grad is not None:
        raise RuntimeError('Zero-update preflight modified .grad buffers')
    if not torch.equal(users.detach(), initial['user']) or not torch.equal(items.detach(), initial['item']):
        raise RuntimeError('Zero-update preflight modified initial tensors')
    rms = [math.sqrt(sum(values) / len(values)) for values in
           (real_squares, sham_squares, image_squares, bpr_squares)]
    if not all(math.isfinite(x) and x > 0 for x in rms):
        raise RuntimeError('Invalid zero-update gradient RMS')
    return {'real_distill_item_rms': rms[0], 'sham_distill_item_rms': rms[1],
            'image_item_rms': rms[2], 'bpr_item_rms': rms[3],
            'sham_over_real_rms': rms[1] / rms[0], 'batches': len(batches)}


def preflight():
    if sha256(PRIOR_BATCH) != PRIOR_BATCH_SHA:
        raise RuntimeError('Prior paired batch report identity changed')
    previous = json.loads(PRIOR_BATCH.read_text(encoding='utf-8'))
    if previous['status'] != 'completed' or len(previous['completed']) != 6:
        raise RuntimeError('Prior strict pair incomplete')
    if sha256(ROOT / 'data/sports/train_mat') != TRAIN_SHA:
        raise RuntimeError('Sports training matrix changed')
    with (ROOT / 'data/sports/train_mat').open('rb') as stream:
        train_matrix = pickle.load(stream)
    if train_matrix.shape != (35598, 18357) or train_matrix.nnz != 218409:
        raise RuntimeError('Sports training matrix structure changed')
    positive_item_support = torch.as_tensor(train_matrix.getnnz(axis=0) > 0)
    teacher_path = ROOT / SPORTS_STUDENT_PROFILES[PROFILES[0]]['defaults']['teacher_checkpoint']
    if sha256(teacher_path) != TEACHER_SHA:
        raise RuntimeError('Frozen teacher changed')
    frozen = teacher_tensors(ROOT)
    output = {'status': 'preparing', 'teacher_asset_sha256': TEACHER_ASSET_SHA,
              'prior_batch_sha256': PRIOR_BATCH_SHA, 'train_sha256': TRAIN_SHA,
              'teacher_checkpoint_sha256': TEACHER_SHA, 'seeds': {}}
    for seed in (2022, 2023, 2024):
        old = [x for x in previous['completed'] if x['seed'] == seed]
        if len(old) != 2 or old[0]['initial']['sha256'] != PAIR_SHA[seed][0] or old[1]['initial'] != old[0]['initial'] or old[0]['triplets']['sha256'] != PAIR_SHA[seed][1] or old[1]['triplets'] != old[0]['triplets']:
            raise RuntimeError('Prior same-seed initial/tape record mismatch')
        session = replay_session(ROOT, seed)
        first_eight = session.tape[0, :8].copy()
        del session.tape
        initial = torch.load(session.initial, map_location='cpu', weights_only=True)
        if set(initial) != {'user', 'item'} or initial['user'].shape != (35598, 64) or initial['item'].shape != (18357, 64):
            raise RuntimeError('Prior initial ID tensor shape mismatch')
        target, valid = sham_text(frozen['image_items'], frozen['text_items'], SHAM_SEEDS[seed])
        if not bool(valid[positive_item_support].all()):
            raise RuntimeError('Some train-positive items have degenerate text residuals')
        geometry = geometry_error(frozen['image_items'], frozen['text_items'], target, valid)
        if geometry['valid_rows'] < 18000 or max(geometry[k] for k in (
                'max_text_norm_error', 'max_image_text_cosine_error',
                'max_combined_norm_error')) > 2e-5:
            raise RuntimeError('Sham geometry hard gate failed: %r' % geometry)
        scale = gradient_scale(initial, frozen['image_items'], frozen['text_items'], target, first_eight)
        if not 0.95 <= scale['sham_over_real_rms'] <= 1.05:
            raise RuntimeError('Sham gradient RMS hard gate failed: %r' % scale)
        seed_dir = OUT / ('seed%d' % seed)
        seed_dir.mkdir(exist_ok=False)
        target_path = seed_dir / 'sham_text.pt'
        with target_path.open('xb') as stream:
            torch.save({'item_text': target, 'seed': SHAM_SEEDS[seed],
                        'teacher_asset_sha256': TEACHER_ASSET_SHA}, stream)
        output['seeds'][str(seed)] = {
            'initial_sha256': PAIR_SHA[seed][0], 'tape_sha256': PAIR_SHA[seed][1],
            'sham_seed': SHAM_SEEDS[seed], 'sham_target': str(target_path),
            'sham_target_sha256': sha256(target_path),
            'geometry': geometry, 'gradient_scale': scale,
        }
    output['status'] = 'passed'
    with PREFLIGHT.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
    return output


def verify(path, profile, preflight_record):
    manifest = json.loads(path.read_text(encoding='utf-8'))
    verify_outcome(manifest, profile)
    seed, arm = SPORTS_SHAM_RESIDUAL_PROFILES[profile]
    args = manifest['resolved_arguments']
    for key, value in SPORTS_STUDENT_PROFILES[profile]['defaults'].items():
        if args.get(key) != value:
            raise RuntimeError('Resolved argument mismatch: ' + key)
    if (manifest.get('sham_residual_seed'), manifest.get('sham_residual_arm')) != (seed, arm):
        raise RuntimeError('Sham profile identity mismatch')
    if manifest.get('paper_ready_eligible') is not False or manifest['evaluation_protocol'] != 'val_test_once_v1' or manifest['selection_split'] != 'validation':
        raise RuntimeError('Validation-only protocol mismatch')
    if manifest['teacher_checkpoint_fingerprint']['sha256'] != TEACHER_SHA or manifest['dataset_identity']['matrices']['train']['sha256'] != TRAIN_SHA or manifest['dataset_identity']['conversion_manifest']['sha256'] != CONVERSION_SHA:
        raise RuntimeError('Teacher/data anchor mismatch')
    record = preflight_record['seeds'][str(seed)]
    semantic = manifest['sham_residual_semantics']
    if semantic['teacher_asset_sha256'] != TEACHER_ASSET_SHA or semantic['sham_target_sha256'] != record['sham_target_sha256'] or semantic['arm'] != arm:
        raise RuntimeError('Teacher/sham semantic source mismatch')
    if sha256(semantic['sham_target']) != record['sham_target_sha256'] or sha256(semantic['teacher_asset']) != TEACHER_ASSET_SHA:
        raise RuntimeError('Semantic asset changed')
    if manifest['paired_cold_initial']['sha256'] != PAIR_SHA[seed][0] or manifest['paired_cold_triplets']['sha256'] != PAIR_SHA[seed][1] or manifest['paired_cold_triplets']['triplet_batches_consumed'] != 64200:
        raise RuntimeError('Initial/tape replay mismatch')
    if manifest['paired_cold_triplets']['shape'] != [300, 214, 3, 1024] or sha256(manifest['paired_cold_initial']['path']) != PAIR_SHA[seed][0] or sha256(manifest['paired_cold_triplets']['path']) != PAIR_SHA[seed][1]:
        raise RuntimeError('Initial/tape artifact changed')
    for fingerprint in manifest['code_fingerprints'].values():
        if sha256(fingerprint['path']) != fingerprint['sha256']:
            raise RuntimeError('Source fingerprint changed')
    with Path(manifest['artifacts']['converge_run']).open('rb') as stream:
        curve = pickle.load(stream)
    for key in ('td_batch_loss_List', 'td_bpr_loss_List', 'td_distill_loss_List',
                'selection_recall20_List', 'selection_ndcg20_List'):
        if len(curve[key]) != 300 or not all(math.isfinite(float(x)) for x in curve[key]):
            raise RuntimeError('Incomplete/nonfinite curve: ' + key)
    best = max(range(300), key=lambda index: curve['selection_recall20_List'][index])
    if best != manifest['best_selection_epoch'] or not math.isclose(float(curve['selection_recall20_List'][best]), manifest['best_selection_recall'], rel_tol=0, abs_tol=1e-12):
        raise RuntimeError('Wrong Validation checkpoint selection')
    checkpoint = Path(manifest['td_full_checkpoint'])
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['best_epoch'] != best or state['td_init_from_teacher'] is not False or not all(torch.isfinite(v).all() for v in state['model_state_dict'].values()):
        raise RuntimeError('Invalid selected checkpoint')
    infer = torch.load(manifest['td_infer_checkpoint'], map_location='cpu', weights_only=False)
    for key in ('user_id_embedding.weight', 'item_id_embedding.weight'):
        if not torch.equal(infer[key], state['model_state_dict'][key]):
            raise RuntimeError('Inference export differs from selected checkpoint')
    return {'profile': profile, 'seed': seed, 'arm': arm, 'manifest': str(path),
            'manifest_sha256': sha256(path), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': sha256(checkpoint), 'best_epoch': best,
            'best_recall20': float(curve['selection_recall20_List'][best]),
            'best_ndcg20': float(curve['selection_ndcg20_List'][best]),
            'initial_sha256': PAIR_SHA[seed][0], 'tape_sha256': PAIR_SHA[seed][1],
            'teacher_asset_sha256': TEACHER_ASSET_SHA,
            'sham_target_sha256': record['sham_target_sha256']}


def run(python=sys.executable):
    if Path(python).resolve() != Path('D:/miniconda/envs/run_5060/python.exe').resolve():
        raise RuntimeError('Declared run_5060 Python required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    check_source(head)
    if OUT.exists():
        raise FileExistsError('Existing sham cohort attempt; no retry: ' + str(OUT))
    for path in RUNS.glob('run_manifest__*.json'):
        if json.loads(path.read_text(encoding='utf-8')).get('student_config_profile') in PROFILES:
            raise FileExistsError('Existing sham profile attempt: ' + str(path))
    OUT.mkdir(parents=True, exist_ok=False)
    report = {'status': 'started', 'started_at': datetime.now().astimezone().isoformat(),
              'launch_commit': head, 'launch_dirty': False, 'python': str(python),
              'launcher_sha256': sha256(__file__),
              'prior_batch_sha256': PRIOR_BATCH_SHA,
              'teacher_semantic_asset_sha256': TEACHER_ASSET_SHA,
              'commands': commands(python), 'completed': []}
    def save():
        REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    with REPORT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    try:
        report['stage'] = 'zero_update_preflight'
        save()
        preflight_record = preflight()
        report['preflight_sha256'] = sha256(PREFLIGHT)
        for profile, command in zip(PROFILES, commands(python)):
            check_source(head)
            if sha256(PREFLIGHT) != report['preflight_sha256']:
                raise RuntimeError('Preflight report changed')
            before = set(RUNS.glob('run_manifest__*.json'))
            report.update(stage='training', active_profile=profile)
            save()
            print('Starting', profile, flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            check_source(head)
            created = set(RUNS.glob('run_manifest__*.json')) - before
            if len(created) != 1:
                raise RuntimeError('Expected exactly one run manifest')
            result = verify(created.pop(), profile, preflight_record)
            if result['arm'] == 'sham':
                real = report['completed'][-1]
                for field in ('seed', 'initial_sha256', 'tape_sha256',
                              'teacher_asset_sha256', 'sham_target_sha256'):
                    if real[field] != result[field]:
                        raise RuntimeError('Real/sham pair differs: ' + field)
            report['completed'].append(result)
            save()
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['updated_at'] = datetime.now().astimezone().isoformat()
        save()
        print('Batch report:', REPORT, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        for command in commands(sys.executable):
            print(subprocess.list2cmdline(command))
    else:
        run()


if __name__ == '__main__':
    main()
