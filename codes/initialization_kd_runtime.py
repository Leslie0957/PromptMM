"""Launch contracts and artifact acceptance shared by smoke and future cohort."""
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys

from initialization_kd_adapter import sha256
from initialization_kd_interaction import ARMS, Protocol, arm_config, select_epochs


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment():
    import numpy
    import psutil
    import scipy
    import torch
    return {'python': platform.python_version(), 'executable': str(Path(sys.executable).resolve()),
            'torch': str(torch.__version__), 'numpy': numpy.__version__,
            'scipy': scipy.__version__, 'psutil': psutil.__version__}


def validate_environment(expected):
    actual = environment()
    if actual != expected:
        raise RuntimeError('Environment mismatch: ' + json.dumps(actual))
    return actual


def resolve_protocol(spec):
    """Every field used by the fixed implementation is checked, never overridden."""
    p = Protocol()
    common = spec['common']
    expected = {
        'dataset': 'sports', 'student': 'td_distill_no_projection',
        'dimensions': {'users': p.n_users, 'items': p.n_items, 'id': p.dim},
        'training': {'epochs': p.epochs, 'batches_per_epoch': p.batches_per_epoch,
                     'batch_size': p.batch_size, 'triplet_batches_per_arm': 64200,
                     'total_triplet_batches': 256800, 'early_stopping_disabled_before_300': True},
        'bpr': {'formula': 'mean(-logsigmoid(dot(u,p)-dot(u,n)))',
                'explicit_embedding_regularizer': 0.0,
                'parser_regs_inherited_but_unused_in_td_path': '[1e-5,1e-5,1e-2]'},
        'kd': {'item_image_rate': 1.0, 'item_text_rate': .3, 'user_image_rate': 0.0,
               'user_text_rate': 0.0, 'full_reduction': 'alpha * (image_mse_mean + 0.3*text_mse_mean)/1.3',
               'positive_items_only': True},
        'optimizer': {'type': 'AdamW', 'learning_rate': p.lr, 'weight_decay': p.weight_decay,
                      'betas': [.9, .999], 'eps': 1e-8, 'amsgrad': False,
                      'fresh_empty_moments_each_arm': True},
        'evaluation': {'split': 'validation_only', 'candidate_exclusion': 'train_only',
                       'Ks': [10, 20, 40, 50], 'test_flag': 'part', 'k': 20,
                       'every_epoch': True, 'epoch_zero_descriptive_only': True,
                       'primary': 'fixed_epoch300_recall20',
                       'secondary': ['earliest_max_validation_recall20', 'same_epoch_ndcg20', 'fixed_epoch_curve'],
                       'test_files_open_allowed': False, 'test_ranking_allowed': False,
                       'final_test_allowed': False}}
    if set(common) != set(expected) | {'assets'}:
        raise RuntimeError('Common contract keys mismatch')
    for key, value in expected.items():
        if common[key] != value:
            raise RuntimeError('Runtime/config mismatch: ' + key)
    allowed_seeds = (2022, 2023, 2024) if spec.get('mode') == 'four_arm' else (2022,)
    if spec['seed'] not in allowed_seeds or set(spec['arms']) != set(ARMS):
        raise RuntimeError('Seed/arm contract mismatch')
    for arm in ARMS:
        config = arm_config(arm, p)
        if spec['arms'][arm] != {key: config[key] for key in ('initial', 'objective', 'alpha')}:
            raise RuntimeError('Arm contract mismatch')
    if spec.get('mode') == 'resource_smoke':
        if spec['numerics'] != {'device': 'cuda:0', 'dtype': 'float32', 'tf32': False,
                                'deterministic_algorithms': True,
                                'cublas_workspace_config': ':4096:8', 'torch_cpu_threads': 4,
                                'pythonhashseed': '2022'}:
            raise RuntimeError('Numerics contract mismatch')
        if spec['hard_caps'] != {'parent_wall_seconds': 900, 'cuda_allocator_bytes': 2147483648,
                                 'process_rss_bytes': 4294967296, 'output_bytes': 268435456,
                                 'minimum_free_disk_bytes': 4294967296, 'attempts': 1}:
            raise RuntimeError('Smoke resource cap mismatch')
        if spec['numerics'] != {'device': 'cuda:0', 'dtype': 'float32', 'tf32': False,
                                'deterministic_algorithms': True,
                                'cublas_workspace_config': ':4096:8', 'torch_cpu_threads': 4,
                                'pythonhashseed': '2022'}:
            raise RuntimeError('Numerics contract mismatch')
        if spec['hard_caps'] != {'parent_wall_seconds': 900, 'cuda_allocator_bytes': 2147483648,
                                 'process_rss_bytes': 4294967296, 'output_bytes': 268435456,
                                 'minimum_free_disk_bytes': 4294967296, 'attempts': 1}:
            raise RuntimeError('Smoke resource cap mismatch')
        if spec['smoke'] != {'arm': 'T1', 'epochs': 1, 'batches_per_epoch': 8,
                             'epoch_zero_ranking': False, 'validation_calls': 1}:
            raise RuntimeError('Smoke contract mismatch')
        from dataclasses import replace
        return replace(p, epochs=1, batches_per_epoch=8)
    if spec.get('mode') == 'four_arm':
        if spec.get('evaluator') != 'exact_topk_v1':
            raise RuntimeError('Formal evaluator mismatch')
        if spec['numerics'] != {'device': 'cuda:0', 'dtype': 'float32', 'tf32': False,
                                'deterministic_algorithms': True,
                                'cublas_workspace_config': ':4096:8', 'torch_cpu_threads': 4,
                                'pythonhashseed': '2022'}:
            raise RuntimeError('Formal numerics mismatch')
        if spec['hard_caps'] != {'parent_wall_seconds': None, 'cuda_allocator_bytes': 2147483648,
                                 'process_rss_bytes': 8589934592, 'output_bytes': 1073741824,
                                 'minimum_free_disk_bytes': 4294967296, 'attempts': 1}:
            raise RuntimeError('Formal resource cap mismatch')
        if spec['candidate_screening'] != {
                'interaction_definition': '(R1-R0)-(T1-T0)',
                'primary_min_positive_interaction_absolute_recall20': .002,
                'candidate_noninferiority_margin_absolute_recall20': .001,
                'single_seed_inference': 'descriptive_only_no_significance_or_equivalence'}:
            raise RuntimeError('Formal screening threshold mismatch')
    return p


def bind_worker(output, expected_output, spec, head, token, parent_pid):
    """Bind internal entry to this parent's manifest, source and one-time claim."""
    output = Path(output)
    if output.resolve() != Path(expected_output).resolve():
        raise RuntimeError('Worker output mismatch')
    manifest = json.loads((output / 'launch_manifest.json').read_text(encoding='utf-8'))
    if (not token or manifest['token_sha256'] != hashlib.sha256(token.encode()).hexdigest()
            or parent_pid != manifest['parent_pid'] or manifest['source_commit'] != head
            or manifest['config_digest'] != digest_json(spec)):
        raise RuntimeError('Worker parent/source/config binding mismatch')
    # Exclusive claim persists even if the worker fails before loading tensors.
    with (output / 'worker_claim.json').open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'parent_pid': parent_pid}, stream)
    return manifest


def validate_artifacts(output, arms, protocol, source_commit, config_digest):
    """Reject exit0/partial/corrupt output; inspect CPU checkpoint state, no ranking."""
    import torch
    output = Path(output)
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report.get('status') != 'completed' or set(report.get('arms', {})) != set(arms)
            or report.get('source_commit') != source_commit
            or report.get('config_digest') != config_digest
            or report.get('test_file_reads') != 0 or report.get('selection_split') != 'Validation'):
        raise RuntimeError('Incomplete or mismatched final report')
    hashes = {'report.json': sha256(report_path)}
    for arm in arms:
        arm_report = report['arms'][arm]
        curve_path = output / arm / 'curve.json'
        curve = json.loads(curve_path.read_text(encoding='utf-8'))
        if len(curve) != protocol.epochs or [row['epoch'] for row in curve] != list(range(1, protocol.epochs + 1)):
            raise RuntimeError('Incomplete epoch curve')
        for row in curve:
            for key in ('recall20', 'ndcg20'):
                value = row['validation'][key]
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise RuntimeError('Invalid Validation metric')
        selection = select_epochs([row['validation']['recall20'] for row in curve])
        if (arm_report['selection'] != selection or arm_report['final_validation'] != curve[-1]['validation']
                or arm_report['optimizer_steps'] != protocol.epochs * protocol.batches_per_epoch):
            raise RuntimeError('Report/curve selection mismatch')
        hashes[curve_path.relative_to(output).as_posix()] = sha256(curve_path)
        for label, index in (('best', selection['best_index']), ('final', protocol.epochs - 1)):
            relative = f'{arm}/{label}.pt'
            if arm_report[label + '_checkpoint'] != relative:
                raise RuntimeError('Checkpoint path mismatch')
            checkpoint = torch.load(output / relative, weights_only=True, map_location='cpu')
            if (checkpoint['arm'] != arm or checkpoint['epoch'] != index + 1
                    or checkpoint['metric'] != curve[index]['validation']
                    or checkpoint['source_commit'] != source_commit
                    or checkpoint['config_digest'] != config_digest):
                raise RuntimeError('Checkpoint identity/epoch mismatch')
            model = checkpoint['model']
            shapes = {'user_id_embedding.weight': (protocol.n_users, protocol.dim),
                      'item_id_embedding.weight': (protocol.n_items, protocol.dim)}
            if set(model) != set(shapes):
                raise RuntimeError('Checkpoint parameter keys mismatch')
            for key, shape in shapes.items():
                if model[key].shape != shape or model[key].dtype != torch.float32 or not torch.isfinite(model[key]).all():
                    raise RuntimeError('Checkpoint tensor invalid')
            optim = checkpoint['optimizer']
            if len(optim['param_groups']) != 1 or len(optim['state']) != 2:
                raise RuntimeError('Optimizer state count mismatch')
            group = optim['param_groups'][0]
            for key, value in {'lr': protocol.lr, 'weight_decay': protocol.weight_decay,
                               'eps': 1e-8, 'betas': (.9, .999), 'amsgrad': False}.items():
                if group[key] != value:
                    raise RuntimeError('Optimizer setting mismatch: ' + key)
            if len(group['params']) != 2 or set(group['params']) != set(optim['state']):
                raise RuntimeError('Optimizer parameter binding mismatch')
            for param_id, shape in zip(group['params'], shapes.values()):
                state = optim['state'][param_id]
                if float(state['step']) != (index + 1) * protocol.batches_per_epoch:
                    raise RuntimeError('Optimizer step mismatch')
                for key in ('exp_avg', 'exp_avg_sq'):
                    if state[key].shape != shape or not torch.isfinite(state[key]).all():
                        raise RuntimeError('Optimizer moments invalid')
            hashes[relative] = sha256(output / relative)
            del checkpoint
    if tuple(arms) == ARMS:
        from initialization_kd_interaction import interaction
        if report.get('interaction') != interaction({a: report['arms'][a]['final_validation']['recall20'] for a in arms}):
            raise RuntimeError('Interaction mismatch')
    return hashes
