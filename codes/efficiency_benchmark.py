import os
from datetime import datetime
from time import time

import numpy as np
import torch


def _sync_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def _resolve_ckpt_path(repo_root, ckpt_path):
    if not ckpt_path:
        return ''
    ckpt_path = str(ckpt_path).strip()
    if not ckpt_path:
        return ''
    if os.path.isabs(ckpt_path):
        return ckpt_path
    return os.path.join(repo_root, ckpt_path)


def resolve_efficiency_student_ckpt(trainer, args):
    cli_path = _resolve_ckpt_path(trainer.repo_root, getattr(args, 'efficiency_student_ckpt', ''))
    if cli_path:
        return cli_path

    if args.student_model_type not in ('td_distill', 'td_distill_no_projection'):
        return ''

    infer_candidates = []
    full_candidates = []
    if os.path.isdir(trainer.td_distill_dir):
        for file_name in os.listdir(trainer.td_distill_dir):
            full_path = os.path.join(trainer.td_distill_dir, file_name)
            if not os.path.isfile(full_path):
                continue
            if file_name.startswith('td_distill_infer_only__') and file_name.endswith('.pth'):
                infer_candidates.append(full_path)
            elif file_name.startswith('td_distill_full__') and file_name.endswith('.pth'):
                full_candidates.append(full_path)

    if infer_candidates:
        infer_candidates.sort(key=os.path.getmtime, reverse=True)
        return infer_candidates[0]
    if full_candidates:
        full_candidates.sort(key=os.path.getmtime, reverse=True)
        return full_candidates[0]
    return ''


def maybe_load_student_checkpoint_for_efficiency(trainer, args):
    ckpt_path = resolve_efficiency_student_ckpt(trainer, args)
    if not ckpt_path:
        trainer.logger.logging('Efficiency benchmark: no student checkpoint provided/found; using current in-memory weights.')
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('Efficiency benchmark student checkpoint not found at %s' % ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location=trainer.device)
    state_dict = checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

    if args.student_model_type in ('td_distill', 'td_distill_no_projection'):
        if not isinstance(state_dict, dict):
            raise ValueError('Unsupported TD student checkpoint format: %s' % ckpt_path)
        compatible_state = {
            key: value for key, value in state_dict.items()
            if isinstance(key, str) and key.endswith('.weight')
        }
        load_result = trainer.td_distill_model.load_state_dict(compatible_state, strict=False)
    else:
        if not isinstance(state_dict, dict):
            raise ValueError('Unsupported student checkpoint format: %s' % ckpt_path)
        load_result = trainer.student_model.load_state_dict(state_dict, strict=False)

    trainer.logger.logging(
        'Efficiency benchmark loaded student checkpoint: %s (missing=%d, unexpected=%d)' % (
            ckpt_path,
            len(load_result.missing_keys),
            len(load_result.unexpected_keys),
        )
    )


def _get_embeddings_for_efficiency(trainer, args, is_teacher):
    if is_teacher:
        trainer.teacher_model.eval()
        trainer.prompt_module.eval()
        with torch.no_grad():
            u_embed, i_embed, *_ = trainer.teacher_model(trainer.ui_graph, trainer.iu_graph, trainer.prompt_module)
        return u_embed, i_embed

    if args.student_model_type in ('td_distill', 'td_distill_no_projection'):
        trainer.td_distill_model.eval()
        with torch.no_grad():
            return trainer.td_distill_model()

    trainer.teacher_model.eval()
    trainer.prompt_module.eval()
    trainer.student_model.eval()
    with torch.no_grad():
        trainer.u_final_embed, trainer.i_final_embed, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds \
            , _, _, _, _ = trainer.teacher_model(trainer.ui_graph, trainer.iu_graph, trainer.prompt_module)

        if args.student_model_type == 'lightgcn':
            return trainer.student_model(trainer.adj, image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds)
        if args.student_model_type == 'gcn':
            return trainer.student_model(trainer.u_final_embed, trainer.i_final_embed, trainer.ui_graph, trainer.iu_graph)
        if args.student_model_type == 'mlp':
            return trainer.student_model(trainer.u_final_embed, trainer.i_final_embed)
        if args.student_model_type == 'mm_light':
            return trainer.student_model(image_item_embeds, text_item_embeds, image_user_embeds, text_user_embeds)

    raise ValueError('Unsupported student_model_type for efficiency benchmark: %s' % args.student_model_type)


def _benchmark_pipeline(trainer, args, users_to_test, is_teacher):
    warmup_runs = max(0, int(getattr(args, 'efficiency_warmup_runs', 3)))
    measure_runs = max(1, int(getattr(args, 'efficiency_measure_runs', 10)))
    user_batch_size = max(1, int(getattr(args, 'efficiency_batch_size', 2048)))
    topk = max(1, int(getattr(args, 'efficiency_topk', 20)))
    total_runs = warmup_runs + measure_runs

    embed_times, rank_times, total_times, peak_mems = [], [], [], []
    users_count = len(users_to_test)

    for run_idx in range(total_runs):
        if trainer.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(trainer.device)

        _sync_cuda(trainer.device)
        t0 = time()
        u_embed, i_embed = _get_embeddings_for_efficiency(trainer, args, is_teacher=is_teacher)
        _sync_cuda(trainer.device)
        t1 = time()

        i_embed_t = torch.transpose(i_embed, 0, 1)
        effective_topk = min(topk, i_embed.size(0))
        for start_idx in range(0, users_count, user_batch_size):
            end_idx = min(start_idx + user_batch_size, users_count)
            batch_users = users_to_test[start_idx:end_idx]
            if not batch_users:
                continue
            batch_user_tensor = torch.as_tensor(batch_users, dtype=torch.long, device=trainer.device)
            batch_scores = torch.matmul(u_embed[batch_user_tensor], i_embed_t)
            _ = torch.topk(batch_scores, k=effective_topk, dim=1)

        _sync_cuda(trainer.device)
        t2 = time()

        if run_idx >= warmup_runs:
            embed_times.append(t1 - t0)
            rank_times.append(t2 - t1)
            total_times.append(t2 - t0)
            if trainer.device.type == 'cuda':
                peak_mems.append(torch.cuda.max_memory_allocated(trainer.device) / (1024.0 * 1024.0))

    module = trainer.teacher_model if is_teacher else (trainer.td_distill_model if args.student_model_type in ('td_distill', 'td_distill_no_projection') else trainer.student_model)
    param_count = _count_parameters(module)
    if is_teacher:
        param_count += _count_parameters(trainer.prompt_module)

    total_time_mean = float(np.mean(total_times))
    users_per_second = users_count / max(total_time_mean, 1e-12)

    return {
        'embed_time_ms_mean': float(np.mean(embed_times) * 1000.0),
        'rank_time_ms_mean': float(np.mean(rank_times) * 1000.0),
        'total_time_ms_mean': total_time_mean * 1000.0,
        'latency_ms_per_user': (total_time_mean / max(users_count, 1)) * 1000.0,
        'throughput_users_per_sec': users_per_second,
        'peak_memory_mb_max': float(max(peak_mems)) if peak_mems else 0.0,
        'parameter_count': int(param_count),
        'users_count': int(users_count),
        'warmup_runs': int(warmup_runs),
        'measure_runs': int(measure_runs),
        'batch_size': int(user_batch_size),
        'topk': int(topk),
    }


def run_efficiency_benchmark(trainer, args, data_generator):
    users_to_test = list(data_generator.test_set.keys())
    if not users_to_test:
        raise ValueError('Efficiency benchmark failed: users_to_test is empty.')

    teacher_metrics = _benchmark_pipeline(trainer, args, users_to_test, is_teacher=True)
    student_metrics = _benchmark_pipeline(trainer, args, users_to_test, is_teacher=False)

    speedup = teacher_metrics['total_time_ms_mean'] / max(student_metrics['total_time_ms_mean'], 1e-12)
    throughput_gain = student_metrics['throughput_users_per_sec'] / max(teacher_metrics['throughput_users_per_sec'], 1e-12)

    trainer.logger.logging('Efficiency teacher: total=%.3fms, embed=%.3fms, rank=%.3fms, latency/user=%.6fms, throughput=%.2f users/s, peak_mem=%.2fMB, params=%d' % (
        teacher_metrics['total_time_ms_mean'],
        teacher_metrics['embed_time_ms_mean'],
        teacher_metrics['rank_time_ms_mean'],
        teacher_metrics['latency_ms_per_user'],
        teacher_metrics['throughput_users_per_sec'],
        teacher_metrics['peak_memory_mb_max'],
        teacher_metrics['parameter_count'],
    ))
    trainer.logger.logging('Efficiency student: total=%.3fms, embed=%.3fms, rank=%.3fms, latency/user=%.6fms, throughput=%.2f users/s, peak_mem=%.2fMB, params=%d' % (
        student_metrics['total_time_ms_mean'],
        student_metrics['embed_time_ms_mean'],
        student_metrics['rank_time_ms_mean'],
        student_metrics['latency_ms_per_user'],
        student_metrics['throughput_users_per_sec'],
        student_metrics['peak_memory_mb_max'],
        student_metrics['parameter_count'],
    ))
    trainer.logger.logging('Efficiency relative: latency speedup=%.3fx, throughput gain=%.3fx' % (speedup, throughput_gain))

    efficiency_dir = os.path.join(trainer.repo_root, 'exp', 'efficiency', args.dataset)
    os.makedirs(efficiency_dir, exist_ok=True)
    benchmark_payload = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': trainer.run_name,
        'dataset': args.dataset,
        'student_model_type': args.student_model_type,
        'teacher': teacher_metrics,
        'student': student_metrics,
        'latency_speedup_x': speedup,
        'throughput_gain_x': throughput_gain,
    }
    benchmark_path = os.path.join(efficiency_dir, 'efficiency__%s.pkl' % trainer.run_name)
    trainer._save_pickle(benchmark_payload, benchmark_path)
    trainer.logger.logging('Efficiency benchmark artifact saved at %s' % benchmark_path)

    return benchmark_payload
