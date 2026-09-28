"""Sealed user-manual seed2023 cohort; import and spec resolution do no asset I/O."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from initialization_kd_adapter import require_hash
from initialization_kd_runtime import resolve_protocol
from run_initialization_kd_cohort import load_spec as seed2022_spec
from run_initialization_kd_interaction import main

DELTA = ROOT / 'docs/research/INITIALIZATION_KD_SEED2023_COHORT_CONFIG_2026-09-28.json'
BASE = ROOT / 'docs/research/INITIALIZATION_KD_COHORT_CONFIG_2026-09-27.json'
PREFLIGHT = ROOT / 'docs/research/INITIALIZATION_KD_SEED2023_PREFLIGHT_2026-09-28.json'


def load_spec():
    delta = json.loads(DELTA.read_text(encoding='utf-8'))
    if set(delta) != {'base_cohort_config', 'base_cohort_sha256', 'asset_preflight',
                      'asset_preflight_sha256', 'seed', 'random_initial', 'triplet_tape',
                      'run_id', 'output_namespace', 'launch_command'}:
        raise RuntimeError('Seed2023 delta keys mismatch')
    if (delta['base_cohort_config'] != BASE.relative_to(ROOT).as_posix()
            or delta['asset_preflight'] != PREFLIGHT.relative_to(ROOT).as_posix()):
        raise RuntimeError('Seed2023 anchor path mismatch')
    require_hash(BASE, delta['base_cohort_sha256'])
    require_hash(PREFLIGHT, delta['asset_preflight_sha256'])
    preflight = json.loads(PREFLIGHT.read_text(encoding='utf-8'))
    if (preflight['status'] != 'passed_cpu_read_only' or preflight['seed'] != 2023
            or preflight['test_reads'] != 0 or preflight['model_instances'] != 0
            or preflight['rankings'] != 0 or preflight['cuda_allocations'] != 0
            or not all(preflight['different_from_seed2022'].values())
            or preflight['tape']['positive_not_in_train'] != 0
            or preflight['tape']['negative_in_train'] != 0
            or preflight['formal_output_exists']):
        raise RuntimeError('Seed2023 preflight not eligible')
    base = seed2022_spec()
    spec = copy.deepcopy(base)
    expected_command = "& 'D:\\miniconda\\envs\\run_5060\\python.exe' -B 'D:\\Download\\PromptMM\\tools\\run_initialization_kd_cohort_seed2023.py'"
    if (delta['seed'] != 2023 or delta['run_id'] != 'sports_init_kd_interaction_seed2023_v1'
            or delta['output_namespace'] != 'exp/initialization_kd_interaction/sports_init_kd_interaction_seed2023_v1'
            or delta['launch_command'] != expected_command):
        raise RuntimeError('Seed2023 run identity mismatch')
    for key, report_name in (('random_initial', 'random_initial'), ('triplet_tape', 'triplet_tape')):
        expected = {part: preflight['assets'][report_name][part] for part in ('path', 'sha256')}
        if delta[key] != expected or expected == base['common']['assets'][key]:
            raise RuntimeError('Seed2023 asset identity mismatch: ' + key)
        spec['common']['assets'][key] = dict(base['common']['assets'][key], **expected)
    for key in ('seed', 'run_id', 'output_namespace', 'launch_command'):
        spec[key] = delta[key]
    resolve_protocol(spec)
    return spec


if __name__ == '__main__':
    main(load_spec(), Path(__file__).resolve())
