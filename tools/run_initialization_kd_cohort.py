"""Sealed user-manual Sports seed2022 four-arm cohort; no CLI overrides."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from initialization_kd_adapter import require_hash
from initialization_kd_runtime import resolve_protocol
from run_initialization_kd_interaction import main


def load_spec():
    delta = json.loads((ROOT / 'docs/research/INITIALIZATION_KD_COHORT_CONFIG_2026-09-27.json').read_text(encoding='utf-8'))
    base_path = ROOT / delta['base_config']
    require_hash(base_path, delta['base_config_sha256'])
    base = json.loads(base_path.read_text(encoding='utf-8'))
    if base['status'] != 'preparation_only_not_launchable' or base['launch_command'] is not None:
        raise RuntimeError('Base candidate identity changed')
    if set(delta) != {'base_config', 'base_config_sha256', 'launch_branch', 'launch_source_rule',
                      'environment', 'numerics', 'status', 'mode', 'evaluator', 'run_id',
                      'output_namespace', 'hard_caps', 'launch_command'}:
        raise RuntimeError('Undeclared cohort delta')
    if (delta['mode'] != 'four_arm' or delta['run_id'] != base['run_id']
            or delta['output_namespace'] != base['output_namespace']):
        raise RuntimeError('Cohort identity mismatch')
    spec = dict(base, **delta)
    resolve_protocol(spec)
    return spec


if __name__ == '__main__':
    main(load_spec(), Path(__file__).resolve())
