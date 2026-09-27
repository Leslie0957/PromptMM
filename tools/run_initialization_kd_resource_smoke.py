"""User-manual one-attempt GPU resource smoke. Import performs no asset I/O."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'codes'))
from initialization_kd_adapter import require_hash
from run_initialization_kd_interaction import main


def load_spec():
    config = ROOT / 'docs/research/INITIALIZATION_KD_RESOURCE_SMOKE_CONFIG_2026-09-27.json'
    smoke = json.loads(config.read_text(encoding='utf-8'))
    base_path = ROOT / smoke['base_config']
    require_hash(base_path, smoke['base_config_sha256'])
    base = json.loads(base_path.read_text(encoding='utf-8'))
    # The candidate cohort itself stays disabled. Only this frozen delta runs.
    if base['status'] != 'preparation_only_not_launchable' or base['launch_command'] is not None:
        raise RuntimeError('Unexpected formal cohort state')
    return dict(base, **smoke)


if __name__ == '__main__':
    main(load_spec(), Path(__file__).resolve())
