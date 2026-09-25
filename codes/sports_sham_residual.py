"""Pinned Sports real/sham text targets with exact cold-ID and batch replay."""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sports_paired_cold import PairedColdSession, sha256

COHORT = Path('exp/sham_residual/sports_three_seed_v1')
TEACHER_ASSET = Path('exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/process0/initial_tensors.pt')
TEACHER_ASSET_SHA = 'e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20'
PAIR_SHA = {
    2022: ('2357868cefeb086dba14c3808415eac97936f67cf5e6bef8382d95ff43bcffff',
           'e8d982c6a2d1be20a934502d4a1ad6ba7e2b699b21098668343341f10db3d77b'),
    2023: ('bb5429aa8a1edfe53b36fed977146460d1101ae79539d2badf856d5514ef9d85',
           '3763128ba003228d60c09924b9370b4cab0b8c16679366742835fc4559d89085'),
    2024: ('e2c6885debdec09a8ff8401fe4f4f96ea5f46dcc1fd90f24dcc4ecdbd2188e37',
           '131cc14ebe6853e79216183f06923f3689d47d26c69eac2521d7f93b1b314bc7'),
}
SHAM_SEEDS = {2022: 9022022, 2023: 9022023, 2024: 9022024}


def teacher_tensors(root):
    path = Path(root) / TEACHER_ASSET
    if sha256(path) != TEACHER_ASSET_SHA:
        raise RuntimeError('Pinned teacher semantic asset changed')
    tensors = torch.load(path, map_location='cpu', weights_only=True)
    if set(tensors) != {'users', 'items', 'image_items', 'text_items', 'image_users', 'text_users'}:
        raise ValueError('Unexpected teacher semantic tensor keys')
    for name, value in tensors.items():
        rows = 35598 if name in ('users', 'image_users', 'text_users') else 18357
        if value.shape != (rows, 64) or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError('Invalid teacher semantic tensor: ' + name)
    return tensors


def sham_text(image, text, random_seed):
    """Keep each item's image/text cosine and norm, randomize only text residual."""
    if image.shape != text.shape or image.ndim != 2 or image.dtype != torch.float32 or text.dtype != torch.float32:
        raise ValueError('Expected matching float32 item targets')
    if image.device.type != 'cpu' or text.device.type != 'cpu':
        raise ValueError('Construct the immutable sham target on CPU')
    if not torch.isfinite(image).all() or not torch.isfinite(text).all():
        raise ValueError('Nonfinite teacher target')
    image_norm = torch.linalg.vector_norm(image, dim=1, keepdim=True)
    text_norm = torch.linalg.vector_norm(text, dim=1, keepdim=True)
    v = F.normalize(image, dim=1)
    t = F.normalize(text, dim=1)
    cosine = (v * t).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    residual = t - cosine * v
    residual_norm = torch.linalg.vector_norm(residual, dim=1, keepdim=True)
    valid = ((image_norm[:, 0] > 1e-12) & (text_norm[:, 0] > 1e-12)
             & (residual_norm[:, 0] > 1e-6))
    generator = torch.Generator(device='cpu').manual_seed(int(random_seed))
    random = torch.randn(text.shape, dtype=torch.float32, generator=generator)
    orthogonal = random - (random * v).sum(dim=1, keepdim=True) * v
    orthogonal_norm = torch.linalg.vector_norm(orthogonal, dim=1, keepdim=True)
    if (orthogonal_norm[valid] < 1e-6).any():
        raise ValueError('Degenerate seeded random residual')
    s = orthogonal / orthogonal_norm.clamp_min(1e-12)
    residual_length = (1.0 - cosine.square()).clamp_min(0.0).sqrt()
    candidate = (cosine * v + residual_length * s) * text_norm
    result = text.clone()
    result[valid] = candidate[valid]
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite sham target')
    return result, valid


def geometry_error(image, text, sham, valid):
    v = F.normalize(image[valid], dim=1)
    t = F.normalize(text[valid], dim=1)
    q = F.normalize(sham[valid], dim=1)
    original_cos = (v * t).sum(dim=1)
    sham_cos = (v * q).sum(dim=1)
    original_combined = torch.linalg.vector_norm(v + 0.3 * t, dim=1)
    sham_combined = torch.linalg.vector_norm(v + 0.3 * q, dim=1)
    return {
        'valid_rows': int(valid.sum()),
        'unchanged_rows': int((~valid).sum()),
        'max_text_norm_error': float((torch.linalg.vector_norm(text[valid], dim=1) -
                                      torch.linalg.vector_norm(sham[valid], dim=1)).abs().max()),
        'max_image_text_cosine_error': float((original_cos - sham_cos).abs().max()),
        'max_combined_norm_error': float((original_combined - sham_combined).abs().max()),
    }


def replay_session(root, seed, epochs=300, batches=214, batch_size=1024):
    session = PairedColdSession(root, seed, 'image_matched', epochs, batches, batch_size)
    if session.digest_path.read_text(encoding='ascii').strip() != PAIR_SHA[seed][1]:
        raise RuntimeError('Declared training tape SHA mismatch')
    if sha256(session.initial) != PAIR_SHA[seed][0]:
        raise RuntimeError('Declared initial tensor SHA mismatch')
    return session


def load_semantics(root, seed, arm, device):
    if arm not in ('real', 'sham') or seed not in SHAM_SEEDS:
        raise ValueError('Unknown Sports sham profile')
    root = Path(root)
    report_path = root / COHORT / 'preflight.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report.get('status') != 'passed' or report.get('teacher_asset_sha256') != TEACHER_ASSET_SHA:
        raise RuntimeError('Sham cohort preflight has not passed')
    record = report['seeds'][str(seed)]
    if record['initial_sha256'] != PAIR_SHA[seed][0] or record['tape_sha256'] != PAIR_SHA[seed][1]:
        raise RuntimeError('Preflight paired asset identity mismatch')
    target_path = root / COHORT / ('seed%d' % seed) / 'sham_text.pt'
    if sha256(target_path) != record['sham_target_sha256']:
        raise RuntimeError('Sham target changed after preflight')
    frozen = teacher_tensors(root)
    if arm == 'sham':
        saved = torch.load(target_path, map_location='cpu', weights_only=True)
        if saved['seed'] != SHAM_SEEDS[seed] or saved['teacher_asset_sha256'] != TEACHER_ASSET_SHA:
            raise RuntimeError('Sham target provenance mismatch')
        frozen['text_items'] = saved['item_text']
        if frozen['text_items'].shape != (18357, 64) or not torch.isfinite(frozen['text_items']).all():
            raise RuntimeError('Sham target shape/finiteness mismatch')
    return {
        'item_image': frozen['image_items'].to(device),
        'item_text': frozen['text_items'].to(device),
        'user_image': frozen['image_users'].to(device),
        'user_text': frozen['text_users'].to(device),
    }, {
        'teacher_asset': str((root / TEACHER_ASSET).resolve()),
        'teacher_asset_sha256': TEACHER_ASSET_SHA,
        'sham_target': str(target_path.resolve()),
        'sham_target_sha256': record['sham_target_sha256'],
        'sham_seed': SHAM_SEEDS[seed],
        'arm': arm,
    }
