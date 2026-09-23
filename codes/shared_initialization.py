"""Pinned diagnostic process0 tensors shared by a newly declared TD pair."""
from pathlib import Path
import torch
from initialization_audit import describe_tensor
from promptmm_release_resource import sha256

ASSET = Path('exp/initialization_checks/sports_initialization_repeatability_seed2022_v1/process0/initial_tensors.pt')
ASSET_SHA = 'e5573bbdbbb609b4bc037a551455578463795c6df4bc4a1b236788d86c24da20'
NAMES = ('users', 'items', 'image_items', 'text_items', 'image_users', 'text_users')


def load_shared(root, device, n_users, n_items, dim=64):
    path = Path(root)/ASSET
    if sha256(path) != ASSET_SHA:
        raise RuntimeError('Shared initialization asset fingerprint mismatch.')
    tensors = torch.load(path, map_location='cpu', weights_only=True)
    if set(tensors) != set(NAMES):
        raise ValueError('Shared tensor keys mismatch.')
    identities = {}
    for name, value in tensors.items():
        rows = n_users if name in ('users','image_users','text_users') else n_items
        if value.shape != (rows,dim) or value.dtype != torch.float32:
            raise ValueError('Shared tensor shape/dtype mismatch: '+name)
        identities[name] = describe_tensor(value)
    return ({k:v.to(device) for k,v in tensors.items()},
            dict(path=str(path.resolve()),sha256=ASSET_SHA,tensors=identities,
                 source='diagnostic process0 first forward; fixed without quality selection'))


def semantic_targets(tensors):
    return {'item_image':tensors['image_items'],'item_text':tensors['text_items'],
            'user_image':tensors['image_users'],'user_text':tensors['text_users']}
