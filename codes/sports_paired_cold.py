"""Exact initial-ID and train-triplet replay for opt-in Sports pairs."""
import hashlib
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class PairedColdSession:
    def __init__(self, root, seed, arm, epochs, batches, batch_size):
        if arm not in ('full', 'image_matched'):
            raise ValueError(arm)
        self.arm = arm
        self.shape = (epochs, batches, 3, batch_size)
        self.directory = Path(root) / 'exp' / 'paired_coldinit' / 'sports_three_seed_v1' / ('seed%d' % seed)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.initial = self.directory / 'initial.pt'
        self.tape_path = self.directory / 'triplets.npy'
        self.digest_path = self.directory / 'triplets.sha256'
        self.count = 0
        if arm == 'full':
            if any(p.exists() for p in (self.initial, self.tape_path, self.digest_path)):
                raise FileExistsError('Paired seed assets already exist: %s' % self.directory)
            self.tape = None
        else:
            if not all(p.exists() for p in (self.initial, self.tape_path, self.digest_path)):
                raise FileNotFoundError('Full-arm paired assets incomplete: %s' % self.directory)
            expected = self.digest_path.read_text(encoding='ascii').strip()
            if sha256(self.tape_path) != expected:
                raise ValueError('Training triplet tape SHA256 mismatch')
            self.tape = np.load(self.tape_path, mmap_mode='r', allow_pickle=False)
            if self.tape.shape != self.shape or self.tape.dtype != np.dtype('<i4'):
                raise ValueError('Training triplet tape shape/dtype mismatch')

    def apply_initial(self, model):
        if self.arm == 'full':
            state = {
                'user': model.user_id_embedding.weight.detach().cpu().clone(),
                'item': model.item_id_embedding.weight.detach().cpu().clone(),
            }
            with self.initial.open('xb') as stream:
                torch.save(state, stream)
        else:
            state = torch.load(self.initial, map_location='cpu', weights_only=True)
            if set(state) != {'user', 'item'}:
                raise ValueError('Unexpected paired initial tensor keys')
            if state['user'].shape != model.user_id_embedding.weight.shape or state['item'].shape != model.item_id_embedding.weight.shape:
                raise ValueError('Paired initial tensor shape mismatch')
            model.init_user_item_embed(state['user'], state['item'])
        if not torch.equal(model.user_id_embedding.weight.detach().cpu(), state['user']) or not torch.equal(model.item_id_embedding.weight.detach().cpu(), state['item']):
            raise ValueError('Paired initial tensor copy failed')
        return {'path': str(self.initial), 'sha256': sha256(self.initial)}

    def sample(self, epoch, batch, sampler):
        if self.arm == 'full':
            if self.tape is None:
                with self.tape_path.open('xb'):
                    pass
                self.tape = np.lib.format.open_memmap(self.tape_path, mode='w+', dtype='<i4', shape=self.shape)
            triplet = sampler()
            values = np.asarray(triplet, dtype=np.int32)
            if values.shape != (3, self.shape[-1]):
                raise ValueError('Sampled triplet shape mismatch')
            self.tape[epoch, batch] = values
        else:
            values = self.tape[epoch, batch]
            triplet = tuple(values[row] for row in range(3))
        self.count += 1
        return triplet

    def finish(self):
        expected = self.shape[0] * self.shape[1]
        if self.count != expected:
            raise RuntimeError('Incomplete paired triplet tape: %d/%d' % (self.count, expected))
        if self.arm == 'full':
            self.tape.flush()
            del self.tape
            digest = sha256(self.tape_path)
            with self.digest_path.open('x', encoding='ascii') as stream:
                stream.write(digest + '\n')
        else:
            digest = self.digest_path.read_text(encoding='ascii').strip()
            del self.tape
        return {'path': str(self.tape_path), 'sha256': digest,
                'shape': list(self.shape), 'triplet_batches_consumed': self.count}
