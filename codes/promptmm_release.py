"""Pinned PromptMM release semantics; intentionally not a corrected algorithm.

Reference: HKUDS/PromptMM 70da1002a35d6f2c7712c16cd0b2ca24c8813008,
codes/Models.py Student_LightGCN and codes/main.py student objective.
See docs/research/PROMPTMM_BASELINE_IDENTITY_2026-09-18.md.
"""
from dataclasses import dataclass
import random

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F

IDENTITY = 'PromptMM-release-Sports-sharedTeacher-v1'
UPSTREAM = '70da1002a35d6f2c7712c16cd0b2ca24c8813008'


@dataclass(frozen=True)
class ResourceConfig:
    # Diagnostic settings, NOT selected/tuned Sports hyperparameters.
    seed: int = 2022
    batch_size: int = 1024
    steps: int = 3
    embedding_dim: int = 64
    layers: int = 1
    learning_rate: float = 0.00002
    weight_decay: float = 0.01  # upstream AdamW default
    decay: float = 0.00001
    pair_rate: float = 1000000.0
    list_rate: float = 1000000.0
    feature_rate: float = 0.1
    sce_power: float = 2.0
    negative_count: int = 10


class ReleaseStudent(nn.Module):
    def __init__(self, n_users, n_items, dim, layers):
        super().__init__()
        self.n_users, self.n_items, self.n_ui_layers = n_users, n_items, layers
        self.user_id_embedding = nn.Embedding(n_users, dim)
        self.item_id_embedding = nn.Embedding(n_items, dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def init_user_item_embed(self, users, items):
        # Do not clone: upstream registers distinct Parameters sharing storage.
        self.user_id_embedding = nn.Embedding.from_pretrained(users, freeze=False)
        self.item_id_embedding = nn.Embedding.from_pretrained(items, freeze=False)
        self.user_id_embedding_pre = nn.Embedding.from_pretrained(users, freeze=False)
        self.item_id_embedding_pre = nn.Embedding.from_pretrained(items, freeze=False)

    def assert_aliases(self):
        for a, b in ((self.user_id_embedding, self.user_id_embedding_pre),
                     (self.item_id_embedding, self.item_id_embedding_pre)):
            if a.weight is b.weight or a.weight.data_ptr() != b.weight.data_ptr():
                raise RuntimeError('Release embedding storage alias was lost.')

    def forward(self, adj):
        ego = torch.cat((self.user_id_embedding.weight + self.user_id_embedding_pre.weight,
                         self.item_id_embedding.weight + self.item_id_embedding_pre.weight), dim=0)
        layers = [ego]
        for _ in range(self.n_ui_layers):
            ego = torch.sparse.mm(adj, ego)
            layers.append(ego)
        result = torch.stack(layers, dim=1).mean(dim=1)
        return torch.split(result, [self.n_users, self.n_items], dim=0)


def normalize_rows(matrix):
    scale = np.power(np.array(matrix.sum(1)) + 1e-8, -0.5).flatten()
    scale[np.isinf(scale)] = 0.
    return sp.diags(scale) * matrix


def release_graphs(train):
    ui, iu = normalize_rows(train), normalize_rows(train.T)
    nu, ni = train.shape
    # Preserve upstream main.py:83 column ordering, NOT conventional [0, UI; IU, 0].
    adj = sp.vstack([sp.hstack([ui, sp.csr_matrix((nu, nu))]),
                     sp.hstack([sp.csr_matrix((ni, ni)), iu])])
    return ui, iu, adj


def sparse_tensor(matrix, device):
    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    return torch.sparse_coo_tensor(indices, torch.from_numpy(coo.data),
                                   coo.shape).float().to(device)


def sample_triplets(train, existing, batch_size):
    if batch_size <= train.shape[0]:
        users = random.sample(existing, batch_size)
    else:
        users = [random.choice(existing) for _ in range(batch_size)]
    positives, negatives = [], []
    for u in users:
        items = train.indices[train.indptr[u]:train.indptr[u + 1]]
        positives.append(int(items[np.random.randint(0, len(items), size=1)[0]]))
        if len(items) >= train.shape[1]:
            raise ValueError('No negative item exists for sampled user.')
        while True:
            neg = int(np.random.randint(0, train.shape[1], size=1)[0])
            if neg not in items:
                negatives.append(neg)
                break
    return users, positives, negatives


def release_candidates(train, users, positives, negative_count, device, dgl):
    row, col = train[users].nonzero()
    # CPU DGL compatibility; retain inferred homogeneous node count and ignored rows.
    graph = dgl.graph((row, col))
    neg_row, neg_col = dgl.sampling.global_uniform_negative_sampling(
        graph, len(users) * negative_count, replace=True)
    neg_row = neg_row.to(device).reshape(len(users), negative_count)
    neg_col = neg_col.to(device).reshape(len(users), negative_count)
    if torch.any(neg_col >= train.shape[1]):
        raise ValueError('Release homogeneous sampler produced out-of-range item ID.')
    return torch.cat((torch.as_tensor(positives, device=device).unsqueeze(1), neg_col), dim=1)


def release_kl(student_values, teacher_values):
    # Deliberately keep negative-log inputs, unsqueeze and element-mean reduction.
    return F.kl_div(student_values.unsqueeze(0), teacher_values.unsqueeze(0), reduction='mean')


def sce(x, y, power):
    x, y = F.normalize(x, p=2, dim=-1), F.normalize(y, p=2, dim=-1)
    return (1 - (x * y).sum(dim=-1)).pow_(power).mean()


def pair_values(user, pos, neg):
    return -F.logsigmoid((user * pos).sum(dim=1) - (user * neg).sum(dim=1))


def release_objective(student_outputs, teacher_outputs, users, pos, neg, candidates, config):
    su, si = student_outputs
    tu, ti, image_i, text_i, image_u, text_u = teacher_outputs[:6]
    u, p, n = su[users], si[pos], si[neg]
    values = pair_values(u, p, n)
    bpr = values.mean()
    reg = config.decay * (0.5 * (u ** 2).sum() + 0.5 * (p ** 2).sum()
                          + 0.5 * (n ** 2).sum()) / config.batch_size
    pair = release_kl(values, pair_values(tu[users], ti[pos], ti[neg]))

    def list_values(uu, ii):
        prob = (uu[users].unsqueeze(1) * ii[candidates]).sum(-1).softmax(-1)
        return -(prob + 1e-8).log()

    student_list = list_values(su, si)
    image_list = release_kl(student_list, list_values(image_u, image_i))
    text_list = release_kl(student_list, list_values(text_u, text_i))
    feature = sce(image_i, si, config.sce_power) + sce(text_i, si, config.sce_power)
    # Upstream computes other terms but does not connect them to this total.
    total = bpr + reg + config.pair_rate * pair + config.list_rate * image_list
    total = total + config.list_rate * text_list + config.feature_rate * feature
    return dict(total=total, bpr=bpr, reg=reg, pair=pair,
                list_image=image_list, list_text=text_list, feature=feature)
