"""Fixed objectives and candidate construction for the seed2022 comparison."""
import numpy as np

from initialization_kd_fast_eval import exact_topk


def candidate_rows(train, teacher_user, teacher_item, limit=None, seed=20260930,
                   user_batch=256, check=None):
    """Use Train and fused teacher tables only; no held-out split is accepted."""
    import torch

    users = np.flatnonzero(np.diff(train.indptr))
    if limit is not None:
        users = users[:limit]
    rng = np.random.default_rng(seed)
    ids = np.empty((len(users), 64), dtype=np.int32)
    logits = np.empty((len(users), 64), dtype=np.float32)
    mu = np.empty(len(users), dtype=np.float32)
    scale = np.empty(len(users), dtype=np.float32)
    items = np.arange(train.shape[1], dtype=np.int32)
    with torch.no_grad():
        for start in range(0, len(users), user_batch):
            stop = min(start + user_batch, len(users))
            scores = (teacher_user[torch.as_tensor(users[start:stop])] @ teacher_item.T).cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite teacher candidate scores')
            for row, uid in enumerate(users[start:stop], start):
                seen = set(train.indices[train.indptr[uid]:train.indptr[uid + 1]].tolist())
                top = np.asarray(exact_topk(scores[row - start], seen, 32), dtype=np.int32)
                eligible = items[~np.isin(items, np.r_[list(seen), top])]
                if len(eligible) < 32:
                    raise ValueError('Insufficient random candidate pool')
                random = rng.choice(eligible, 32, replace=False)
                chosen = np.r_[top, random]
                if len(np.unique(chosen)) != 64 or any(int(i) in seen for i in chosen):
                    raise ValueError('Candidate exclusion/uniqueness failed')
                raw = scores[row - start, chosen]
                center = float(raw.mean(dtype=np.float64))
                spread = max(float(raw.std(dtype=np.float64)), 1e-6)
                ids[row] = chosen
                mu[row] = center
                scale[row] = spread
                logits[row] = (raw - center) / spread
            if check is not None:
                check()
    return users.astype(np.int32), ids, logits, mu, scale


def rank_kl(user, candidate_item, teacher_logits, mu, scale):
    """Occurrence-weighted KL, teacher detached, no division by candidate count."""
    import torch
    import torch.nn.functional as F

    student = (user[:, None, :] * candidate_item).sum(dim=-1)
    student_logq = F.log_softmax((student - mu[:, None]) / scale[:, None], dim=-1)
    target_logq = F.log_softmax(teacher_logits.detach(), dim=-1)
    target_q = target_logq.exp()
    return (target_q * (target_logq - student_logq)).sum(dim=-1).mean()


def anchor_loss(user, pos, neg, initial_user, initial_pos, initial_neg,
                user_scale, item_scale):
    return (((user - initial_user).square().mean(dim=-1) / user_scale
             + (pos - initial_pos).square().mean(dim=-1) / item_scale
             + (neg - initial_neg).square().mean(dim=-1) / item_scale) / 3).mean()


def initial_scales(user, item):
    return max(float(user.square().mean()), 1e-12), max(float(item.square().mean()), 1e-12)


def mixed_tables(student_user, student_item, teacher_user, teacher_item):
    """Dot of concatenated 128D tables equals fixed half/half score mixture."""
    import torch
    factor = 2 ** -0.5
    return (torch.cat((student_user * factor, teacher_user * factor), dim=-1),
            torch.cat((student_item * factor, teacher_item * factor), dim=-1))
