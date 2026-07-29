# ===== TD-Distill (No Projection) =====
# Formal structure-control baseline branch.
# Keep this file in the repository: it is used for projection-vs-no-projection
# ablation and residual-gap localization under the frozen-teacher protocol.
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDDistillNoProjectionModel(nn.Module):
    """TD-Distill no-projection student.

    This is a maintained control variant (not a temporary script). It removes
    train-time projection heads and aligns student embeddings directly to
    teacher semantic targets for structure-disambiguation experiments.
    """
    def __init__(
        self,
        n_users,
        n_items,
        embedding_dim,
        item_teacher_dim,
        user_teacher_dim=None,
        item_head_names=('image', 'text'),
        user_head_names=('image', 'text'),
    ):
        super().__init__()

        if item_teacher_dim != embedding_dim:
            raise ValueError(
                f'No-projection TD-Distill requires item_teacher_dim == embedding_dim, '
                f'but got item_teacher_dim={item_teacher_dim}, embedding_dim={embedding_dim}'
            )

        resolved_user_teacher_dim = item_teacher_dim if user_teacher_dim is None else user_teacher_dim
        if resolved_user_teacher_dim != embedding_dim:
            raise ValueError(
                f'No-projection TD-Distill requires user_teacher_dim == embedding_dim, '
                f'but got user_teacher_dim={resolved_user_teacher_dim}, embedding_dim={embedding_dim}'
            )

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.item_teacher_dim = item_teacher_dim
        self.user_teacher_dim = resolved_user_teacher_dim

        self.item_head_names = tuple(item_head_names)
        self.user_head_names = tuple(user_head_names)

        self.user_id_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def init_user_item_embed(self, user_embeddings=None, item_embeddings=None):
        if user_embeddings is not None:
            if user_embeddings.size(-1) != self.embedding_dim:
                raise ValueError(
                    f'user embedding dim mismatch: got {user_embeddings.size(-1)}, '
                    f'expected {self.embedding_dim}'
                )
            self.user_id_embedding.weight.data.copy_(user_embeddings.detach())

        if item_embeddings is not None:
            if item_embeddings.size(-1) != self.embedding_dim:
                raise ValueError(
                    f'item embedding dim mismatch: got {item_embeddings.size(-1)}, '
                    f'expected {self.embedding_dim}'
                )
            self.item_id_embedding.weight.data.copy_(item_embeddings.detach())

    def forward(self):
        return self.user_id_embedding.weight, self.item_id_embedding.weight

    def project_items(self, item_emb):
        # no projection: directly reuse the student item embedding
        return {head_name: item_emb for head_name in self.item_head_names}

    def project_users(self, user_emb):
        # no projection: directly reuse the student user embedding
        return {head_name: user_emb for head_name in self.user_head_names}


def bpr_loss(user, pos, neg):
    pos_scores = torch.sum(user * pos, dim=-1)
    neg_scores = torch.sum(user * neg, dim=-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def directional_distillation_loss(student, teacher):
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher.detach(), dim=-1)
    return F.mse_loss(student, teacher)


def export_infer_state_dict(model):
    return {
        'user_id_embedding.weight': model.user_id_embedding.weight.detach().cpu(),
        'item_id_embedding.weight': model.item_id_embedding.weight.detach().cpu(),
        'embedding_dim': model.embedding_dim,
        'n_users': model.n_users,
        'n_items': model.n_items,
        'variant': 'td_distill_no_projection',
    }
