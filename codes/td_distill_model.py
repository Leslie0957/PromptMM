# ===== TD-Distill =====
import torch
import torch.nn as nn
import torch.nn.functional as F


class TDDistillBridge(nn.Module):
    def __init__(self, student_dim, teacher_dim):
        super().__init__()
        self.proj = nn.Linear(student_dim, teacher_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return self.proj(x)


class TDDistillProjectionGroup(nn.Module):
    def __init__(self, student_dim, teacher_dim, head_names):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: TDDistillBridge(student_dim, teacher_dim)
            for name in head_names
        })

    def forward(self, x):
        return {k: head(x) for k, head in self.heads.items()}


class TDDistillModel(nn.Module):
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

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        # Keep explicit semantic-dimension metadata for checkpoint compatibility.
        self.item_teacher_dim = item_teacher_dim
        self.user_teacher_dim = item_teacher_dim if user_teacher_dim is None else user_teacher_dim
        self.item_head_names = tuple(item_head_names)
        self.user_head_names = tuple(user_head_names)

        self.user_id_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, embedding_dim)

        self.item_projection_heads = TDDistillProjectionGroup(
            embedding_dim,
            item_teacher_dim,
            item_head_names
        )

        self.user_projection_heads = TDDistillProjectionGroup(
            embedding_dim,
            self.user_teacher_dim,
            user_head_names
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

    def forward(self):
        return self.user_id_embedding.weight, self.item_id_embedding.weight

    def project_items(self, item_emb):
        return self.item_projection_heads(item_emb)

    def project_users(self, user_emb):
        return self.user_projection_heads(user_emb)


# ===== Loss =====
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
    }
