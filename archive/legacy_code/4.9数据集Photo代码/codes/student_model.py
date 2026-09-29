import torch
import torch.nn as nn
import torch.nn.functional as F

class LightweightStudent(nn.Module):
    def __init__(self, num_users, num_items, teacher_dim=1536, latent_dim=64):
        super(LightweightStudent, self).__init__()
        
        # 1. 极度轻量化的核心：只需要 64 维的 ID 嵌入
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)
        
        # 2. 专利创新点：非对称投影头 (映射 R^64 -> R^1536)
        # 仅在训练阶段使用，推断阶段直接丢弃！
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, teacher_dim)
        )
        
        # 初始化权重
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_idx=None, item_idx=None):
        """推断与评估逻辑"""
        # 如果什么都不传，直接返回全量 Embedding 用于评估
        if user_idx is None:
            return self.user_emb.weight, self.item_emb.weight
        
        # 正常推断流程
        u_e = self.user_emb(user_idx)
        if item_idx is not None:
            i_e = self.item_emb(item_idx)
            # 返回：打分, 用户特征, 物品特征
            return torch.sum(u_e * i_e, dim=1), u_e, i_e
        return u_e

def distillation_loss(student_item_emb_64d, teacher_feat_1536d, model):
    """
    改进的蒸馏 Loss：先通过投影头，再计算 MSE
    """
    # 映射特征
    projected_feat = model.projector(student_item_emb_64d)
    # L2 归一化 (对齐方向，忽略绝对长度)
    projected_feat = F.normalize(projected_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat_1536d, p=2, dim=1)
    
    return F.mse_loss(projected_feat, teacher_feat)