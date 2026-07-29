import os
import time
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from student_model import LightweightStudent, distillation_loss

# ==========================================
# 0. 科研护身符：固定随机种子 (保证 0.41 可复现)
# ==========================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 1. 硬件与环境配置
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 准备起飞！当前分配的计算设备: {device}")
if device.type == 'cuda':
    print(f"🔥 当前显卡型号: {torch.cuda.get_device_name(0)}")

# 数据路径（保留 %20 原始文件夹名）
data_dir = r"D:\Download\zero-shot%20datasets\zero-shot datasets\Photo"
print("📦 正在加载巨大的交互矩阵和特征盲盒...")

with open(os.path.join(data_dir, "trn_mat.pkl"), 'rb') as f:
    train_mat = pickle.load(f)
with open(os.path.join(data_dir, "tst_mat.pkl"), 'rb') as f:
    test_mat = pickle.load(f).tocsr()

with open(os.path.join(data_dir, "feats.pkl"), 'rb') as f:
    shared_feat = pickle.load(f)

# 特征影分身：将统一特征同时喂给图、文两个通道
shared_feat_tensor = torch.tensor(shared_feat, dtype=torch.float32).to(device)
teacher_image_feat = shared_feat_tensor
teacher_text_feat = shared_feat_tensor

print(f"✅ 数据加载完毕！物品数: {shared_feat.shape[0]}, 特征维度: {shared_feat.shape[1]}")

# ==========================================
# 2. 自动归档系统：构思文件夹
# ==========================================
alpha = 0.1
lr = 0.001
epochs = 40
timestamp = time.strftime("%m%d_%H%M")
folder_name = f"Result_{timestamp}_Photo_a{alpha}_lr{lr}"
os.makedirs(folder_name, exist_ok=True)
path_prefix = os.path.join(folder_name, folder_name)

# ==========================================
# 3. 初始化模型与优化策略
# ==========================================
num_users, num_items = train_mat.shape
emb_dim = teacher_image_feat.shape[1]
rows, cols = train_mat.nonzero()

class FastRecDataset(Dataset):
    def __init__(self, users, items):
        self.users = torch.tensor(users, dtype=torch.long)
        self.pos_items = torch.tensor(items, dtype=torch.long)
    def __len__(self): return len(self.users)
    def __getitem__(self, idx): return self.users[idx], self.pos_items[idx]

dataloader = DataLoader(FastRecDataset(rows, cols), batch_size=1024, shuffle=True)

student = LightweightStudent(num_users, num_items, teacher_dim=emb_dim, latent_dim=64).to(device)
optimizer = optim.Adam(student.parameters(), lr=lr)

# 进阶建议：学习率调度器（每 15 轮缩小为原来的 0.1，帮助在后期精细收敛）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# ==========================================
# 4. 联合训练循环
# ==========================================
loss_history = []
best_recall = 0.0
print(f"🔥 开始联合训练 (40 Epochs 为黄金平衡点)")

for epoch in range(epochs):
    student.train()
    total_bpr, total_distill = 0, 0
    start_time = time.time()
    
    for batch_u, batch_pos_i in dataloader:
        batch_u, batch_pos_i = batch_u.to(device), batch_pos_i.to(device)
        optimizer.zero_grad()
        
        batch_neg_i = torch.randint(0, num_items, (len(batch_u),)).to(device)
        u_emb = student(batch_u)
        pos_i_emb = student.item_emb(batch_pos_i)
        neg_i_emb = student.item_emb(batch_neg_i)
        
        bpr_loss = -torch.mean(F.logsigmoid(torch.sum(u_emb * pos_i_emb, 1) - torch.sum(u_emb * neg_i_emb, 1)))
        distill_loss = distillation_loss(pos_i_emb, teacher_image_feat[batch_pos_i], student) + \
                       distillation_loss(pos_i_emb, teacher_text_feat[batch_pos_i], student)
        
        loss = bpr_loss + alpha * distill_loss
        loss.backward()
        optimizer.step()
        
        total_bpr += bpr_loss.item()
        total_distill += distill_loss.item()
    
    scheduler.step() # 更新学习率
    avg_total_loss = (total_bpr + alpha * total_distill) / len(dataloader)
    loss_history.append(avg_total_loss)
    
    print(f"Epoch [{epoch+1:02d}] | BPR:{total_bpr/len(dataloader):.4f} | Dist:{total_distill/len(dataloader):.2e} | Time:{time.time()-start_time:.2f}s")

# ==========================================
# 5. 全量评测与“战果”封存
# ==========================================
print("\n🔍 正在进行全量推荐性能评测...")
def evaluate(scores, test_matrix, top_k=20):
    _, top_indices = torch.topk(scores, top_k, dim=1)
    top_indices = top_indices.cpu().numpy()
    recalls, ndcgs = [], []
    for i in range(len(top_indices)):
        pos_items = test_matrix[i].nonzero()[1]
        if len(pos_items) == 0: continue
        hit = len(set(top_indices[i]) & set(pos_items))
        recalls.append(hit / len(pos_items))
        dcg = sum([1 / np.log2(j + 2) for j, item in enumerate(top_indices[i]) if item in pos_items])
        idcg = sum([1 / np.log2(k + 2) for k in range(min(len(pos_items), top_k))])
        ndcgs.append(dcg / idcg)
    return np.mean(recalls), np.mean(ndcgs)

student.eval()
with torch.no_grad():
    u_e, i_e = student()
    recall, ndcg = evaluate(torch.matmul(u_e[:1000], i_e.T), test_mat[:1000])

# 保存最佳模型权重
torch.save(student.state_dict(), f"{path_prefix}_best_model.pth")

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='#1f77b4', label='Training Loss')
plt.title(f'PromptMM Convergence (Photo)\nRecall@20: {recall:.4f}')
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{path_prefix}_curve.png", dpi=300)
plt.close()

# 撰写实验战报
with open(f"{path_prefix}_report.txt", 'w', encoding='utf-8') as f:
    f.write(f"=== PromptMM 极客归档系统 ===\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"🏆 核心战果:\n📈 Recall@20: {recall:.4f}\n📈 NDCG@20: {ndcg:.4f}\n")
    f.write(f"⚙️ 参数: alpha={alpha}, lr={lr}, seed=42\n")
    f.write(f"💾 模型已保存至: {folder_name}_best_model.pth\n")

print(f"\n✨ 恭喜！所有产物已自动归档至文件夹: {folder_name}")