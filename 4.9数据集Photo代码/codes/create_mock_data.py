import os
import numpy as np
import pickle
from scipy.sparse import csr_matrix

# 定义 PromptMM 默认读取的路径
data_path = "D:/Download/PromptMM/data/amazon/"
if not os.path.exists(data_path):
    os.makedirs(data_path)

# 1. 模拟生成用户和物品数量 (设为 1000 规模，CPU 跑起来飞快)
num_users, num_items = 1000, 1000

# 2. 模拟生成多模态特征 (.npy 文件)
# PromptMM 默认特征维度通常是 64 或 128
image_feat = np.random.rand(num_items, 64).astype(np.float32)
text_feat = np.random.rand(num_items, 64).astype(np.float32)

np.save(os.path.join(data_path, "image_feat.npy"), image_feat)
np.save(os.path.join(data_path, "text_feat.npy"), text_feat)

# 3. 模拟生成交互矩阵 (train_mat)
# 随机生成 5000 条交互记录
data = np.ones(5000)
row = np.random.randint(0, num_users, 5000)
col = np.random.randint(0, num_items, 5000)
train_mat = csr_matrix((data, (row, col)), shape=(num_users, num_items))

with open(os.path.join(data_path, "train_mat"), 'wb') as f:
    pickle.dump(train_mat, f)

print(f"✅ 搞定！模拟数据已塞进: {data_path}")
print("现在你可以去跑 main.py 验证你的专利代码了！")