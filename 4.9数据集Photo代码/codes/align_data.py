import os
import shutil
import pickle
import numpy as np

src_dir = "D:/Download/data/yelp/"  # 你刚才截图里原始解压的路径
dst_dir = "D:/Download/PromptMM/data/yelp/" # 你的项目路径

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

print("🚀 开始搬运并转换数据...")

# 1. 搬运并重命名核心交互矩阵
# PromptMM 需要的是没有后缀的 train_mat, val_mat, test_mat
shutil.copy(src_dir + "trn_mat.pkl", dst_dir + "train_mat")
shutil.copy(src_dir + "val_mat.pkl", dst_dir + "val_mat")
shutil.copy(src_dir + "tst_mat.pkl", dst_dir + "test_mat")
print("✅ 交互矩阵搬运完成！")

# 2. 转换多模态特征
# RLMRec 将特征存成了 pkl，我们把它读出来存成 PromptMM 需要的 npy 格式
with open(src_dir + "itm_emb_np.pkl", "rb") as f:
    itm_emb = pickle.load(f)

# 因为你的学生模型是纯 ID 的，教师模型的特征用来做蒸馏 Loss
# 我们直接把这个高维特征复制给 image 和 text，满足代码的读取要求
np.save(dst_dir + "image_feat.npy", itm_emb)
np.save(dst_dir + "text_feat.npy", itm_emb)
print("✅ 多模态特征转换完成！")

print(f"🎉 搞定！现在 {dst_dir} 里的数据已经完全适配 PromptMM！")