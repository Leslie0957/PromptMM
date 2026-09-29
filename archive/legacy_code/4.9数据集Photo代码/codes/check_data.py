import pickle
import numpy as np

# 换成 Photo 的路径
data_path = r"D:\Download\zero-shot%20datasets\zero-shot datasets\Photo\feats.pkl"
print("📦 正在潜入 Photo 数据集，破解特征盲盒...")

with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(f"盲盒类型: {type(data)}")

if isinstance(data, dict):
    print("🎉 太棒了！这是一个字典，里面的钥匙有：")
    print(data.keys())
elif isinstance(data, np.ndarray):
    print(f"🔥 这是一个巨大的矩阵，形状是: {data.shape}")
else:
    print("这到底是个什么怪物？")