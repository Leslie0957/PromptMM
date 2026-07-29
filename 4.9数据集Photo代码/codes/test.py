import torch
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"当前显卡型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无'}")