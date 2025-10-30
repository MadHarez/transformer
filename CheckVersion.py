import torch
print(torch.__version__)  # 如 2.7.0+cu124，表示 CUDA 版本
print(torch.cuda.is_available())  # 应返回 True
print(torch.cuda.get_device_name(0))  # 显示你的 GPU 型号，如 "NVIDIA GeForce RTX 3080"