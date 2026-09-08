import torch
import torch.nn as nn

# nn.Linear(...) 创建参数时，如果没有指定 dtype，默认使用：torch.float32
model = nn.Linear(
    10000,
    10000,
    bias=False
).cuda()  # 参数量10000 * 10000 = 100M

params = sum(p.numel() for p in model.parameters())

print(f"Parameters: {params / 1e6:.1f} M")

print(
    "Parameter memory:",
    params * 4 / 1024**3,
    "GiB"
)

print(
    "GPU allocated:",
    torch.cuda.memory_allocated() / 1024**3,
    "GiB"
)

input()