import torch
import time

device = "cuda"

torch.cuda.empty_cache()

print("Before:")
print(torch.cuda.memory_allocated() / 1024**3, "GB")

x = torch.empty(
    1024, 1024, 1024,
    dtype=torch.float16,
    device=device
) # 构造1024 * 1024 * 1024个FP32，也就是1B的FP32，

print("After:")
print(torch.cuda.memory_allocated() / 1024**3, "GB")

time.sleep(30)