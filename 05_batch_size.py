import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.ReLU(),
).cuda()

for batch_size in [1, 8, 32, 64, 128]:

    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(
        batch_size,
        4096,
        device="cuda"
    )

    y = model(x)
    loss = y.sum()
    loss.backward()

    peak = torch.cuda.max_memory_allocated()

    print(
        batch_size,
        f"{peak / 1024**2:.1f} MB"
    )

    model.zero_grad(set_to_none=True)