import torch
import torch.nn as nn

def show(name):
    print(
        f"{name:<20}",
        f"{torch.cuda.memory_allocated()/1024**2:.1f} MB"
    )

model = nn.Linear(
    10000,
    10000,
    bias=False
).cuda() # FP32

show("model created")

x = torch.randn(
    32,
    10000,
    device="cuda"
)

y = model(x)

show("after forward")

loss = y.sum()

optimizer = torch.optim.AdamW(model.parameters())

show("optimizer created")

loss.backward()

show("after backward")

optimizer.step()

show("after optimizer step")

input()