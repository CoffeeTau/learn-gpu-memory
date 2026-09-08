---
number headings: off
---
你现在其实不需要先把截图这篇文章从头啃完。对于初学者，**最快的方式是先建立一个“显存账本”模型，然后在 L20 上做 5～6 个非常小的实验，把每一笔显存亲眼测出来。**

你可以先把整件事压缩成下面这张图：

```text
                    GPU 显存
                       │
          ┌────────────┴────────────┐
          │                         │
       模型本身                  运行时显存
          │                         │
   ┌──────┴──────┐          ┌──────┴────────┐
   │             │          │               │
 参数 Parameters  优化器状态   激活 Activations   临时 Buffer
                 Optimizer
                                  │
                     训练：通常很多
                     推理：主要变成 KV Cache
```

理解这几个东西以后，你截图里大部分内容都会变得非常直观。

保存激活值是因为==反向传播计算梯度==时，需要使用这些前向结果

---

# 1 先建立最重要的「显存账本」

先记住一个基本公式：

$$
\boxed{
\text{显存}
\approx
\text{参数}
+\text{梯度}
+\text{优化器状态}
+\text{激活值}
+\text{其他临时显存}
}
$$

但是**训练和推理的账本不一样**。

## 1.1 训练

```text
训练显存
=
模型参数
+ 梯度
+ 优化器状态
+ 激活值
+ 临时显存
```

## 1.2 推理

```text
推理显存
=
模型参数
+ KV Cache
+ 激活值
+ 临时显存
```

因为推理：

```text
不 backward
→ 不需要保存梯度
→ 不需要 optimizer
```

所以同一个模型，通常：

$$
\boxed{\text{训练显存} \gg \text{推理显存}}
$$

这就是你首先应该建立起来的对应关系。

---

# 2 第一笔账：模型参数到底占多少显存？

假设一个模型有：

$$
N=4B
$$

也就是 40 亿参数。

不同数据类型，每个参数占：

| dtype |     每个参数 |
| ----- | -------: |
| FP32  |   4 Byte |
| FP16  |   2 Byte |
| BF16  |   2 Byte |
| INT8  |   1 Byte |
| INT4  | 0.5 Byte |
（浮点数都要先转换为二进制，也就是 $1.xxxx \times 2^n$， n就代表指数，.xxxx代表尾数）

所以一个 4B 模型：

## 2.1 FP32

$10^9\ \text{Byte} \approx 1\ \text{GB}$

所以：
$$
4B\times4Byte=16GB
$$

## 2.2 BF16 / FP16

$$
4B\times2Byte=8GB
$$

因此你以后看到：

```text
Qwen 4B
BF16 inference
```

第一反应应该是：

> **光模型权重大概就需要 8GB。**

注意实际 `nvidia-smi/nvitop` 会因为 CUDA context、allocator、workspace 等因素与理论值有差异。

---

# 3 第二笔账：为什么训练突然贵很多？

因为训练除了 Parameter，还需要 Gradient。

假设 BF16：

```text
Parameter
   │
   ▼
Forward
   │
   ▼
Loss
   │
   ▼
Backward
   │
   ├── Gradient
   │
   ▼
Optimizer
   │
   ▼
更新 Parameter
```

那么最简单地看：

```text
BF16 参数：2 Byte / param
BF16 梯度：2 Byte / param
```

仅仅这两个：

$$
4B\times(2+2)=16GB
$$

但真正的大头之一还没出现。

---

# 4 第三笔账：Optimizer State

> - **优化器 `optimizer.step()` 负责使用梯度更新参数**

比如你最近接触过的 AdamW。

Adam 对每个参数通常维护两个状态：

$$
m_t
$$

一阶动量，以及

$$
v_t
$$

二阶动量。

如果它们使用 FP32：

```text
Parameter
Gradient
Adam m   ← 4 Byte
Adam v   ← 4 Byte
```

所以光：

$$
m+v=8Byte/param
$$

4B 模型就是理论上：

$$
4B\times8=32GB
$$

这也是为什么你之前问过“为什么优化器状态这么占显存”。

训练时不是简单地：

```text
4B BF16模型
≈ 8GB
```

而可能变成：

```text
Parameter        8 GB
Gradient         8 GB
Adam m          16 GB
Adam v          16 GB
────────────────────
                48 GB
```

而且**这还没有计算 activation**。

具体训练框架是否保留 FP32 master weights、optimizer state 的 dtype 等会改变数字，所以把这个当作“显存账本思维”，不要死记某个固定倍数。

---

# 5 第四笔账：Activation 是理解训练显存的关键

这一项初学时特别容易忽略。

假设 Transformer：

```text
Input
 ↓
Embedding
 ↓
Transformer Layer 1
 ↓
Transformer Layer 2
 ↓
Transformer Layer 3
 ↓
...
 ↓
Loss
```

训练的时候不能把 Layer 1 的中间结果算完就全部删除。

因为：

```text
Forward
    ↓
保存一些中间结果
    ↓
Backward
    ↓
利用这些结果计算梯度
```

这些保存下来的中间 Tensor 就形成了大量 **Activation Memory**。

因此你会看到一个非常重要的现象：

$$
\boxed{
\text{Batch Size ↑}
\Rightarrow
\text{Activation Memory ↑}
}
$$

以及：

$$
\boxed{
\text{Sequence Length ↑}
\Rightarrow
\text{Activation Memory ↑}
}
$$

这也是你接下来最值得亲手实验的东西。

---

# 6 推理为什么又是另一套逻辑？

LLM 自回归推理：

```text
prompt: 今天天气
        ↓
Transformer
        ↓
       很
        ↓
今天天气很
        ↓
Transformer
        ↓
       好
        ↓
今天天气很好
```

如果每生成一个 token 都重新计算之前所有 token，效率很低。

所以 Transformer 会保存过去 token 的：

$$
K,\quad V
$$

也就是：

$$
\boxed{\text{KV Cache}}
$$

因此推理的账本变成：

```text
Parameter        固定为主
KV Cache         随序列增长
Activation       相对较少
CUDA/Buffer      一部分
```

所以：

$$
\boxed{
\text{Sequence Length ↑}
\Rightarrow
\text{KV Cache ↑}
\Rightarrow
\text{推理显存 ↑}
}
$$

这和训练时 sequence length 导致 activation 增长虽然表面现象类似，**背后的主要原因并不完全相同**。

---

# 7 你现在不要继续看理论，直接在 L20 上做实验

我建议你按照下面这个顺序做。

每次开两个终端。

Terminal A：

```bash
watch -n 0.5 nvidia-smi
```

或者你已经安装好的：

```bash
nvitop
```

Terminal B 跑 Python。

---

## 7.1 实验 1：亲眼看到 Parameter 占显存

新建：

```bash
vim memory_01.py
```

写：

```python
import torch
import time

device = "cuda"

torch.cuda.empty_cache()

print("Before:")
print(torch.cuda.memory_allocated() / 1024**3, "GB")

x = torch.empty(
    1024, 1024, 1024,
    dtype=torch.float32,
    device=device
)

print("After:")
print(torch.cuda.memory_allocated() / 1024**3, "GB")

time.sleep(30)
```

这里一共有：

$$
1024^3
$$

个 FP32。

理论显存：

$$
1024^3\times4Byte
=
4GiB
$$

运行：

```bash
python memory_01.py
```

你应该能明显看到显存增加约 4 GiB。

然后改：

```python
dtype=torch.float16
```

再跑。

理论变成：

$$
1024^3\times2Byte=2GiB
$$

**你通过这个实验要建立**：

```text
Tensor 元素数量
        ×
dtype 每元素字节数
        ↓
      显存
```

这是整个显存分析最底层的规律。

---

## 7.2 实验 2：Parameter 到底有多大

不要直接上 Qwen，我们自己造一个模型。

```python
import torch
import torch.nn as nn

model = nn.Linear(
    10000,
    10000,
    bias=False
).cuda()

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
```

参数量：

$$
10000\times10000
=
100M
$$

FP32 理论占：

$$
100M\times4
=
400MB
$$

然后改成：

```python
model = nn.Linear(
    10000,
    10000,
    bias=False,
    dtype=torch.float16
).cuda()
```

看看是不是接近减半。

---

## 7.3 实验 3：亲眼看 Gradient 出现

这是我最推荐你做的实验。

```python
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
).cuda()

show("model created")

x = torch.randn(
    32,
    10000,
    device="cuda"
)

y = model(x)

show("after forward")

loss = y.sum()

loss.backward()

show("after backward")

input()
```

重点观察：

```text
model created
       ↓
after forward
       ↓
after backward
```

你会发现 backward 后显存明显增加。

然后检查：

```python
for p in model.parameters():
    print(p.shape)
    print(p.grad.shape)
```

你会看到：

```text
parameter
10000 × 10000

gradient
10000 × 10000
```

于是你脑子里应该形成：

```text
Parameter
10000 × 10000
       │
       └── 对应一个 Gradient
             10000 × 10000
```

---

## 7.4 实验 4：AdamW 为什么又增加显存？

在刚才代码里加：

```python
optimizer = torch.optim.AdamW(model.parameters())

show("optimizer created")
```

这时候你可能发现：

> 怎么 optimizer 创建之后显存没增加多少？

这反而是一个非常好的实验现象。

继续：

```python
loss.backward()

show("after backward")

optimizer.step()

show("after optimizer step")
```

完整过程：

```text
Model created
      ↓
Forward
      ↓
Backward
      ↓
optimizer.step()
```

观察：

```text
model created
after forward
after backward
after optimizer step
```

AdamW 的 optimizer state 通常是**第一次 `step()` 时才初始化**。

所以你会亲眼看到：

```text
optimizer = AdamW(...)
        ↓
显存变化可能不大

optimizer.step()
        ↓
Adam state 创建
        ↓
显存明显增加
```

这个实验比背“Adam 占几倍显存”有用得多。

---

## 7.5 实验 5：Batch Size 为什么影响训练显存？

这次用一个简单 MLP：

```python
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
```

观察：

```text
batch = 1
batch = 8
batch = 32
batch = 64
batch = 128
```

对应 peak memory 怎么变化。

你第一次真正看到：

$$
\boxed{
Batch Size
\rightarrow
Activation 数量
\rightarrow
训练显存
}
$$

---

# 8 然后再做一个真正的 Transformer 实验

做到前面这些以后，再进入你真正关心的 LLM。

这时候实验矩阵可以设计成：

| 实验 | 改什么 | 观察什么 |
|---|---|---|
| A | FP32 → BF16 | Parameter memory |
| B | inference → training | Gradient / Optimizer |
| C | batch 1 → 2 → 4 → 8 | Activation |
| D | seq 128 → 512 → 2048 | Activation |
| E | AdamW step 前后 | Optimizer State |
| F | inference seq 增长 | KV Cache |

你甚至可以做一个非常有价值的表：

```text
                    GPU Peak Memory

BF16 inference
batch=1 seq=128        ? GB
batch=1 seq=512        ? GB
batch=1 seq=2048       ? GB

BF16 training
batch=1 seq=128        ? GB
batch=1 seq=512        ? GB
batch=1 seq=2048       ? GB

BF16 training
batch=2 seq=512        ? GB
batch=4 seq=512        ? GB
batch=8 seq=512        ? GB
```

把 `?` 全部在你的 L20 上亲手测出来。

你会很快对“模型训练到底为什么爆显存”产生非常具体的感觉。

---

# 9 测显存时一定区分这三个指标

PyTorch 里你以后会经常用：

```python
torch.cuda.memory_allocated()
```

表示：

> 当前 Tensor 真正占用的显存。

还有：

```python
torch.cuda.memory_reserved()
```

表示：

> PyTorch CUDA allocator 已经向 CUDA 申请并保留的显存。

以及：

```python
torch.cuda.max_memory_allocated()
```

表示：

> 从统计重置后到现在，**最高实际分配显存**。

训练实验里我最建议你关注：

```python
torch.cuda.reset_peak_memory_stats()

# training ...

peak = torch.cuda.max_memory_allocated()
```

也就是：

$$
\boxed{\text{Peak Memory}}
$$

因为 OOM 往往不是看“训练结束后还剩多少”，而是：

> **训练过程最高峰需要多少显存。**

---

# 10 最后把整套知识串起来

你以后看到一个训练任务：

```text
Qwen3-4B
BF16
batch_size = 8
seq_len = 2048
AdamW
```

脑子里不要直接想：

> “大概要多少 GB？”

而应该开始**记账**：

```text
                  Training Memory
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
    Model State       Activation        Others
        │
   ┌────┼─────┐
   ↓    ↓     ↓
Param  Grad  Optimizer
```

然后逐项分析：

```text
4B Parameters
      ↓
BF16
      ↓
参数 ≈ 8GB

Gradient
      ↓
又需要一部分

AdamW
      ↓
m + v
      ↓
又需要大量显存

Batch = 8
Seq = 2048
      ↓
Activation
      ↓
继续增加

CUDA / temporary buffer
      ↓
还有额外开销
```

而推理换成：

```text
                Inference Memory
                       │
          ┌────────────┼───────────┐
          ↓            ↓           ↓
      Parameter     KV Cache     Buffer
```

这样你的知识体系就建立起来了。

---

# 11 我建议你的学习顺序

你现在已经有 **L20 + PyTorch + nvitop**，环境非常适合边测边学。不要先深入 ZeRO、FSDP、FlashAttention、量化训练这些东西，容易把基本关系搞乱。

按照这个路线走：

**Tensor 显存 → Parameter → Gradient → Optimizer State → Activation → Peak Memory → KV Cache → Mixed Precision → Gradient Checkpointing → 最后再看 ZeRO/FSDP。**

其中前 **6 个实验最好真的自己敲一遍**。做完以后再回头看你截图这篇文章，你会发现里面的 FP32/FP16/BF16、训练显存估算、推理显存估算基本都有一个明确的位置，而不是一堆散乱的公式。

如果你接下来想继续，我比较建议直接从你的 L20 开始做一个**“显存实验 Lab 01”**：我可以给你一份几十行的 PyTorch 实验代码，让它自动测试 `FP32/BF16 × batch size × forward/backward/AdamW step`，最后输出一张显存变化表。这样大概半小时就能把上面最重要的概念跑通。
