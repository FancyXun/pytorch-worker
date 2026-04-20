# PyTorch 分布式训练（六）：DDP 一次迭代源码时序

**源码与 API 版本**：文中函数名与行为均以官方 **PyTorch 2.5.x** 为准；源码定位使用 GitHub 标签 **[v2.5.0](https://github.com/pytorch/pytorch/releases/tag/v2.5.0)**。

前面 01-03 讲了概念，这一篇只做一件事：把 DDP 在一次迭代中的关键函数按时间顺序串起来。后续讲魔改时，你可以直接拿这一篇当“原生时序对照表”。

## 1. 初始化阶段：把普通 Module 变成 DDP

训练脚本里通常是：

```python
model = torch.nn.parallel.DistributedDataParallel(model, ...)
```

对应源码入口：

- `torch/nn/parallel/distributed.py` 中 `class DistributedDataParallel`  
  <https://github.com/pytorch/pytorch/blob/v2.5.0/torch/nn/parallel/distributed.py>
- C++ 侧 `Reducer` 构造与状态机：  
  <https://github.com/pytorch/pytorch/blob/v2.5.0/torch/csrc/distributed/c10d/reducer.hpp>

初始化阶段关键动作：

1. 校验设备与进程组。  
2. 收集参数列表，构建 bucket 元数据。  
3. 注册 autograd 相关 hook，使 backward 时能回调到 `Reducer`。  
4. 按配置决定是否同步 buffer、是否追踪 unused parameter。

## 2. forward 前后：DDP 在做什么

每轮 forward 的高层顺序可以概括为：

1. 进入 DDP 包装层。  
2. 必要时进行参数或 buffer 同步（取决于配置与上下文）。  
3. 执行原始模块 forward。  
4. 记录反向阶段所需状态（例如本轮是否需要梯度同步、图结构相关信息）。

可对照的函数位点（同文件）：

- `DistributedDataParallel.forward(...)`  
- 与 forward 同步点相关的内部逻辑（例如 `require_forward_param_sync`、`find_unused_parameters` 相关分支）

## 3. backward 阶段：真正的核心时序

`loss.backward()` 之后，并不是“反向全算完再一次性通信”，而是更细粒度的流水线：

1. 某个参数梯度 ready。  
2. autograd hook 触发，`Reducer` 标记对应参数就绪。  
3. 当一个 bucket 内参数都 ready 时，`Reducer` 对该 bucket 发起通信（默认 allreduce）。  
4. 其它 bucket 继续等待或并行推进。  
5. 通信完成后将结果写回梯度视图（具体取决于 `gradient_as_bucket_view` 等设置）。

对应源码：

- `reducer.hpp` / `reducer.cpp`：`autograd_hook(...)`、`run_allreduce_hook(...)`、`run_comm_hook(...)`  
  <https://github.com/pytorch/pytorch/blob/v2.5.0/torch/csrc/distributed/c10d/reducer.cpp>

这也是为什么 profiler 常看到 **CUDA 计算核**与 **NCCL 通信核**交错：计算与通信在 bucket 粒度做了重叠。

## 4. optimizer.step 前后：一致性是如何维持的

当 backward + 梯度对齐完成后，各 rank 拿到一致梯度，再执行各自的 `optimizer.step()`。标准 DDP 语义里，权重一致性依赖：

- 同步后的梯度一致。  
- 每个 rank 采用相同优化器逻辑与超参。  
- 训练控制流一致（例如没有某个 rank 额外多 step 一次）。

因此在排错时，一旦出现“同一步 rank 间参数漂移”，通常先排查：

1. 是否某些 rank 走了不同分支。  
2. 是否存在未同步梯度路径（`no_sync()`、unused 分支）。  
3. 是否优化器状态或 step 触发条件不一致。

## 5. 两个最值得盯的源码文件

在不改代码前，建议始终把以下两份文件并排读：

- Python 侧：`torch/nn/parallel/distributed.py`  
  <https://github.com/pytorch/pytorch/blob/v2.5.0/torch/nn/parallel/distributed.py>
- C++ 侧：`torch/csrc/distributed/c10d/reducer.cpp`  
  <https://github.com/pytorch/pytorch/blob/v2.5.0/torch/csrc/distributed/c10d/reducer.cpp>

前者告诉你“策略与入口”，后者告诉你“时序与触发点”。后续所有魔改讨论，几乎都会落回这两处。

## 参考资料

- PyTorch 2.5：`DistributedDataParallel`  
  <https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html>
- PyTorch 源码（v2.5.0）：`distributed.py` / `reducer.cpp` / `reducer.hpp`
- arXiv:2006.15704  
  <https://arxiv.org/abs/2006.15704>
