# PyTorch 分布式训练（七）：DDP 关键开关与语义边界

**源码与 API 版本**：本文以官方 **PyTorch 2.5.x** 为准，源码参考 **[v2.5.0](https://github.com/pytorch/pytorch/releases/tag/v2.5.0)**。

写魔改前，最容易被忽略的不是“怎么改”，而是“原生 DDP 哪些行为由开关决定”。这一篇把最关键的开关放在一张清单里，避免后面把“原生语义”记错。

## 1. `find_unused_parameters`

- 作用：处理动态图或条件分支导致的“本轮未参与 loss 的参数”。  
- 代价：额外图遍历与状态管理开销。  
- 常见建议：结构稳定、每轮全参数参与时尽量保持 `False`。

文档入口：
<https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html>

## 2. `broadcast_buffers`

- 作用：每轮将 rank0 的 buffer（如 BN running stats）同步到其它 rank。  
- 关闭后：可能得到更高吞吐，但要自己保证 buffer 语义正确。

与“参数不广播、梯度 allreduce”不是一回事，二者是两条独立机制。

## 3. `gradient_as_bucket_view`

- 作用：让梯度直接视作 bucket 内存视图，减少复制与显存压力。  
- 风险点：某些自定义梯度后处理若假设 `.grad` 独立张量，可能触发兼容性问题。

## 4. `bucket_cap_mb`

- 作用：控制 bucket 大小，影响通信粒度与重叠效果。  
- 太小：通信次数过多，碎片化严重。  
- 太大：等待桶凑满时间变长，重叠变差。

这是调性能时最常见的 DDP 参数之一，但不能脱离模型结构与网络带宽独立讨论。

## 5. `no_sync()`

- 作用：在梯度累积时临时关闭某几步的梯度同步，仅在最后一步同步。  
- 典型用法：micro-batch 累积，减少通信频率。  
- 语义边界：关闭同步期间，各 rank 梯度是本地态；只有同步步后才重新对齐。

## 6. 通信 Hook（`register_comm_hook`）

- 作用：在 bucket 粒度接管默认 allreduce 路径。  
- 用途：压缩、量化、延迟同步、分层通信策略等。  
- 边界：这是“改通信路径”，不是“改优化器数学定义”。

源码入口：
- `reducer.hpp` 中 `register_comm_hook` / `run_comm_hook`  
  <https://github.com/pytorch/pytorch/blob/v2.5.0/torch/csrc/distributed/c10d/reducer.hpp>

## 7. `static_graph`

- 作用：在图结构稳定时减少 DDP 每轮动态处理成本。  
- 前提：计算图与参数使用模式足够稳定。  
- 风险：若实际图并不静态，可能出现语义或性能异常。

## 8. `join()`（uneven inputs）

- 作用：处理各 rank 样本数不均、提前结束导致的同步问题。  
- 场景：数据不整除、部分 rank 提前耗尽输入。  
- 不使用时：常见现象是某些 rank 卡死在 collective 等待。

文档入口：
<https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.Join>

## 9. 这篇清单如何服务后续魔改

后续要改 DDP 时，每一处改动都建议先标注它触碰的是哪类语义：

- 是在改 bucket 调度？  
- 还是改同步开关？  
- 还是改通信原语（allreduce → 其它）？

把这一步做好，后面的技术文会更清楚：读者能知道你是在“换策略”还是“换定义”。

## 参考资料

- PyTorch 2.5：`DistributedDataParallel` API  
  <https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html>
- PyTorch 源码（v2.5.0）：`distributed.py` / `reducer.hpp`
