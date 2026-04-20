# PyTorch 分布式训练（三）：DDP 内部、`Reducer` 与梯度分桶

**源码与 API 版本**：文中摘录的代码与 API 行为均以官方 **PyTorch 2.5.x** 为准；代码定位使用 GitHub 标签 **[v2.5.0](https://github.com/pytorch/pytorch/releases/tag/v2.5.0)**，与 [PyTorch 2.5 文档](https://pytorch.org/docs/2.5/) 一致。

数据并行要对齐各 rank 上的梯度；`torch.distributed` 提供 `ProcessGroup` 与 `all_reduce`；`DistributedDataParallel` 在模块外再包一层，在反向时触发对齐。本篇说明：**梯度不是在 Python 的 `forward()` 里一次性发完的**，而是在反向传播过程中，由 C++ 侧的 **`Reducer`** 按 **bucket（桶）** 分批触发通信，并由此带来「通信与计算重叠」「桶字节上限」等工程细节。


## 1. 为什么需要 `Reducer`

反向传播沿计算图从输出往输入走，**各参数的梯度就绪时间不同**。若每就绪一个参数就立刻发起一次小规模 `all_reduce`，会产生大量小消息，链路上**有效带宽**往往很差。PyTorch 在 C++ 里维护 **`Reducer`**：把多个参数按 **bucket** 分组，某个桶内梯度都就绪后，再对该桶做一次（或一组）通信，使单次 payload 更大、更接近线性带宽区段。

与桶相关的默认常量及「单桶累加器」类型（节选）：

出处：[torch/csrc/distributed/c10d/reducer.hpp（v2.5.0，约 L29–L42）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/csrc/distributed/c10d/reducer.hpp#L29-L42)

```cpp
constexpr int kDefaultFirstBucketBytes = int(1024 * 1024);
constexpr int kDefaultBucketBytesCap = int(25 * 1024 * 1024);
// Collect runtime stats once for every kDDPRuntimeLoggingSampleRate iterations.
constexpr int kDDPRuntimeLoggingSampleRate = 100;

// Forward declaration
class Logger;

// Local accumulator type for a single bucket.
struct BucketAccumulator {
  std::vector<size_t> indices;
  size_t size = 0;
  size_t size_limit = 0;
};
```

`Reducer` 的**完整构造函数签名与全部参数**以官方文件为准，可直接打开：

[torch/csrc/distributed/c10d/reducer.hpp @ v2.5.0](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/csrc/distributed/c10d/reducer.hpp)

类前的注释说明了：`bucket_indices` 给出每个桶包含哪些参数下标；`ProcessGroup` 用于实际通信。实现细节见同目录下的 **`reducer.cpp`**。


## 2. `DistributedDataParallel` 与 `Reducer` 的分工

- **`DistributedDataParallel`（Python）**：模块包装、设备、`find_unused_parameters`、混合精度、`Join` 等与训练脚本直接交互的部分。  
- **`Reducer`（C++）**：在 backward 路径上注册与调度 autograd hook，维护 bucket 就绪状态，在合适时机调用 **`ProcessGroup`** 上的通信。

训练脚本里仍是 `loss.backward()`；**何时**把某一桶梯度交给后端，由 `Reducer` 决定。


## 3. 默认 allreduce 与通信 hook 的入口

出处：[torch/csrc/distributed/c10d/reducer.hpp（v2.5.0，约 L90–L99、L108–L114）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/csrc/distributed/c10d/reducer.hpp#L90-L114)

```cpp
  // Registers a hook to the reducer. The hook is `CommHookInterface`
  // type to allow both Python and CPP hooks. This function can only
  // be called once before calling backward.
  // Cannot combine with the call of `register_builtin_comm_hook`.
  void register_comm_hook(std::unique_ptr<CommHookInterface> iface);

  // Registers a built-in C++ comm hook to the reducer. This function can only
  // be called once before calling backward.
  // Cannot combine with the call of `register_comm_hook`.
  void register_builtin_comm_hook(c10d::BuiltinCommHookType comm_hook_type);
```

```cpp
  // Runs allreduce or installed communication hook given GradBucket instance.
  c10::intrusive_ptr<c10::ivalue::Future> run_comm_hook(
      GradBucket& grad_bucket);

  // Runs default allreduce hook.
  c10::intrusive_ptr<c10::ivalue::Future> run_allreduce_hook(
      GradBucket& grad_bucket);
```

默认路径下，未安装自定义 hook 时，桶内梯度通过 **`run_allreduce_hook`** 走 **`all_reduce`** 语义；安装 hook 后则由 **`run_comm_hook`** 接管。详见 [DistributedDataParallel](https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html) 与源码 **`reducer.cpp`** 中的调用链。


## 4. 与论文 arXiv:2006.15704 的对应

PyTorch 团队在 *PyTorch Distributed: Experiences on Accelerating Data Parallel Training*（arXiv:2006.15704，<https://arxiv.org/abs/2006.15704>）中概括的加速手段，与上文机制一一对应：

- **Bucketing**：对应 `bucket_indices` 与桶字节上限等策略。  
- **Overlapping computation with communication**：已就绪的桶可先发起通信，与尚未结束的 backward 重叠。  
- **Skipping gradient synchronization**：训练策略与 API 选项，需在理解语义后使用。

公开文档中与 **bucket 大小**、`bucket_cap_mb` 等相关的参数名，以 **PyTorch 2.5 文档** 当前页为准。


## 5. `find_unused_parameters`

若某轮前向中部分参数未参与计算图，对应参数可能没有梯度。`find_unused_parameters=True` 会启用额外逻辑以保证一类正确性，通常有额外开销；若每次前向都用到全部参数，应保持默认 **`False`**。说明见 [DistributedDataParallel](https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html)。


## 6. 一次反向里与 DDP 相关的第 4 步

4. 反向过程中，`Reducer` 按 bucket 收集就绪梯度；条件满足时通过 `ProcessGroup` 对该桶执行 **`all_reduce`**（或 comm hook 定义的等价通信）；在实现允许的情况下，该通信可与仍在进行的 backward **重叠**。Profiler 时间轴上 NCCL 与 CUDA kernel 交错，多与该重叠有关。


## 参考资料

- PyTorch 2.5：[DistributedDataParallel](https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html)  
- 源码：[reducer.hpp / reducer.cpp @ v2.5.0](https://github.com/pytorch/pytorch/tree/v2.5.0/torch/csrc/distributed/c10d)  
- Shen Li 等，arXiv:2006.15704：<https://arxiv.org/abs/2006.15704>
