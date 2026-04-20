# PyTorch 分布式训练（二）：`torch.distributed`、进程组与集合通信

**源码与 API 版本**：文中摘录的代码与 API 行为均以官方 **PyTorch 2.5.x** 为准；代码定位使用 GitHub 标签 **[v2.5.0](https://github.com/pytorch/pytorch/releases/tag/v2.5.0)**，与 [PyTorch 2.5 文档](https://pytorch.org/docs/2.5/) 一致。

多 GPU / 多机训练时，程序往往被启动成**多个操作系统进程**（每个进程通常绑定一张 GPU）。它们之间要能交换张量、要能在关键点上彼此对齐，这套能力由 **`torch.distributed`** 提供。本篇说明：进程组是什么、常见后端各自适合什么场景、训练代码里反复出现的 **collective** 语义，以及 **`all_reduce` 在 API 层简单、在系统层并不简单** 的原因。


## 1. 启动之后第一件事：进程组与 `rank`

`torch.distributed.init_process_group` 会在每个进程里建立默认进程组（也可显式创建子组）。每个进程有一个整数 **`rank`**（从 0 到 `world_size - 1`），以及 **`world_size`** 表示参与该组的进程总数。

出处：[torch/distributed/distributed_c10d.py（v2.5.0，`init_process_group` 文档串节选，约 L1372–L1406）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/distributed/distributed_c10d.py#L1372-L1406)

```text
    """
    Initialize the default distributed process group.

    This will also initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
            to discover peers. Optionally specify ``rank`` and ``world_size``,
            or encode all required parameters in the URL and omit them.

    If neither is specified, ``init_method`` is assumed to be "env://".
...
        rank (int, optional): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).
                              Required if ``store`` is specified.
```

日常训练里常见模式：用 **`torchrun`** 拉起 `world_size` 个进程（环境变量 `RANK`、`WORLD_SIZE`、`LOCAL_RANK` 等）；每个进程调用 **`init_process_group`**；GPU 场景下再 **`torch.cuda.set_device(local_rank)`**。若 `init_process_group` 未成功执行，`DistributedDataParallel` 等依赖进程组的 API 无法按设计工作。


## 2. 后端：NCCL、Gloo、MPI、UCC

**后端**决定底层用什么库实现 `all_reduce`、`broadcast` 等。`init_process_group` 的文档中列举的有效取值包括 **`mpi`、`gloo`、`nccl`**，以及实验性的 **`ucc`**（见上引同一文件 `Args` 段中 `backend` 说明）。工程上常见选择是：**GPU 训练默认 `nccl`**；CPU 或跨平台场景可用 **gloo**；集群已统一 MPI 栈时可选 **mpi**。NCCL 在多进程 per-GPU 时有「每进程独占所使用 GPU」等约束，文档里亦有说明。


## 3. 点对点与集合通信

两进程之间单独 `send` / `recv` 叫**点对点**通信。训练脚本里更常见的是**集合通信（collective）**：一组预先约定的进程同时参与，各自提供输入缓冲区，按规则写回结果。

### 3.1 `broadcast` 与 `barrier`

- **`broadcast`**：根 rank 上的张量复制到组内其它 rank。  
- **`barrier`**：全组到达后才继续；调试或阶段同步有用，训练热路径不宜滥用。

### 3.2 `reduce_scatter` 与 `all_gather`

- **`reduce_scatter`**：各 rank 持有一块大向量的不同分片，规约后每个 rank 只保留结果的一部分；在 **FSDP / ZeRO** 等分片优化里与通信模式强相关（见同系列第五篇《其他分布式并行范式概览》）。  
- **`all_gather`**：各 rank 一份本地张量，通信后每 rank 拿到拼接后的全局视图；常用于指标汇总或某些并行策略中的激活拼接。

### 3.3 `all_reduce`：语义一层，实现一层

**在 PyTorch / c10d 这一层**，`all_reduce` 的语义可以一句话说完：组内每个 rank 提供同形张量，通信结束后，**每个 rank** 都得到各 rank 数据按给定 **ReduceOp**（如 SUM、MAX）规约后的**同一份结果**。数据并行里把各 rank 梯度对齐，用的就是「SUM + 必要时再缩放」这一套语义。

**在 NCCL / 网络栈这一层**，实现并不简单：同一语义可以由多种算法完成（例如 **ring**、**tree**、以及依赖拓扑与硬件的 **CollNet / SHARP** 等路径），调度器会根据进程数、消息大小、是否 GPU Direct、NVLink / PCIe / 跨机网络拓扑、是否多线程等选择或组合策略；消息还会被 **分块（chunk）**、**流水化**，以在带宽与延迟之间折中。因此：

- 你在 **Python** 里调用的 `dist.all_reduce(tensor, op=ReduceOp.SUM)` 是**稳定、简单的抽象**；  
- 你在 **profiler** 里看到的 `ncclKernel` 或类似条目，背后是**可能随 NCCL 版本变化的复杂实现**；  
- 读 **allreduce 经典论文**有助于建立**带宽与延迟**的直觉，但不能代替对当前 NCCL 行为与实测通信占比的分析。

`ProcessGroup` 在 C++ 侧对 `allreduce` 的虚接口（节选）：经 dispatcher 落到具体后端实现。

出处：[torch/csrc/distributed/c10d/ProcessGroup.hpp（v2.5.0，约 L154–L174）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/csrc/distributed/c10d/ProcessGroup.hpp#L154-L174)

```cpp
  virtual c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) {
    static auto op =
        c10::Dispatcher::singleton()
            .findSchemaOrThrow("c10d::allreduce_", "")
            .typed<
                std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
                    at::TensorList,
                    const c10::intrusive_ptr<::c10d::ProcessGroup>&,
                    const c10::intrusive_ptr<::c10d::ReduceOp>&,
                    const std::optional<at::Tensor>& sparse_indices,
                    int64_t)>();

    return std::get<1>(op.call(
        tensors,
        c10::intrusive_ptr<ProcessGroup>::unsafe_reclaim_from_nonowning(this),
        c10::make_intrusive<ReduceOp>(opts.reduceOp),
        opts.sparseIndices,
        opts.timeout.count()));
  }
```

标准 **DDP** 的默认梯度路径，以通过 `Reducer` 触发的 **`all_reduce`** 为主；`reduce_scatter` / `all_gather` 等更多出现在 **FSDP、自定义 comm hook、梯度压缩** 等广义分布式训练场景。


## 4. PyTorch API 与 NCCL 文档的分工

- **PyTorch 2.5 文档**：[`torch.distributed`](https://pytorch.org/docs/2.5/distributed.html) — collective 语义、超时、`device_id` 等与进程组相关的选项。  
- **NCCL 用户指南**：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html> — 集合通信在 NVIDIA 栈上的行为、环境变量与调优线索。


## 5. 与 DDP 的衔接（脚本层）

典型顺序是：各 rank 调用 `init_process_group` → 构造 `DistributedDataParallel(...)`。`DDP` 内部持有 `ProcessGroup`，反向时由 C++ `Reducer` 驱动何时发起 `all_reduce`（或 comm hook 的等价通信）。

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# init_process_group 已成功
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```


## 6. 其它并行范式

**张量并行、流水线并行、全分片数据并行（FSDP）、RPC** 等与「单副本 DDP + allreduce 梯度」不同的模型，见同系列第五篇《其他分布式并行范式概览》。


## 参考资料

- PyTorch 2.5：`torch.distributed` — <https://pytorch.org/docs/2.5/distributed.html>  
- NVIDIA NCCL 文档：<https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html>  
- Shen Li 等，arXiv:2006.15704 — <https://arxiv.org/abs/2006.15704>
