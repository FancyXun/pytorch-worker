# PyTorch 分布式训练（一）：数据并行在算什么，DDP 负责哪一段

**源码与 API 版本**：文中摘录的代码与 API 行为均以官方 **PyTorch 2.5.x** 为准；代码定位使用 GitHub 标签 **[v2.5.0](https://github.com/pytorch/pytorch/releases/tag/v2.5.0)**，与 [PyTorch 2.5 文档](https://pytorch.org/docs/2.5/) 一致。若你本地安装的是 2.5 补丁版本，行为以该版本为准，行号仍以 `v2.5.0` 树为参照。

单卡训练里，一轮迭代可以概括成：前向得到损失 → 反向把梯度写到各参数的 `.grad` → `optimizer.step()` 更新权重。多卡、多机时，最常用的一种扩展方式叫**数据并行**：每个进程里各放一份**结构相同**的模型，各读一份不同的 mini-batch，各自算出本地梯度，再用网络通信把「该对齐的量」对齐，最后各进程用**同一套优化规则**更新本地权重。

PyTorch 里与这条路径直接对应的两块是 **`torch.distributed`**（多进程怎么连上、怎么发起集合通信）和 **`torch.nn.parallel.DistributedDataParallel`**（常写作 DDP，在 `nn.Module` 外包一层，把梯度同步接到反向传播里）。本篇只说明**对称数据并行**在算什么、DDP 不做什么。


## 1. 数据并行要对齐的是梯度，不是「各算各的就算了」

设第 \(i\) 个进程上的 batch 损失为 \(\mathcal{L}_i\)。若把全局损失定义为各进程损失的**算术平均**：

\[
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \mathcal{L}_i
\]

则对共享参数 \(\theta\) 的梯度为

\[
\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \mathcal{L}_i.
\]

也就是说：每个进程先算出本地的 \(\nabla_\theta \mathcal{L}_i\)，需要的是**对所有进程的梯度取平均**（或等价地先对所有进程的梯度做 **sum**，再在每一步乘上同一个常数因子）。集合通信中的 **`all_reduce`** 所表达的，正是「每个参与者都拿到全局规约后的结果」这一类语义；在 DDP 的默认路径里，参与对齐的主要是**各参数的梯度**，使得规约之后各进程上的梯度一致，随后的 `optimizer.step()` 才会让各份权重仍保持同构。


## 2. 损失怎么聚合，会一路传到「等效 batch」和学习率

若你在**单条样本或小批量内部**对损失用 **sum** 而不是常见的 **mean** 来聚合，再叠加「多进程上对梯度取平均」，则最终梯度与「假想中单机一次吃齐所有样本」之间的比例关系，会和「全程用 mean」时不同。`DistributedDataParallel` 的官方文档里专门有一条说明：在多节点、每节点 batch 为 \(N\) 的配置下，若损失在实例维上是 **sum** 而非 **mean**，相对单机 batch 为 \(M \times N\) 的情形，梯度会差一个与进程数相关的倍数；若你要在数值上对齐单机大 batch 的训练，需要从**损失定义 → 梯度缩放 → 学习率与正则**整条链上自洽。

出处：[torch/nn/parallel/distributed.py（v2.5.0，约 L379–L388）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/nn/parallel/distributed.py#L379-L388)

```text
    .. note:: When a model is trained on ``M`` nodes with ``batch=N``, the
        gradient will be ``M`` times smaller when compared to the same model
        trained on a single node with ``batch=M*N`` if the loss is summed (NOT
        averaged as usual) across instances in a batch (because the gradients
        between different nodes are averaged). You should take this into
        consideration when you want to obtain a mathematically equivalent
        training process compared to the local training counterpart. But in most
        cases, you can just treat a DistributedDataParallel wrapped model, a
        DataParallel wrapped model and an ordinary model on a single GPU as the
        same (E.g. using the same learning rate for equivalent batch size).
```

这一点和「DDP 有没有 bug」无关，是**分布式只是把同一张计算图拆到多台机器上**，数学定义仍由你在代码里写的 `loss` 决定。


## 3. DDP 不替你切 batch：数据从哪来仍由 DataLoader / Sampler 决定

`DistributedDataParallel` 的职责是：**在模块级别，基于已初始化的进程组，把各副本上的梯度同步起来**。官方文档写明：**它不会**把输入在参与训练的设备上自动切分；典型做法是用 `DistributedSampler` 等，让每个 rank 的 DataLoader 在每个 epoch 里分到不重叠的子集，从而近似实现「全局一个打乱顺序的大池子，各进程各取一瓢」。

出处：[torch/nn/parallel/distributed.py（v2.5.0，约 L312–L321）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/nn/parallel/distributed.py#L312-L321)

```python
class DistributedDataParallel(Module, Joinable):
    r"""Implement distributed data parallelism based on ``torch.distributed`` at module level.

    This container provides data parallelism by synchronizing gradients
    across each model replica. The devices to synchronize across are
    specified by the input ``process_group``, which is the entire world
    by default. Note that ``DistributedDataParallel`` does not chunk or
    otherwise shard the input across participating GPUs; the user is
    responsible for defining how to do so, for example through the use
    of a :class:`DistributedSampler`.
```

与 DDP 配套的数据侧典型用法在 `DistributedSampler` 的文档串里写在一起。

出处：[torch/utils/data/distributed.py（v2.5.0，约 L16–L23）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/utils/data/distributed.py#L16-L23)

```python
class DistributedSampler(Sampler[_T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.
```

因此读代码时可以固定分工：

- **谁管数据**：`Dataset` + `Sampler` + `DataLoader`（以及你是否做梯度累积）。  
- **谁管梯度对齐**：DDP 及其内部的 C++ `Reducer` 与 `ProcessGroup`。  
- **谁管更新**：各进程上同一个优化器类、同一组超参（除非你自己刻意写成分歧逻辑，那就已不是标准数据并行）。


## 4. 参数与 buffer：DDP 文档里的两条不同同步规则

官方说明里有两句容易一起背混、但机制不同的话：

- **参数**：文档写明，**不会在进程之间广播参数**；各副本之间的一致性，来自「规约后的梯度相同 + 各进程用优化器对参数做相同更新」。  
- **Buffer**（例如 BatchNorm 的 running mean / variance）：文档写明，由 **rank 0** 上的那份模块向其余副本在迭代中广播，以保证 buffer 一致。

出处：[torch/nn/parallel/distributed.py（v2.5.0，约 L390–L395）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/nn/parallel/distributed.py#L390-L395)

```text
    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.
```

因此调试「多卡数值和单卡不一致」时，要分清是梯度规约、损失缩放问题，还是 BN 统计、buffer 版本与 `eval()`/`train()` 切换等问题。


## 5. 为什么多卡场景更常用 DDP 而不是 `nn.DataParallel`

**单机多 GPU**（一台机器上多张卡）做数据并行时，常见有两种用法：`nn.DataParallel` 是**单进程**里把 batch 维切到多 GPU；`DistributedDataParallel` 则是**多进程**，典型 **每进程一卡**。二者针对的是**同一类部署形态**（仍是单机、仍是数据并行），差别在进程模型与通信路径，而不是「一个单机、一个多机」。官方文档给出的结论是：在这种 **single-node multi-GPU data parallel** 设定下，**DDP 通常比 `DataParallel` 快得多**。

出处：[torch/nn/parallel/distributed.py（v2.5.0，约 L329–L331）](https://github.com/pytorch/pytorch/blob/v2.5.0/torch/nn/parallel/distributed.py#L329-L331)

```text
    ``DistributedDataParallel`` is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.
```

工程上的主要原因包括：多进程模型更易与 CUDA 设备绑定；梯度规约走 `torch.distributed` 后端的高效路径；与 autograd 的集成方式更适合做分桶与通信重叠等优化（见第三篇《DDP 内部、Reducer 与分桶》）。


## 6. 一次标准数据并行迭代里，各组件按什么顺序协作

下面按时间顺序列出**典型**一步里发生的事，便于和源码、profiler 时间线对照：

1. 各 rank 从自己的 DataLoader 取 batch（数据互不重复，由 Sampler 保证）。  
2. 各 rank 对本地 `model` 做前向，得到损失。  
3. 各 rank 对损失做 `backward()`，梯度写入参数的 `.grad`。  
4. DDP 侧的 `Reducer` 在反向过程中按 bucket 就绪情况，通过 `ProcessGroup` 发起 **all_reduce**（或你注册的通信 hook 的等价语义）。  
5. 各 rank 在已对齐的梯度上执行 `optimizer.step()`。  
6. 涉及 BN 等 buffer 时，按框架规则由 rank 0 广播到其它 rank。

profiler 时间轴里看到的通信算子，多半出现在第 4 步附近及其与后续计算的重叠区域。


## 参考资料

- PyTorch 2.5：`torch.distributed` — <https://pytorch.org/docs/2.5/distributed.html>  
- PyTorch 2.5：`DistributedDataParallel` — <https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html>  
- `Join` / uneven inputs：<https://pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.Join>
