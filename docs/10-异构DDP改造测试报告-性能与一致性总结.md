# PyTorch 分布式训练（十）：异构 DDP 改造测试报告（性能与一致性）

## 1. 测试目的

本次测试的目标很明确：验证异构 DDP（Trainer=GPU，Follower=CPU，Follower 跳过前向）在真实训练中的两件事：

1. **训练语义是否正确**：异构路径下 loss 曲线是否与本地基线一致。  
2. **性能收益是否成立**：异构方案相对本地 CPU 是否提速，以及相对本地 GPU 还存在多少差距。

---

## 2. 测试方案与日志

统一使用同一份数据配置、模型配置、优化器参数，仅改变执行模式。共采集四份日志：

- `local_gpu.log`：本地单进程 GPU（`train.py`）
- `local_cpu.log`：本地单进程 CPU（`train.py`）
- `hetero_trainer.log`：异构 Trainer（GPU，rank0）
- `hetero_follower.log`：异构 Follower（CPU，rank1，跳过前向，仅同步参数和指标）

分析脚本：

- `pytorch_fork_asymmetric_ddp/STGNN-for-Covid-in-Korea-master/analyze_bench_logs.py`
- `pytorch_fork_asymmetric_ddp/STGNN-for-Covid-in-Korea-master/plot_bench_logs.py`

本次统计基于导出的日志结果（共 30 个 epoch，性能统计默认去掉第 1 个 warmup epoch，因此计入 29 个 epoch）。

---

## 3. 核心结果

### 3.1 性能结果（`epoch_sec`）

- `local_gpu`：mean **0.414s**
- `local_cpu`：mean **14.862s**
- `hetero_trainer`：mean **4.798s**
- `hetero_follower`：mean **4.797s**

可以看到 `hetero_trainer` 与 `hetero_follower` 几乎重合，说明异构模式下整体节奏由同步点锁步推进，这是预期行为。

### 3.2 速度倍率

- `local_cpu / hetero_trainer` = **3.098x**  
  -> 异构方案相对纯 CPU 基线显著提速。

- `hetero_trainer / local_gpu` = **11.578x**  
  -> 异构方案仍显著慢于单机纯 GPU。

### 3.3 指标一致性（`epoch_mse`）

- `local_gpu` vs `hetero_trainer`：max abs diff = **0.0**
- `hetero_trainer` vs `hetero_follower`：max abs diff = **0.0**
- `metric_consistency_ok=True`（阈值 `1e-6`）

结论：本次改造在训练语义上是稳定的，Follower 即使不做前向，仍可通过内部指标同步拿到与 Trainer 一致的 loss。

---

## 4. 为什么异构仍明显慢于本地 GPU

这不是“GPU 算不动”，而是系统瓶颈不同：

1. **异构每 step 都有跨 rank 同步开销**  
   当前策略是同步间隔 `sync_interval=1`，每一步都要做参数同步与协调。

2. **CPU Follower 仍在关键路径上**  
   即便跳过前向，Follower 仍需要参与 step 级同步、参数接收、状态推进；Trainer 不能单边“飞跑”。

3. **通信栈与链路成本不可忽略**  
   Gloo/TCP 的延迟与 Python 调度开销，在小模型或较小 batch 下更容易放大。

4. **本地单机 GPU 的基线天然更“纯”**  
   单进程路径几乎没有分布式协调成本，所以 `0.4s/epoch` 是“纯算力上限”参考，不是异构系统目标值。

---

## 5. 当前改造是否值得

如果目标是“在 CPU-only 与异构之间做工程取舍”，答案是 **值得**：

- 相比本地 CPU，异构已获得约 **3.1x** 提升；
- 并且 loss 完全对齐，说明优化没有破坏训练语义。

如果目标是“逼近本地单机 GPU”，当前方案仍有优化空间，且这是预期现象。

---

## 6. 下一步优化方向（按优先级）

### P1：降低同步频率（收益大，改动小）

- 将 `TORCH_DDP_SYNC_INTERVAL` 从 `1` 提高到 `2/4/8` 做 AB test。  
- 预期：显著降低通信频率，提高吞吐；代价是参数陈旧度上升，需要验证收敛。

### P1：参数同步聚合（减少小包广播）

- 优化 `sync_params_from_trainer()`，考虑 flatten/coalesce 同步，减少逐 tensor 广播次数。  
- 预期：降低通信调度开销与 Python 层循环成本。

### P2：增加有效计算密度

- 适度增大 batch 或进行梯度累积，降低“每样本通信成本”。  
- 预期：单位 step 的固定开销被摊薄。

### P2：通信与计算重叠

- 在 Trainer 侧探索异步同步或 pipeline 化，尽量减少“硬 barrier”等待。  
- 预期：降低锁步等待时间。

### P3：运行时与内核层优化

- 使用 `torch.compile`、混合精度、内核融合等手段降低 Trainer 本地 step 时间。  
- 这部分无法消除分布式固定开销，但可进一步压缩总时延。

---

## 7. 图表占位与解读（可直接用于知乎）

下面给出三张图的**占位模板**。你只需要把 `TODO` 路径替换成你实际上传后的图片地址（本地路径或图床 URL）。

### 图 1：每个 Epoch 耗时对比

```markdown
![图1：每个Epoch耗时对比](TODO/epoch_time_comparison.png)
```

图注建议：  
四种运行模式（local_gpu、local_cpu、hetero_trainer、hetero_follower）的每 epoch 耗时曲线。`hetero_trainer` 与 `hetero_follower` 基本重合，说明异构训练由同步节奏锁步推进；`hetero_trainer` 明显快于 `local_cpu`，但慢于 `local_gpu`。

文中解读建议：

- 异构路径已显著降低 CPU-only 的训练时间；
- 异构双 rank 同步紧密，说明参数同步语义稳定；
- 与单机 GPU 的差距主要来自同步与通信成本，而非指标偏移。

### 图 2：每个 Epoch 指标（MSE）对比

```markdown
![图2：每个Epoch指标对比](TODO/epoch_mse_comparison.png)
```

图注建议：  
四种运行模式的 epoch_mse 曲线对比。异构 Trainer 与本地 GPU 曲线重合，异构 Follower 通过内部指标同步得到一致结果，表明本次改造没有破坏训练语义。

文中解读建议：

- 指标曲线重合意味着“快”不是以牺牲正确性为代价；
- follower 不做前向后，仍可得到可观测、可追踪的训练指标；
- 可以把“性能收益 + 指标一致”作为改造有效性的核心证据。

### 图 3：平均 Epoch 耗时柱状图

```markdown
![图3：平均Epoch耗时柱状图](TODO/mean_epoch_time_bar.png)
```

图注建议：  
去除 warmup epoch 后的平均 epoch 时间对比。异构方案相对 CPU-only 具有明显吞吐优势，但相较单机 GPU 仍存在系统级开销差距。

文中解读建议：

- 重点强调“异构是工程折中最优解”，不是“绝对最快”；
- 当前版本可作为可用基线，后续通过同步间隔、同步聚合等继续优化；
- 该图最适合作为结论图放在文末。

---

## 8. 总结

这次异构 DDP 改造的结论可以归纳为一句话：

**正确性上，已经与本地基线对齐；性能上，已显著优于纯 CPU，但仍受分布式同步路径制约，距离单机 GPU 上限存在可解释的系统差距。**

因此，这个版本可以作为“可用且可持续优化”的工程基线进入下一轮迭代。
