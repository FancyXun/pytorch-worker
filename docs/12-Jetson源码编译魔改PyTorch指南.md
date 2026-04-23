# Jetson 源码编译魔改 PyTorch 指南

本文给出在 **NVIDIA Jetson**（ARM64）上，从源码编译当前仓库的魔改 PyTorch（含异构 DDP 逻辑）的可执行步骤。目标是产出可安装的 wheel，并在 Jetson 上直接运行。

> 适用目录：`pytorch_fork_asymmetric_ddp/pytorch`

---

## 1. 前置确认

先确认 Jetson 基础信息：

```bash
uname -a
python3 --version
nvcc --version
cat /etc/nv_tegra_release
```

建议：

- JetPack / CUDA / Python 版本保持一致性（不要混装多个 CUDA）。
- 编译过程非常吃内存，建议至少准备 **8GB+ RAM + 8GB+ swap**。

### 你的当前环境（已确认）

根据你提供的输出：

- Kernel: `5.15.148-tegra`
- Python: `3.10.12`
- CUDA: `12.6` (`nvcc 12.6.68`)
- L4T: `36.4.3`（`nvidia-l4t-core 36.4.3`）

这对应 **JetPack 6.x 系列（R36）**。本文后续步骤可直接套用。

---

## 2. 准备系统依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  ninja-build \
  git \
  libopenblas-dev \
  libjpeg-dev \
  zlib1g-dev \
  libpython3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel
```

针对你当前 Python 3.10，建议补齐：

```bash
sudo apt install -y python3.10-dev
```

> 如果你需要分布式 Gloo 网络功能，系统里保留常用网络工具（例如 `iproute2`）会更方便排障。

---

## 3. 准备 Python 构建依赖

推荐用虚拟环境：

```bash
python3 -m venv ~/venvs/torch_build
source ~/venvs/torch_build/bin/activate
python -m pip install --upgrade pip
```

安装构建依赖：

```bash
pip install \
  pyyaml \
  numpy \
  typing_extensions \
  setuptools \
  wheel \
  ninja \
  cmake
```

---

## 4. 进入源码目录

```bash
cd /code/pytorch-worker/pytorch_fork_asymmetric_ddp/pytorch
```

确保子模块就绪（非常关键）：

```bash
git submodule sync
git submodule update --init --recursive
```

---

## 5. Jetson 上建议的构建环境变量

先设置 CUDA 架构（按机型选择）：

- Jetson Nano / TX1: `5.3`
- TX2: `6.2`
- Xavier NX / AGX Xavier: `7.2`
- Orin NX / AGX Orin: `8.7`

例如 Xavier：

```bash
export TORCH_CUDA_ARCH_LIST="7.2"
```

例如 Orin：

```bash
export TORCH_CUDA_ARCH_LIST="8.7"
```

再设置构建选项（Jetson 上更稳的一组）：

```bash
export USE_CUDA=1
export USE_CUDNN=1
export USE_DISTRIBUTED=1
export USE_GLOO=1
export USE_NCCL=0
export BUILD_TEST=0
export MAX_JOBS=4
```

> `USE_NCCL=0` 是为了 Jetson 场景更稳（单机多卡通常也不是 Jetson 主场景）；你当前异构 DDP 主路径基于 Gloo 可工作。

> 对你当前环境（L4T 36.4.3 + CUDA 12.6），若设备是 Orin 系列，继续使用 `TORCH_CUDA_ARCH_LIST="8.7"`。

---

## 6. 编译并打包 wheel

```bash
python setup.py bdist_wheel
```

输出通常在：

```bash
dist/torch-*.whl
```

安装：

```bash
pip install dist/torch-*.whl
```

---

## 7. 编译后快速验证

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("ddp class:", torch.nn.parallel.DistributedDataParallel)
PY
```

验证你的魔改接口是否在：

```bash
python - <<'PY'
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
print("DDP has is_trainer_rank:", hasattr(DDP, "is_trainer_rank"))
print("DDP has trainer_step:", hasattr(DDP, "trainer_step"))
print("DDP has sync_scalar_from_trainer:", hasattr(DDP, "sync_scalar_from_trainer"))
PY
```

---

## 8. 典型故障与处理

## 8.1 OOM / 编译被杀死

现象：`cc1plus: out of memory` 或进程被 kill。  
处理：

- 降低并行度：`export MAX_JOBS=2`
- 增加 swap（至少 8GB）
- 关闭不必要模块（如测试）

## 8.2 子模块缺失导致编译报错

现象：找不到 `third_party/...` 或 CMake 目标缺失。  
处理：

```bash
git submodule update --init --recursive
```

## 8.3 CUDA 检测失败

现象：`USE_CUDA=1` 但 `torch.cuda.is_available()` 为 false。  
处理：

- 确认 `nvcc --version`
- 确认 JetPack CUDA 运行时完整
- 确认没有混乱的 `LD_LIBRARY_PATH`

## 8.4 运行分布式报 Gloo 网络问题

处理建议：

- 显式设置 `GLOO_SOCKET_IFNAME`
- 保证 `MASTER_ADDR/MASTER_PORT` 可达
- 双向连通（容器场景注意 host/network 配置）

---

## 9. 给当前项目的最小实操流程

```bash
# 1) 激活环境
source ~/venvs/torch_build/bin/activate

# 2) 进入源码
cd /code/pytorch-worker/pytorch_fork_asymmetric_ddp/pytorch

# 3) 子模块
git submodule update --init --recursive

# 4) 构建参数（以 Orin 为例）
export TORCH_CUDA_ARCH_LIST="8.7"
export USE_CUDA=1 USE_CUDNN=1
export USE_DISTRIBUTED=1 USE_GLOO=1 USE_NCCL=0
export BUILD_TEST=0 MAX_JOBS=4

# 5) 编译
python setup.py bdist_wheel

# 6) 安装
pip install dist/torch-*.whl
```

如果你要严格按“你这台机器”一键执行，可用下面这段（等价于上面流程）：

```bash
source ~/venvs/torch_build/bin/activate
cd /code/pytorch-worker/pytorch_fork_asymmetric_ddp/pytorch
git submodule update --init --recursive

export TORCH_CUDA_ARCH_LIST="8.7"
export USE_CUDA=1
export USE_CUDNN=1
export USE_DISTRIBUTED=1
export USE_GLOO=1
export USE_NCCL=0
export BUILD_TEST=0
export MAX_JOBS=4

python setup.py bdist_wheel
pip install dist/torch-*.whl
```

---

## 10. 备注

如果你后续要在 CI/容器里重复构建，建议把本文步骤固化为一个 `build_jetson_torch.sh`，并把 `TORCH_CUDA_ARCH_LIST` 做成参数化输入（不同机型可复用同一脚本）。

