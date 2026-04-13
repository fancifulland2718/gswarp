# gswarp

[English](README.md) · **中文**

gswarp 是一个基于 **NVIDIA Warp** 的 3D Gaussian Splatting 加速后端，用纯 Python 重新实现了 3DGS 训练管线中的三个核心 CUDA 模块——光栅化器、SSIM 损失和 KNN 初始化。无需 C++/CUDA 编译，通过 pip 安装后即可直接替换原有 CUDA 实现。

> **许可证**：[Apache License 2.0](LICENSE)。第三方归属见 [NOTICE](NOTICE)。

---

## 目录

- [三个替换模块](#三个替换模块)
- [系统要求](#系统要求)
- [安装](#安装)
- [在 3DGS 框架中替换 CUDA 后端](#在-3dgs-框架中替换-cuda-后端)
  - [光栅化器替换](#光栅化器替换)
  - [SSIM 替换](#ssim-替换)
  - [KNN 替换](#knn-替换)
  - [推荐：一次性全部替换](#推荐一次性全部替换)
- [整体性能](#整体性能)
- [质量指标](#质量指标)
- [详细文档](#详细文档)
- [致谢](#致谢)

---

## 三个替换模块

| 模块 | 替换目标 | 导入路径 | 说明 |
|------|----------|----------|------|
| **光栅化器** | `diff_gaussian_rasterization` | `gswarp` | 完整的可微高斯光栅化 + 自动调优 |
| **SSIM** | `fused_ssim` | `gswarp.fused_ssim` | 可分离高斯卷积，带 launch 缓存 |
| **KNN** | `simple_knn` | `gswarp.knn` | Morton 排序 + 包围盒剪枝的 3-NN |

---

## 系统要求

| 组件 | 最低版本 |
|------|---------|
| Python | 3.10+ |
| NVIDIA GPU | 计算能力 ≥ 7.0 (Volta) |
| PyTorch | 1.13+（含 CUDA 支持） |
| NVIDIA Warp | 1.8.0+ |

---

## 安装

```bash
pip install gswarp
```

`gswarp` 会通过依赖自动安装 `warp-lang`。如果你希望显式固定 Warp 版本，可以使用：

```bash
pip install "warp-lang>=1.12.0" gswarp
```

或从源码安装：

```bash
git clone https://github.com/fancifulland2718/gswarp.git
cd gswarp
pip install .
```

安装后无需额外编译步骤。首次调用任何 Warp 内核时会触发 JIT 编译（约数秒），之后使用缓存。

---

## 在 3DGS 框架中替换 CUDA 后端

以下以 [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 官方代码库为例，说明如何将三个 CUDA 后端替换为 gswarp。

### 光栅化器替换

原始代码（`gaussian_renderer/__init__.py`）：

```python
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
```

替换为：

```python
from gswarp import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
```

`GaussianRasterizationSettings` 是 `NamedTuple`，字段与原版完全一致（额外的 Warp 专有字段均有默认值）。`GaussianRasterizer.forward()` 的返回值比原版**多额外输出**：

```python
# 原版 CUDA：
color, radii = rasterizer(means3D=..., means2D=..., ...)

# gswarp：
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, \
    gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(...)
# 仅使用 color 和 radii 时可以忽略其余输出
```

**可选的运行时配置**（在训练循环开始前调用一次）：

```python
from gswarp import initialize_runtime_tuning, set_binning_sort_mode

# 检测 GPU 并自动选择最优 block_dim（推荐）
initialize_runtime_tuning(device="cuda:0", verbose=True)

# 手动选择排序模式（默认 warp_depth_stable_tile 通常最优）
set_binning_sort_mode("warp_depth_stable_tile")  # 推荐（大场景优势明显）
# set_binning_sort_mode("warp_radix")            # 备选
# set_binning_sort_mode("torch")                 # 回退
```

### SSIM 替换

原始代码（`train.py`）：

```python
from fused_ssim import fused_ssim
```

替换为：

```python
from gswarp.fused_ssim import fused_ssim
```

函数签名完全一致：

```python
loss_ssim = fused_ssim(img1, img2, padding="same", train=True)
```

### KNN 替换

原始代码（`scene/gaussian_model.py`）：

```python
from simple_knn._C import distCUDA2
```

替换为：

```python
from gswarp.knn import distCUDA2
```

函数签名一致：

```python
dist2 = distCUDA2(points)  # points: (N, 3) float32 CUDA tensor
```

### 推荐：一次性全部替换

将上述三处修改合并，在 `train.py` 开头添加：

```python
try:
    from gswarp import GaussianRasterizationSettings, GaussianRasterizer
    from gswarp.fused_ssim import fused_ssim
    from gswarp.knn import distCUDA2
    GSWARP_AVAILABLE = True
except ImportError:
    GSWARP_AVAILABLE = False
```

并在各自的使用位置通过 `GSWARP_AVAILABLE` 标志切换后端。参考实现见 [gaussian-splatting/train.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py)（已集成 gswarp 切换逻辑）。

---

## 整体性能

以下数据来自 12 个标准 3DGS 数据集的完整 30K 步训练，测试环境为 RTX 5090D V2（sm_120，24 GiB），Python 3.14，PyTorch 2.11.0+cu130，Warp 1.12.0。**三个模块均使用 Warp 后端**，含 Python 层开销优化。

| 数据集 | CUDA (it/s) | Warp (it/s) | 速度比 |
|--------|------------|------------|--------|
| chair | 103.6 | 113.1 | ×1.09 |
| drums | 103.0 | 115.3 | ×1.12 |
| ficus | 139.5 | 148.2 | ×1.06 |
| hotdog | 144.5 | 156.5 | ×1.08 |
| lego | 117.5 | 126.5 | ×1.08 |
| materials | 134.0 | 144.9 | ×1.08 |
| mic | 95.4 | 105.1 | ×1.10 |
| ship | 107.0 | 113.2 | ×1.06 |
| train | 55.6 | 58.3 | ×1.05 |
| truck | 39.4 | 40.1 | ×1.02 |
| drjohnson | 30.8 | 32.0 | ×1.04 |
| playroom | 46.9 | 47.5 | ×1.01 |

**NeRF Synthetic（8 场景）**：平均 ×1.08 加速；**Tanks & Temples / Deep Blending（4 大场景）**：平均 ×1.03 加速。大场景（drjohnson、playroom）中 Warp 排序模式优势不如中等规模场景明显，原因是这类场景的高斯点数更多，Python 层开销占比更低，光栅化内核本身差异被稀释（详见[光栅化器文档](docs/rasterizer_zh.md)）。

---

## 质量指标

训练 30K 步后在测试集上的评估结果：

**NeRF Synthetic（8 场景均值）**

| 指标 | CUDA | Warp | 差异 |
|------|------|------|------|
| PSNR (dB) | 33.31 | 33.33 | +0.02 |
| SSIM | 0.9692 | 0.9693 | +0.0001 |
| LPIPS | 0.0303 | 0.0302 | −0.0001 |

**Tanks & Temples（2 场景均值）**

| 指标 | CUDA | Warp | 差异 |
|------|------|------|------|
| PSNR (dB) | 23.74 | 23.79 | +0.04 |
| SSIM | 0.8512 | 0.8515 | +0.0003 |
| LPIPS | 0.1711 | 0.1707 | −0.0004 |

**Deep Blending（2 场景均值）**

| 指标 | CUDA | Warp | 差异 |
|------|------|------|------|
| PSNR (dB) | 29.77 | 30.01 | +0.04 |
| SSIM | 0.9062 | 0.9063 | +0.0001 |
| LPIPS | 0.2390 | 0.2388 | −0.0002 |

各场景 PSNR 差异均在 ±0.25 dB 以内，SSIM 差异 < 0.001。Warp 后端在所有场景中的训练质量与 CUDA 基线等价。

---

## 详细文档

| 文档 | 内容 |
|------|------|
| [docs/rasterizer_zh.md](docs/rasterizer_zh.md) | 光栅化器架构、CUDA 实现差异、微基准、正确性、已知限制 |
| [docs/ssim_zh.md](docs/ssim_zh.md) | SSIM 内核优化、性能分析、正确性验证 |
| [docs/knn_zh.md](docs/knn_zh.md) | KNN 算法、Morton 排序、性能分析 |

---

## 致谢

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)（INRIA / MPII）
- [fused-ssim](https://github.com/rahul-goel/fused-ssim)（Rahul Goel et al.）
- [simple-knn](https://github.com/camenduru/simple-knn)（graphdeco-inria）
- [Fast Converging 3DGS](https://arxiv.org/abs/2601.19489)（Zhang et al., 2025）——AABB 剔除启发来源
- [NVIDIA Warp](https://nvidia.github.io/warp/)

