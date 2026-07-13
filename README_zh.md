# gswarp

[English](README.md) · **中文**

gswarp 是一个基于 **NVIDIA Warp** 的 3D Gaussian Splatting 加速后端，用纯 Python 重新实现了 3DGS 训练管线中的三个核心 CUDA 模块——光栅化器、SSIM 损失和 KNN 初始化。无需 C++/CUDA 编译，通过 pip 安装后即可直接替换原有 CUDA 实现。

> **许可证**：[Apache License 2.0](LICENSE)。第三方归属见 [NOTICE](NOTICE)。

---

## 目录

- [三个替换模块](#三个替换模块)
- [当前架构](#当前架构)
- [系统要求](#系统要求)
- [安装](#安装)
- [在 3DGS 框架中替换 CUDA 后端](#在-3dgs-框架中替换-cuda-后端)
  - [光栅化器替换](#光栅化器替换)
  - [SSIM 替换](#ssim-替换)
  - [KNN 替换](#knn-替换)
  - [推荐：一次性全部替换](#推荐一次性全部替换)
- [基准测试](#基准测试)
- [详细文档](#详细文档)
- [致谢](#致谢)

---

## 三个替换模块

| 模块 | 替换目标 | 导入路径 | 说明 |
|------|----------|----------|------|
| **光栅化器** | `diff_gaussian_rasterization` | `gswarp` | 类型化分阶段光栅化、手写反向与 stream-safe 缓存 |
| **SSIM** | `fused_ssim` | `gswarp.fused_ssim` | 可分离高斯卷积，使用随 autograd graph 持有的可复用计划 |
| **KNN** | `simple_knn` | `gswarp.knn` | 在当前 PyTorch stream 上执行 Morton 排序和包围盒剪枝 3-NN |

---

## 当前架构

公开模块保持为轻量兼容层。光栅化器内部解析并缓存不可变的方法计划，再执行明确的阶段：

```text
公开 API -> 输入校验/autograd -> 方法计划
        -> preprocess -> features -> binning -> render -> typed forward state
        -> manual backward
```

标准 3DGS 与可选 flow 后端共用预处理、分箱、调用级运行时选项、stream 互操作、workspace 管理和大部分反向实现。方法专属后端只适配阶段输入输出、辅助数据和需要保留的反向状态。

每次调用都会冻结自己的运行时选项，并将 Warp 绑定到当前 PyTorch CUDA 设备和 stream。可复用 workspace 与 recorded launch 按设备和 stream 有界缓存；forward 所有的张量随 autograd graph 保留，直到 backward 释放。`clear_warp_caches()` 和 `get_warp_cache_report()` 用于显式管理缓存生命周期。

stable 后端面向声明的最低 Warp 版本。可选 advanced 后端只有在最低版本与所需 Warp 能力同时满足时才会启用；当前发布尚未启用任何 advanced 后端。

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
pip install "warp-lang>=1.8.0" gswarp
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

`GaussianRasterizationSettings` 是 `NamedTuple`，包含 CUDA 兼容字段以及带默认值的 Warp 专属字段。`GaussianRasterizer.forward()` **返回三值元组 `(color, radii, meta)`**，其中 `meta` 是 `RasterizerMeta` NamedTuple，携带五个附加输出：

| `meta` 字段 | 含义 |
|-------------|------|
| `meta.depth` | 每像素累积深度 |
| `meta.alpha` | 每像素累积不透明度 |
| `meta.proj_2D` | 各高斯在图像平面上的 2D 投影坐标 |
| `meta.conic_2D` | 2D 协方差的 conic 表示 |
| `meta.conic_2D_inv` | 2D conic 的逆 |

```python
# 原版 CUDA（仅两个输出）：
color, radii = rasterizer(means3D=..., means2D=..., ...)

# gswarp（三值元组）：
color, radii, meta = rasterizer(means3D=..., means2D=..., ...)

# 使用 meta 中的附加输出：
depth      = meta.depth        # (1, H, W)
alpha      = meta.alpha        # (1, H, W)
proj_2D    = meta.proj_2D      # (N, 2)
conic_2D   = meta.conic_2D     # (N, 3)
conic_2D_inv = meta.conic_2D_inv  # (N, 3)

# 仅需 color 与 radii 时可忽略 meta：
color, radii, _ = rasterizer(means3D=..., means2D=..., ...)
```

迁移现有代码时应使用关键字参数。`shs`、`colors_precomp`、`scales`、`rotations` 与 `cov3D_precomp` 遵守 CUDA 光栅化器的互斥输入规则；`dc` 可作为 `shs` 的兼容别名。当前 stable 后端尚不支持 `prefiltered=True`，也不会执行 CUDA 的 antialiasing 路径。

### 光流后端（可选）

光流管线需要逐像素 Top-K 贡献信息时，请改用独立的光流后端 `gswarp.rasterizer_flow`。

与非光流版本的主要差异：

**1. `GaussianRasterizationSettings` 新增两个专有字段**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_flow_grad` | `bool` | `True` | 启用光流梯度计算（非光流版本无此字段） |
| `compute_flow_aux` | `bool \| None` | `None`（运行时默认 `True`） | 控制是否填充 Top-K 辅助输出 |

**2. `forward()` 返回完整的 10 元组，不再使用 `meta`**

```python
from gswarp.rasterizer_flow import GaussianRasterizationSettings, GaussianRasterizer

raster_settings = GaussianRasterizationSettings(
    ...,                    # 与非光流版本相同的公共字段
    enable_flow_grad=True,  # 默认 True
    compute_flow_aux=True,  # 默认 None（运行时 True）
)

color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, \
    gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(...)
```

三个辅助输出的形状与含义：

| 输出名称 | 形状 | dtype | 含义 |
|----------|------|-------|------|
| `gs_per_pixel` | `(K, H, W)` | `int32` | 每像素 Top-K 贡献高斯的索引（未填满槽位置为 `-1`） |
| `weight_per_gs_pixel` | `(K, H, W)` | `float32` | 对应高斯的混合权重 |
| `x_mu` | `(2, K, H, W)` | `float32` | 对应高斯投影中心到像素中心的偏移 `(dx, dy)` |

其中 `K` 默认为 `20`，可通过运行时 API 调整：

```python
from gswarp.rasterizer_flow import set_flow_topk, set_compute_flow_aux

set_flow_topk(32)            # 修改 K（需在首次渲染前调用，改变 K 会清空 Warp 内核启动缓存）
set_compute_flow_aux(False)  # 临时禁用辅助输出以节省显存
```

**可选的运行时配置**（在训练循环开始前调用一次）：

```python
from gswarp import initialize_runtime_tuning, set_binning_sort_mode

# 初始化 Warp 并记录当前设备的 launch/tuning 报告
initialize_runtime_tuning(device="cuda:0", verbose=True)

# stable 默认使用 32 位深度排序，再进行稳定的 tile 排序
set_binning_sort_mode("warp_depth_stable_tile")
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

## 基准测试

当前数据由本地构建的包复测得到，而非直接从 checkout 导入。CUDA 对照使用原生 `diff_gaussian_rasterization` 扩展；gswarp 被安装到独立目标目录，并在训练前核对源码哈希。该隔离方式避免两个同名包通过 `PYTHONPATH` 相互遮蔽。

**环境。** NVIDIA GeForce RTX 5090（32 GiB，sm_120）、NVIDIA 驱动 610.62、Python 3.14.3、PyTorch 2.11.0+cu130、Warp 1.12.0，以及由当前工作区构建的 gswarp 1.1.0。所有运行均采用原始 3DGS 默认优化参数、`--data_device cpu`、默认 Adam 和 30,000 次迭代。Warp 路径使用 gswarp 光栅化器、fused SSIM 与 KNN；该参考训练的 loss 不消费 depth，因此关闭 Warp depth accumulation。

当前矩阵覆盖全部三个标准评测族：8 个 NeRF Synthetic 场景、2 个 Tanks and Temples 场景和 2 个 Deep Blending 场景。这是完成的 12 场景、30K 迭代套件，而非代表性子集。

### 端到端训练

墙钟吞吐包含完整训练循环、最终评估和 checkpoint 保存。稳定 GPU 吞吐取自 CUDA event 在迭代 25,000-29,999 的记录，排除最终评估与保存。峰值显存为训练脚本重置计数器后记录的 PyTorch peak allocated。

| 场景 | 数据集族 | CUDA min | Warp min | 墙钟比 | CUDA 稳定 it/s | Warp 稳定 it/s | CUDA/Warp 最终高斯数 | CUDA/Warp 峰值 MiB |
|------|----------|---------:|---------:|-------:|---------------:|---------------:|---------------------:|-------------------:|
| chair | NeRF Synthetic | 6.95 | 6.69 | 1.04x | 73.33 | 73.32 | 298,936 / 297,746 | 616 / 778 |
| drums | NeRF Synthetic | 6.58 | 6.61 | 1.00x | 77.61 | 74.90 | 319,786 / 319,116 | 649 / 797 |
| ficus | NeRF Synthetic | 5.60 | 5.93 | 0.95x | 90.87 | 84.89 | 177,831 / 177,639 | 401 / 563 |
| hotdog | NeRF Synthetic | 7.40 | 6.02 | 1.23x | 73.45 | 83.88 | 157,488 / 157,839 | 433 / 529 |
| lego | NeRF Synthetic | 6.78 | 6.54 | 1.04x | 76.70 | 75.82 | 298,102 / 299,285 | 648 / 770 |
| materials | NeRF Synthetic | 6.27 | 6.07 | 1.03x | 82.93 | 82.57 | 238,715 / 236,993 | 520 / 670 |
| mic | NeRF Synthetic | 6.58 | 6.92 | 0.95x | 76.49 | 70.99 | 277,299 / 276,471 | 606 / 730 |
| ship | NeRF Synthetic | 9.31 | 6.80 | 1.37x | 61.01 | 73.29 | 309,749 / 310,845 | 811 / 789 |
| train | Tanks and Temples | 11.22 | 9.29 | 1.21x | 44.88 | 50.35 | 1,094,613 / 1,089,398 | 1,902 / 2,108 |
| truck | Tanks and Temples | 12.47 | 11.38 | 1.10x | 39.89 | 41.49 | 2,051,896 / 2,052,183 | 3,351 / 3,696 |
| drjohnson | Deep Blending | 23.34 | 16.58 | 1.41x | 21.34 | 27.93 | 3,109,310 / 3,109,155 | 5,270 / 5,639 |
| playroom | Deep Blending | 18.31 | 20.38 | 0.90x | 28.12 | 37.85 | 1,842,952 / 1,842,580 | 3,149 / 3,474 |

完整套件中，CUDA 总墙钟为 120.81 分钟，全 Warp 为 109.23 分钟，墙钟比为 1.106x。这是全套件汇总，不是逐场景比值的算术平均。Warp 在 12 个完整任务中的 8 个更快，在 12 个稳定迭代窗口中的 6 个更快。Warp 的训练 peak allocated 通常更高，这是实测 tradeoff，并非显存降低声明。稳定窗口阶段拆分和当前 Nsight 证据见[光栅化器文档](docs/rasterizer_zh.md)。

### 热启动推理

对冻结的 Warp 30K checkpoint，测量三次已预热的完整测试视图 pass，并报告 CUDA event 时间中位数。每次 pass 先预热 100 个视图。该设置对应实际集成路径：当调用方不消费 depth 时，Warp 关闭 depth accumulation。

| 场景 | 后端 | GPU ms/view | Warp/CUDA 比 | 峰值 allocated |
|------|------|------------:|-------------:|---------------:|
| Lego | CUDA | 1.8224 | 1.00x | 290 MiB |
| Lego | Warp | 1.6382 | 1.11x | 206 MiB |
| Truck | CUDA | 4.2975 | 1.00x | 1,317 MiB |
| Truck | Warp | 4.0478 | 1.06x | 1,207 MiB |

### 独立训练质量

以下测试图像指标来自 CUDA 配置和全 Warp 配置分别完成 30K 训练后，运行原始 3DGS `render.py` 与 `metrics.py` 得到的结果。两种配置使用相同的优化参数，并统一由原生 CUDA 光栅化器完成最终评估。两条训练路径的梯度不要求位级一致：浮点归约顺序的差异可能在非凸优化过程中改变优化器和 densification 决策。因此，该表衡量端到端训练结果，而非逐像素光栅化器等价性。

| 场景 | CUDA PSNR | Warp PSNR | PSNR 差值 | CUDA SSIM | Warp SSIM | CUDA LPIPS | Warp LPIPS |
|------|----------:|----------:|----------:|----------:|----------:|-----------:|-----------:|
| chair | 35.6934 | 35.7688 | +0.0754 | 0.987447 | 0.987523 | 0.011773 | 0.011646 |
| drums | 26.1614 | 26.1687 | +0.0073 | 0.954811 | 0.954778 | 0.036494 | 0.036379 |
| ficus | 34.8947 | 34.9049 | +0.0102 | 0.987307 | 0.987330 | 0.011735 | 0.011737 |
| hotdog | 37.6701 | 37.7385 | +0.0684 | 0.985379 | 0.985416 | 0.019953 | 0.019955 |
| lego | 35.9071 | 35.9231 | +0.0161 | 0.983264 | 0.983284 | 0.015307 | 0.015234 |
| materials | 30.1175 | 30.1108 | -0.0066 | 0.961664 | 0.961612 | 0.032918 | 0.032874 |
| mic | 35.8806 | 35.6974 | -0.1832 | 0.992079 | 0.991871 | 0.005762 | 0.005878 |
| ship | 31.0945 | 31.0405 | -0.0540 | 0.907392 | 0.907365 | 0.105282 | 0.105641 |
| train | 22.2565 | 21.9888 | -0.2677 | 0.821652 | 0.820027 | 0.195832 | 0.196647 |
| truck | 25.5052 | 25.5192 | +0.0141 | 0.884745 | 0.884810 | 0.142212 | 0.142430 |
| drjohnson | 29.3829 | 29.4829 | +0.1000 | 0.904947 | 0.905391 | 0.236216 | 0.235705 |
| playroom | 30.1062 | 30.1593 | +0.0531 | 0.909675 | 0.908975 | 0.241131 | 0.240396 |

Warp 在 12 个独立训练场景中的 8 个获得更高 PSNR。最大负差值是 train 的 -0.2677 dB，最大正差值是 drjohnson 的 +0.1000 dB。该套件没有呈现统一的质量变化方向；场景级差值需要结合下方冻结 checkpoint 与模块级验证共同解释。

### 冻结 Checkpoint 一致性

为了从训练动力学中隔离光栅化器，使用原生 CUDA 扩展和 gswarp 分别渲染同一个 Warp 训练 checkpoint。每个视图使用完全相同的相机、高斯和背景输入。

| 场景 | 测试视图 | 图像 MAE | CUDA/Warp PSNR | 可见性 Jaccard | 最大绝对误差 |
|------|---------:|---------:|---------------:|---------------:|-------------:|
| Lego | 200 | 1.79e-7 | 105.54 dB | 1.000000 | 0.00996 |
| Truck | 32 | 4.64e-7 | 100.43 dB | 1.000000 | 0.01326 |
| Train | 38 | 5.10e-7 | 100.33 dB | 0.99999992 | 0.00478 |

Lego 和 Truck 的可见高斯集合完全相同。Train 的 Jaccard 与 1.0 的差异，对应约 2560 万次 Gaussian-view 观测中的 2 次单侧可见性判定；原生 CUDA 与 Warp 图像相对 ground truth 的全局 PSNR 也均为 20.955083 dB（按展示精度一致）。表中的 CUDA/Warp PSNR 衡量两种渲染输出之间的一致性，不是相对 ground truth 的重建质量。

这些结果支持光栅化器在已测 FP32 容差内等价，且没有显示系统性漏绘；但它们不意味着两条独立的非凸训练轨迹必须收敛到相同 checkpoint。阶段耗时、来源隔离规则及解读限制见[光栅化器文档](docs/rasterizer_zh.md)，SSIM 梯度与训练路径的可控对照见 [SSIM 文档](docs/ssim_zh.md)。

---

## 详细文档

| 文档 | 内容 |
|------|------|
| [docs/rasterizer_zh.md](docs/rasterizer_zh.md) | 光栅化器架构、CUDA 实现差异、微基准、正确性、已知限制 |
| [docs/ssim_zh.md](docs/ssim_zh.md) | SSIM 内核优化、性能分析、正确性验证 |
| [docs/knn_zh.md](docs/knn_zh.md) | KNN 算法、Morton 排序、执行模型、正确性验证 |

---

## 致谢

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)（INRIA / MPII）
- [fused-ssim](https://github.com/rahul-goel/fused-ssim)（Rahul Goel et al.）
- [simple-knn](https://github.com/camenduru/simple-knn)（graphdeco-inria）
- [Fast Converging 3DGS](https://arxiv.org/abs/2601.19489)（Zhang et al., 2025）——AABB 剔除启发来源
- [NVIDIA Warp](https://nvidia.github.io/warp/)

