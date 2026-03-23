# gswarp

[English](README.md) · **中文**

一个基于 **纯 Python + NVIDIA Warp** 的可微高斯光栅化管线重新实现，原始版本以 CUDA C++ 编写，用于 [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)。本后端提供了与原生 CUDA 光栅化器的 **直接替换**（drop-in replacement）——无需 C++/CUDA 编译。

> **许可证**：本项目继承 [Gaussian-Splatting 许可证](LICENSE.md)（Inria 和 MPII，仅限非商业研究使用）。

---

## 目录

- [特性](#特性)
- [系统要求](#系统要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [架构概览](#架构概览)
- [与 CUDA 基线的差异](#与-cuda-基线的差异)
- [正确性](#正确性)
- [性能特征](#性能特征)
- [已知限制](#已知限制)
- [未来优化方向](#未来优化方向)
- [致谢](#致谢)

---

## 特性

- **完整的光栅化管线**：预处理 → 分箱 → 前向渲染 → 反向渲染，全部由 Warp 内核实现。
- **直接替换**：API 与原生 CUDA 后端兼容——只需更改导入路径即可切换后端。
- **无需编译**：纯 Python + Warp JIT。不需要 `setup.py build_ext`，不需要 CUDA 工具链头文件，没有平台相关的构建问题。
- **球谐函数**：0–3 阶，与原始实现一致。
- **自动调优**：基于 GPU SM 架构的占用率感知内核块维度选择（支持 Volta 到 Blackwell）。
- **多种分箱排序模式**：`warp_depth_stable_tile`（默认，推荐）、`warp_radix`、`torch`、`torch_count`。
- **融合反向内核**：合并分配和梯度累积步骤，减少内核启动开销。
- **紧凑 AABB 瓦片剔除**：使用逐轴 3σ 包围盒进行高斯到瓦片的分配，减少细长高斯的不必要瓦片重叠。
- **前向状态打包**：高效的前向状态序列化，供反向传播重用，避免冗余重算。

---

## 系统要求

| 组件 | 最低版本 | 测试版本 |
|------|---------|---------|
| **Python** | 3.10+ | 3.10 |
| **NVIDIA GPU** | 计算能力 ≥ 7.0（Volta） | RTX 4060 Laptop (sm_89) |
| **NVIDIA 驱动** | 兼容 CUDA 12.x | 13.2 |
| **PyTorch** | 2.0+（含 CUDA 支持） | 2.7.0+cu126 |
| **NVIDIA Warp** | 1.12.0+ | 1.12.0 |

SM 架构自动调优表覆盖以下架构：

| 架构 | 计算能力 |
|------|---------|
| Volta | 7.0 |
| Turing | 7.5 |
| Ampere (GA100) | 8.0 |
| Ampere (GA10x) | 8.6 |
| Ada Lovelace | 8.9 |
| Hopper | 9.0 |
| Blackwell | 10.0 |

不在此表中的 GPU 仍可正常工作——自动调优会回退到保守的默认参数。

---

## 安装

### 1. 安装依赖

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install warp-lang>=1.12.0
```

### 2. 克隆并集成

将本仓库作为 3DGS 项目（或任何使用 `gswarp` 的项目）的子模块：

```bash
git clone <本仓库地址> submodules/gswarp
```

### 3.（可选）构建原生 CUDA 基线用于对比

如果您还需要原生 CUDA 光栅化器（用于 A/B 测试或回退）：

```bash
cd submodules/gswarp
pip install .
```

> **注意**：Warp 后端本身 **不需要** `pip install .` 或任何原生编译。直接通过 Python 导入即可运行。

---

## 快速开始

### 使用 Warp 后端

```python
# 导入 Warp 后端（替代原生 CUDA 后端）
from gswarp.warp import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    initialize_runtime_tuning,
)

# （可选）初始化运行时自动调优——检测 GPU 并选择最优参数
initialize_runtime_tuning(device="cuda:0", verbose=True)

# 配置光栅化设置
raster_settings = GaussianRasterizationSettings(
    image_height=height,
    image_width=width,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg_color,          # [3] 张量，背景颜色
    scale_modifier=1.0,
    viewmatrix=viewmatrix,     # [4, 4] 世界到相机矩阵
    projmatrix=projmatrix,     # [4, 4] 完整投影矩阵
    sh_degree=active_sh_degree,
    campos=campos,             # [3] 世界空间中的相机位置
    prefiltered=False,
    # Warp 特有的可选字段：
    backward_mode="manual",              # 仅支持 "manual"
    binning_sort_mode="warp_depth_stable_tile",  # 或："warp_radix", "torch", "torch_count"
    auto_tune=True,
    auto_tune_verbose=True,
)

# 创建光栅化器并运行前向传播
rasterizer = GaussianRasterizer(raster_settings=raster_settings)
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(
    means3D=means3D,           # [N, 3]
    means2D=means2D,           # [N, 3]（屏幕空间，接收梯度）
    opacities=opacities,       # [N, 1]
    shs=shs,                   # [N, K, 3] 球谐系数
    scales=scales,             # [N, 3]
    rotations=rotations,       # [N, 4] 四元数
)

# 反向传播通过 PyTorch autograd 自动完成
loss = compute_loss(color, target)
loss.backward()
```

### 从原生 CUDA 切换到 Warp

替换您的导入语句：

```python
# 之前（原生 CUDA）：
from gswarp import GaussianRasterizationSettings, GaussianRasterizer

# 之后（Warp 后端）：
from gswarp.warp import GaussianRasterizationSettings, GaussianRasterizer
```

Warp 后端的前向输出元组比原生版本返回更多输出：

```python
# 原生 CUDA 输出：
color, radii = rasterizer(...)

# Warp 后端输出：
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(...)
```

---

## API 参考

### 核心函数

| 函数 | 描述 |
|------|------|
| `rasterize_gaussians(...)` | 完整前向传播：预处理 + 分箱 + 渲染 |
| `rasterize_gaussians_backward(...)` | 完整反向传播：反向渲染 + 反向预处理 |
| `mark_visible(positions, viewmatrix, projmatrix)` | 返回 3D 位置的可见性掩码 |
| `preprocess_gaussians(...)` | 仅预处理（不渲染），用于调试/分析 |

### 运行时配置

| 函数 | 描述 |
|------|------|
| `initialize_runtime_tuning(device, verbose)` | 一次性 GPU 检测和参数调优 |
| `get_runtime_tuning_report(device)` | 获取当前调优报告（内存、瓦片大小、排序模式） |
| `get_runtime_auto_tuning_config()` | 获取自动调优开关状态 |
| `set_binning_sort_mode(mode)` | 运行时设置分箱排序模式 |
| `get_default_parameter_info()` | 获取编译时常量（TOP_K、BLOCK_X 等） |
| `is_available()` | 检查 Warp 是否可导入 |

### GaussianRasterizationSettings 字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `image_height` | `int` | 输出图像高度（像素） |
| `image_width` | `int` | 输出图像宽度（像素） |
| `tanfovx` | `float` | tan(FoV_x / 2) |
| `tanfovy` | `float` | tan(FoV_y / 2) |
| `bg` | `Tensor[3]` | 背景颜色 |
| `scale_modifier` | `float` | 全局缩放乘数 |
| `viewmatrix` | `Tensor[4,4]` | 世界到相机变换矩阵 |
| `projmatrix` | `Tensor[4,4]` | 完整投影矩阵 |
| `sh_degree` | `int` | 活跃球谐阶数（0–3） |
| `campos` | `Tensor[3]` | 世界空间中的相机位置 |
| `prefiltered` | `bool` | 点是否已预过滤 |
| `backward_mode` | `str \| None` | `"manual"`（唯一支持的模式） |
| `binning_sort_mode` | `str \| None` | 分箱排序算法 |
| `auto_tune` | `bool` | 启用自动调优（默认：`True`） |
| `auto_tune_verbose` | `bool` | 打印调优信息（默认：`True`） |

---

## 架构概览

### 管线阶段

```
输入高斯（means3D, SH, scales, rotations, opacities）
    │
    ▼
┌─────────────────────────────────┐
│  1. 预处理（PREPROCESS）         │
│  - 从 scale+rotation 计算 Cov3D  │
│  - 投影到 2D（cov2d, conic）     │
│  - 视锥 + 近平面剔除             │
│  - SH → RGB 颜色评估             │
│  - 紧凑 AABB 瓦片矩形            │
│  - 前向状态打包                   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. 分箱（BINNING）              │
│  - 瓦片重叠计数（scan）           │
│  - 高斯→瓦片复制                 │
│  - 深度排序 + 瓦片排序            │
│  - 瓦片范围标识                   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3. 前向渲染（FORWARD RENDER）    │
│  - 逐像素 alpha 混合             │
│  - 从前到后合成                   │
│  - TOP_K 提前终止                │
│  - 颜色、深度、alpha 输出         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. 反向渲染（BACKWARD RENDER）   │
│  - 渲染关于 conic、opacity、      │
│    color、pos 的梯度              │
│  - atomic_add 累积               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  5. 反向预处理（BACKWARD PREPROCESS）│
│  - 预处理关于 means3D、scales、   │
│    rotations 的梯度               │
│  - SH 反向传播                    │
│  - Cov3D → scale/rotation 梯度   │
└─────────────────────────────────┘
```

### 单文件设计

整个 Warp 后端包含在一个 Python 文件中（约 4400 行），包括：

- 所有 Warp 内核定义（`@wp.kernel`）
- 所有 Warp 辅助函数（`@wp.func`）
- 缓冲区缓存管理
- 运行时自动调优
- 公共 API 函数

这是有意为之的——Warp 的 JIT 编译模型要求所有内核代码及其依赖项必须在同一个 `wp.Module` 作用域内。拆分到多个文件会破坏 Warp 在 JIT 时对 `@wp.func` 交叉引用的解析能力。

### 关键常量

| 常量 | 值 | 描述 |
|------|---|------|
| `TOP_K` | 20 | 每像素最大高斯数（提前终止） |
| `BLOCK_X` | 16 | 瓦片宽度（像素） |
| `BLOCK_Y` | 16 | 瓦片高度（像素） |
| `NUM_CHANNELS` | 3 | RGB 输出通道 |
| `PREPROCESS_CULL_SIGMA` | 3.0 | 视锥剔除 sigma 乘数 |
| `PREPROCESS_CULL_FOV_SCALE` | 1.3 | 剔除用 FoV 边界缩放 |
| `VISIBILITY_NEAR_PLANE` | 0.2 | 剔除用近平面距离 |

---

## 与 CUDA 基线的差异

### 1. 紧凑 AABB vs 各向同性半径

**CUDA 基线**使用两个特征值派生半径中的最大值计算正方形包围盒：

```c
// CUDA: auxiliary.h getRect()
int max_radius = ...;  // max(ceil(3σ_max), 0), 各向同性
rect_min = {min(grid.x, max((int)((point.x - max_radius) / BLOCK_X), 0)), ...};
rect_max = {min(grid.x, max((int)((point.x + max_radius + BLOCK_X - 1) / BLOCK_X), 0)), ...};
```

**Warp 后端**使用 2D 协方差矩阵的对角元素计算逐轴的紧凑包围盒：

```python
# Warp: _compute_tile_rect_tight_wp()
radius_x = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(cov_xx, 0.01))))
radius_y = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(cov_yy, 0.01))))
```

**影响**：
- 对于 **细长高斯**（高各向异性），Warp 后端分配的瓦片更少，减少了 `num_rendered`，提高了分箱/渲染效率。
- 对于 **圆形高斯**，两种方法等价。
- 这引入了微小的不匹配：CUDA 基线由于过于保守的各向同性半径而包含的一些边界瓦片，被 Warp 的更紧凑边界排除。视觉差异可忽略不计，但在数值比较中可检测到。

### 2. 无共享内存协作获取

**CUDA 基线**使用 `__shared__` 内存进行协作式瓦片级数据获取：

```c
// CUDA: forward.cu renderCUDA()
__shared__ int collected_id[BLOCK_SIZE];
__shared__ float2 collected_xy[BLOCK_SIZE];
__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

// 瓦片中的所有线程协作地从全局内存加载一批高斯到共享内存，
// 然后遍历该批次。
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
    // 从共享内存读取——快速，广播给瓦片中的所有线程
    float2 xy = collected_xy[j];
    float4 con_o = collected_conic_opacity[j];
    ...
}
```

**Warp 后端**无法使用 `__shared__` 内存（Warp 不提供此功能）。每个线程独立从全局内存读取：

```python
# Warp：每个线程独立读取
g_idx = point_list[tile_start + j]
xy_x = means2d_x[g_idx]
xy_y = means2d_y[g_idx]
con_x = conic_x[g_idx]
...
```

**影响**：
- 这是 Warp 与 CUDA 在中小规模下 **最大的性能差距**。CUDA 的共享内存模式将全局内存读取分摊到瓦片中的所有线程（256 个线程共享一次加载），而 Warp 的方法每个高斯发出 256 次独立的全局读取。
- 在大规模场景下（num_rendered > ~30K），混合计算和内存带宽主导性能，该差距减小。

### 3. 排序差异

**CUDA 基线**使用单次 CUB `DeviceRadixSort`，键为 64 位打包格式（`(tile_id << 32) | depth_bits`）。

**Warp 后端**默认模式（`warp_depth_stable_tile`）使用两次排序：
1. 第一次：按深度排序（Warp 基数排序）
2. 第二次：按瓦片 ID 稳定排序（Warp 基数排序）

这导致每个瓦片内的高斯排序与 CUDA 基线不同，进而在 alpha 混合期间产生不同的浮点累积顺序。由于浮点运算的非结合性，像素级输出会有微小差异。

### 4. 剔除参数

Warp 后端应用显式的视锥剔除参数：
- `PREPROCESS_CULL_SIGMA = 3.0`：3σ 包围盒完全在图像外的高斯被剔除
- `PREPROCESS_CULL_FOV_SCALE = 1.3`：略微扩展 FoV 以避免过度边界剔除

这与 CUDA 基线的隐式剔除行为略有不同。

### 5. 逐像素 vs 逐瓦片调度

CUDA 基线以瓦片块的 2D 网格调度渲染内核：
```c
dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
dim3 block(BLOCK_X, BLOCK_Y, 1);
```

Warp 后端以 1D 方式调度，每个像素一个线程：
```python
wp.launch(kernel, dim=image_height * image_width, ...)
```

瓦片索引在内核内通过 `wp.tid()` 算术推导。功能等价，但占用率和调度特性不同。

---

## 正确性

Warp 后端已针对原生 CUDA 基线进行了充分测试。以下是测试套件的代表性残差量级（4096 个随机高斯，128×128 图像）：

### 前向传播

| 输出 | 最大绝对差 | 备注 |
|------|-----------|------|
| `color` | ~4.8e-7 | 接近机器精度（FP32） |
| `depth` | ~3.6e-7 | 接近机器精度 |
| `alpha` | ~3.0e-7 | 接近机器精度 |
| `radii` | 精确匹配 | 整数值 |
| `proj_2D` | ~3.8e-6 | 投影精度 |
| `conic_2D` | ~1.5e-7 | 接近机器精度 |
| `conic_2D_inv` | ~0.031 | 矩阵逆运算放大误差 |
| `gs_per_pixel` | 精确匹配 | 整数值 |
| `weight_per_gs_pixel` | ~1.1e-7 | 接近机器精度 |
| `x_mu` | ~1.9e-6 | 位置精度 |

### 反向传播（预计算颜色）

| 梯度 | 最大绝对差 |
|------|-----------|
| `grad_means3D` | ~3.8e-5 |
| `grad_means2D` | ~5.7e-6 |
| `grad_colors` | ~3.8e-6 |
| `grad_opacity` | ~4.6e-5 |
| `grad_cov3D` | ~1.7e-5 |

### 反向传播（scale + rotation 模式）

| 梯度 | 最大绝对差 |
|------|-----------|
| `grad_means3D` | ~1.5e-5 |
| `grad_means2D` | ~1.3e-6 |
| `grad_colors` | ~7.6e-6 |
| `grad_opacity` | ~4.8e-6 |
| `grad_scales` | ~3.6e-6 |
| `grad_rotations` | ~1.1e-5 |

### 已知前向不匹配：`conic_2D_inv`

`conic_2D_inv`（2D 协方差 conic 的逆矩阵）具有最大的前向残差（~0.031）。这是由于 2×2 矩阵逆运算路径中的浮点精度差异造成的。逆运算放大了微小的输入差异，尤其对于近奇异的协方差矩阵（细长高斯）。使用 `scale + rotation` 输入路径（而非预计算的 `cov3D`）可能产生多达 38 个不匹配元素。此残差对训练收敛没有可观察的影响。

---

## 性能特征

所有基准测试在 **NVIDIA GeForce RTX 4060 Laptop GPU**（sm_89, 8 GiB, 24 SMs）上进行，Warp 1.12.0，PyTorch 2.7.0+cu126。数值为 30 次迭代的均值（共 50 次，前 20 次丢弃用于预热）。

### 汇总表

| 点数 | 分辨率 | num_rendered | 原生前向 | Warp 前向 | 比率 | 原生反向 | Warp 反向 | 比率 |
|------|--------|-------------|---------|----------|------|---------|----------|------|
| 4,096 | 128×128 | 2,059 | 0.53 ms | 1.54 ms | 2.9× | 1.25 ms | 2.29 ms | 1.8× |
| 16,384 | 128×128 | 8,254 | 0.57 ms | 1.97 ms | 3.4× | 1.73 ms | 2.36 ms | 1.4× |
| 65,536 | 256×256 | 33,292 | 11.56 ms | 2.43 ms | **0.21×** | 11.62 ms | 2.83 ms | **0.24×** |

### 分析

- **小规模（num_rendered < ~5K）**：Warp 比原生 CUDA **慢 2–3 倍**。开销主要来自：
  - Python 调度和 Warp 启动开销
  - 分箱排序固定成本（基数排序初始化）
  - 渲染内核中缺少 `__shared__` 内存协作获取
  - 计算工作量小时，这些固定成本分摊不佳。

- **中等规模（num_rendered ~5K–20K）**：Warp **慢 1.4–2 倍**。计算工作量开始主导固定开销，但渲染内核中的共享内存差距仍然显著。

- **大规模（num_rendered > ~30K）**：Warp 可以 **显著更快**（在 65K 点 / 256×256 下实测快 4–5 倍）。在此规模下：
  - 原生 CUDA 基线的 `num_rendered` 因各向同性 AABB（正方形边界）而膨胀，而 Warp 的紧凑 AABB 消除了冗余的瓦片分配。
  - 渲染内核计算主导性能，紧凑 AABB 的逐像素调度具有竞争力。
  - Warp 的分箱更高效，因为重复条目更少。

> **注意**：65,536 点 / 256×256 的基准测试显示了戏剧性的反转，因为原生 CUDA 基线生成了更多的瓦片分配（各向同性半径），而 Warp 的紧凑 AABB 保持 `num_rendered` 可控。更大的分辨率和更高的高斯数量将进一步有利于 Warp 后端。

### Warp 内核时间分解（65K 点，256×256）

| 阶段 | 时间 | 占比 |
|------|------|------|
| 预处理 | 1.14 ms | 21.6% |
| 分箱 | 1.44 ms | 27.3% |
| 渲染 | 0.39 ms | 7.4% |
| 反向渲染 | 1.21 ms | 22.9% |
| （反向预处理） | ~1.10 ms | 20.8% |

分箱阶段（排序）是最大的单一组件。渲染内核本身由于紧凑 AABB 减少了每瓦片工作量而相对较快。

---

## 已知限制

### 1. 无共享内存（Warp 限制）

NVIDIA Warp 不提供 CUDA `__shared__` 内存。这阻止了实现 CUDA 基线中的协作式瓦片级获取模式——在该模式中，瓦片中的 256 个线程协作地将高斯数据从全局内存加载到共享内存，然后遍历共享缓冲区。取而代之的是每个线程独立从全局内存读取，导致：
- 每个瓦片对相同数据的全局内存事务约多 ~256 倍
- 内存合并度差（按高斯索引的离散读取）
- NCU 性能分析显示反向渲染内核中 **48.4% LG 限制（长记分板阻塞）**，直接由非合并全局内存读取导致

### 2. 无 Warp 级原语

Warp 不提供 CUDA warp 级原语（`__shfl_sync`、`__ballot_sync`、`__any_sync`）。这阻止了：
- 梯度累积的 warp 级归约
- warp 级提前终止投票
- 高级 CUDA 光栅化器中使用的跨通道通信模式

### 3. `tile_atomic_add` 仅支持标量

`wp.tile_atomic_add` 仅支持 `float32` 标量操作数。向量/矩阵级别的原子归约需要分解为单独的标量原子操作，产生过多的同步开销。这已通过实验验证——瓦片归约的反向内核（C2）比非瓦片版本 **慢 2–3 倍**，原因是每个高斯需要 10 次标量原子操作。

### 4. 编译时瓦片形状

`BLOCK_X` 和 `BLOCK_Y` 被定义为 `wp.constant()` 值（16×16）。更改瓦片形状需要修改源代码并触发 Warp 模块重新编译。CUDA 基线也有固定的瓦片大小，但 Warp 的 JIT 模型使此限制更为明显，因为常量在编译时已烘焙到内核中。

### 5. 首次运行 JIT 编译开销

首次调用任何 Warp 内核会触发整个模块的 JIT 编译。在典型系统上，这需要 **约 6–15 秒**（取决于内核数量和 GPU）。后续调用使用 Warp 内核缓存，几乎即时完成。

### 6. 反向传播非确定性

反向渲染内核使用 `wp.atomic_add` 累积梯度，当多个线程写入同一地址时，这本质上是非确定性的。这意味着：
- 使用相同输入的两次运行可能产生略有不同的梯度值
- 差异通常在 FP32 精度内（大多数梯度 < 1e-4）
- 在极端情况下，`conic_opacity` 梯度可能出现高达 ~1.6e+05 的最大差异（由于梯度幅值很大）
- 这与 CUDA 基线的行为一致（`atomicAdd` 同样是非确定性的）

### 7. Python 级编排开销

与 CUDA 基线整个管线由 C++ 编排、Python 交互极少不同，Warp 后端使用 Python 来：
- 分配中间张量（通过 PyTorch）
- 依次启动各个 Warp 内核
- 通过 Python 变量在各阶段间传递数据

这引入了可测量的固定开销（每次完整前向 + 反向约 0.5–1.0 ms），在小规模时显著，在大规模时可忽略。

### 8. 单一反向模式

仅支持 `backward_mode="manual"`。CUDA 基线的 `autograd` 级别微分不适用，因为 Warp 内核未原生集成到 PyTorch 的 autograd 计算图中——反向传播使用手工推导的梯度显式编码。

---

## 未来优化方向

以下是最有影响力的潜在改进，大致按预期收益排序：

### 1. 共享内存支持（等待 Warp 新特性）

如果 NVIDIA Warp 在未来版本中添加 `__shared__` 内存支持，渲染和反向渲染内核可以重写为协作式瓦片级获取，有望弥合中小规模下 **2–3 倍的差距**。这是单一影响最大的优化。

### 2. 大场景的紧凑聚集

实验性的紧凑聚集优化（`_ENABLE_COMPACT_GATHER`）在渲染内核之前，按 `point_list` 顺序将离散的逐高斯数据预拷贝到紧凑的 SoA 缓冲区。这将离散全局读取转换为顺序读取，改善内存合并度。性能分析结果：
- **65K 点，256×256**：反向渲染 **快 27%**，总前向+反向 **快 16%**
- **4K–16K 点**：**更慢**（48–135% 回退），因为拷贝开销占主导

目前默认禁用。应在大场景训练时有选择地启用。

### 3. 运行时瓦片形状自适应

目前 `BLOCK_X=16, BLOCK_Y=16` 在编译时固定。允许运行时选择瓦片形状（例如小图像用 8×8，宽图像用 32×16）可以改善占用率并减少瓦片边界开销。需要 Warp 支持动态内核参数化或类模板机制。

### 4. TOP_K 外部化

`TOP_K = 20` 常量控制每像素在提前终止前考虑的最大高斯数。将其外部化为运行时参数将允许：
- 降低 TOP_K 以在可接受的质量损失下加速训练
- 提高 TOP_K 以用于对质量要求高的渲染
- 基于场景复杂度自适应 TOP_K

### 5. Warp 级原语（等待 Warp 新特性）

如果 Warp 中可用 warp 级内建函数，逐 warp 归约可以替代反向内核中的 `atomic_add`，有望消除非确定性问题并减少 LG 限制阻塞。

---

## 致谢

本 Warp 后端基于以下项目构建：

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)，作者 Bernhard Kerbl、Georgios Kopanas、Thomas Leimkühler 和 George Drettakis（INRIA, MPII）。原始 CUDA 光栅化器是参考实现。
- [NVIDIA Warp](https://nvidia.github.io/warp/)——用于高性能 GPU 仿真和计算的 Python 框架。
- [PyTorch](https://pytorch.org/)——用于张量管理、autograd 集成和 CUDA 内存分配。

---

## 引用

如果您在研究中使用了原始 3D Gaussian Splatting，请引用：

```bibtex
@article{kerbl3Dgaussians,
    author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal   = {ACM Transactions on Graphics},
    number    = {4},
    volume    = {42},
    month     = {July},
    year      = {2023},
    url       = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
