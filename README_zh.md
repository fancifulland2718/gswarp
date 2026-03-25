# gswarp

[English](README.md) · **中文**

一个基于 **纯 Python + NVIDIA Warp** 的可微高斯光栅化管线重新实现，原始版本以 CUDA C++ 编写，用于 [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)。本后端提供了与原生 CUDA 光栅化器的 **直接替换**（drop-in replacement）——无需 C++/CUDA 编译。

> **许可证**：本项目继承 [Gaussian-Splatting 许可证](LICENSE)（Inria 和 MPII，仅限非商业研究使用）。

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
- **紧凑 AABB 瓦片剔除**：使用逐轴 3σ 包围盒进行高斯到瓦片的分配（受 [Zhang et al., 2025](https://arxiv.org/abs/2601.19489) 启发），减少细长高斯的不必要瓦片重叠。
- **前向状态打包**：高效的前向状态序列化，供反向传播重用，避免冗余重算。

---

## 系统要求

| 组件 | 最低版本 | 测试版本 |
|------|---------|---------|
| **Python** | 3.10+ | 3.10 |
| **NVIDIA GPU** | 计算能力 ≥ 7.0（Volta） | RTX 4060 Laptop
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
git clone https://github.com/fancifulland2718/gswarp.git submodules/gswarp
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
| `preprocess_gaussians(...)` | 仅预处理（不渲染），用于分析 |

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

本项目采用的紧凑 AABB 包围盒方法受到论文 *[Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction](https://arxiv.org/abs/2601.19489)*（Ziyu Zhang, Tianle Liu, Diantao Tu, Shuhan Shen, arXiv:2601.19489）的启发。该技术使用从 2D 协方差矩阵派生的逐轴范围替代各向同性的正方形半径，为各向异性高斯提供更紧凑的瓦片分配。本项目未引入额外的代码依赖，仅采纳了包围盒计算逻辑。

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

**Warp 后端**使用 `__shared__` 内存相对困难，目前每个线程独立从全局内存读取：

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

**Warp 后端**共有四种排序模式，共用同一套 **preprocess / render / backward-render** 内核，区别只在 **binning 阶段如何构造与排序 `(tile, point)` 对**：

- `warp_radix`：先在 Warp 内核里直接复制出 `(tile_id, depth_bits)` 的 64 位 packed key，再走 **单次 Warp radix sort**。路径最短、scratch 最轻，但 tie-break 与 CUDA/CUB 不完全一致；
- `warp_depth_stable_tile`：先把点按深度做一次 **Warp i32 radix sort**，再按这个深度顺序复制 `(tile_id, point_id)`，最后只对 `tile_id` 做一次 **稳定的 Warp radix sort**，从而保留“tile 内深度顺序”；
- `torch`：先用 Warp 内核生成 `tile_id / point_id`，再在 PyTorch 侧做 **stable argsort**。当 `num_rendered <= TORCH_SINGLE_SORT_THRESHOLD` 时走单次 packed-key 排序；超过阈值后退化成“先深度、后 tile”的两段 stable sort；
- `torch_count`：同样先由 Warp 生成 `tile_id / point_id`，但会先按深度 stable sort，再按 `tile_id` 做第二次 PyTorch 排序；当 tile 数较小时还会走 `int16` 快路径，以减少第二次排序的键宽。

换句话说，四条路径的差别不是“有没有算对 preprocess / render”，而是 **binning 阶段把同一批高斯以什么顺序送进每个 tile**。这导致每个瓦片内的高斯排序与 CUDA 基线不同，进而在 alpha 混合期间产生不同的浮点累积顺序。由于浮点运算的非结合性，像素级输出会有微小差异。


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

测试配置如下：

- `mode = sh_scale_rotation`
- `backward_mode="manual"`
- `binning_sort_mode ∈ {"warp_radix", "torch", "warp_depth_stable_tile", "torch_count"}`（每次运行显式指定）
- `auto_tune=True`
- `auto_tune_verbose=True`
- 测试规模：**4,096 / 16,384 / 65,536 / 262,144** 个粒子

这次四种排序模式的全量扫描里，16 组组合的 preprocess 诊断项都保持对齐，最终 forward 输出也都在检查阈值内精确一致。剩余差异仍然只出现在稀疏的 backward outlier 上，而且它会随着粒子规模和排序后端而变化。


### 四排序模式正确性汇总

> 这组实验里每次都显式指定 `binning_sort_mode`，因此“请求模式”和“实际执行模式”在文档表格里不再重复列出。

<table>
    <thead>
        <tr>
            <th>点数</th>
            <th>排序模式</th>
            <th>Preprocess</th>
            <th>Backward 超阈值占比</th>
            <th>Backward 最大误差</th>
        </tr>
    </thead>
    <tbody>
        <tr><td rowspan="4"><strong>4,096</strong></td><td><code>warp_radix</code></td><td>aligned</td><td>clean</td><td>0.037109</td></tr>
        <tr><td><code>torch</code></td><td>aligned</td><td>clean</td><td><strong>0.011719</strong></td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>aligned</td><td><code>means3D</code>: 1 / 12,288 (0.008138%)</td><td>0.039062</td></tr>
        <tr><td><code>torch_count</code></td><td>aligned</td><td><code>opacity</code>: 1 / 4,096 (0.024414%)</td><td>0.048828</td></tr>
        <tr><td rowspan="4"><strong>16,384</strong></td><td><code>warp_radix</code></td><td>aligned</td><td>clean</td><td>0.014648</td></tr>
        <tr><td><code>torch</code></td><td>aligned</td><td>clean</td><td>0.015625</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>aligned</td><td>clean</td><td><strong>0.010742</strong></td></tr>
        <tr><td><code>torch_count</code></td><td>aligned</td><td>clean</td><td>0.011230</td></tr>
        <tr><td rowspan="4"><strong>65,536</strong></td><td><code>warp_radix</code></td><td>aligned</td><td><code>means3D</code>: 3 / 196,608 (0.001526%)<br><code>opacity</code>: 2 / 65,536 (0.003052%)<br><code>shs</code>: 7 / 3,145,728 (0.000223%)</td><td>0.375000</td></tr>
        <tr><td><code>torch</code></td><td>aligned</td><td><code>means3D</code>: 3 / 196,608 (0.001526%)<br><code>opacity</code>: 1 / 65,536 (0.001526%)<br><code>scales</code>: 2 / 196,608 (0.001017%)<br><code>rotations</code>: 4 / 262,144 (0.001526%)</td><td><strong>0.171875</strong></td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>aligned</td><td><code>means3D</code>: 3 / 196,608 (0.001526%)<br><code>opacity</code>: 1 / 65,536 (0.001526%)<br><code>shs</code>: 7 / 3,145,728 (0.000223%)</td><td>0.343750</td></tr>
        <tr><td><code>torch_count</code></td><td>aligned</td><td><code>means3D</code>: 6 / 196,608 (0.003052%)<br><code>opacity</code>: 2 / 65,536 (0.003052%)<br><code>shs</code>: 13 / 3,145,728 (0.000413%)<br><code>rotations</code>: 3 / 262,144 (0.001144%)</td><td>0.218750</td></tr>
        <tr><td rowspan="4"><strong>262,144</strong></td><td><code>warp_radix</code></td><td>aligned</td><td><code>means3D</code>: 6 / 786,432 (0.000763%)<br><code>means2D</code>: 5 / 786,432 (0.000636%)<br><code>opacity</code>: 2 / 262,144 (0.000763%)<br><code>shs</code>: 49 / 12,582,912 (0.000389%)<br><code>scales</code>: 5 / 786,432 (0.000636%)<br><code>rotations</code>: 5 / 1,048,576 (0.000477%)</td><td>0.578125</td></tr>
        <tr><td><code>torch</code></td><td>aligned</td><td><code>means3D</code>: 3 / 786,432 (0.000381%)<br><code>means2D</code>: 2 / 786,432 (0.000254%)<br><code>opacity</code>: 3 / 262,144 (0.001144%)<br><code>shs</code>: 7 / 12,582,912 (0.000056%)<br><code>scales</code>: 3 / 786,432 (0.000381%)<br><code>rotations</code>: 4 / 1,048,576 (0.000381%)</td><td><strong>0.156250</strong></td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>aligned</td><td><code>means3D</code>: 9 / 786,432 (0.001144%)<br><code>means2D</code>: 5 / 786,432 (0.000636%)<br><code>opacity</code>: 2 / 262,144 (0.000763%)<br><code>shs</code>: 79 / 12,582,912 (0.000628%)<br><code>scales</code>: 6 / 786,432 (0.000763%)<br><code>rotations</code>: 9 / 1,048,576 (0.000858%)</td><td>0.429688</td></tr>
        <tr><td><code>torch_count</code></td><td>aligned</td><td><code>means3D</code>: 7 / 786,432 (0.000890%)<br><code>means2D</code>: 4 / 786,432 (0.000509%)<br><code>opacity</code>: 3 / 262,144 (0.001144%)<br><code>shs</code>: 70 / 12,582,912 (0.000556%)<br><code>scales</code>: 4 / 786,432 (0.000509%)<br><code>rotations</code>: 7 / 1,048,576 (0.000668%)</td><td>0.734375</td></tr>
    </tbody>
</table>

结论与归因：

- 四种排序模式都没有引入 preprocess 张量对不齐的问题，forward 输出也在阈值内完全一致，这说明问题 **不在 preprocess / render 主链路**；
- backward 的超阈值覆盖率始终非常低：4K 最坏也只有 **1 / 4,096 = 0.024414%**，到 65K / 262K 时单字段通常已经降到 **$10^{-3}\%$ 到 $10^{-4}\%$** 量级，属于极稀疏 outlier；
- 残差与排序路径强相关，说明它主要来自 **binning 改变了 tile 内遍历顺序**：一旦高斯进入每个 tile 的先后顺序变化，alpha 混合和 backward `atomic_add` 的浮点累积顺序也会变化，进而放大为少量离散梯度尖峰；
- `warp_radix` 还显式关闭了 `WARP_RADIX_DETERMINISTIC_TIEBREAK`，因此相同 key 的 tie-break 更容易与 CUDA/CUB 参考实现不同；`warp_depth_stable_tile` 与 `torch_count` 的两阶段重排也会额外改变边界样本的相对次序；
- 从结果上看，`torch` 在 65K 和 262K 时残差最小，说明它的 stable sort 路径在大规模下更接近 CUDA 基线的最终遍历顺序；但这是一种**经验性现象**，不是 preprocess 或公式层面的更改。

---

## 性能特征

以下数据同样来自**当前代码状态**，测试平台为 **NVIDIA GeForce RTX 4060 Laptop GPU**（sm_89，8 GiB，24 个 SM）、**Warp 1.12.0**、**PyTorch 2.7.0+cu126**。

测试方法：

- **稳态耗时**：使用公共 API（`diff_gaussian_rasterization.GaussianRasterizer` 与 `diff_gaussian_rasterization.warp.GaussianRasterizer`）先做单独预热，再用 CUDA event 对正式采样段做批量计时；下表同时展示原生 **CUDA 基线** 与四种显式 Warp 排序后端的平均值。
- **显存占用**：先预热，再分别测一次 forward 阶段和 backward 阶段，记录 CUDA 分配器的峰值增量（`peak_allocated_delta_mib`）。
- **阶段计时 / 阶段显存**：仅为分析热点，使用内部 `_warp_backend` 辅助函数做诊断性测量；因此 **binning 阶段耗时 / 内部 binning scratch** 只对 Warp 后端可用，CUDA 基线在表中记为 `—`。这些阶段数据**不保证**与公共 API 的端到端时间或峰值显存逐项严格相加一致。

其中 4K / 16K / 65K / 262K 四档规模分别采用 **12+24 / 10+20 / 6+12 / 4+8**（预热次数 + 正式采样次数）的配置，并对每种排序模式分别执行。

### CUDA 基线 + 四排序模式性能汇总

<table>
    <thead>
        <tr>
            <th>点数</th>
            <th>后端 / 排序模式</th>
            <th>Warp 前向</th>
            <th>Warp 反向</th>
            <th>Binning 阶段</th>
        </tr>
    </thead>
    <tbody>
        <tr><td rowspan="5"><strong>4,096</strong></td><td><strong>CUDA 基线</strong></td><td>7.416 ms</td><td>8.270 ms</td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td>7.020 ms</td><td>8.474 ms</td><td>2.256 ms</td></tr>
        <tr><td><code>torch</code></td><td>4.178 ms</td><td>5.650 ms</td><td>3.967 ms</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>3.533 ms</td><td>4.199 ms</td><td><strong>1.343 ms</strong></td></tr>
        <tr><td><code>torch_count</code></td><td><strong>2.441 ms</strong></td><td><strong>3.162 ms</strong></td><td>6.255 ms</td></tr>
        <tr><td rowspan="5"><strong>16,384</strong></td><td><strong>CUDA 基线</strong></td><td>3.196 ms</td><td>2.638 ms</td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td>3.748 ms</td><td><strong>1.811 ms</strong></td><td><strong>1.748 ms</strong></td></tr>
        <tr><td><code>torch</code></td><td>3.110 ms</td><td>2.325 ms</td><td>6.178 ms</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>3.771 ms</td><td>2.292 ms</td><td>2.024 ms</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>3.049 ms</strong></td><td>2.775 ms</td><td>2.691 ms</td></tr>
        <tr><td rowspan="5"><strong>65,536</strong></td><td><strong>CUDA 基线</strong></td><td>20.214 ms</td><td>5.024 ms</td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td>20.506 ms</td><td>4.998 ms</td><td>22.415 ms</td></tr>
        <tr><td><code>torch</code></td><td>21.709 ms</td><td>3.781 ms</td><td>34.787 ms</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>19.999 ms</strong></td><td>5.035 ms</td><td><strong>13.537 ms</strong></td></tr>
        <tr><td><code>torch_count</code></td><td>22.914 ms</td><td><strong>3.532 ms</strong></td><td>34.513 ms</td></tr>
        <tr><td rowspan="5"><strong>262,144</strong></td><td><strong>CUDA 基线</strong></td><td>183.858 ms</td><td><strong>9.276 ms</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>183.562 ms</strong></td><td>10.957 ms</td><td>171.128 ms</td></tr>
        <tr><td><code>torch</code></td><td>189.218 ms</td><td><strong>10.266 ms</strong></td><td>975.769 ms</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>186.276 ms</td><td>10.895 ms</td><td><strong>110.787 ms</strong></td></tr>
        <tr><td><code>torch_count</code></td><td>316.262 ms</td><td>36.428 ms</td><td>1195.816 ms</td></tr>
    </tbody>
</table>

### 公共 API 峰值显存 + 内部 binning scratch

<table>
    <thead>
        <tr>
            <th>点数</th>
            <th>后端 / 排序模式</th>
            <th>Warp 前向峰值</th>
            <th>Warp 反向峰值</th>
            <th>内部 binning 峰值</th>
        </tr>
    </thead>
    <tbody>
        <tr><td rowspan="5"><strong>4,096</strong></td><td><strong>CUDA 基线</strong></td><td><strong>1.57 MiB</strong></td><td><strong>4.41 MiB</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>1.57 MiB</strong></td><td>4.42 MiB</td><td><strong>0.00 MiB</strong></td></tr>
        <tr><td><code>torch</code></td><td><strong>1.57 MiB</strong></td><td>4.42 MiB</td><td>4.69 MiB</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>1.57 MiB</strong></td><td>4.42 MiB</td><td>0.02 MiB</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>1.57 MiB</strong></td><td>4.42 MiB</td><td>5.34 MiB</td></tr>
        <tr><td rowspan="5"><strong>16,384</strong></td><td><strong>CUDA 基线</strong></td><td><strong>5.18 MiB</strong></td><td><strong>17.63 MiB</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>5.18 MiB</strong></td><td>17.69 MiB</td><td><strong>0.00 MiB</strong></td></tr>
        <tr><td><code>torch</code></td><td><strong>5.18 MiB</strong></td><td>17.69 MiB</td><td>18.98 MiB</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>5.18 MiB</strong></td><td>17.69 MiB</td><td>0.06 MiB</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>5.18 MiB</strong></td><td>17.69 MiB</td><td>22.13 MiB</td></tr>
        <tr><td rowspan="5"><strong>65,536</strong></td><td><strong>CUDA 基线</strong></td><td><strong>41.79 MiB</strong></td><td><strong>70.57 MiB</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>41.79 MiB</strong></td><td>70.82 MiB</td><td><strong>0.00 MiB</strong></td></tr>
        <tr><td><code>torch</code></td><td><strong>41.79 MiB</strong></td><td>70.82 MiB</td><td>348.00 MiB</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>41.79 MiB</strong></td><td>70.82 MiB</td><td>0.25 MiB</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>41.79 MiB</strong></td><td>70.82 MiB</td><td>330.50 MiB</td></tr>
        <tr><td rowspan="5"><strong>262,144</strong></td><td><strong>CUDA 基线</strong></td><td><strong>302.13 MiB</strong></td><td><strong>282.00 MiB</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td>303.13 MiB</td><td>284.00 MiB</td><td><strong>0.00 MiB</strong></td></tr>
        <tr><td><code>torch</code></td><td><strong>302.13 MiB</strong></td><td>283.00 MiB</td><td>3038.64 MiB</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>302.13 MiB</strong></td><td>283.00 MiB</td><td>1.00 MiB</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>302.13 MiB</strong></td><td>284.32 MiB</td><td>2891.15 MiB</td></tr>
    </tbody>
</table>

从公共 API 视角看，CUDA 基线与四种 Warp 排序模式在前向峰值显存上整体仍处在同一量级；反向峰值显存则是 CUDA 基线 consistently 更低一些。真正拉开差距的仍然是**内部 binning scratch**：`warp_radix` 和 `warp_depth_stable_tile` 很轻，而 `torch` / `torch_count` 会在 65K 与 262K 时迅速放大。

综合 CUDA 基线与四种 Warp 排序模式，可以总结出四条更准确的观察：

- **4K 小规模**：四种 Warp 排序模式里 `torch_count` 端到端最快，且前向 / 反向都明显快于这次测得的 CUDA 基线；但它的内部 binning scratch 也是四者里最重的；
- **16K 中规模**：`torch_count` 仍给出最短前向，`warp_radix` 拿到最短反向和最轻 binning；不过这一档 CUDA 基线的前向峰值显存与 Warp 基本持平、反向峰值显存仍更低；
- **65K / 262K 大规模**：`warp_depth_stable_tile` 继续保持最稳的 Warp 侧 binning 耗时，而 **262K 反向端到端最快的是 CUDA 基线**（9.276 ms），说明 Warp 当前仍未在所有大规模路径上全面超越原生实现；
- **默认模式选择**：`warp_depth_stable_tile` 之所以仍被保留为默认值，不是因为它在所有指标上绝对最优，而是因为它在 Warp 四模式里提供了**最平衡的 binning 性能、极低 scratch 显存、以及较小的 correctness residual**。

---


## 已知限制

### 1. 无共享内存（Warp 限制）

NVIDIA Warp 缺乏显式控制共享内存的手段，在对齐、填充、复杂结构的构造上存在困难，不方便自由处理 CUDA `__shared__` 内存。这阻碍了实现 CUDA 基线中的协作式瓦片级获取模式——在该模式中，瓦片中的 256 个线程协作地将高斯数据从全局内存加载到共享内存，然后遍历共享缓冲区。取而代之的是每个线程独立从全局内存读取，导致：
- 每个瓦片对相同数据的全局内存事务多数倍
- 内存合并度差（按高斯索引的离散读取）
- NCU 性能分析显示**几乎所有的  Kernel 都受内存瓶颈限制**。

### 2. 无 Warp 级原语

Warp 未暴露 CUDA warp 级原语（`__shfl_sync`、`__ballot_sync`、`__any_sync`），提供的同步手段比较有限，并且不能与某些特性兼容。这阻止了：
- 梯度累积的 warp 级归约
- warp 级提前终止投票
- 高级 CUDA 光栅化器中使用的跨通道通信模式

### 3. `tile_atomic_add` 仅支持标量

`wp.tile_atomic_add` 仅支持如 `float32` 等少数标量操作数。向量/矩阵级别的原子归约需要分解为单独的标量原子操作，产生过多的同步开销。这已通过实验验证——瓦片归约的反向内核（C2）比非 Warp 版本 **慢 2–3 倍**，每个高斯需要 10 次标量原子操作。

### 4. 编译时瓦片形状

`BLOCK_X` 和 `BLOCK_Y` 被定义为 `wp.constant()` 值（16×16）。更改瓦片形状需要修改源代码并触发 Warp 模块重新编译。CUDA 基线也有固定的瓦片大小，但 Warp 的 JIT 模型使此限制更为明显，因为常量在编译时已烘焙到内核中。

### 5. 首次运行 JIT 编译开销

首次调用任何 Warp 内核会触发整个模块的 JIT 编译。在典型系统上，这需要数秒（取决于内核数量和 GPU）。后续调用使用 Warp，几乎即时完成。

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

这引入了可测量的固定开销，在小规模时显著，在大规模时可忽略。**这可能会使得 Warp 版本的训练速率波动巨大**。

### 8. 单一反向模式

仅支持 `backward_mode="manual"`。CUDA 基线的 `autograd` 级别微分不适用，因为 Warp 内核未原生集成到 PyTorch 的 autograd 计算图中——反向传播使用手工推导的梯度显式编码。

---

## 未来优化方向

以下是最有影响力的潜在改进，大致按预期收益排序：

### 1. 共享内存支持

如果 NVIDIA Warp 在未来版本中添加更丰富的 `__shared__` 内存支持，渲染和反向渲染内核可以重写为协作式瓦片级获取，有望弥合中小规模下 **2–3 倍的差距**。这是单一影响最大的优化。在目前的 Warp 版本上实现这一点还比较困难。

### 2. 运行时瓦片形状自适应

目前 `BLOCK_X=16, BLOCK_Y=16` 在编译时固定。允许运行时选择瓦片形状（例如小图像用 8×8，宽图像用 32×16）可以改善占用率并减少瓦片边界开销。需要 Warp 支持动态内核参数化或类模板机制。

### 3. TOP_K 外部化

`TOP_K = 20` 常量控制每像素在提前终止前考虑的最大高斯数。将其外部化为运行时参数将允许：
- 降低 TOP_K 以在可接受的质量损失下加速训练
- 提高 TOP_K 以用于对质量要求高的渲染
- 基于场景复杂度自适应 TOP_K

### 4. Warp 级原语（等待 Warp 新特性）

如果 Warp 中可用 warp 级内建函数，逐 warp 归约可以替代反向内核中的 `atomic_add`，有望消除非确定性问题并减少 LG 限制阻塞。

---

## 致谢

本 Warp 后端基于以下项目构建：

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)，作者 Bernhard Kerbl、Georgios Kopanas、Thomas Leimkühler 和 George Drettakis（INRIA, MPII）。原始 CUDA 光栅化器是参考实现。
- [Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction](https://arxiv.org/abs/2601.19489)，作者 Ziyu Zhang、Tianle Liu、Diantao Tu 和 Shuhan Shen。本项目使用的紧凑 AABB 包围盒技术受该工作启发。
- [NVIDIA Warp](https://nvidia.github.io/warp/)——用于高性能 GPU 仿真和计算的 Python 框架。
- [PyTorch](https://pytorch.org/)——用于张量管理和 CUDA 内存分配。

