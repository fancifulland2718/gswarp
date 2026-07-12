# 光栅化器后端

[English](rasterizer.md) · **中文**

本文档记录 `gswarp` 光栅化后端的技术细节、实现差异、性能数据与正确性验证。日常使用请参考[主文档](../README_zh.md)。

---

## 目录

- [概述](#概述)
- [架构概览](#架构概览)
- [与 CUDA 基线的差异](#与-cuda-基线的差异)
- [Python 层开销优化](#python-层开销优化)
- [正确性](#正确性)
- [性能特征（早期基准，仅 Warp 光栅化器）](#性能特征早期基准仅-warp-光栅化器)
- [更新基准（bench30k_plots，三模块全 Warp）](#更新基准bench30k_plots三模块全-warp)
- [已知限制](#已知限制)
- [未来优化方向](#未来优化方向)

---

## 概述

`gswarp` 光栅化后端以纯 Python + NVIDIA Warp 实现了兼容 [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) 接口的可微高斯光栅化管线，包含以下阶段：

1. **预处理**：3D 高斯投影到 2D，计算协方差、color（SH 评估）、AABB 瓦片矩形
2. **特征选择**：选择预计算 RGB 或 SH 评估得到的颜色
3. **分箱**：高斯-瓦片映射、深度排序、瓦片范围标识
4. **前向渲染**：block_dim=256，协作式 tile 加载，alpha 从前到后混合
5. **状态组装**：保留 backward 所需的 typed preprocess/binning/render 状态
6. **手写反向**：计算渲染、投影、协方差、scale/rotation 与 SH 梯度

每个公开调用都会冻结自己的运行时选项，并进入调用级 execution context。整个提交期间，Warp 绑定到当前 PyTorch CUDA 设备和 stream，包括非默认 stream。

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
│  - typed 预处理输出               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. 特征 + 分箱                   │
│  - 选择预计算/SH RGB              │
│  - 瓦片重叠计数（scan）           │
│  - 高斯→瓦片复制                 │
│  - 深度排序 + 瓦片排序            │
│  - 瓦片范围标识                   │
└─────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  3. 前向渲染 + 状态组装                   │
│  - block_dim=256 协作瓦片加载             │
│    (wp.tile + wp.tile_extract)           │
│  - 逐像素 alpha 混合                     │
│  - 从前到后合成                           │
│  - 透射率阈值提前终止                     │
│  - 颜色、深度、alpha 输出                 │
│  - 手写反向使用的 typed state             │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  4. 手写反向（MANUAL BACKWARD）           │
│  - block_dim=32 warp 级梯度归约           │
│    (wp.tile_reduce → warp shuffle)       │
│  - 渲染关于 conic、opacity、              │
│    color、pos 的梯度                      │
│  - 32× 更少的 atomic_add 写入            │
└──────────────────────────────────────────┘
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

### 模块边界

公开模块是兼容层，内部实现按功能所有权拆分：

| 层次 | 职责 |
|------|------|
| `gswarp/rasterizer*.py` | 公开 settings、返回结构与 CUDA 风格模块封装 |
| `_internal/frontend/` | PyTorch autograd 适配和输出转换 |
| `_internal/methods/` | 不可变方法描述、阶段计划与共享 typed executor |
| `_internal/backends/select.py` | stable/advanced 后端解析与能力校验 |
| `_internal/backends/warp/backend_3dgs*.py` | 轻量 standard/flow 阶段适配与 raw 兼容入口 |
| `*_ops.py` | 单一算法域的 Torch/Warp 编排 |
| `*_kernels.py` | 对应算法域的 Warp kernel 与 device function |
| `state.py`、`memory.py`、`runtime.py` | typed retained state、有界 workspace 与调用级运行时策略 |

kernel 模块仍与其调用的 device function 放在同一算法域内，Python 编排和公开兼容代码则保持在外部。这既保留 Warp JIT 的符号解析，也不再把整个后端重新合并成单体文件。

standard 与 flow 后端遵守同一阶段协议，共用预处理、分箱、渲染、stream、缓存和反向组件；flow 专属辅助输出只存在于 flow 路径。

### 关键常量

| 常量 | 值 | 描述 |
|------|---|------|
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

### 2. 部分协作式瓦片加载（通过 Warp Tile API）

**CUDA 基线**使用显式 `__shared__` 内存进行协作式数据获取——瓦片中的 256 个线程协作地将高斯数据从全局内存加载到共享内存，然后遍历共享缓冲区：

```c
// CUDA: forward.cu renderCUDA()
__shared__ int collected_id[BLOCK_SIZE];
__shared__ float2 collected_xy[BLOCK_SIZE];
__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
    float2 xy = collected_xy[j];
    float4 con_o = collected_conic_opacity[j];
    ...
}
```

**Warp 后端**无法直接声明和操作 `__shared__` 变量，但通过 Warp 的 Tile API（`wp.tile()` + `wp.tile_extract()`）间接实现了协作式加载。前向渲染 kernel 使用 block_dim=256（对应一个 16×16 像素瓦片），每轮迭代中每个线程加载 1 个高斯，然后所有线程通过 `wp.tile_extract()` 从 tile 中读取其他线程加载的数据：

```python
# Warp: _render_tiles_tiled256_warp_kernel
# 每个线程加载 1 个高斯（协作式）
my_xy = points_xy_image[my_id]
my_co = conic_opacity[my_id]
t_xy = wp.tile(my_xy, preserve_type=True)
t_co = wp.tile(my_co, preserve_type=True)

# 所有 256 个线程共享这批数据
for j in range(batch_count):
    xy_j = wp.tile_extract(t_xy, j)
    co_j = wp.tile_extract(t_co, j)
    ...
```

这在效果上等价于 CUDA 的共享内存协作获取模式——全局内存读取被分摊到瓦片中的所有线程。但受限于 Warp Tile API 的约束：
- `wp.tile()` 总是创建 block 级别的 tile，无法创建 warp 级别的 tile
- 每次 `wp.tile()` 调用会隐式涉及 `__syncthreads`
- 无法直接控制共享内存的布局或对齐方式

**反向渲染 kernel** 使用 block_dim=32（单 warp），采用 `wp.tile_reduce()` 进行梯度归约。在单 warp 配置下，`tile_reduce` 编译为纯 warp shuffle（`__shfl_down_sync`），无需 `__syncthreads` 或共享内存，是 Warp API 下反向梯度归约的最优配置。

**影响**：
- 前向渲染在所有规模下都已接近 CUDA 基线的效率，协作加载消除了原先每高斯 256× 的冗余全局内存读取。
- 反向渲染同样高效——warp 级归约将 atomic 写入减少了 32 倍。
- 但 Warp Tile API 在灵活性上仍不如直接操作 `__shared__` 内存，例如无法实现复杂的 double buffering 或自定义 bank conflict 规避策略。

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

### 5. 调度方式差异

CUDA 基线以瓦片块的 2D 网格调度渲染内核：
```c
dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
dim3 block(BLOCK_X, BLOCK_Y, 1);
```

Warp 后端以 1D 方式调度，但同样以瓦片为单位——每个瓦片对应 256 个线程（16×16 像素）：
```python
# 前向：block_dim=256，每个 block = 一个瓦片
_dim = num_tiles * 256
wp.launch(kernel, dim=_dim, block_dim=256, ...)
```

瓦片索引在内核内通过 `wp.tid()` 算术推导（`tile_id = tid // 256`，`local_id = tid % 256`）。功能等价，但 1D 调度的线程映射方式略有不同。

反向渲染 kernel 使用 block_dim=32（单 warp），每个 warp 覆盖一个相同的 16×16 瓦片的全部像素，但每个线程仅对应 tile 中的约 8 个像素。

---

## Python 层开销优化

Warp 的 Python 层需要进行类型检查、kernel 参数打包和 Torch/Warp array 转换，在每个训练迭代中会形成固定开销。当前实现通过以下方式降低这部分成本：

- 缓存不可变 MethodPlan，不在每个高斯或 kernel 上重新解析阶段
- 使用调用级不可变 ExecutionOptions，不再修改并恢复进程全局配置
- 使用带上限的 recorded launch 缓存
- standard backward 复用 forward 创建的 non-autograd Warp view
- 动态参数使用 ownerless launch descriptor，真实 tensor owner 留在调用或 graph 状态中
- 普通 frontend 使用 typed forward state，不再把状态打包为 opaque byte buffer
- workspace slot 按 stream 持有并有界淘汰，同时提供缓存报告

这些机制不改变公开返回结构和 stable 排序策略。这里不再声明当前设备上的性能数字；下方数值章节在新显卡上同时复测 CUDA/Warp 前均视为历史记录。

---

---

## 正确性

> **历史数值快照，等待复测。** 本节数值来自上一张显卡和较早代码快照。当前仓库已覆盖空场景、全裁剪、梯度、retain_graph、stream 所有权和缓存生命周期，但 CUDA/Warp 数值对照将在新硬件上一起重新生成。

### 随机粒子正确性（256K–2048K）

在 256K–2048K 规模的随机粒子下，使用公共 API 对 Native CUDA 与 Warp 后端的前向渲染输出和反向梯度进行数值一致性验证。测试配置：`backward_mode="manual"`、`binning_sort_mode="warp_depth_stable_tile"`、`auto_tune=True`。测试平台：**NVIDIA RTX 5090D V2**，PyTorch 2.11.0+cu130，Warp 1.12.0。

#### 前向：渲染颜色差异

| 规模 | 分辨率 | 最大绝对误差 | 平均绝对误差 |
|------|--------|------------|------------|
| 256K | 384×384 | 0.0108 | 4.01e-05 |
| 512K | 512×512 | 0.0064 | 3.37e-05 |
| 1024K | 640×640 | 0.0077 | 2.78e-05 |
| 2048K | 800×800 | 0.0033 | 1.74e-05 |

最大单像素误差 < 0.011（值域 [0,1]），平均绝对误差在 1e-5 数量级。随规模增大，误差反而略有下降，说明差异并非系统性偏差而是稀疏离群像素。两个后端使用不同的 tile 排序实现和浮点累加顺序，产生数值级别的微小差异。

#### 反向：梯度差异

| 规模 | 梯度字段 | 最大绝对误差 | 平均绝对误差 |
|------|----------|------------|------------|
| 256K | `grad_means3D` | 35.78 | 0.00114 |
| | `grad_shs` | 1.75 | 6.68e-05 |
| | `grad_scales` | 152.56 | 0.00607 |
| | `grad_rotations` | 12.11 | 3.95e-04 |
| | `grad_opacities` | 2.38 | 1.56e-04 |
| 1024K | `grad_means3D` | 89.53 | 5.63e-04 |
| | `grad_shs` | 6.36 | 4.97e-05 |
| | `grad_scales` | 1014.70 | 0.00324 |
| | `grad_rotations` | 64.96 | 2.14e-04 |
| | `grad_opacities` | 12.29 | 1.00e-04 |
| 2048K | `grad_means3D` | 43.93 | 2.95e-04 |
| | `grad_shs` | 3.68 | 2.21e-05 |
| | `grad_scales` | 461.47 | 0.00173 |
| | `grad_rotations` | 19.03 | 1.05e-04 |
| | `grad_opacities` | 1.85 | 5.37e-05 |

反向梯度的最大绝对误差看似较大，但需注意：
- 梯度值域极宽（如 `grad_scales` 可达 ±10⁴），最大绝对误差仅出现在少数梯度极值点。
- **平均绝对误差**极小（`grad_means3D` < 0.0012，`grad_shs` < 7e-05），绝大多数点的梯度高度一致。
- 随规模从 256K → 2048K 增大，平均绝对误差持续下降（如 `grad_means3D` 从 0.00114 降至 0.00030），说明差异源于稀疏离群点而非系统性偏差。
- 差异来源于 tile 排序顺序不同导致 alpha blending 累加的浮点舍入差异，以及原子写入梯度的非确定性顺序。

### 端到端训练质量（12 数据集 × 30K 迭代）

> **基准条件**：本节的端到端训练质量数据收集于较早的基准测试：仅光栅化器使用 Warp 后端；SSIM 使用 cuda-fused；KNN 使用 CUDA simple-knn；Python 层开销优化尚未应用。三模块全替换后的结果请参考[更新基准](#更新基准bench30k_plots三模块全-warp)节。

以下数据基于 3DGS 原始仓库的**默认训练参数**（30,000 次迭代），在 12 个标准数据集上测试。测试平台为 **NVIDIA RTX 5090D V2**（24 GiB），PyTorch 2.11.0+cu130，Warp 1.12.0。

**NeRF Synthetic（800×800）：**

| 数据集 | | PSNR (dB) | SSIM | LPIPS |
|--------|---|-----------|------|-------|
| chair | Warp | 35.614 | 0.9871 | 0.0119 |
| | CUDA | 35.854 | 0.9876 | 0.0116 |
| drums | Warp | 26.083 | 0.9540 | 0.0371 |
| | CUDA | 26.173 | 0.9548 | 0.0367 |
| ficus | Warp | 34.827 | 0.9871 | 0.0119 |
| | CUDA | 34.901 | 0.9873 | 0.0117 |
| hotdog | Warp | 37.552 | 0.9849 | 0.0206 |
| | CUDA | 37.624 | 0.9854 | 0.0200 |
| lego | Warp | 35.758 | 0.9829 | 0.0154 |
| | CUDA | 35.903 | 0.9832 | 0.0154 |
| materials | Warp | 29.998 | 0.9609 | 0.0334 |
| | CUDA | 30.102 | 0.9616 | 0.0329 |
| mic | Warp | 35.596 | 0.9916 | 0.0061 |
| | CUDA | 35.998 | 0.9922 | 0.0057 |
| ship | Warp | 30.884 | 0.9057 | 0.1054 |
| | CUDA | 31.062 | 0.9074 | 0.1057 |

**Tanks & Temples / Deep Blending（各场景原始分辨率）：**

| 数据集 | | PSNR (dB) | SSIM | LPIPS |
|--------|---|-----------|------|-------|
| train | Warp | 22.101 | 0.8183 | 0.1982 |
| | CUDA | 22.060 | 0.8213 | 0.1962 |
| truck | Warp | 25.372 | 0.8828 | 0.1445 |
| | CUDA | 25.479 | 0.8850 | 0.1420 |
| drjohnson | Warp | 29.383 | 0.9043 | 0.2372 |
| | CUDA | 29.455 | 0.9053 | 0.2357 |
| playroom | Warp | 30.150 | 0.9076 | 0.2414 |
| | CUDA | 30.072 | 0.9091 | 0.2399 |

大部分场景中 Warp 与 CUDA 的 PSNR 差距在 **-0.07 ~ -0.40 dB** 之间，属于可忽略的差异。两个场景（train、playroom）Warp 的 PSNR 甚至略优于 CUDA。SSIM 和 LPIPS 指标同样极为接近。这种差异可能是由 Warp 后端的训练收敛速率稍慢导致的。

---

## 性能特征（早期基准）

> **历史 benchmark。** 本节仅用于保留测试来源，不代表当前版本的性能声明。


> **基准条件**：本节数据收集于较早的基准测试：仅光栅化器使用 Warp 后端；SSIM 使用 cuda-fused（非 Warp SSIM）；KNN 使用 CUDA simple-knn（非 Warp KNN）；Python 层开销优化尚未应用。三模块全替换后的训练速度请参考[更新基准](#更新基准bench30k_plots三模块全-warp)节。由于使用的测试脚本有所差异，存在性能上的巨大不同。

### 端到端训练性能（12 数据集 × 30K 迭代）

以下数据在 **NVIDIA RTX 5090D V2**（24 GiB，sm_120）、**PyTorch 2.11.0+cu130**、**Warp 1.12.0** 上测试，使用 3DGS 原始仓库的默认训练参数。

#### 训练速度与训练总时间

| 数据集 | | 平均 FPS (30K) | 训练总时间 (s) | 峰值显存 (MB) |
|--------|---|-------:|--------:|--------:|
| chair | CUDA | 139.6 | 215 | 609 |
| | Stable Tile | 125.5 | 239 | 567 |
| | Radix | 111.3 | 270 | 566 |
| | Torch Sort | 117.9 | 255 | 565 |
| drums | CUDA | 160.9 | 186 | 648 |
| | Stable Tile | 128.2 | 234 | 607 |
| | Radix | 120.1 | 250 | 609 |
| | Torch Sort | 118.4 | 253 | 620 |
| ficus | CUDA | 214.7 | 140 | 394 |
| | Stable Tile | 152.9 | 196 | 386 |
| | Radix | 145.7 | 206 | 387 |
| | Torch Sort | 147.1 | 204 | 385 |
| hotdog | CUDA | 124.3 | 241 | 427 |
| | Stable Tile | 156.1 | 192 | 344 |
| | Radix | 139.4 | 215 | 345 |
| | Torch Sort | 141.1 | 213 | 348 |
| lego | CUDA | 151.6 | 198 | 640 |
| | Stable Tile | 144.1 | 208 | 565 |
| | Radix | 130.1 | 231 | 560 |
| | Torch Sort | 133.2 | 225 | 567 |
| materials | CUDA | 169.5 | 177 | 512 |
| | Stable Tile | 162.6 | 184 | 468 |
| | Radix | 145.2 | 207 | 464 |
| | Torch Sort | 144.1 | 208 | 467 |
| mic | CUDA | 154.1 | 195 | 603 |
| | Stable Tile | 116.7 | 257 | 561 |
| | Radix | 112.7 | 266 | 568 |
| | Torch Sort | 110.5 | 271 | 566 |
| ship | CUDA | 83.8 | 358 | 801 |
| | Stable Tile | 121.7 | 247 | 642 |
| | Radix | 120.1 | 250 | 645 |
| | Torch Sort | 121.3 | 247 | 642 |
| train | CUDA | 66.5 | 451 | 1,888 |
| | Stable Tile | 73.6 | 408 | 1,869 |
| | Radix | 71.6 | 419 | 1,870 |
| | Torch Sort | 72.7 | 413 | 1,874 |
| truck | CUDA | 62.9 | 477 | 3,353 |
| | Stable Tile | 60.6 | 495 | 3,597 |
| | Radix | 59.3 | 506 | 3,596 |
| | Torch Sort | 61.7 | 486 | 3,614 |
| drjohnson | CUDA | 32.9 | 913 | 5,254 |
| | Stable Tile | 51.2 | 586 | 5,305 |
| | Radix | 48.8 | 615 | 5,304 |
| | Torch Sort | 49.4 | 607 | 5,323 |
| playroom | CUDA | 41.9 | 716 | 3,126 |
| | Stable Tile | 70.2 | 427 | 3,219 |
| | Radix | 67.3 | 446 | 3,229 |
| | Torch Sort | 65.2 | 460 | 3,229 |

**关键发现：**
- NeRF Synthetic（小场景，~15–35 万高斯）：CUDA 通常更快。三种 Warp 排序后端中，**Stable Tile** 整体最快，其次是 Torch Sort 和 Radix。
- Tanks & Temples / Deep Blending（大场景，~100–310 万高斯）：**三种 Warp 后端均明确优于 CUDA**——例如 drjohnson：Stable Tile 1.56×、Radix 1.49×、Torch Sort 1.50×；playroom：Stable Tile 1.68×、Radix 1.61×、Torch Sort 1.56×。
- 三种 Warp 排序后端（Stable Tile、Radix、Torch Sort）性能非常接近：最大速度差异通常在 10–15% 以内。**推荐使用 Stable Tile（默认值）**，它提供了最佳的整体吞吐量。
- 显存占用：所有后端在同一量级，Warp 变体在大多数 NeRF Synthetic 场景上峰值显存更低。

#### 训练过程曲线

下图展示了 12 个数据集上 CUDA（蓝色）、Stable Tile（红色）、Radix（绿色）和 Torch Sort（紫色）在 30K 迭代训练中的迭代速度（FPS）、总 loss（对数）、峰值 GPU 显存和高斯点数的变化：

![排序后端训练概览](../figures/warp%20rasterizer/sort_backend_overview.png)

### 微基准测试（随机粒子）

以下数据来自随机粒子测试，测试平台为 **NVIDIA GeForce RTX 5090D**（sm_120，24 GiB，170 个 SM）、**Warp 1.12.0**、**PyTorch 2.11.0+cu130**。

测试方法：

- **稳态耗时**：使用公共 API（`diff_gaussian_rasterization.GaussianRasterizer` 与 `diff_gaussian_rasterization.warp.GaussianRasterizer`）先做单独预热，再用 CUDA event 对正式采样段做批量计时，主表展示正式采样的平均值。
- **峰值显存**：预热后，`reset_peak_memory_stats` 后运行一次完整阶段，记录 `max_memory_allocated` 绝对峰值。前向峰值仅含 forward；反向峰值含 forward+backward 全流程。
- **阶段计时 / 阶段显存**：仅为分析热点，使用内部 `_warp_backend` 辅助函数做诊断性测量；这些阶段数据**不保证**与公共 API 的端到端时间或峰值显存逐项严格相加一致。

其中 256K / 512K / 1024K / 2048K 四档规模分别采用 **4+8 / 3+6 / 3+6 / 2+4**（预热次数 + 正式采样次数）的配置。

### 公共 API 稳态耗时

| 点数 | 分辨率 | `num_rendered` | 原生前向 | Warp 前向 | 前向比值 | 原生反向 | Warp 反向 | 反向比值 |
|------|--------|----------------|----------|-----------|----------|----------|-----------|----------|
| 262,144 | 384×384 | 265,106 | 0.315 ms | 0.999 ms | 3.17× | 1.772 ms | 1.012 ms | 0.57× |
| 524,288 | 512×512 | 874,049 | 0.461 ms | 1.101 ms | 2.39× | 3.252 ms | 1.632 ms | 0.50× |
| 1,048,576 | 640×640 | 2,556,549 | 0.857 ms | 1.594 ms | 1.86× | 5.363 ms | 2.464 ms | 0.46× |
| 2,097,152 | 800×800 | 7,644,361 | 2.695 ms | 2.697 ms | 1.00× | 8.839 ms | 4.418 ms | 0.50× |

在 256K–2048K 规模下，Warp 前向因 Python 级编排开销（张量分配、kernel launch、阶段间数据传递）仍慢于原生（1.00×–3.17×），但随粒子数增大、GPU 计算量占主导，差距快速收窄——2048K 时前向达到 **1.00×**（与原生持平）。**反向方面，Warp 在所有规模下均快于原生**（比值 0.46×–0.57×），这得益于 `_backward_render_tiles_warp32` kernel 的 warp shuffle（block_dim=32）快速路径在大规模 `num_rendered` 下的计算效率优势。

### 公共 API 峰值显存占用

| 点数 | 分辨率 | 原生前向峰值 | 原生反向峰值 | Warp 前向峰值 | Warp 反向峰值 |
|------|--------|------------|------------|-------------|-------------|
| 262,144 | 384×384 | 128.61 MiB | 205.67 MiB | 145.63 MiB | 336.08 MiB |
| 524,288 | 512×512 | 274.00 MiB | 427.51 MiB | 285.69 MiB | 662.50 MiB |
| 1,048,576 | 640×640 | 587.75 MiB | 891.75 MiB | 570.97 MiB | 1324.02 MiB |
| 2,097,152 | 800×800 | 1300.70 MiB | 1910.72 MiB | 1149.20 MiB | 2645.05 MiB |

> **说明**：此处为绝对峰值（`max_memory_allocated`），包含输入张量、模型参数、前向中间结果及 autograd 保存张量等全部分配。反向峰值包含 forward+backward 全流程。

Warp 前向峰值在 256K 时比原生高约 1.13×，但 1024K 起 Warp 前向峰值（571 MiB）反而**低于**原生（588 MiB），2048K 时为原生的 0.88×。反向峰值 Warp 约为原生的 1.4×–1.6×（2048K 时 2645 vs 1911 MiB），主要由 Warp 的额外中间张量（深度、alpha、投影坐标、逐像素权重等）所致。

### 内部阶段热点

| 点数 | 分辨率 | 预处理 | 分箱 | 渲染 | 反向渲染 | 选定排序模式 |
|------|--------|--------|------|------|----------|--------------|
| 262,144 | 384×384 | 0.335 ms | 0.495 ms | 0.221 ms | 0.386 ms | `warp_depth_stable_tile` |
| 524,288 | 512×512 | 0.390 ms | 0.492 ms | 0.248 ms | 0.430 ms | `warp_depth_stable_tile` |
| 1,048,576 | 640×640 | 0.580 ms | 0.583 ms | 0.249 ms | 0.561 ms | `warp_depth_stable_tile` |
| 2,097,152 | 800×800 | 0.944 ms | 1.275 ms | 0.323 ms | 0.790 ms | `warp_depth_stable_tile` |

### 内部阶段累计峰值显存（仅作诊断解释）

| 点数 | 分辨率 | 预处理后峰值 | 预处理+分箱后峰值 | 前向全流程峰值 | 前向+反向全流程峰值 |
|------|--------|-------------|-----------------|-------------|-------------------|
| 262,144 | 384×384 | 134.76 MiB | 137.76 MiB | 138.76 MiB | 136.07 MiB |
| 524,288 | 512×512 | 264.67 MiB | 271.01 MiB | 270.83 MiB | 266.01 MiB |
| 1,048,576 | 640×640 | 529.08 MiB | 541.08 MiB | 540.08 MiB | 526.71 MiB |
| 2,097,152 | 800×800 | 1053.43 MiB | 1076.07 MiB | 1076.07 MiB | 1052.18 MiB |

> **说明**：每列为从 `empty_cache()` 后重新执行到该阶段结束时的绝对峰值。由于各阶段间有临时张量释放和复用，"前向+反向全流程峰值"可能略低于"前向全流程峰值"。

从内部阶段看，preprocess 和 binning 随粒子数近线性增长，是可扩展性瓶颈；render 和 backward render 在 2048K 时仍不到 1 ms，表明 Warp tile kernel 在纯 GPU 计算侧效率良好。preprocess 是显存主要消耗者（2048K 时预处理后峰值 1053 MiB）；binning 在此基础上仅增加约 23 MiB（2048K），render 几乎不增加额外峰值。

### Kernel 级剖析（Nsight Systems）

> **说明**：该 GPU SKU（RTX 5090D V2）不支持 Nsight Compute 硬件性能计数器采集（`ERR_NVGPU`）。以下数据通过 **Nsight Systems 2025.6** 时间线追踪获取，分别在 **256K@384×384** 和 **1024K@640×640** 两个规模下进行。每次运行 5 次预热 + 3 次 NVTX 标记的正式迭代（前向+反向），排序模式 `warp_depth_stable_tile`。

#### 每迭代端到端耗时分解（NVTX）

| 规模 | 前向（NVTX） | 反向（NVTX） | 迭代总耗时 |
|------|------------|------------|-----------|
| 256K@384×384 | 1.42 ms | 1.37 ms | 2.79 ms |
| 1024K@640×640 | 2.84 ms | 1.36 ms | 4.20 ms |

#### 按 GPU 耗时排序的 Kernel 热点

下表列出 256K 与 1024K 规模下全管线（前向 + 反向）中每迭代各 kernel 的平均 GPU 耗时，数据来自 nsys 时间线统计（取 8 次迭代的每实例均值）。

| Kernel | 调用/迭代 | 256K 平均耗时 | 1024K 平均耗时 | 所属阶段 |
|--------|---------|-------------|--------------|----------|
| `_backward_render_tiles_warp32` | 1 | 230.0 µs | 423.2 µs | 反向渲染 |
| `_render_tiles_fast_warp` | 1 | 93.6 µs | 127.7 µs | 前向渲染 |
| `_backward_rgb_from_sh_v3` | 1 | 59.4 µs | 408.5 µs | 反向预处理 |
| `_forward_rgb_from_sh_v3` | 1 | 48.6 µs | 227.9 µs | 前向预处理（SH→RGB） |
| CUB `DeviceRadixSort` (Onesweep) | 8 | 7.4 µs × 8 | 12.5 µs × 8 | 分箱（排序） |
| `_duplicate_with_keys_from_order` | 1 | 34.7 µs | 129.9 µs | 分箱（重叠展开） |
| `_fused_project_cov3d_cov2d_preprocess_sr` | 1 | 25.2 µs | 131.2 µs | 前向预处理（投影+协方差） |
| `_fused_backward_preprocess_accumulate` | 1 | 24.2 µs | 102.3 µs | 反向预处理 |
| PyTorch `elementwise_copy` | ~3 | 41.5 µs × 3 | 366.5 µs × 3 | PyTorch 张量拷贝 |
| PyTorch `fill` / `zero_` | ~10 | 3.6 µs × 10 | 18.8 µs × 10 | PyTorch 初始化 |
| `_identify_tile_ranges` | 1 | 1.4 µs | 6.9 µs | 分箱 |
| `_gather_i32_by_index` | 1 | 1.7 µs | 5.9 µs | 分箱 |

#### CPU vs GPU 开销分解

| 指标 | 256K@384×384 | 1024K@640×640 |
|------|-------------|--------------|
| 每迭代墙钟时间 | 2.79 ms | 4.20 ms |
| Warp kernel GPU 时间 | 578 µs (20.7%) | 1,664 µs (39.6%) |
| PyTorch kernel GPU 时间 | 186 µs (6.7%) | 1,383 µs (32.9%) |
| **GPU 合计** | **764 µs (27.4%)** | **3,047 µs (72.5%)** |
| **CPU 开销（差值）** | **2,026 µs (72.6%)** | **1,153 µs (27.5%)** |

CPU 开销的主要来源（由 nsys CUDA API Summary 统计）：

| CUDA API 调用 | 每迭代调用次数（近似） | 中位延迟 | 每迭代合计 |
|---------------|---------------------|---------|-----------|
| `cudaLaunchKernel`（PyTorch 侧） | ~38 | 5.8 µs | ~220 µs |
| `cuLaunchKernel`（Warp 侧） | ~9 | 15.4 µs | ~139 µs |
| `cudaMemsetAsync` | ~15 | 4.6 µs | ~69 µs |
| `cudaMemcpyAsync` | ~4 | 18.8 µs | ~75 µs |
| `cudaMallocAsync` | ~5 | 4.3 µs | ~22 µs |
| `cudaFreeAsync` | ~6 | 2.3 µs | ~14 µs |
| **CUDA API 合计** | | | **~539 µs** |
| **Python/Warp 运行时（张量创建、属性查找、调度）** | | | **~1,487 µs** |

#### 关键分析

1. **CPU 开销近似恒定**：无论 256K 还是 1024K，CPU 侧开销均约 1.2–2.0 ms/迭代。256K 时 CPU 占 73%，1024K 时降至 28%。到 2048K 时 GPU 计算量进一步增大，CPU 开销比例继续下降，这解释了 2048K 前向比值降至 1.00× 的原因。
2. **反向渲染是最大 GPU 热点**：`_backward_render_tiles_warp32`（block_dim=32，warp shuffle 快速路径）在 256K 时 230 µs，1024K 时 423 µs，占 Warp kernel GPU 时间的 25%–40%。
3. **SH 颜色计算随 N 线性增长**：`_forward_rgb_from_sh_v3` + `_backward_rgb_from_sh_v3` 在 1024K 时合计 636 µs，成为仅次于 backward render 的第二大热点。SH degree=3 时每点 16 个系数 × 3 通道的读写量大。
4. **PyTorch `elementwise_copy` 随数据量剧增**：256K 时单次 42 µs，1024K 时单次 367 µs。这是 PyTorch autograd 框架的张量拷贝/类型转换，并非 Warp 自身 kernel。在 1024K 规模占 GPU 时间的 33%。
5. **实际 CUDA API 调用仅占 CPU 开销的 ~26%**（256K 时 539 µs / 2026 µs）。其余 ~74% 是 Python GIL 下的对象创建、Warp 运行时调度、PyTorch autograd 图构建等纯 CPU 开销。这是 Warp-on-Python 架构的固有代价，短期内只能通过 CUDA Graph 或减少管线阶段数来缓解。

---

## 更新基准（bench30k_plots，三模块全 Warp）

> **历史 benchmark，等待替换。** 在新显卡上同时复测 CUDA 与 Warp 之前，本节不视为当前结论。

**测试平台**：NVIDIA RTX 5090D V2（24 GiB，sm_120），PyTorch 2.11.0+cu130，Warp 1.12.0。训练参数同 3DGS 原始仓库默认值。

> **测试条件**：光栅化器、SSIM、KNN 均使用 Warp 后端；Python 层开销优化已应用；12 数据集 × 30K 迭代。可通过 [SSIM 文档](ssim_zh.md#端到端训练影响)和 [KNN 文档](knn_zh.md#端到端训练影响)查看各模块的单独贡献。

### 训练吞吐量与总时长

| 数据集 | 场景类型 | CUDA (it/s) | Warp (it/s) | 加速比 | CUDA 总时长 | Warp 总时长 | 最终高斯数 |
|--------|---------|------------|------------|-------|-----------|-----------|---------|
| chair | NeRF合成 | 103.6 | 113.1 | ×1.09 | 4.7 min | 4.3 min | ~30 万 |
| drums | NeRF合成 | 103.0 | 115.3 | ×1.12 | 4.6 min | 4.2 min | ~33 万 |
| ficus | NeRF合成 | 139.5 | 148.2 | ×1.06 | 3.5 min | 3.3 min | ~19 万 |
| hotdog | NeRF合成 | 144.5 | 156.5 | ×1.08 | 3.6 min | 3.2 min | ~17 万 |
| lego | NeRF合成 | 117.5 | 126.5 | ×1.08 | 4.2 min | 3.9 min | ~30 万 |
| materials | NeRF合成 | 134.0 | 144.9 | ×1.08 | 3.7 min | 3.4 min | ~24 万 |
| mic | NeRF合成 | 95.4 | 105.1 | ×1.10 | 4.9 min | 4.5 min | ~30 万 |
| ship | NeRF合成 | 107.0 | 113.2 | ×1.06 | 4.5 min | 4.2 min | ~35 万 |
| train | Tanks&Temples | 55.6 | 58.3 | ×1.05 | 8.7 min | 8.2 min | ~110 万 |
| truck | Tanks&Temples | 39.4 | 40.1 | ×1.02 | 11.7 min | 11.3 min | ~210 万 |
| drjohnson | Deep Blending | 30.8 | 32.0 | ×1.04 | 14.3 min | 13.8 min | ~310 万 |
| playroom | Deep Blending | 46.9 | 47.5 | ×1.01 | 9.9 min | 9.7 min | ~190 万 |

**规律小结**：
- NeRF 合成场景（小场景，15–35 万高斯）：Warp 平均快 **~8%**，主要得益于反向渲染的 warp shuffle 归约和更紧凑的 AABB 分箱。
- 大场景（100–310 万高斯）：加速比收窄至 1–5%，瓶颈转移到 Adam 优化器，Warp 的渲染/反向优化贡献比例下降。

### 各阶段耗时分布（最后 5K 迭代均值，ms）

以下数据来自几个代表性场景，展示 Warp 相比 CUDA 在哪个阶段有优势：

| 阶段 | chair CUDA | chair Warp | drjohnson CUDA | drjohnson Warp |
|------|-----------|-----------|---------------|---------------|
| render | 1.85 | 1.63 | 6.11 | 5.79 |
| loss (SSIM+L1) | 0.29 | 0.35 | 0.31 | 0.38 |
| backward | 4.88 | 4.27 | 11.21 | 10.60 |
| densify | 0.48 | 0.44 | 1.48 | 1.44 |
| optim | 0.98 | 0.98 | 8.24 | 8.14 |

前向渲染（render）和反向计算（backward）是 Warp 加速最明显的两个阶段；优化器（optim）由 PyTorch Adam 控制，两者基本持平。loss 阶段 Warp 略慢，因为 Warp SSIM 核在某些分辨率下比 CUDA fused-ssim 慢——细节见 [SSIM 文档](ssim_zh.md)。

---

---

## 已知限制

### 1. CUDA 兼容边界

常见的关键字参数迁移路径已支持，但公开合同并不与每个版本的 `diff-gaussian-rasterization` 完全逐字节一致：

- 标准 gswarp 返回 `(color, radii, RasterizerMeta)`，而不是直接返回单独的 depth tensor
- `dc` 是位于 `shs` 之前的额外兼容别名，因此不建议按位置迁移参数
- 尚未实现 `prefiltered=True`
- `antialiasing` 字段可用于构造兼容，但 stable 后端不会执行对应 CUDA 路径
- `debug=True` 会检查公开输入中的非有限值，但不会复现 CUDA 扩展的快照导出

### 2. 手写反向与原子累积顺序

当前仅支持 `backward_mode="manual"`。梯度累积使用原子操作，因此重复运行可能因浮点累积顺序产生微小差异，CUDA 基线也有相同行为。typed forward state 与显式反向通过自定义 PyTorch autograd function 接入。

### 3. 固定瓦片形状与首次 JIT

`BLOCK_X=16`、`BLOCK_Y=16` 是编译期常量，修改后需要重新编译 Warp kernel。新设备或新 kernel 配置的第一次调用会触发 Warp JIT；后续进程可以复用 Warp 的磁盘缓存。

### 4. Python 编排与同步

CUDA 基线的大部分编排发生在 C++。gswarp 仍需从 Python 提交多个 Warp kernel 和 PyTorch 操作；分箱阶段还要把精确的渲染引用数量读回主机，再分配并启动后续工作。recorded launch 与 ownerless descriptor 能降低但不能消除这部分开销。

### 5. 缓存高水位

可复用 workspace 按设备和 stream 数量有界，launch cache 也有条目上限。每个 workspace slot 内的 buffer 会保留该 slot 曾请求的最大容量，直到被淘汰或显式调用 `clear_warp_caches()`。可以用 `get_warp_cache_report()` 检查保留字节和 stream 分布。

### 6. Advanced Warp 后端

resolver 已能按 Warp 版本和公开能力门控可选实现。当前 advanced 模块仍是不可执行的占位，因此自动选择会使用 stable 后端。

---

## 未来优化方向

优化优先级将在新显卡上完成 CUDA/Warp 配对 profile 后重新编写。下一轮 benchmark 必须区分完整训练、完整推理生命周期与单 kernel 时间，并同时报告固定主机开销、同步、分配和缓存保留行为。

其他 Gaussian 方法和任何可执行的 advanced Warp 后端都将使用独立规划。当前 baseline 只保留阶段执行器与能力门控基础，不提前增加外部插件 API 或未经验证的后端占位。
