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
- [当前基准](#当前基准)
- [已知限制](#已知限制)

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

这些机制不改变公开返回结构和 stable 排序策略。当前端到端、可控 microbenchmark 和 profiler 结果见[当前基准](#当前基准)。

---

## 正确性

当前正确性评估由合同测试、组件对照与冻结 checkpoint 对照共同组成：

- 公开 API 回归测试覆盖空场景、全裁剪场景、手写梯度、retain_graph、stream 所有权、缓存生命周期和 CUDA 兼容调用行为
- 当前组件检查将 SSIM forward/backward 与 KNN 初始化结果分别和原生 CUDA 对照
- 冻结 checkpoint 检查向原生 CUDA 和 gswarp 提供完全相同的相机、高斯与背景输入

当前冻结检查覆盖 Lego、Truck 和 Train。图像 MAE 为 1.79e-7 至 5.10e-7，两种渲染输出之间的 PSNR 为 100.33 dB 至 105.54 dB；可见集合完全相同，或仅在 Train 约 2560 万次 Gaussian-view 观测中出现 2 次单侧判定。详细数值与解释见[当前基准](#当前基准)。

在已测范围内，结果没有显示光栅化器公式、公开接口合同或系统性覆盖方面的缺陷。FP32 原子累积与归约顺序仍可能产生微小残差，并改变长时间训练轨迹；逐字节一致不是正确性判据。

---

## 当前基准

当前端到端矩阵维护在[项目 README](../README_zh.md#基准测试)。数据在 RTX 5090（32 GiB，sm_120）、Python 3.14.3、PyTorch 2.11.0+cu130 与 Warp 1.12.0 上，以 30K 迭代重新生成。它覆盖 8 个 NeRF Synthetic、2 个 Tanks and Temples 和 2 个 Deep Blending 场景。

### 来源与测量规则

CUDA 和 Warp 的对比必须刻意隔离两个包：

1. 将当前 gswarp 源码构建到独立安装目录，并校验安装后源码哈希。
2. 仅通过包含原生 diff_gaussian_rasterization 扩展的包根暴露 CUDA 后端，避免意外导入其他已安装的 gswarp。
3. 在接纳结果前，断言 CUDA 光栅化器解析到原生扩展，Warp 光栅化器解析到独立 gswarp 目录。
4. 对每次运行记录解析出的模块与原生扩展路径；对独立训练的 checkpoint，通过相同的原始 3DGS CUDA 渲染器运行 render 和 metrics 流程。

两个后端均使用默认 3DGS 优化参数、CPU 数据加载、默认 Adam 与 30K 迭代。Warp 使用 gswarp 光栅化器、fused SSIM 与 KNN。仅 Warp 关闭 depth 累积，因为参考 loss 不消费 depth。墙钟时间包含最终评估和 checkpoint 保存；阶段时间使用迭代 25,000--29,999 的 CUDA event。

### 稳定训练阶段

| 场景 | 后端 | Render ms | Loss ms | Backward ms | Densify ms | Optimizer ms |
|------|------|----------:|--------:|------------:|-----------:|-------------:|
| Lego | CUDA | 1.789 | 0.503 | 3.509 | 0.001 | 1.130 |
| Lego | Warp | 1.608 | 0.562 | 3.734 | 0.002 | 1.127 |
| Train | CUDA | 3.916 | 0.582 | 9.358 | 0.002 | 3.252 |
| Train | Warp | 3.134 | 0.637 | 7.630 | 0.001 | 3.236 |
| Truck | CUDA | 4.347 | 0.579 | 9.471 | 0.001 | 5.548 |
| Truck | Warp | 4.099 | 0.668 | 8.578 | 0.001 | 5.629 |
| DrJohnson | CUDA | 6.356 | 0.683 | 20.112 | 0.001 | 8.472 |
| DrJohnson | Warp | 5.143 | 0.749 | 10.278 | 0.001 | 8.458 |

这四行展示了套件汇总掩盖的差异。Lego 上 Warp 的 render 降低 10.1%，但 loss 和 backward 更高，完整任务快 3.5%。Train 上，render 降低 20.0%、backward 降低 18.5%，墙钟低 17.2%。Truck 上，render 降低 5.7%、backward 降低 9.4%，但 Warp SSIM loss 更高，墙钟低 8.7%。DrJohnson 上，render 降低 19.1%、backward 降低 48.9%，墙钟低 29.0%。这里有意报告混合的端到端结果，而不把全部吞吐量归因到单个光栅化器 kernel。

### 当前可控 Microbenchmark

下列结果使用当前 RTX 5090 上已安装的包产物，而非从 checkout 导入。它们是可控的 kernel 栈测量，不能被解读为完整训练吞吐。

| 工作负载 | 原生 CUDA GPU 中位数 | Warp GPU 中位数 | CUDA host 中位数 | Warp host 中位数 |
|----------|--------------------:|----------------:|----------------:|----------------:|
| 300K Gaussian，800x800，SH degree 3，光栅化反向 | 3.809 ms | 1.638 ms | 0.274 ms | 0.530 ms |

该合成视角有 36,842 个可见 Gaussian，使用 20 次预热和 100 次测量，GPU 时间来自 CUDA event。这个刻意隔离的反向工作负载中 Warp GPU 时间更低，并不代表总 Python 侧调度开销更低：其 host 中位数更高。单独的 Warp 全前向加反向、预计算 RGB 测量中，256K Gaussian、384x384 为 2.215 ms 和 85.8 MiB；1024K Gaussian、640x640 为 4.733 ms 和 345.5 MiB。重复输出的最大绝对差分别为 5.59e-9 和 5.96e-8。

### 当前 Nsight 证据

在 RTX 5090 上采集了 Nsight Systems 2026.1.2 的 CUDA/NVTX trace 与 Nsight Compute 2026.1 profile。trace 工作负载是 300K Gaussian、800x800、SH degree 3，且与可控基准相同，具有 36,842 个可见 Gaussian。trace 的 kernel 时长只用于识别工作结构；CUDA event 中位数仍是延迟参考，因为 profiler replay 和 trace 会扰动计时。

Warp 中，反向 tile renderer 占 trace kernel 时间的 57.4%，每个 trace launch 平均 1.069 ms；tiled forward renderer 占 11.2%，为 0.209 ms。每次重复有 8 次 radix-sort dispatch，单次 radix-sort launch 平均 14.0 microseconds。原生 CUDA 中，主反向 render kernel 占 trace kernel 时间的 73.8%，每个 trace launch 为 3.339 ms；前向 render 占 7.2%，为 0.327 ms。它每次重复有 6 次 sort pass，sort launch 平均 49.4 microseconds。

Nsight Compute 显示 Warp 反向 tile kernel 没有 local 或 shared-memory spill。其报告的 theoretical occupancy 为 50.0%，achieved occupancy 为 31.14%，issue-slot utilization 为 29.23%，L2 hit rate 为 96.91%，L1 hit rate 为 43.38%，DRAM-throughput utilization 为 0.8%。一次 launch 有 4.9 个 wave/SM，报告估计 tail effect 约为 20%。对于这个已测工作负载，occupancy、发射效率和工作分布比 DRAM 带宽更具解释力。这些数值仅描述 RTX 5090 上的该工作负载，不构成跨设备性能声明。

### 数值结果解释

README 分开报告独立训练质量与冻结 checkpoint 的 CUDA/Warp 对比，因为两者回答的问题不同。独立 30K 训练衡量完整训练栈的最终结果；冻结 checkpoint 则向两个渲染器提供相同的相机、高斯和背景输入，以隔离光栅化过程。

在 Lego、Truck 和 Train 上，冻结 checkpoint 的图像 MAE 为 1.79e-7 至 5.10e-7，两种渲染输出之间的 PSNR 为 100.33 dB 至 105.54 dB。Lego 和 Truck 的可见集合完全相同。Train 的可见性 Jaccard 为 0.99999992，对应约 2560 万次 Gaussian-view 观测中的 2 次单侧判定；两种输出相对 ground truth 的全局 PSNR 也均为 20.955083 dB（按展示精度一致）。原子累积顺序会留下很小的像素残差，因此不要求也不预期逐字节一致。

在已测范围内，这些证据没有显示光栅化器公式、公开接口合同或系统性覆盖方面的缺陷。特别是，冻结 Train 对比没有复现独立训练 checkpoint 之间 -0.2677 dB 的差值。可控模块检查还显示，Train 上的 KNN 初始化距离位级一致；CUDA fused SSIM 与 Warp SSIM 的公式和 padding 语义一致，但 FP32 梯度存在微小差异。因此，已观察到的 30K 差值符合非凸优化与 densification 对数值轨迹敏感的特征，而不是冻结渲染器带来的质量损失。可控数值与训练路径测量见 [SSIM 文档](ssim_zh.md)。

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

resolver 可以按 Warp 版本和公开能力门控可选实现。当前发行版未包含可选的高版本后端，因此自动选择使用 stable 后端。

---
