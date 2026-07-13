# SSIM 模块

[English](ssim.md) · **中文**

本文档记录 `gswarp.fused_ssim` 模块的实现细节、内核性能分析与正确性验证。

---

## 目录

- [概述](#概述)
- [算法](#算法)
- [实现细节](#实现细节)
- [Python 层开销与启动缓存](#python-层开销与启动缓存)
- [正确性验证](#正确性验证)
- [已知限制](#已知限制)

---

## 概述

`gswarp.fused_ssim` 是 [fused-ssim](https://github.com/rahul-goel/fused-ssim) 的 Warp 替换实现，计算可微的结构相似性损失 L_SSIM = 1 − SSIM(rendered, gt)，约等于 3DGS 损失函数中权重 0.2 的项。

接口与 `fused_ssim.fused_ssim()` 完全兼容：

```python
from gswarp.fused_ssim import fused_ssim
loss_ssim = warp_ssim(image, gt_image)  # 等价替换
```

---

## 算法

SSIM 基于高斯权重均值（μ）、方差（σ²）和互协方差（σ₁₂）：

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

高斯模糊（σ=1.5，window=11）作用于均值和方差计算。分离式卷积将二维 11×11 高斯分解为水平（H）和垂直（V）两个一维 1×11 卷积，共 4 个内核：

| 内核 | 操作 | 方向 |
|------|-----|------|
| `fwd_h` | 计算行向高斯均值 | → |
| `fwd_v` | 从行均值计算列向均值，得到 μ_x、μ_y、σ_xy 等 | ↓ |
| `bwd_h` | 反向传播到行方向 | → |
| `bwd_v` | 反向传播到列方向，输出梯度 | ↓ |

`fwd_v` 同时完成 SSIM 值计算和 loss 标量输出；`bwd_v` 同时完成 SSIM 梯度计算和链式法则传播。

---

## 实现细节

代码位于 `gswarp/fused_ssim.py`，应用了四个主要优化：

### 常数高斯权重展开

11 个高斯权重（σ=1.5）在编译期展开为常数数组，避免运行时从全局内存加载权重表。Warp 将常数直接嵌入 PTX 立即数。

### `_SSIMPlan` 启动缓存

训练路径使用有界 `_SSIMPlan` 池，key 包含设备、当前 PyTorch stream、shape、padding、常数和调优后的 block dimension。每个 plan 拥有自己的中间缓冲区和四个 recorded command。

每次 forward 获取一个私有 lease，并随 autograd context 保留到 graph 被释放。这样后续 forward 不会覆盖前一个 graph 的 workspace，同时支持多个 live graph、`retain_graph=True` 和非默认 stream。返回梯度每次独立分配，不进入复用池。

动态图像、上游梯度和输出指针通过调用级 ownerless descriptor 传入；recorded command 不会持有输入 autograd graph。

### 平坦数组布局

中间缓冲区（行均值、列均值等）使用一维平坦布局（`[C, H, W]`），而非结构化张量，便于内核直接指针偏移访问，减少 Warp 内核的索引计算。

### 基于生命周期的 workspace 别名

水平反向 workspace 与水平前向缓冲区的三个通道复用。前向值在同一 stream 上进入 backward 前已经失效，因此可以减少 plan 保留空间而不会覆盖仍存活的数据。

---

## Python 层开销与启动缓存

forward 与 backward 使用和 rasterizer 相同的调用级 PyTorch/Warp execution context。recorded command 保留静态 workspace 参数，每次调用只更新动态指针。空闲 plan 池按 key 和设备有界；淘汰或显式清理时只同步受影响的 Warp stream。

### 当前 RTX 5090 可控计时

在 RTX 5090（32 GiB）、Python 3.14.3、PyTorch 2.11.0+cu130、Warp 1.12.0 上，对已安装的包产物在 800x800 分辨率进行测量。该 benchmark 使用 30 次预热和 200 次重复，报告 CUDA event 中位数，并固定输入和归约语义。它是 loss 栈 microbenchmark，不是端到端训练吞吐声明。

| 操作 | CUDA fused SSIM | Warp SSIM | Warp/CUDA |
|------|----------------:|----------:|----------:|
| Forward | 0.166 ms | 0.143 ms | 0.86x |
| 仅 Backward | 0.193 ms | 0.240 ms | 1.24x |
| SSIM 训练路径 | 0.280 ms | 0.359 ms | 1.28x |
| L1 加 SSIM 训练路径 | 0.469 ms | 0.624 ms | 1.33x |

Warp forward 在该可控工作负载中更低，但 backward 路径更高。结果支持将 SSIM backward 和 loss 栈编排视为实质训练热点；它不支持把端到端场景结果单独归因于 SSIM。当前完整的 12 场景训练矩阵见[项目 README](../README_zh.md#基准测试)。

### 当前数值行为

CUDA fused SSIM 与 Warp SSIM 使用相同的 SSIM 公式、高斯系数和 padding 语义，但两者的 FP32 归约与 backward 执行顺序并非位级一致。下表比较相对于渲染图像的梯度；梯度相对 L1 定义为绝对梯度差之和除以 CUDA 梯度绝对值之和。

| 输入 | SSIM forward 绝对差 | 梯度最大绝对差 | 梯度相对 L1 |
|------|--------------------:|---------------:|-------------:|
| 随机 800x800 图像对 | 0.00 | 3.18e-12 | 2.04e-7 |
| Train 实际训练视图，545x980 | 1.79e-7 | 4.83e-9 | 1.45e-5 |

在该 Train 视图上，CUDA fused SSIM 为 0.9185836315，Warp SSIM 为 0.9185838103。这些是单次算子层面的微小差异，但在非凸优化器中反复应用后，可能改变后续参数更新和 densification 决策。

以下固定种子 Train 组件对照均完成 30K 训练，并使用原始 3DGS `render.py` 与 `metrics.py` 评估；所有最终 checkpoint 统一通过同一个原生 CUDA 光栅化器渲染：

| 训练光栅化器 | 训练 SSIM/KNN | PSNR | SSIM | LPIPS | 最终高斯数 |
|-------------|---------------|-----:|-----:|------:|-----------:|
| CUDA | CUDA / CUDA | 22.256477 | 0.821652 | 0.195832 | 1,094,613 |
| Warp | CUDA / CUDA | 22.123102 | 0.821906 | 0.194714 | 1,094,296 |
| CUDA | Warp / Warp | 21.958277 | 0.820587 | 0.196497 | 1,091,900 |
| Warp | Warp / Warp | 21.988764 | 0.820027 | 0.196647 | 1,089,398 |

Train 的实际初始化点云包含 182,686 个点，Warp KNN 与原生 `simple-knn` 的全部平方距离输出位级一致，因此 KNN 不能解释该对照中的辅助路径差异。该表说明的是：SSIM backward 的微小数值差异可能使优化过程进入不同轨迹。光栅化器与辅助路径的影响并非线性相加；这组单种子组件对照也不是对场景期望质量的统计估计。

这些结果没有显示 SSIM 公式或微分实现缺陷。它们支持一个更严格的有限结论：CUDA 与 Warp 的单次 forward/backward 高度一致，但不能要求两条 FP32 执行路径在 30K 后得到位级一致的 checkpoint。

---

## 正确性验证

当前包产物同时接受单次算子与训练路径检查。上方数值表在随机 800x800 输入和实际 545x980 Train 视图上对比 CUDA fused SSIM 与 Warp SSIM：forward 绝对差分别为 0.00 和 1.79e-7，梯度最大绝对差分别为 3.18e-12 和 4.83e-9。

回归测试还覆盖多个 live forward、retain_graph、非默认和并发 stream、有界 plan 池、缓存清理以及返回梯度所有权。CUDA 与 Warp 使用相同的 SSIM 公式、高斯系数和 padding 语义，但 FP32 执行顺序并非位级一致，因此正确性使用有界数值对照判断，而不是要求非凸训练后的 checkpoint 完全相同。

---

## 已知限制

- stable 实现保留四阶段可分离算法，因此需要物化与图像尺寸相关的中间缓冲区。
- 训练 plan 池有界，但每个存活的 autograd graph 都需要独占 lease，直到 graph 被释放。
- `padding="valid"` 要求两个空间维度均不小于 11。
- 输入必须是 shape 相同、设备相同的 CUDA `float32` tensor。
- 性能和显存行为取决于图像尺寸、并发 live graph 数、stream 数以及当前 Warp/PyTorch 版本；单设备比值不能作为可移植结论。
