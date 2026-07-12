# SSIM 模块

[English](ssim.md) · **中文**

本文档记录 `gswarp.fused_ssim` 模块的实现细节、内核性能分析与正确性验证。

---

## 目录

- [概述](#概述)
- [算法](#算法)
- [实现细节](#实现细节)
- [Python 层开销与启动缓存](#python-层开销与启动缓存)
- [内核性能分析](#内核性能分析)
- [端到端训练影响](#端到端训练影响)
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

### C0：常数高斯权重展开

11 个高斯权重（σ=1.5）在编译期展开为常数数组，避免运行时从全局内存加载权重表。Warp 将常数直接嵌入 PTX 立即数。

### C4：_SSIMPlan 启动缓存（最重要）

训练路径使用有界 `_SSIMPlan` 池，key 包含设备、当前 PyTorch stream、shape、padding、常数和调优后的 block dimension。每个 plan 拥有自己的中间缓冲区和四个 recorded command。

每次 forward 获取一个私有 lease，并随 autograd context 保留到 graph 被释放。这样后续 forward 不会覆盖前一个 graph 的 workspace，同时支持多个 live graph、`retain_graph=True` 和非默认 stream。返回梯度每次独立分配，不进入复用池。

动态图像、上游梯度和输出指针通过调用级 ownerless descriptor 传入；recorded command 不会持有输入 autograd graph。

### M1：平坦数组布局

中间缓冲区（行均值、列均值等）使用一维平坦布局（`[C, H, W]`），而非结构化张量，便于内核直接指针偏移访问，减少 Warp 内核的索引计算。

### O3：基于生命周期的 workspace 别名

水平反向 workspace 与水平前向缓冲区的三个通道复用。前向值在同一 stream 上进入 backward 前已经失效，因此可以减少 plan 保留空间而不会覆盖仍存活的数据。

---

## Python 层开销与启动缓存

forward 与 backward 使用和 rasterizer 相同的调用级 PyTorch/Warp execution context。recorded command 保留静态 workspace 参数，每次调用只更新动态指针。空闲 plan 池按 key 和设备有界；淘汰或显式清理时只同步受影响的 Warp stream。

这里不再声明当前设备上的计时；下方数值章节在新显卡上同时复测 CUDA fused-ssim 与 Warp SSIM 前均视为历史记录。

---

## 内核性能分析

> **历史 benchmark，等待复测。** 下列数值来自上一张显卡和较早实现快照。

**测试平台**：NVIDIA RTX 5090D V2（sm_120），PyTorch 2.11.0+cu130，Warp 1.12.0。

### 各内核 GPU 执行时长（drjohnson，分辨率 1332×876）

| 内核 | 操作 | 执行时长 | 占内核总时间 |
|------|-----|---------|------------|
| `fwd_h` | 水平高斯卷积 | 0.040 ms | 16% |
| `fwd_v` | 垂直卷积 + SSIM 计算 | 0.101 ms | 40% |
| `bwd_h` | 反向水平传播 | 0.036 ms | 14% |
| `bwd_v` | 反向垂直传播 | 0.073 ms | 29% |
| **合计（内核）** | | **0.250 ms** | 100% |
| **端到端（含 Python 开销）** | | **0.303 ms** | — |

`fwd_v` 是热点内核：V 向缓冲区（行均值中间结果）在 drjohnson 分辨率下约 **124 MB**，超过 L2 缓存容量（sm_120 为 64 MB），是带宽受限的主要来源。

### 与 CUDA fused-ssim 的对比

| 分辨率 | fused-ssim e2e | Warp SSIM e2e | 比值 |
|--------|--------------|--------------|------|
| 1080p (1920×1080) | 0.647 ms | 0.738 ms | 1.14× |
| drjohnson (1332×876) | — | 0.303 ms | — |

在 drjohnson 分辨率下，Warp 内核执行时间（0.25 ms）已快于 fused-ssim 的内核总时间（~0.35 ms），但 Python 层开销（0.053 ms）使端到端总耗时仍略高。在 1080p 分辨率下，V 向缓冲区溢出 L2 带来额外的 DRAM 访问，使比值升至 1.14×。

---

## 端到端训练影响

> **历史 benchmark，等待复测。** 在新硬件、同一训练配置下同时重跑 CUDA/Warp loss 后端之前，不应把这些数值作为当前声明。

> **基准条件**：以下消融数据在光栅化器固定为 Warp 后端、SSIM/KNN 各取两种组合的条件下采集，Python 层开销优化尚未应用。数据反映的是 SSIM 后端对训练速度的独立贡献，而非最终全栈 Warp 的综合表现。

消融实验在 drjohnson 和 playroom 两个场景、rasterizer 固定为 Warp 后端（降低光栅化器差异的干扰）的条件下进行，对比四种 SSIM/KNN 组合。

**drjohnson（1332×876，~310 万高斯，30K 迭代）**

| SSIM 后端 | KNN 后端 | 吞吐量 (it/s) | 壁钟时间 (s) | PSNR@30K |
|----------|---------|------------|------------|---------|
| cuda-fused | cuda | 30.0 | 853 | 29.504 |
| cuda-fused | warp | 29.9 | 879 | 29.465 |
| **warp** | cuda | **29.6** | **854** | **29.462** |
| warp | warp | 29.1 | 900 | 29.435 |

**playroom（1584×1008，~190 万高斯，30K 迭代）**

| SSIM 后端 | KNN 后端 | 吞吐量 (it/s) | 壁钟时间 (s) | PSNR@30K |
|----------|---------|------------|------------|---------|
| cuda-fused | cuda | 45.0 | 619 | 30.458 |
| cuda-fused | warp | 46.3 | 620 | 30.326 |
| **warp** | cuda | **45.0** | **622** | **30.420** |
| warp | warp | 46.7 | 578 | 30.332 |

**解读**：
- 在 drjohnson（大场景），Warp SSIM 比 cuda-fused 慢约 **1.3%**（30.0 → 29.6 it/s）
- 在 playroom，两者持平（45.0 → 45.0 it/s）
- PSNR 差异（~±0.05 dB）在训练随机性范围内，与 SSIM 后端选择无关
- Warp SSIM 的影响在大场景下更显著，因为此时 Adam 优化器的 GPU 计算耗时不变，SSIM 的相对权重更高

在完整的 bench30k 12 数据集比测中（三个模块均为 Warp），NeRF 合成场景平均快 ~8%，已经超过 SSIM 后端带来的 ~1% 额外开销，总体呈净加速。这是因为反向渲染的 warp shuffle 优化和紧凑 AABB 分箱的贡献超过了 SSIM 的损耗。

---

## 正确性验证

> **历史数值快照，等待复测。** 当前测试覆盖多个 live forward、`retain_graph`、非默认/双 stream、plan 池上限、缓存清理和返回梯度所有权；CUDA/PyTorch 数值表将在新显卡上重新生成。

**前向（损失值精度）**：

以 PyTorch 参考实现（`torchvision.transforms.functional` + `F.conv2d`）为基准：

| 数据集 | 分辨率 | L_SSIM 差值 |
|--------|--------|------------|
| drjohnson（随机帧） | 1332×876 | 0.00e+00（bit-exact） |
| playroom（随机帧） | 1584×1008 | 0.00e+00（bit-exact） |
| 800×800 合成场景 | 800×800 | 0.00e+00（bit-exact） |

前向在大多数测试条件下与 PyTorch 参考完全一致（位级精确），原因是高斯核系数为常数，Warp FP32 计算路径与 PyTorch FP32 Conv2d 相同。

**反向（梯度精度）**：

以 `torch.autograd.gradcheck` 有限差分为基准：

| 数据集 | 分辨率 | 梯度 L∞ 差值 |
|--------|--------|------------|
| drjohnson | 1332×876 | ~1e-11 |
| playroom | 1584×1008 | ~1e-11 |

梯度误差在数值分析的机器精度范围内，不会对训练收敛产生影响。

---

## 已知限制

- stable 实现保留四阶段可分离算法，因此需要物化与图像尺寸相关的中间缓冲区。
- 训练 plan 池有界，但每个存活的 autograd graph 都需要独占 lease，直到 graph 被释放。
- `padding="valid"` 要求两个空间维度均不小于 11。
- 输入必须是 shape 相同、设备相同的 CUDA `float32` tensor。
- 性能和显存行为取决于图像尺寸、并发 live graph 数、stream 数以及当前 Warp/PyTorch 版本；旧单设备比值不能作为可移植结论。
