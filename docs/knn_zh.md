# KNN 模块

[English](knn.md) · **中文**

本文档记录 `gswarp.knn` 模块的算法实现、性能数据与正确性验证。

---

## 目录

- [概述](#概述)
- [算法](#算法)
- [实现细节](#实现细节)
- [端到端训练影响](#端到端训练影响)
- [正确性验证](#正确性验证)

---

## 概述

`gswarp.knn` 是 [simple-knn](https://github.com/camenduru/simple-knn) 中 `distCUDA2()` 的 Warp 替换实现，计算每个 3D 高斯到其最近 3 个邻居的平均距离平方，用于初始化高斯尺度。

接口与 simple-knn 完全兼容：

```python
from gswarp.knn import distCUDA2 as warp_distCUDA2
dist2 = warp_distCUDA2(means3D.cuda())  # 直接替换
```

KNN 计算仅在高斯密度化阶段触发（每 ~100 次迭代一次），在整个 30K 迭代训练中的累计时间占比约 1–2%，对总训练时长影响有限。

---

## 算法

算法流程与 simple-knn 的 CUDA 实现保持一致：

1. **Morton 编码**：计算所有高斯中心的 AABB，将每个点归一化到 [0, 1)³ 范围内，映射为 30 位 Morton 码（3D Z 曲线）
2. **基数排序**：按 Morton 码对高斯排序，空间上相邻的高斯在索引上也连续
3. **SoA 聚合**：按排序结果重排坐标，生成 Structure-of-Arrays 布局（x[]、y[]、z[] 独立数组）
4. **盒 AABB**：以 BOX_SIZE=1024 个高斯为单位，计算每个盒的 AABB（min/max xyz）
5. **3-NN 盒剪枝搜索**：对每个高斯，根据当前 k-NN 距离上限动态剪枝远处的盒，再对剩余候选点精确计算 Euclidean 距离
6. **均值距离²输出**：返回每个高斯到其 3 个最近邻的距离平方的均值

这与 simple-knn 的算法完全一致，不引入额外近似。

---

## 实现细节

代码位于 `gswarp/knn.py`，共 4 个 Warp 内核：

| 内核 | Kernel Family | 寄存器 | 说明 |
|------|--------------|-------|------|
| `knn_morton` | FAMILY_COMPUTE | 32 | 计算 AABB + Morton 码 |
| `knn_gather` | FAMILY_MEMORY | 32 | 按排序索引重排点数据（SoA） |
| `knn_box_minmax` | FAMILY_ATOMIC | 32 | 计算各盒 AABB（atomic min/max） |
| `knn_box_dist` | FAMILY_COMPUTE | 64 | 盒剪枝 3-NN 搜索 |

Morton 码排序和逆排列索引由 PyTorch `torch.sort()` 完成（简化内核逻辑，避免在 Warp 中实现基数排序）。全局 AABB 归约同样使用 torch reduce 完成。

`knn_box_dist` 使用 64 个寄存器（较宽裕），是为了在盒剪枝内层循环中维持多个局部最小值而不溢出到 local memory。

---

## 端到端训练影响

> **基准条件**：以下消融数据在光栅化器固定为 Warp 后端、SSIM/KNN 各取两种组合的条件下采集，Python 层开销优化尚未应用。数据反映的是 KNN 后端对训练速度的独立贡献，而非最终全栈 Warp 的综合表现。

消融实验与 SSIM 消融相同条件（rasterizer=warp，30K 迭代）：

**drjohnson（~310 万高斯）**

| SSIM 后端 | KNN 后端 | 吞吐量 (it/s) | 壁钟时间 (s) | PSNR@30K |
|----------|---------|------------|------------|---------|
| cuda-fused | **cuda** | **30.0** | **853** | **29.504** |
| cuda-fused | **warp** | **29.9** | **879** | **29.465** |

drjohnson：Warp KNN 比 CUDA KNN 慢约 **0.3%**（30.0 → 29.9 it/s）。壁钟时间差异更大（+26 s），说明 Warp KNN 在高斯数量极多（310 万）时，盒剪枝阶段的 kernel launch 开销较 CUDA 高。

**playroom（~190 万高斯）**

| SSIM 后端 | KNN 后端 | 吞吐量 (it/s) | 壁钟时间 (s) | PSNR@30K |
|----------|---------|------------|------------|---------|
| cuda-fused | **cuda** | **45.0** | **619** | **30.458** |
| cuda-fused | **warp** | **46.3** | **620** | **30.326** |

playroom：Warp KNN 快约 **2.9%**（45.0 → 46.3 it/s），壁钟时间相近（+1 s）。推测是 playroom 场景的高斯空间分布与 Morton 曲线更匹配，盒剪枝效率更高。

PSNR 差异（~±0.1 dB）在训练随机性范围内，与 KNN 后端选择无关。

---

## 正确性验证

Warp KNN 与 simple-knn CUDA 实现使用相同算法（相同的 Morton 码、相同的 BOX_SIZE、相同的盒剪枝逻辑），在给定点云上的输出在浮点精度范围内一致。

唯一可能的差异来源：float32 的 atomic min/max 在并发更新时的写入顺序，理论上可能导致盒 AABB 有 ULP 级别的差异，进而影响距离排序的边界情况。实测中在随机点云（100K–3M 点）上未观测到任何差异。
