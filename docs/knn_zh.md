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

在参考 3DGS 管线中，`distCUDA2()` 用于根据输入点云初始化 Gaussian scale，并不位于每次迭代的 rasterizer 热路径中。若集成方会重新初始化尺度，也可能在其他生命周期节点再次调用。

---

## 算法

算法流程与 simple-knn 的 CUDA 实现保持一致：

1. **Morton 编码**：计算所有高斯中心的 AABB，将每个点归一化到 [0, 1)³ 范围内，映射为 30 位 Morton 码（3D Z 曲线）
2. **Warp 基数排序**：按 Morton 码对高斯排序，空间上相邻的高斯在索引上也连续
3. **SoA 聚合**：按排序结果重排坐标，生成 Structure-of-Arrays 布局（x[]、y[]、z[] 独立数组）
4. **盒 AABB**：以 BOX_SIZE=1024 个高斯为单位，计算每个盒的 AABB（min/max xyz）
5. **3-NN 盒剪枝搜索**：对每个高斯，根据当前 k-NN 距离上限动态剪枝远处的盒，再对剩余候选点精确计算 Euclidean 距离
6. **均值距离²输出**：返回每个高斯到其 3 个最近邻的距离平方的均值

这与 simple-knn 的算法完全一致，不引入额外近似。

---

## 实现细节

代码位于 `gswarp/knn.py`，由四个 Warp kernel 和 Warp radix-sort utility 组成：

| 实现 | 职责 |
|------|------|
| `_coord2morton_kernel` | 将归一化坐标转换为 Morton 码 |
| `wp.utils.radix_sort_pairs` | 排序 Morton 码与原始点索引 |
| `_gather_sorted_soa_kernel` | 将排序后的坐标收集为 SoA 数组 |
| `_box_min_max_kernel` | 为每个固定大小的点盒计算 AABB |
| `_box_mean_dist_kernel` | 使用盒剪枝执行精确 3-NN 搜索 |

全局场景 AABB 由 PyTorch 归约，再复制为主机标量用于 Morton 归一化。完整操作进入调用级 execution context，因此 PyTorch 归约、Warp 排序和 kernel 会在当前 CUDA stream 上按序提交。

公开小样本与非法输入合同是明确的：0 个点返回空 tensor，1 个点返回 0；2 或 3 个点因不存在三个邻居而报错；非有限坐标会在 Morton 转换前被拒绝。

---

## 端到端训练影响

> **历史 benchmark，等待复测。** 下列消融使用上一张显卡和较早代码快照。CUDA simple-knn 与 Warp KNN 将在新显卡上一起重跑。

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

> **历史数值快照，等待复测。** 当前合同测试覆盖空输入、单点，以及 2/3 点拒绝行为；CUDA/Warp 随机点云与退化分布对照将在新显卡上重新生成。

Warp KNN 与 simple-knn CUDA 实现使用相同算法（相同的 Morton 码、相同的 BOX_SIZE、相同的盒剪枝逻辑），在给定点云上的输出在浮点精度范围内一致。

唯一可能的差异来源：float32 的 atomic min/max 在并发更新时的写入顺序，理论上可能导致盒 AABB 有 ULP 级别的差异，进而影响距离排序的边界情况。实测中在随机点云（100K–3M 点）上未观测到任何差异。
