# KNN 模块

[English](knn.md) · **中文**

本文档记录 `gswarp.knn` 模块的算法、执行模型与正确性验证。

---

## 目录

- [概述](#概述)
- [算法](#算法)
- [实现细节](#实现细节)
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

## 正确性验证

在当前 Train benchmark 输入上，初始化点云包含 182,686 个点。Warp KNN 与原生 simple-knn 对每个点输出的平方距离均位级一致，最大差、平均差和相对差均为零。

当前合同测试还覆盖空输入和单点输入，并明确拒绝不支持的 2 点与 3 点输入。Warp KNN 与 simple-knn 使用相同的 Morton 码、盒划分、剪枝和近邻语义。float32 并发盒边界更新理论上可能影响边界 tie，因此 Train 的实测结果按特定工作负载结论报告，不作为所有输入逐字节一致的普遍承诺。
