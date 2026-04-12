# KNN Module

**中文** · [English](knn.md)

This document covers the algorithm, implementation, performance, and correctness of the `gswarp.knn` module.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Implementation Details](#implementation-details)
- [End-to-End Training Impact](#end-to-end-training-impact)
- [Correctness](#correctness)

---

## Overview

`gswarp.knn` is a Warp-based drop-in replacement for the `distCUDA2()` function in [simple-knn](https://github.com/camenduru/simple-knn). It computes the mean squared distance from each 3D Gaussian to its 3 nearest neighbors, used to initialize Gaussian scales.

The API is fully compatible with simple-knn:

```python
from gswarp.knn import distCUDA2 as warp_distCUDA2
dist2 = warp_distCUDA2(means3D.cuda())  # drop-in replacement
```

KNN is only invoked during Gaussian densification (approximately every ~100 iterations). Its cumulative contribution to total 30K training time is ~1–2%, making it a minor factor in overall performance.

---

## Algorithm

The algorithm matches the CUDA implementation in simple-knn:

1. **Morton encoding**: Compute the AABB of all Gaussian centers, normalize each point to [0, 1)³, and map to a 30-bit Morton code (3D Z-curve)
2. **Radix sort**: Sort Gaussians by Morton code, placing spatially adjacent Gaussians adjacent in index order
3. **SoA gather**: Reorder coordinates by sort result into Structure-of-Arrays layout (x[], y[], z[] separate arrays)
4. **Box AABB**: For groups of BOX_SIZE=1024 Gaussians, compute AABB (min/max xyz) of each box
5. **3-NN box-pruning search**: For each Gaussian, dynamically prune distant boxes based on the current k-NN distance bound, then compute exact Euclidean distances to remaining candidates
6. **Mean dist² output**: Return the mean of each Gaussian's 3 nearest-neighbor squared distances

This exactly matches the simple-knn CUDA algorithm without additional approximations.

---

## Implementation Details

The code lives in `gswarp/knn.py`, comprising 4 Warp kernels:

| Kernel | Kernel Family | Registers | Description |
|--------|--------------|-----------|-------------|
| `knn_morton` | FAMILY_COMPUTE | 32 | Compute scene AABB + Morton codes |
| `knn_gather` | FAMILY_MEMORY | 32 | Reorder points into SoA by sort order |
| `knn_box_minmax` | FAMILY_ATOMIC | 32 | Compute per-box AABB (atomic min/max) |
| `knn_box_dist` | FAMILY_COMPUTE | 64 | Box-pruning 3-NN search |

Morton code sorting and the inverse permutation index are handled by `torch.sort()` (simplifies kernel logic, avoids implementing radix sort in Warp). Global AABB reduction also uses torch reduce.

`knn_box_dist` uses 64 registers (generous) to maintain multiple running minimums in the box-pruning inner loop without spilling to local memory.

---

## End-to-End Training Impact

> **Benchmark conditions**: The ablation data below was collected with the rasterizer fixed to the Warp backend and SSIM/KNN each taking two values. Python-layer overhead optimizations had not yet been applied. The numbers reflect KNN's isolated contribution to training speed, not the final all-Warp stack performance.

Ablation runs under the same conditions as the SSIM ablation (rasterizer=warp, 30K iters):

**drjohnson (~3.1M Gaussians)**

| SSIM backend | KNN backend | Throughput (it/s) | Wall time (s) | PSNR@30K |
|-------------|------------|-------------------|--------------|---------|
| cuda-fused | **cuda** | **30.0** | **853** | **29.504** |
| cuda-fused | **warp** | **29.9** | **879** | **29.465** |

drjohnson: Warp KNN is ~**0.3% slower** than CUDA KNN (30.0 → 29.9 it/s). The wall-time gap is larger (+26 s), indicating that Warp KNN has higher kernel-launch overhead at very high Gaussian counts (3.1M).

**playroom (~1.9M Gaussians)**

| SSIM backend | KNN backend | Throughput (it/s) | Wall time (s) | PSNR@30K |
|-------------|------------|-------------------|--------------|---------|
| cuda-fused | **cuda** | **45.0** | **619** | **30.458** |
| cuda-fused | **warp** | **46.3** | **620** | **30.326** |

playroom: Warp KNN is ~**2.9% faster** (45.0 → 46.3 it/s), wall time nearly unchanged (+1 s). The playroom Gaussian distribution likely aligns better with Morton-curve locality, improving box-pruning efficiency.

PSNR differences (~±0.1 dB) are within training noise; not attributable to KNN backend choice.

---

## Correctness

Warp KNN and simple-knn CUDA use the same algorithm (same Morton codes, same BOX_SIZE, same box-pruning logic). Outputs agree within floating-point precision on the same point cloud.

The only potential source of difference: float32 `atomic_min/max` concurrent write ordering during box AABB computation can theoretically cause ULP-level differences in box boundaries, which could affect distance-sort tie breaks. No differences were observed in practice on random point clouds from 100K to 3M points.
