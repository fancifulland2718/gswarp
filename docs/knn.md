# KNN Module

[中文](knn_zh.md) · **English**

This document covers the algorithm, execution model, and correctness of the `gswarp.knn` module.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Implementation Details](#implementation-details)
- [Correctness](#correctness)

---

## Overview

`gswarp.knn` is a Warp-based drop-in replacement for the `distCUDA2()` function in [simple-knn](https://github.com/camenduru/simple-knn). It computes the mean squared distance from each 3D Gaussian to its 3 nearest neighbors, used to initialize Gaussian scales.

The API is fully compatible with simple-knn:

```python
from gswarp.knn import distCUDA2 as warp_distCUDA2
dist2 = warp_distCUDA2(means3D.cuda())  # drop-in replacement
```

In the reference 3DGS pipeline, `distCUDA2()` is used when Gaussian scales are initialized from the input point cloud. It is not part of the per-iteration rasterizer path; integrations that reinitialize scales may call it at additional lifecycle points.

---

## Algorithm

The algorithm matches the CUDA implementation in simple-knn:

1. **Morton encoding**: Compute the AABB of all Gaussian centers, normalize each point to [0, 1)³, and map to a 30-bit Morton code (3D Z-curve)
2. **Warp radix sort**: Sort Gaussians by Morton code, placing spatially adjacent Gaussians adjacent in index order
3. **SoA gather**: Reorder coordinates by sort result into Structure-of-Arrays layout (x[], y[], z[] separate arrays)
4. **Box AABB**: For groups of BOX_SIZE=1024 Gaussians, compute AABB (min/max xyz) of each box
5. **3-NN box-pruning search**: For each Gaussian, dynamically prune distant boxes based on the current k-NN distance bound, then compute exact Euclidean distances to remaining candidates
6. **Mean dist² output**: Return the mean of each Gaussian's 3 nearest-neighbor squared distances

This exactly matches the simple-knn CUDA algorithm without additional approximations.

---

## Implementation Details

The code lives in `gswarp/knn.py` and uses four Warp kernels plus Warp's radix-sort utility:

| Implementation | Responsibility |
|----------------|----------------|
| `_coord2morton_kernel` | Convert normalized coordinates to Morton codes |
| `wp.utils.radix_sort_pairs` | Sort Morton codes and original point indices |
| `_gather_sorted_soa_kernel` | Gather sorted coordinates into SoA arrays |
| `_box_min_max_kernel` | Compute one AABB per fixed-size point box |
| `_box_mean_dist_kernel` | Perform exact 3-NN search with box pruning |

The global scene AABB is reduced with PyTorch, then copied to host scalars for Morton normalization. The complete operation enters a call-scoped execution context, so PyTorch reductions, Warp sorting, and kernels are submitted in order on the active CUDA stream.

The public contract is explicit for small and invalid inputs: zero points return an empty tensor, one point returns zero, two or three points raise because three neighbours do not exist, and non-finite coordinates are rejected before Morton conversion.

---

## Correctness

On the current Train benchmark input, the initialization point cloud contains 182,686 points. Warp KNN and native simple-knn produced bitwise-identical squared distances for every point: maximum, mean, and relative differences were all zero.

Current contract tests also cover empty and singleton inputs and explicitly reject unsupported two- and three-point inputs. Warp KNN follows the same Morton-code, box partition, pruning, and nearest-neighbor semantics as simple-knn. Float32 concurrent box-bound updates can theoretically affect boundary ties, so the measured Train result is reported as a workload-specific result rather than a universal bytewise-identity guarantee.
