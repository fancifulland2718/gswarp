# Rasterizer Backend

**中文** · [English](rasterizer.md)

This document covers technical details, implementation differences, performance data, and correctness verification for the `gswarp` rasterizer backend. For everyday usage, see the [main README](../README.md).

---

## Table of Contents

- [Overview](#overview)
- [Architecture Overview](#architecture-overview)
- [Differences from the CUDA Baseline](#differences-from-the-cuda-baseline)
- [Python-Layer Overhead Optimization](#python-layer-overhead-optimization)
- [Correctness](#correctness)
- [Performance Characteristics (Early Benchmark)](#performance-characteristics-early-benchmark)
- [Updated Benchmark (bench30k_plots, All-Warp Stack)](#updated-benchmark-bench30k_plots-all-warp-stack)
- [Known Limitations](#known-limitations)
- [Future Optimization Directions](#future-optimization-directions)

---

## Overview

The `gswarp` rasterizer backend implements a differentiable Gaussian rasterization
pipeline equivalent to [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
in pure Python + NVIDIA Warp, comprising the following stages:

1. **Preprocessing**: Project 3D Gaussians to 2D, compute covariances, evaluate SH colors, compute AABB tile rects
2. **Binning**: Gaussian-to-tile mapping, depth sort, tile range identification
3. **Forward render**: block_dim=256 cooperative tile loading, front-to-back alpha compositing
4. **Backward render**: block_dim=32 (single warp), warp shuffle gradient reduction, 32× fewer atomics
5. **Backward preprocess**: Gradients for means3D / scales / rotations / SH

The entire pipeline runs on the same PyTorch CUDA stream (aligned via `_stream.py` `ensure_aligned()`), cooperating correctly with PyTorch autograd.

---

## Architecture Overview

### Pipeline Stages

```
Input Gaussians (means3D, SH, scales, rotations, opacities)
    │
    ▼
┌─────────────────────────────────┐
│  1. PREPROCESS                  │
│  - Cov3D from scale+rotation    │
│  - Project to 2D (cov2d, conic) │
│  - Frustum + near-plane culling │
│  - SH → RGB color evaluation    │
│  - Tight AABB tile rectangle    │
│  - Forward-state packing        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. BINNING                     │
│  - Tile-overlap counting (scan) │
│  - Gaussian→tile duplication    │
│  - Depth sort + tile sort       │
│  - Tile-range identification    │
└─────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  3. FORWARD RENDER                       │
│  - block_dim=256 cooperative tile load   │
│    (wp.tile + wp.tile_extract)           │
│  - Per-pixel alpha blending              │
│  - Front-to-back compositing             │
│  - Transmittance threshold termination   │
│  - Color, depth, alpha outputs           │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  4. BACKWARD RENDER                      │
│  - block_dim=32 warp-level grad reduce   │
│    (wp.tile_reduce → warp shuffle)       │
│  - Gradients w.r.t. conic,              │
│    opacity, color, and pos               │
│  - 32× fewer atomic_add writes           │
└──────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  5. BACKWARD PREPROCESS             │
│  - Gradients w.r.t. means3D, scales,│
│    and rotations                    │
│  - SH backward                      │
│  - Cov3D → scale/rotation gradients │
└─────────────────────────────────────┘
```

### Single-File Design

The entire Warp backend is contained in a single Python file (about 4100 lines), including:

- all Warp kernel definitions (`@wp.kernel`)
- all Warp helper functions (`@wp.func`)
- runtime auto-tuning
- public API functions

This is intentional — Warp's JIT compilation model requires all kernel code and dependencies to remain within the same `wp.Module` scope. Splitting the code across multiple files would break Warp's ability to resolve `@wp.func` cross-references during JIT.

### Key Constants

| Constant | Value | Description |
|------|---|------|
| `BLOCK_X` | 16 | Tile width in pixels |
| `BLOCK_Y` | 16 | Tile height in pixels |
| `NUM_CHANNELS` | 3 | RGB output channels |
| `RENDER_TILE_BATCH` | 32 | Number of Gaussians cooperatively loaded into shared memory per round in the forward kernel |
| `PREPROCESS_CULL_SIGMA` | 3.0 | Frustum-culling sigma multiplier |
| `PREPROCESS_CULL_FOV_SCALE` | 1.3 | FoV-boundary scale used for culling |
| `VISIBILITY_NEAR_PLANE` | 0.2 | Near-plane distance used for culling |

---

## Differences from the CUDA Baseline

### 1. Tight AABB vs Isotropic Radius

The tight AABB bounding-box method used in this project is inspired by *[Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction](https://arxiv.org/abs/2601.19489)* (Ziyu Zhang, Tianle Liu, Diantao Tu, Shuhan Shen, arXiv:2601.19489). This technique replaces the isotropic square radius with per-axis extents derived from the 2D covariance matrix, yielding tighter tile assignment for anisotropic Gaussians. This project introduces no extra code dependency — only the bounding-box computation logic is adopted.

**CUDA baseline** computes a square bounding box using the maximum of the two eigenvalue-derived radii:

```c
// CUDA: auxiliary.h getRect()
int max_radius = ...;  // max(ceil(3σ_max), 0), isotropic
rect_min = {min(grid.x, max((int)((point.x - max_radius) / BLOCK_X), 0)), ...};
rect_max = {min(grid.x, max((int)((point.x + max_radius + BLOCK_X - 1) / BLOCK_X), 0)), ...};
```

**Warp backend** computes a per-axis tight bounding box using the diagonal elements of the 2D covariance matrix:

```python
# Warp: _compute_tile_rect_tight_wp()
radius_x = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(cov_xx, 0.01))))
radius_y = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(cov_yy, 0.01))))
```

**Impact**:
- For **elongated Gaussians** (high anisotropy), the Warp backend assigns fewer tiles, reduces `num_rendered`, and improves binning/render efficiency.
- For **circular Gaussians**, the two approaches are equivalent.
- This introduces a small mismatch: some boundary tiles included by the CUDA baseline due to its overly conservative isotropic radius are excluded by Warp's tighter bounds. The visual difference is negligible, but measurable in numerical comparisons.

### 2. Partial Cooperative Tile Loading (via Warp Tile API)

**CUDA baseline** uses explicit `__shared__` memory for cooperative data fetching — all 256 threads in a tile collaboratively load Gaussian data from global memory into shared memory, then iterate over the shared buffer:

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

The **Warp backend** cannot directly declare or manipulate `__shared__` variables, but achieves cooperative loading indirectly through the Warp Tile API (`wp.tile()` + `wp.tile_extract()`). The forward render kernel uses block_dim=256 (one 16×16 pixel tile per block). In each iteration, each thread loads 1 Gaussian, then all threads read other threads' data via `wp.tile_extract()`:

```python
# Warp: _render_tiles_tiled256_warp_kernel
# Each thread loads 1 Gaussian (cooperative)
my_xy = points_xy_image[my_id]
my_co = conic_opacity[my_id]
t_xy = wp.tile(my_xy, preserve_type=True)
t_co = wp.tile(my_co, preserve_type=True)

# All 256 threads share this batch of data
for j in range(batch_count):
    xy_j = wp.tile_extract(t_xy, j)
    co_j = wp.tile_extract(t_co, j)
    ...
```

This is functionally equivalent to CUDA's shared-memory cooperative fetch — global memory reads are amortized across all threads in the tile. However, the Warp Tile API has certain constraints:
- `wp.tile()` always creates block-level tiles — cannot create warp-level tiles
- Each `wp.tile()` call implicitly involves `__syncthreads`
- No direct control over shared memory layout or alignment

The **backward render kernel** uses block_dim=32 (single warp), employing `wp.tile_reduce()` for gradient reduction. In the single-warp configuration, `tile_reduce` compiles to pure warp shuffles (`__shfl_down_sync`), requiring no `__syncthreads` or shared memory — the optimal configuration for backward gradient reduction under the Warp API.

**Impact**:
- The forward render is now close to CUDA baseline efficiency at all scales — cooperative loading eliminates the previous 256× redundant global memory reads per Gaussian.
- The backward render is also efficient — warp-level reduction reduces atomic writes by 32×.
- However, the Warp Tile API is still less flexible than direct `__shared__` memory manipulation — for example, complex double buffering or custom bank-conflict avoidance strategies cannot be implemented.

### 3. Sorting Differences

**CUDA baseline** uses a single CUB `DeviceRadixSort` pass with a packed 64-bit key (`(tile_id << 32) | depth_bits`).

**Warp backend** default mode (`warp_depth_stable_tile`) uses two sorts:
1. First pass: sort by depth (Warp radix sort)
2. Second pass: stable sort by tile ID (Warp radix sort)

This leads to different Gaussian ordering within each tile compared with the CUDA baseline, which in turn changes the floating-point accumulation order during alpha blending. Because floating-point arithmetic is non-associative, the pixel-level outputs differ slightly.

### 4. Culling Parameters

The Warp backend applies explicit frustum-culling parameters:
- `PREPROCESS_CULL_SIGMA = 3.0`: Gaussians whose 3σ bounding boxes lie fully outside the image are culled.
- `PREPROCESS_CULL_FOV_SCALE = 1.3`: Slightly enlarges the FoV to avoid over-aggressive boundary culling.

This differs slightly from the CUDA baseline's implicit culling behavior.

### 5. Dispatch Differences

The CUDA baseline dispatches the render kernel on a 2D grid of tile blocks:

```c
dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
dim3 block(BLOCK_X, BLOCK_Y, 1);
```

The Warp backend dispatches in 1D, but also on a per-tile basis — each tile corresponds to 256 threads (16×16 pixels):

```python
# Forward: block_dim=256, each block = one tile
_dim = num_tiles * 256
wp.launch(kernel, dim=_dim, block_dim=256, ...)
```

The tile index is derived inside the kernel using arithmetic on `wp.tid()` (`tile_id = tid // 256`, `local_id = tid % 256`). The functionality is equivalent, but the 1D dispatch thread mapping is slightly different.

The backward render kernel uses block_dim=32 (single warp), where each warp covers the same 16×16 tile but each thread handles approximately 8 pixels within the tile.

---

## Python-Layer Overhead Optimization

Warp's Python layer (type checking, kernel parameter packing, torch→wp.array conversion)
introduces measurable Python function-call overhead on the per-iteration hot path.
The following optimizations have been applied to `_rasterizer.py`:

**Key measures:**
- `GaussianRasterizationSettings` uses `dataclasses.dataclass` instead of a plain class,
  reducing field access from dict lookup to attribute lookup
- The `device` object is cached on first call, avoiding repeated `tensor.device` queries
- Parameter validation and format conversion are moved from the call hot-path into `_prep()`,
  reducing conditional branches

**Measured effect (RTX 5090D V2):**

| Scale | Before optimization | After optimization | Reduction |
|-------|--------------------|--------------------|-----------|
| 256K | 0.786 ms | 0.760 ms | -3.2% |
| 512K | 0.906 ms | 0.868 ms | -4.2% |
| 1024K | 1.090 ms | 1.061 ms | -2.7% |
| 2048K | 1.622 ms | 1.580 ms | -2.6% |

*Public API end-to-end (forward + backward) steady-state latency; bit-exact output unchanged.*

Real-world 30K training effect varies by scene: small scenes (lego, ~300K Gaussians) see ~2.5%
improvement; large scenes (truck, drjohnson, 2M+ Gaussians) see <0.1% — GPU compute dominates
and Python overhead is negligible.

---

## Correctness

### Random Particle Correctness (256K–2048K)

Numerical consistency between Native CUDA and Warp backends was verified at 256K–2048K random particle scales using the public API. Test configuration: `backward_mode="manual"`, `binning_sort_mode="warp_depth_stable_tile"`, `auto_tune=True`. Test platform: **NVIDIA RTX 5090D V2**, PyTorch 2.11.0+cu130, Warp 1.12.0.

#### Forward: Rendered Color Difference

| Scale | Resolution | Max abs error | Mean abs error |
|------|--------|------------|------------|
| 256K | 384×384 | 0.0108 | 4.01e-05 |
| 512K | 512×512 | 0.0064 | 3.37e-05 |
| 1024K | 640×640 | 0.0077 | 2.78e-05 |
| 2048K | 800×800 | 0.0033 | 1.74e-05 |

Max single-pixel error < 0.011 (value range [0,1]), with mean absolute error on the order of 1e-5. Error slightly decreases with increasing scale, indicating the differences arise from sparse outlier pixels rather than systematic bias. The two backends use different tile sorting implementations and floating-point accumulation orders, producing numerically small differences.

#### Backward: Gradient Difference

| Scale | Gradient field | Max abs error | Mean abs error |
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

The large max absolute errors in backward gradients require context:
- Gradient values span a very wide range (e.g. `grad_scales` can reach ±10⁴); max absolute errors occur only at a few extreme gradient points.
- **Mean absolute errors** are tiny (`grad_means3D` < 0.0012, `grad_shs` < 7e-05), indicating the vast majority of per-point gradients are highly consistent.
- As scale increases from 256K → 2048K, mean absolute errors consistently decrease (e.g. `grad_means3D` drops from 0.00114 to 0.00030), confirming differences arise from sparse outliers rather than systematic bias.
- Differences stem from different tile sorting orders causing floating-point rounding divergence in alpha blending accumulation, plus non-deterministic ordering of atomic gradient writes.

### End-to-End Training Quality (12 Datasets × 30K Iterations)

> **Benchmark conditions**: The end-to-end training quality data below was collected in an earlier benchmark run: only the rasterizer used the Warp backend; SSIM used cuda-fused; KNN used CUDA simple-knn; Python-layer overhead optimizations had not yet been applied. For all-three-module results, see [Updated Benchmark](#updated-benchmark-bench30k_plots-all-warp-stack) below.

The following data is based on the **default training parameters** from the original 3DGS repository (30,000 iterations), tested on 12 standard datasets. The test platform is **NVIDIA RTX 5090D V2** (24 GiB), PyTorch 2.11.0+cu130, Warp 1.12.0.

**NeRF Synthetic (800×800):**

| Dataset | | PSNR (dB) | SSIM | LPIPS |
|---------|---|-----------|------|-------|
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

**Tanks & Temples / Deep Blending (native resolution per scene):**

| Dataset | | PSNR (dB) | SSIM | LPIPS |
|---------|---|-----------|------|-------|
| train | Warp | 22.101 | 0.8183 | 0.1982 |
| | CUDA | 22.060 | 0.8213 | 0.1962 |
| truck | Warp | 25.372 | 0.8828 | 0.1445 |
| | CUDA | 25.479 | 0.8850 | 0.1420 |
| drjohnson | Warp | 29.383 | 0.9043 | 0.2372 |
| | CUDA | 29.455 | 0.9053 | 0.2357 |
| playroom | Warp | 30.150 | 0.9076 | 0.2414 |
| | CUDA | 30.072 | 0.9091 | 0.2399 |

In most scenes the PSNR gap between Warp and CUDA is within **-0.07 to -0.40 dB**, a negligible difference. Two scenes (train, playroom) show Warp PSNR slightly higher than CUDA. SSIM and LPIPS metrics are similarly close. This difference may come from warp-backend's lower convergence rate.

---

## Performance Characteristics (Early Benchmark)


> **Benchmark conditions**: Data in this section was collected in an earlier benchmark run: only the rasterizer used the Warp backend; SSIM used cuda-fused (not Warp SSIM); KNN used CUDA simple-knn (not Warp KNN); Python-layer overhead optimizations had not yet been applied. For all-three-module Warp results, see [Updated Benchmark](#updated-benchmark-bench30k_plots-all-warp-stack) below.

### End-to-End Training Performance (12 Datasets × 30K Iterations)

The following data was tested on **NVIDIA RTX 5090D V2** (24 GiB, sm_120), **PyTorch 2.11.0+cu130**, **Warp 1.12.0**, using the default training parameters from the original 3DGS repository.

#### Training Speed and Total Training Time

| Dataset | | Avg FPS (30K) | Total Time (s) | Peak Mem (MB) |
|---------|---|-------:|--------:|--------:|
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

**Key findings:**
- NeRF Synthetic (small scenes, ~150K–350K Gaussians): CUDA is generally faster. Among the three Warp sort backends, **Stable Tile** is the fastest overall, followed by Torch Sort and Radix.
- Tanks & Temples / Deep Blending (large scenes, ~1M–3.1M Gaussians): **All three Warp backends clearly outperform CUDA** — e.g. drjohnson: Stable Tile 1.56×, Radix 1.49×, Torch Sort 1.50×; playroom: Stable Tile 1.68×, Radix 1.61×, Torch Sort 1.56×.
- The three Warp sort backends (Stable Tile, Radix, Torch Sort) show very similar performance: the maximum speed difference among them is typically within 10–15%. **Stable Tile (default) is recommended** as it delivers the best overall throughput.
- Memory usage: all backends are at the same order of magnitude; Warp variants have lower peak memory on most NeRF Synthetic scenes.

#### Training Curves

The following figure shows the iteration speed (FPS), total loss (log), peak GPU memory, and Gaussian count over 30K training iterations for all 12 datasets across CUDA (blue), Stable Tile (red), Radix (green), and Torch Sort (purple):

![Sort Backend Training Overview](../figures/sort_backend_overview.png)

### Micro-Benchmarks (Random Particles)

The following data comes from random-particle tests. The test platform is **NVIDIA GeForce RTX 5090D** (sm_120, 24 GiB, 170 SMs), **Warp 1.12.0**, and **PyTorch 2.11.0+cu130**.

Methodology:

- **Steady-state runtime**: measured via the public API (`diff_gaussian_rasterization.GaussianRasterizer` and `diff_gaussian_rasterization.warp.GaussianRasterizer`) with dedicated warmup runs first, then a batched CUDA-event timing pass over the measured iterations; the main table reports the mean over the measured runs.
- **Peak memory**: after warmup, `reset_peak_memory_stats` is called before running a full stage, and the absolute peak `max_memory_allocated` is recorded. Forward peak includes forward only; backward peak includes the full forward+backward flow.
- **Stage timing / stage memory**: used only for hotspot analysis, measured diagnostically with internal `_warp_backend` helper functions; these stage-wise values are **not guaranteed** to sum strictly to public-API end-to-end time or peak memory item by item.

For the 256K / 512K / 1024K / 2048K cases, the evaluation uses **4+8 / 3+6 / 3+6 / 2+4** (warmup count + measured count), respectively.

### Public API Steady-State Runtime

| Points | Resolution | `num_rendered` | Native FW | Warp FW | FW ratio | Native BW | Warp BW | BW ratio |
|------|--------|----------------|----------|-----------|----------|----------|-----------|----------|
| 262,144 | 384×384 | 265,106 | 0.315 ms | 0.999 ms | 3.17× | 1.772 ms | 1.012 ms | 0.57× |
| 524,288 | 512×512 | 874,049 | 0.461 ms | 1.101 ms | 2.39× | 3.252 ms | 1.632 ms | 0.50× |
| 1,048,576 | 640×640 | 2,556,549 | 0.857 ms | 1.594 ms | 1.86× | 5.363 ms | 2.464 ms | 0.46× |
| 2,097,152 | 800×800 | 7,644,361 | 2.695 ms | 2.697 ms | 1.00× | 8.839 ms | 4.418 ms | 0.50× |

At the 256K–2048K scale, Warp forward is still slower than native (1.00×–3.17×) due to Python-level orchestration overhead (tensor allocation, kernel launch, inter-stage data passing), but the gap narrows rapidly as particle count grows and GPU compute dominates—at 2048K the forward ratio reaches **1.00×** (parity with native). **On the backward side, Warp is faster than native at all scales** (ratio 0.46×–0.57×), thanks to the computational efficiency of the `_backward_render_tiles_warp32` kernel's warp shuffle (block_dim=32) fast path at large `num_rendered` volumes.

### Public API Peak Memory

| Points | Resolution | Native FW peak | Native BW peak | Warp FW peak | Warp BW peak |
|------|--------|------------|------------|-------------|-------------|
| 262,144 | 384×384 | 128.61 MiB | 205.67 MiB | 145.63 MiB | 336.08 MiB |
| 524,288 | 512×512 | 274.00 MiB | 427.51 MiB | 285.69 MiB | 662.50 MiB |
| 1,048,576 | 640×640 | 587.75 MiB | 891.75 MiB | 570.97 MiB | 1324.02 MiB |
| 2,097,152 | 800×800 | 1300.70 MiB | 1910.72 MiB | 1149.20 MiB | 2645.05 MiB |

> **Note**: These are absolute peak values (`max_memory_allocated`), including input tensors, model parameters, forward intermediates, and autograd saved tensors. Backward peak includes the full forward+backward flow.

Warp forward peak is about 1.13× higher than native at 256K, but from 1024K onward, Warp forward peak (571 MiB) is actually **lower** than native (588 MiB); at 2048K it is 0.88× of native. Backward peak for Warp is about 1.4×–1.6× of native (2645 vs 1911 MiB at 2048K), mainly due to Warp's additional intermediate tensors (depth, alpha, projected coordinates, per-pixel weights, etc.).

### Internal Stage Hotspots

| Points | Resolution | Preprocess | Binning | Render | Backward Render | Selected Sort Mode |
|------|--------|--------|------|------|----------|--------------|
| 262,144 | 384×384 | 0.335 ms | 0.495 ms | 0.221 ms | 0.386 ms | `warp_depth_stable_tile` |
| 524,288 | 512×512 | 0.390 ms | 0.492 ms | 0.248 ms | 0.430 ms | `warp_depth_stable_tile` |
| 1,048,576 | 640×640 | 0.580 ms | 0.583 ms | 0.249 ms | 0.561 ms | `warp_depth_stable_tile` |
| 2,097,152 | 800×800 | 0.944 ms | 1.275 ms | 0.323 ms | 0.790 ms | `warp_depth_stable_tile` |

### Internal Stage Cumulative Peak Memory (diagnostic only)

| Points | Resolution | After preprocess | After preprocess+binning | After full forward | After forward+backward |
|------|--------|-------------|-----------------|-------------|-------------------|
| 262,144 | 384×384 | 134.76 MiB | 137.76 MiB | 138.76 MiB | 136.07 MiB |
| 524,288 | 512×512 | 264.67 MiB | 271.01 MiB | 270.83 MiB | 266.01 MiB |
| 1,048,576 | 640×640 | 529.08 MiB | 541.08 MiB | 540.08 MiB | 526.71 MiB |
| 2,097,152 | 800×800 | 1053.43 MiB | 1076.07 MiB | 1076.07 MiB | 1052.18 MiB |

> **Note**: Each column is the absolute peak from `empty_cache()` through completion of that stage. Due to temporary tensor release and reuse between stages, "after forward+backward" may be slightly lower than "after full forward".

Looking at the internal stages, preprocess and binning grow near-linearly with particle count and are the scalability bottleneck; render and backward render remain under 1 ms even at 2048K, indicating good pure-GPU-compute efficiency in the Warp tile kernels. Preprocess is the main memory consumer (1053 MiB peak after preprocess at 2048K); binning adds only about 23 MiB on top (at 2048K), and render adds virtually no extra peak.

### Kernel-Level Profiling (Nsight Systems)

> **Note**: This GPU SKU (RTX 5090D V2) does not support Nsight Compute hardware performance counter collection (`ERR_NVGPU`). The following data was captured via **Nsight Systems 2025.6** timeline tracing at **256K@384×384** and **1024K@640×640** scales. Each run: 5 warmup + 3 NVTX-annotated profiled iterations (forward+backward), sort mode `warp_depth_stable_tile`.

#### Per-Iteration End-to-End Breakdown (NVTX)

| Scale | Forward (NVTX) | Backward (NVTX) | Iteration total |
|------|------------|------------|-----------|
| 256K@384×384 | 1.42 ms | 1.37 ms | 2.79 ms |
| 1024K@640×640 | 2.84 ms | 1.36 ms | 4.20 ms |

#### Kernel Hotspots by GPU Time

The table below lists per-iteration average GPU time for each kernel across the full pipeline (forward + backward) at 256K and 1024K scales, based on nsys timeline statistics (per-instance mean over 8 iterations).

| Kernel | Calls/iter | 256K avg time | 1024K avg time | Pipeline stage |
|--------|---------|-------------|--------------|----------|
| `_backward_render_tiles_warp32` | 1 | 230.0 µs | 423.2 µs | Backward Render |
| `_render_tiles_fast_warp` | 1 | 93.6 µs | 127.7 µs | Forward Render |
| `_backward_rgb_from_sh_v3` | 1 | 59.4 µs | 408.5 µs | Backward Preprocess |
| `_forward_rgb_from_sh_v3` | 1 | 48.6 µs | 227.9 µs | Forward Preprocess (SH→RGB) |
| CUB `DeviceRadixSort` (Onesweep) | 8 | 7.4 µs × 8 | 12.5 µs × 8 | Binning (sort) |
| `_duplicate_with_keys_from_order` | 1 | 34.7 µs | 129.9 µs | Binning (overlap expansion) |
| `_fused_project_cov3d_cov2d_preprocess_sr` | 1 | 25.2 µs | 131.2 µs | Forward Preprocess (proj+cov) |
| `_fused_backward_preprocess_accumulate` | 1 | 24.2 µs | 102.3 µs | Backward Preprocess |
| PyTorch `elementwise_copy` | ~3 | 41.5 µs × 3 | 366.5 µs × 3 | PyTorch tensor copy |
| PyTorch `fill` / `zero_` | ~10 | 3.6 µs × 10 | 18.8 µs × 10 | PyTorch initialization |
| `_identify_tile_ranges` | 1 | 1.4 µs | 6.9 µs | Binning |
| `_gather_i32_by_index` | 1 | 1.7 µs | 5.9 µs | Binning |

#### CPU vs GPU Overhead Breakdown

| Metric | 256K@384×384 | 1024K@640×640 |
|------|-------------|--------------|
| Wall-clock per iteration | 2.79 ms | 4.20 ms |
| Warp kernel GPU time | 578 µs (20.7%) | 1,664 µs (39.6%) |
| PyTorch kernel GPU time | 186 µs (6.7%) | 1,383 µs (32.9%) |
| **GPU total** | **764 µs (27.4%)** | **3,047 µs (72.5%)** |
| **CPU overhead (remainder)** | **2,026 µs (72.6%)** | **1,153 µs (27.5%)** |

CPU overhead breakdown (from nsys CUDA API Summary):

| CUDA API call | Calls per iter (approx.) | Median latency | Per-iter total |
|---------------|---------------------|---------|-----------|
| `cudaLaunchKernel` (PyTorch side) | ~38 | 5.8 µs | ~220 µs |
| `cuLaunchKernel` (Warp side) | ~9 | 15.4 µs | ~139 µs |
| `cudaMemsetAsync` | ~15 | 4.6 µs | ~69 µs |
| `cudaMemcpyAsync` | ~4 | 18.8 µs | ~75 µs |
| `cudaMallocAsync` | ~5 | 4.3 µs | ~22 µs |
| `cudaFreeAsync` | ~6 | 2.3 µs | ~14 µs |
| **CUDA API total** | | | **~539 µs** |
| **Python/Warp runtime (tensor creation, attribute lookup, dispatch)** | | | **~1,487 µs** |

#### Key Analysis

1. **CPU overhead is approximately constant**: whether at 256K or 1024K, CPU-side overhead is about 1.2–2.0 ms/iteration. At 256K, CPU accounts for 73%; at 1024K it drops to 28%. At 2048K, GPU compute grows further, reducing the CPU ratio even more—explaining why the 2048K forward ratio reaches 1.00×.
2. **Backward render is the largest GPU hotspot**: `_backward_render_tiles_warp32` (block_dim=32, warp shuffle fast path) takes 230 µs at 256K and 423 µs at 1024K, accounting for 25%–40% of Warp kernel GPU time.
3. **SH color computation scales linearly with N**: `_forward_rgb_from_sh_v3` + `_backward_rgb_from_sh_v3` total 636 µs at 1024K, making them the second largest hotspot after backward render. At SH degree=3, each point requires 16 coefficients × 3 channels of read/write bandwidth.
4. **PyTorch `elementwise_copy` grows sharply with data size**: 42 µs at 256K, 367 µs at 1024K per call. These are PyTorch autograd tensor copy/type-conversion operations (not Warp kernels), accounting for 33% of GPU time at 1024K.
5. **Actual CUDA API calls account for only ~26% of CPU overhead** (539 µs / 2026 µs at 256K). The remaining ~74% is pure-CPU work under the Python GIL: object creation, Warp runtime dispatch, PyTorch autograd graph construction. This is an inherent cost of the Warp-on-Python architecture, only mitigable via CUDA Graphs or reducing the number of pipeline stages.

---

## Updated Benchmark (bench30k\_plots, All-Warp Stack)

> **Benchmark conditions**: Rasterizer, SSIM, and KNN all use Warp backends; Python-layer
> overhead optimizations applied; 12 datasets × 30K iterations. See [SSIM docs](ssim.md#end-to-end-training-impact)
> and [KNN docs](knn.md#end-to-end-training-impact) for per-module contributions.Attention, the result differs a lot from all-warp version as different test scripts were used.

**Platform**: NVIDIA RTX 5090D V2 (24 GiB, sm_120), PyTorch 2.11.0+cu130, Warp 1.12.0.
Default 3DGS training hyperparameters.

### Training Throughput and Total Time

| Dataset | Scene type | CUDA (it/s) | Warp (it/s) | Speedup | CUDA time | Warp time | Final Gaussians |
|---------|-----------|------------|------------|---------|-----------|-----------|----------------|
| chair | NeRF Synthetic | 103.6 | 113.1 | ×1.09 | 4.7 min | 4.3 min | ~300K |
| drums | NeRF Synthetic | 103.0 | 115.3 | ×1.12 | 4.6 min | 4.2 min | ~330K |
| ficus | NeRF Synthetic | 139.5 | 148.2 | ×1.06 | 3.5 min | 3.3 min | ~190K |
| hotdog | NeRF Synthetic | 144.5 | 156.5 | ×1.08 | 3.6 min | 3.2 min | ~170K |
| lego | NeRF Synthetic | 117.5 | 126.5 | ×1.08 | 4.2 min | 3.9 min | ~300K |
| materials | NeRF Synthetic | 134.0 | 144.9 | ×1.08 | 3.7 min | 3.4 min | ~240K |
| mic | NeRF Synthetic | 95.4 | 105.1 | ×1.10 | 4.9 min | 4.5 min | ~300K |
| ship | NeRF Synthetic | 107.0 | 113.2 | ×1.06 | 4.5 min | 4.2 min | ~350K |
| train | Tanks&Temples | 55.6 | 58.3 | ×1.05 | 8.7 min | 8.2 min | ~1.1M |
| truck | Tanks&Temples | 39.4 | 40.1 | ×1.02 | 11.7 min | 11.3 min | ~2.1M |
| drjohnson | Deep Blending | 30.8 | 32.0 | ×1.04 | 14.3 min | 13.8 min | ~3.1M |
| playroom | Deep Blending | 46.9 | 47.5 | ×1.01 | 9.9 min | 9.7 min | ~1.9M |

**Summary**:
- NeRF Synthetic (small scenes, 150K–350K Gaussians): Warp averages **~8% faster**, driven by backward-render warp shuffle reduction and compact AABB binning.
- Large scenes (1M–3.1M Gaussians): speedup narrows to 1–5%; Adam optimizer becomes the bottleneck and Warp's render/backward gains contribute proportionally less.

### Per-Phase Timing Breakdown (last 5K iter average, ms)

| Phase | chair CUDA | chair Warp | drjohnson CUDA | drjohnson Warp |
|-------|-----------|-----------|---------------|---------------|
| render | 1.85 | 1.63 | 6.11 | 5.79 |
| loss (SSIM+L1) | 0.29 | 0.35 | 0.31 | 0.38 |
| backward | 4.88 | 4.27 | 11.21 | 10.60 |
| densify | 0.48 | 0.44 | 1.48 | 1.44 |
| optim | 0.98 | 0.98 | 8.24 | 8.14 |

Render and backward are the phases with the largest Warp advantage. The optimizer (Adam) is
controlled by PyTorch and is near-identical for both backends. The loss phase is slightly slower
with Warp SSIM at some resolutions — see [SSIM docs](ssim.md) for details.

---

## Known Limitations

### 1. Warp Tile API Flexibility Limitations

NVIDIA Warp provides shared memory and warp-level operations indirectly through the Tile API (`wp.tile()`, `wp.tile_extract()`, `wp.tile_reduce()`), but compared to directly operating CUDA `__shared__` memory, there are the following limitations:
- **`wp.tile()` always creates block-level tiles** — it is not possible to create warp-level or other granularity sub-block tiles.
- **Each `wp.tile()` call implicitly triggers `__syncthreads`** — creating tiles multiple times in an inner loop causes a barrier storm (verified experimentally: the tiled-256 backward kernel triggers 12 `__syncthreads` per Gaussian, resulting in a 2–3× performance regression).
- **No control over shared memory layout** — alignment, padding, and bank conflict avoidance cannot be manually optimized.
- **`tile_reduce` uses shared memory + `__syncthreads` when block_dim > 32** — only when block_dim=32 (single warp) does it take the pure warp shuffle fast path.

Current solution: forward uses block_dim=256 + `wp.tile()`/`wp.tile_extract()` for cooperative loading; backward uses block_dim=32 to ensure `tile_reduce` takes the warp shuffle fast path.

### 2. `tile_atomic_add` Only Supports Scalars

`wp.tile_atomic_add` only supports a few scalar operands such as `float32`. Vector- and matrix-level atomic reductions must be decomposed into separate scalar atomic operations. This has been verified experimentally — the tiled-256 + `tile_reduce` + `tile_atomic_add` backward kernel is **2–3× slower** than the warp32 version, requiring 10 scalar atomic operations plus 12 `__syncthreads` barriers per Gaussian.

### 3. Compile-Time Tile Shape

`BLOCK_X` and `BLOCK_Y` are defined as `wp.constant()` values (16×16). Changing the tile shape requires modifying the source code and triggering Warp module recompilation. The CUDA baseline also uses fixed tile sizes, but Warp's JIT model makes this limitation more obvious because the constants are baked into the kernel at compile time.

### 4. First-Run JIT Compilation Overhead

The first call to any Warp kernel triggers JIT compilation of the whole module. On a typical system, this takes several seconds (depending on the number of kernels and the GPU). Subsequent calls use the Warp kernel cache and complete almost instantly.

### 5. Backward Non-Determinism

The backward render kernel uses `wp.atomic_add` to accumulate gradients. When multiple threads write to the same address, this is inherently non-deterministic. This means:
- two runs with identical inputs may produce slightly different gradient values
- the difference is usually within FP32 precision (most gradients < 1e-4)
- this is consistent with the CUDA baseline (`atomicAdd` is also non-deterministic)

### 6. Python-Level Orchestration Overhead

Unlike the CUDA baseline, where the full pipeline is orchestrated in C++ with very little Python interaction, the Warp backend uses Python to:

- allocate intermediate tensors (through PyTorch)
- launch Warp kernels sequentially
- pass data between stages through Python variables

This introduces measurable fixed overhead. It is significant at small scales and negligible at large scales. **This may cause large fluctuations in Warp training throughput.**

### 7. Single Backward Mode

Only `backward_mode="manual"` is supported. The CUDA baseline's `autograd`-level differentiation is not applicable because Warp kernels are not natively integrated into PyTorch's autograd graph — the backward pass is explicitly encoded with manually derived gradients.

---

## Future Optimization Directions

The following are the most impactful potential improvements, roughly ordered by expected benefit:

### 1. More Efficient Backward Tile Reduction

The current backward kernel uses block_dim=32 (single warp) to ensure `tile_reduce` takes the pure warp shuffle fast path. This approach is already 3–4× faster than the block_dim=256 `tile_reduce + tile_atomic_add` approach, but each pixel still requires multiple atomic writes. If Warp adds support in a future release for:
- **Sub-block / warp-level tile creation** (allowing creation of 32-thread warp tiles within a 256-thread block)
- **Direct exposure of `__shfl_down_sync` and other warp-level intrinsics**

then the block_dim=256 backward kernel could implement intra-warp reduction followed by cross-warp reduction, combining the benefits of cooperative loading and efficient reduction.

### 2. Runtime Tile-Shape Adaptation

Currently `BLOCK_X=16, BLOCK_Y=16` is fixed at compile time. Allowing runtime selection of tile shapes (for example, 8×8 for small images and 32×16 for wide images) could improve occupancy and reduce tile-boundary overhead. This would require Warp to support dynamic kernel parameterization or template-like mechanisms.

### 3. `.item()` Sync Point Elimination

The binning stage synchronizes via `.item()` to obtain `num_rendered` (GPU→CPU), which breaks the GPU pipeline. If speculative launch or GPU-side branch resolution based on this value could be implemented, pipeline efficiency could be further improved. The effect is limited at low Gaussian counts, but becomes meaningful in multi-view / batch training.

### 4. Smart Buffer Pool Recycling

Currently, radix sort, index gather, scan, and other buffers use a grow-only cache — once allocated, they only grow and never shrink. After densification/pruning, old buffers may be far larger than actually needed. Introducing aging or shrink strategies could reduce memory fragmentation.
