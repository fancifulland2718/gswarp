# Rasterizer Backend

[中文](rasterizer_zh.md) · **English**

This document covers technical details, implementation differences, performance data, and correctness verification for the `gswarp` rasterizer backend. For everyday usage, see the [main README](../README.md).

---

## Table of Contents

- [Overview](#overview)
- [Architecture Overview](#architecture-overview)
- [Differences from the CUDA Baseline](#differences-from-the-cuda-baseline)
- [Python-Layer Overhead Optimization](#python-layer-overhead-optimization)
- [Correctness](#correctness)
- [Current Benchmark](#current-benchmark)
- [Known Limitations](#known-limitations)

---

## Overview

The `gswarp` rasterizer backend implements a differentiable Gaussian rasterization
pipeline compatible with [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
in pure Python + NVIDIA Warp, comprising the following stages:

1. **Preprocessing**: Project 3D Gaussians to 2D, compute covariances, evaluate SH colors, compute AABB tile rects
2. **Feature selection**: Select precomputed RGB or the SH-evaluated colors
3. **Binning**: Gaussian-to-tile mapping, depth sort, tile range identification
4. **Forward render**: block_dim=256 cooperative tile loading, front-to-back alpha compositing
5. **State assembly**: Retain typed preprocess/binning/render state needed by backward
6. **Manual backward**: Render, projection, covariance, scale/rotation, and SH gradients

Each public call snapshots its runtime options and enters a call-scoped execution context. Warp is bound to the current PyTorch CUDA device and stream for the complete submission, including non-default streams.

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
│  - Typed preprocess outputs     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. FEATURES + BINNING          │
│  - Select precomputed/SH RGB    │
│  - Tile-overlap counting (scan) │
│  - Gaussian→tile duplication    │
│  - Depth sort + tile sort       │
│  - Tile-range identification    │
└─────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  3. FORWARD RENDER + STATE               │
│  - block_dim=256 cooperative tile load   │
│    (wp.tile + wp.tile_extract)           │
│  - Per-pixel alpha blending              │
│  - Front-to-back compositing             │
│  - Transmittance threshold termination   │
│  - Color, depth, alpha outputs           │
│  - Typed state for manual backward       │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  4. MANUAL BACKWARD                      │
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

### Module Boundaries

The public modules are compatibility layers; implementation code is split by functional ownership:

| Layer | Responsibility |
|-------|----------------|
| `gswarp/rasterizer*.py` | Public settings, return schema, and CUDA-style module wrappers |
| `_internal/frontend/` | PyTorch autograd adapters and output adaptation |
| `_internal/methods/` | Immutable method specifications, stage plans, and the shared typed executor |
| `_internal/backends/select.py` | Stable/advanced backend resolution and capability validation |
| `_internal/backends/warp/backend_3dgs*.py` | Thin standard/flow stage adapters and raw compatibility entry points |
| `*_ops.py` | Torch/Warp orchestration for one algorithm domain |
| `*_kernels.py` | Warp kernels and device functions for that domain |
| `state.py`, `memory.py`, `runtime.py` | Typed retained state, bounded workspaces, and call-scoped runtime policy |

Kernel modules remain grouped with the device functions they call, while Python orchestration and public compatibility code are kept outside those modules. This preserves Warp JIT symbol resolution without rebuilding a monolithic backend.

The standard and flow backends use the same stage protocol. Their adapters reuse common preprocessing, binning, rendering, stream, cache, and backward components; flow-specific auxiliary outputs remain confined to the flow path.

### Key Constants

| Constant | Value | Description |
|------|---|------|
| `BLOCK_X` | 16 | Tile width in pixels |
| `BLOCK_Y` | 16 | Tile height in pixels |
| `NUM_CHANNELS` | 3 | RGB output channels |
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
The current implementation reduces this cost through:

- cached immutable method plans rather than resolving stages per Gaussian or per kernel
- call-scoped immutable execution options rather than mutating and restoring global settings
- recorded Warp launches with bounded caches
- forward-created, non-autograd Warp views reused by the standard backward path
- ownerless dynamic launch descriptors whose tensor owners remain in the call or graph state
- typed forward state instead of packing normal frontend state into opaque byte buffers
- stream-owned workspace slots with bounded eviction and explicit cache reporting

These mechanisms do not change the public return schema or the stable sorting policy. Current end-to-end, controlled microbenchmark, and profiler results are reported in [Current Benchmark](#current-benchmark).

---

## Correctness

Current correctness evaluation uses complementary contract, component, and frozen-checkpoint checks:

- public API regression tests cover empty and fully culled scenes, manual gradients, retained graphs, stream ownership, cache lifecycle, and CUDA-compatible call behavior
- current component checks compare SSIM forward/backward values and KNN initialization against their native CUDA counterparts
- frozen-checkpoint comparisons render identical cameras, Gaussians, and backgrounds through native CUDA and gswarp

The current frozen checks cover Lego, Truck, and Train. Image MAE ranges from 1.79e-7 to 5.10e-7, inter-renderer PSNR ranges from 100.33 dB to 105.54 dB, and visibility is either identical or differs by two one-sided Train decisions across approximately 25.6 million Gaussian-view observations. Detailed values and interpretation are reported in [Current Benchmark](#current-benchmark).

Within this measured scope, the results do not indicate a rasterizer formula, public-contract, or systematic-coverage defect. FP32 atomic and reduction order can still produce small residuals and different long-horizon training trajectories; bytewise identity is not the correctness criterion.

---

## Current Benchmark

The current end-to-end matrix is maintained in the [project README](../README.md#benchmarks). It was regenerated with 30K iterations on an RTX 5090 (32 GiB, sm_120), Python 3.14.3, PyTorch 2.11.0+cu130, and Warp 1.12.0. It covers eight NeRF Synthetic scenes, two Tanks and Temples scenes, and two Deep Blending scenes.

### Provenance and Measurement Rules

The CUDA and Warp packages must be deliberately isolated during a comparison:

1. Build the current gswarp source into a dedicated install target and verify the installed source hash.
2. Expose the native `diff_gaussian_rasterization` extension through a package root containing only that package, so it cannot pull in an unrelated installed gswarp.
3. Assert the CUDA rasterizer resolves to the native extension and the Warp rasterizer resolves to the dedicated gswarp target before accepting a result.
4. Record the resolved module and native-extension paths with each run, then evaluate independently trained checkpoints with the original 3DGS render and metrics workflow through the same native CUDA renderer.

Both backends use default 3DGS optimization settings, CPU data loading, default Adam, and 30K iterations. Warp selects the gswarp rasterizer, fused SSIM, and KNN. Depth accumulation is disabled only for Warp because the reference loss does not consume depth. Wall time includes final evaluation and checkpoint saving; phase results use CUDA events over iterations 25,000-29,999.

### Stable Training Phases

| Scene | Backend | Render ms | Loss ms | Backward ms | Densify ms | Optimizer ms |
|-------|---------|----------:|--------:|------------:|-----------:|-------------:|
| Lego | CUDA | 1.789 | 0.503 | 3.509 | 0.001 | 1.130 |
| Lego | Warp | 1.608 | 0.562 | 3.734 | 0.002 | 1.127 |
| Train | CUDA | 3.916 | 0.582 | 9.358 | 0.002 | 3.252 |
| Train | Warp | 3.134 | 0.637 | 7.630 | 0.001 | 3.236 |
| Truck | CUDA | 4.347 | 0.579 | 9.471 | 0.001 | 5.548 |
| Truck | Warp | 4.099 | 0.668 | 8.578 | 0.001 | 5.629 |
| DrJohnson | CUDA | 6.356 | 0.683 | 20.112 | 0.001 | 8.472 |
| DrJohnson | Warp | 5.143 | 0.749 | 10.278 | 0.001 | 8.458 |

These four rows show the range hidden by the suite aggregate. On Lego, Warp reduces render by 10.1%, but its loss and backward work are higher; the complete job is 3.5% faster. On Train, Warp reduces render by 20.0% and backward by 18.5%, producing a 17.2% lower wall time. On Truck, Warp reduces render by 5.7% and backward by 9.4%, but its SSIM loss step is higher; wall time is 8.7% lower. On DrJohnson, render is 19.1% lower and backward is 48.9% lower, producing 29.0% lower wall time. Results are deliberately reported as mixed end-to-end outcomes rather than attributed to a single rasterizer kernel.

### Current Controlled Microbenchmarks

The following measurements use the installed package artifact on the current RTX 5090, not a checkout import. They are controlled kernel-stack measurements and must not be read as complete training throughput.

| Workload | Native CUDA GPU median | Warp GPU median | CUDA host median | Warp host median |
|----------|----------------------:|----------------:|----------------:|----------------:|
| 300K Gaussians, 800x800, SH degree 3, raster backward | 3.809 ms | 1.638 ms | 0.274 ms | 0.530 ms |

The synthetic view has 36,842 visible Gaussians. It uses 20 warmups and 100 measured repetitions, with GPU time from CUDA events. The lower Warp GPU time in this deliberately isolated backward workload does not imply lower total Python-side dispatch overhead; its host median is higher. Separate Warp full forward-plus-backward runs with precomputed RGB measured 2.215 ms and 85.8 MiB at 256K Gaussians and 384x384, and 4.733 ms and 345.5 MiB at 1024K Gaussians and 640x640. Repeated-output maximum absolute differences were 5.59e-9 and 5.96e-8 respectively.

### Current Nsight Evidence

Nsight Systems 2026.1.2 CUDA and NVTX traces and Nsight Compute 2026.1 profiling were collected on the RTX 5090. The traced workload is 300K Gaussians, 800x800, SH degree 3, with the same 36,842 visible Gaussians as the controlled benchmark. Trace kernel durations are used for workload structure only; CUDA-event medians above remain the latency reference because profiler replay and tracing perturb timing.

For Warp, the backward tile renderer accounts for 57.4% of traced kernel time, with a 1.069 ms mean per traced launch; the tiled forward renderer is 11.2%, at 0.209 ms. Eight radix-sort dispatches occur per repeat, and individual radix-sort launches average 14.0 microseconds. For native CUDA, the primary backward render kernel accounts for 73.8% of traced kernel time, at 3.339 ms per traced launch; forward render is 7.2%, at 0.327 ms. Its sort launches average 49.4 microseconds, with six sort passes per repeat.

Nsight Compute reports no local or shared-memory spilling in the Warp backward tile kernel. It reports 50.0% theoretical occupancy, 31.14% achieved occupancy, 29.23% issue-slot utilization, 96.91% L2 hit rate, 43.38% L1 hit rate, and 0.8% DRAM-throughput utilization. The launch has 4.9 waves per SM, for which the report estimates a tail effect near 20%. For this measured workload, occupancy, issue efficiency, and work distribution are more relevant than DRAM bandwidth. These values characterize one RTX 5090 workload and are not a cross-device performance claim.

### Numerical Interpretation

The README reports independent-training quality and frozen-checkpoint CUDA/Warp comparison separately because they answer different questions. Independent 30K runs measure the outcome of the complete training stack. Frozen checkpoints isolate rasterization by supplying identical camera, Gaussian, and background inputs to both renderers.

Across Lego, Truck, and Train, the frozen-checkpoint image MAE is between 1.79e-7 and 5.10e-7, and inter-renderer PSNR is between 100.33 dB and 105.54 dB. Lego and Truck have identical visible sets. Train has a visibility Jaccard score of 0.99999992, corresponding to two one-sided decisions across approximately 25.6 million Gaussian-view observations; both outputs have the same 20.955083 dB global PSNR against ground truth to the shown precision. Atomic accumulation order leaves small pixel residuals, so bytewise identity is neither expected nor required.

This evidence does not indicate a rasterizer formula, public-contract, or systematic-coverage defect in the measured scope. In particular, the frozen Train result does not reproduce the -0.2677 dB difference between independently trained checkpoints. Controlled component checks also found bitwise-identical KNN initialization distances on Train, while CUDA fused SSIM and Warp SSIM have small FP32 gradient differences despite matching formula and padding semantics. The observed 30K difference is therefore consistent with numerical trajectory sensitivity in non-convex optimization and densification, not with a frozen-renderer quality loss. See the [SSIM documentation](ssim.md) for the controlled numerical and training-path measurements.

---

## Known Limitations

### 1. CUDA Compatibility Boundaries

The common keyword-based 3DGS integration path is supported, but the public contract is not byte-for-byte identical to every revision of `diff-gaussian-rasterization`:

- standard gswarp returns `(color, radii, RasterizerMeta)` rather than a bare depth tensor
- `dc` is an additional compatibility alias before `shs`, so positional migration is discouraged
- `prefiltered=True` is not implemented
- the `antialiasing` setting is accepted for construction compatibility but is not applied by the stable backend
- `debug=True` enables public-input finite-value checks; it does not reproduce the CUDA extension's snapshot dump

### 2. Manual Backward and Atomic Ordering

Only `backward_mode="manual"` is supported. Gradient accumulation uses atomic operations, so repeated runs can differ slightly in floating-point order, as can the CUDA baseline. The typed forward state and explicit backward implementation are integrated into PyTorch through a custom autograd function.

### 3. Fixed Tile Shape and First-Run JIT

`BLOCK_X=16` and `BLOCK_Y=16` are compile-time constants. Changing them requires recompiling the Warp kernels. The first invocation on a new device/kernel configuration triggers Warp JIT compilation; later processes can reuse Warp's on-disk cache.

### 4. Python Orchestration and Synchronization

The CUDA baseline performs most orchestration in C++. gswarp still submits several Warp kernels and PyTorch operations from Python. Binning also reads the exact rendered-reference count back to the host before allocating and launching dependent work. Recorded launches and ownerless descriptors reduce this cost but do not eliminate it.

### 5. Cache High-Water Marks

Reusable workspaces are bounded by device and stream count, and launch caches have entry limits. Individual workspace buffers retain the largest capacity requested by their slot until eviction or an explicit `clear_warp_caches()`. Use `get_warp_cache_report()` to inspect retained bytes and stream distribution.

### 6. Advanced Warp Backends

The resolver can gate optional implementations by Warp version and required public capabilities. No optional high-version backend is currently distributed, so automatic selection uses the stable backend.

---
