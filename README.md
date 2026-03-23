# gswarp

[中文](README_zh.md) · **English**

A **pure-Python + NVIDIA Warp** reimplementation of the differentiable Gaussian rasterization pipeline, originally implemented in CUDA C++ for [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). This backend provides a **drop-in replacement** for the native CUDA rasterizer — no C++/CUDA compilation required.

> **License**: This project inherits the [Gaussian-Splatting License](LICENSE.md) (Inria & MPII, non-commercial research use only).

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Architecture Overview](#architecture-overview)
- [Differences from CUDA Baseline](#differences-from-cuda-baseline)
- [Correctness](#correctness)
- [Performance Characteristics](#performance-characteristics)
- [Known Limitations](#known-limitations)
- [Future Optimization Targets](#future-optimization-targets)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Full rasterization pipeline**: Preprocess → Binning → Forward render → Backward render, all in Warp kernels.
- **Drop-in replacement**: API-compatible with the original native CUDA backend — switch backends by changing the import path.
- **No compilation**: Pure Python + Warp JIT. No `setup.py build_ext`, no CUDA toolkit headers, no platform-specific build issues.
- **Spherical Harmonics**: Degree 0–3, matching the original implementation.
- **Auto-tuning**: Occupancy-aware kernel block dimension selection based on GPU SM architecture (Volta through Blackwell).
- **Multiple binning sort modes**: `warp_depth_stable_tile` (default, recommended), `warp_radix`, `torch`, `torch_count`.
- **Fused backward kernels**: Merged allocation and gradient accumulation passes to reduce kernel launch overhead.
- **Tight AABB tile culling**: Per-axis 3σ bounding box for Gaussian-to-tile assignment, reducing unnecessary tile overlap for elongated Gaussians.
- **Forward state packing**: Efficient forward state serialization for backward reuse, avoiding redundant recomputation.

---

## Requirements

| Component | Minimum Version | Tested Version |
|-----------|----------------|----------------|
| **Python** | 3.10+ | 3.10 |
| **NVIDIA GPU** | Compute Capability ≥ 7.0 (Volta) | RTX 4060 Laptop (sm_89) |
| **NVIDIA Driver** | Compatible with CUDA 12.x | 13.2 |
| **PyTorch** | 2.0+ (with CUDA) | 2.7.0+cu126 |
| **NVIDIA Warp** | 1.12.0+ | 1.12.0 |

The SM architecture auto-tuning table covers:

| Architecture | Compute Capability |
|-------------|--------------------|
| Volta | 7.0 |
| Turing | 7.5 |
| Ampere (GA100) | 8.0 |
| Ampere (GA10x) | 8.6 |
| Ada Lovelace | 8.9 |
| Hopper | 9.0 |
| Blackwell | 10.0 |

GPUs outside this table still work — the auto-tuner falls back to conservative defaults.

---

## Installation

### 1. Install dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install warp-lang>=1.12.0
```

### 2. Clone and integrate

Place this repository as a submodule of your 3DGS project (or any project that uses `gswarp`):

```bash
git clone <this-repo> submodules/gswarp
```

### 3. (Optional) Build native CUDA baseline for comparison

If you also want the native CUDA rasterizer (for A/B testing or fallback):

```bash
cd submodules/gswarp
pip install .
```

> **Note**: The Warp backend itself does **not** require `pip install .` or any native compilation. It runs directly via Python imports.

---

## Quick Start

### Using the Warp backend

```python
# Import the Warp backend (instead of the native CUDA backend)
from gswarp.warp import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    initialize_runtime_tuning,
)

# (Optional) Initialize runtime auto-tuning — detects GPU and selects optimal parameters
initialize_runtime_tuning(device="cuda:0", verbose=True)

# Configure rasterization settings
raster_settings = GaussianRasterizationSettings(
    image_height=height,
    image_width=width,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg_color,          # [3] tensor, background color
    scale_modifier=1.0,
    viewmatrix=viewmatrix,     # [4, 4] world-to-camera
    projmatrix=projmatrix,     # [4, 4] full projection
    sh_degree=active_sh_degree,
    campos=campos,             # [3] camera position in world space
    prefiltered=False,
    # Warp-specific optional fields:
    backward_mode="manual",              # Only "manual" is supported
    binning_sort_mode="warp_depth_stable_tile",  # Or: "warp_radix", "torch", "torch_count"
    auto_tune=True,
    auto_tune_verbose=True,
)

# Create rasterizer and run forward pass
rasterizer = GaussianRasterizer(raster_settings=raster_settings)
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(
    means3D=means3D,           # [N, 3]
    means2D=means2D,           # [N, 3] (screen-space, receives gradients)
    opacities=opacities,       # [N, 1]
    shs=shs,                   # [N, K, 3] spherical harmonics coefficients
    scales=scales,             # [N, 3]
    rotations=rotations,       # [N, 4] quaternions
)

# Backward pass is automatic via PyTorch autograd
loss = compute_loss(color, target)
loss.backward()
```

### Switching from native CUDA to Warp

Replace your import:

```python
# Before (native CUDA):
from gswarp import GaussianRasterizationSettings, GaussianRasterizer

# After (Warp backend):
from gswarp.warp import GaussianRasterizationSettings, GaussianRasterizer
```


The forward output tuple of the Warp backend returns additional outputs compared to native:

```python
# Native CUDA output:
color, radii = rasterizer(...)

# Warp backend output:
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(...)
```

---

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `rasterize_gaussians(...)` | Full forward pass: preprocess + binning + render |
| `rasterize_gaussians_backward(...)` | Full backward pass: backward render + backward preprocess |
| `mark_visible(positions, viewmatrix, projmatrix)` | Returns visibility mask for 3D positions |
| `preprocess_gaussians(...)` | Preprocess-only (without render) for debugging/analysis |

### Runtime Configuration

| Function | Description |
|----------|-------------|
| `initialize_runtime_tuning(device, verbose)` | One-shot GPU detection and parameter tuning |
| `get_runtime_tuning_report(device)` | Get current tuning report (memory, tile size, sort mode) |
| `get_runtime_auto_tuning_config()` | Get auto-tuning on/off state |
| `set_binning_sort_mode(mode)` | Set binning sort mode at runtime |
| `get_default_parameter_info()` | Get compile-time constants (TOP_K, BLOCK_X, etc.) |
| `is_available()` | Check if Warp is importable |

### GaussianRasterizationSettings Fields

| Field | Type | Description |
|-------|------|-------------|
| `image_height` | `int` | Output image height in pixels |
| `image_width` | `int` | Output image width in pixels |
| `tanfovx` | `float` | tan(FoV_x / 2) |
| `tanfovy` | `float` | tan(FoV_y / 2) |
| `bg` | `Tensor[3]` | Background color |
| `scale_modifier` | `float` | Global scale multiplier |
| `viewmatrix` | `Tensor[4,4]` | World-to-camera transform |
| `projmatrix` | `Tensor[4,4]` | Full projection matrix |
| `sh_degree` | `int` | Active SH degree (0–3) |
| `campos` | `Tensor[3]` | Camera position in world space |
| `prefiltered` | `bool` | Whether points are pre-filtered |
| `backward_mode` | `str \| None` | `"manual"` (only supported mode) |
| `binning_sort_mode` | `str \| None` | Binning sort algorithm |
| `auto_tune` | `bool` | Enable auto-tuning (default: `True`) |
| `auto_tune_verbose` | `bool` | Print tuning info (default: `True`) |

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
│  - Frustum + near-plane culling  │
│  - SH → RGB color evaluation     │
│  - Tight AABB tile rect          │
│  - Forward state packing         │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. BINNING                     │
│  - Tile overlap counting (scan)  │
│  - Gaussian→tile duplication     │
│  - Depth sort + tile sort        │
│  - Tile range identification     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3. FORWARD RENDER              │
│  - Per-pixel alpha blending      │
│  - Front-to-back compositing     │
│  - TOP_K early termination       │
│  - Color, depth, alpha output    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. BACKWARD RENDER             │
│  - Gradient of render w.r.t.     │
│    conic, opacity, color, pos    │
│  - atomic_add accumulation       │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  5. BACKWARD PREPROCESS         │
│  - Gradient of preprocess w.r.t. │
│    means3D, scales, rotations    │
│  - SH backward                   │
│  - Cov3D → scale/rotation grads  │
└─────────────────────────────────┘
```

### Single-file Design

The entire Warp backend is contained in a single Python file (~4400 lines), including:

- All Warp kernel definitions (`@wp.kernel`)
- All Warp function helpers (`@wp.func`)
- Buffer cache management
- Runtime auto-tuning
- Public API functions

This is intentional — Warp's JIT compilation model requires all kernel code and their dependencies to be within the same `wp.Module` scope. Splitting across multiple files would break Warp's ability to resolve `@wp.func` cross-references at JIT time.

### Key Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `TOP_K` | 20 | Max Gaussians per pixel (early termination) |
| `BLOCK_X` | 16 | Tile width in pixels |
| `BLOCK_Y` | 16 | Tile height in pixels |
| `NUM_CHANNELS` | 3 | RGB output channels |
| `PREPROCESS_CULL_SIGMA` | 3.0 | Frustum culling sigma multiplier |
| `PREPROCESS_CULL_FOV_SCALE` | 1.3 | FoV boundary scale for culling |
| `VISIBILITY_NEAR_PLANE` | 0.2 | Near plane distance for culling |

---

## Differences from CUDA Baseline

### 1. Tight AABB vs Isotropic Radius

**CUDA baseline** computes a square bounding box using the maximum of the two eigenvalue-derived radii:

```c
// CUDA: auxiliary.h getRect()
int max_radius = ...;  // max(ceil(3σ_max), 0), isotropic
rect_min = {min(grid.x, max((int)((point.x - max_radius) / BLOCK_X), 0)), ...};
rect_max = {min(grid.x, max((int)((point.x + max_radius + BLOCK_X - 1) / BLOCK_X), 0)), ...};
```

**Warp backend** computes a tight per-axis bounding box using the individual diagonal elements of the 2D covariance matrix:

```python
# Warp: _compute_tile_rect_tight_wp()
radius_x = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(cov_xx, 0.01))))
radius_y = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(cov_yy, 0.01))))
```

**Impact**:
- For **elongated Gaussians** (high anisotropy), the Warp backend assigns fewer tiles, reducing `num_rendered` and improving binning/render efficiency.
- For **circular Gaussians**, the two approaches are equivalent.
- This introduces a small mismatch: some boundary tiles that the CUDA baseline includes (due to the overly conservative isotropic radius) are excluded by Warp's tighter bounds. The visual difference is negligible but measurable in numerical comparisons.

### 2. No Shared Memory Cooperative Fetch

**CUDA baseline** uses `__shared__` memory for cooperative tile-level data fetching:

```c
// CUDA: forward.cu renderCUDA()
__shared__ int collected_id[BLOCK_SIZE];
__shared__ float2 collected_xy[BLOCK_SIZE];
__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

// All threads in a tile cooperatively load a batch of Gaussians
// from global memory into shared memory, then iterate over the batch.
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
    // Read from shared memory — fast, broadcast to all threads in tile
    float2 xy = collected_xy[j];
    float4 con_o = collected_conic_opacity[j];
    ...
}
```

**Warp backend** cannot use `__shared__` memory (Warp does not expose this feature). Each thread reads directly from global memory:

```python
# Warp: each thread reads independently
g_idx = point_list[tile_start + j]
xy_x = means2d_x[g_idx]
xy_y = means2d_y[g_idx]
con_x = conic_x[g_idx]
...
```

**Impact**:
- This is the **single largest performance gap** between Warp and CUDA at small-to-medium scales. CUDA's shared memory pattern amortizes global memory reads across all threads in a 16×16 tile (256 threads share one load), while Warp's approach issues 256 independent global reads per Gaussian.
- At large scales (num_rendered > ~30K), the arithmetic and memory bandwidth are dominated by the actual blending computation, and this gap diminishes.

### 3. Sorting Differences

**CUDA baseline** uses a single-pass CUB `DeviceRadixSort` with a packed 64-bit key (`(tile_id << 32) | depth_bits`).

**Warp backend** default mode (`warp_depth_stable_tile`) uses a two-pass sort:
1. First pass: sort by depth (Warp radix sort)
2. Second pass: stable sort by tile ID (Warp radix sort)

This produces different Gaussian ordering within each tile compared to the CUDA baseline, which in turn causes different floating-point accumulation order during alpha blending. Due to the non-associativity of floating-point arithmetic, this means the pixel-level outputs differ slightly.

### 4. Culling Parameters

The Warp backend applies explicit frustum culling parameters:
- `PREPROCESS_CULL_SIGMA = 3.0` — Gaussians whose 3σ bounding box lies entirely outside the image are culled
- `PREPROCESS_CULL_FOV_SCALE = 1.3` — Slight FoV extension to avoid aggressive boundary culling

These differ slightly from the CUDA baseline's implicit culling behavior.

### 5. Per-Pixel vs Per-Tile Dispatch

The CUDA baseline dispatches render kernels as a 2D grid of tile blocks:
```c
dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
dim3 block(BLOCK_X, BLOCK_Y, 1);
```

The Warp backend dispatches one thread per pixel in a 1D launch:
```python
wp.launch(kernel, dim=image_height * image_width, ...)
```

The tile index is derived arithmetically within the kernel from `wp.tid()`. This is functionally equivalent but has different occupancy and scheduling characteristics.

---

## Correctness

The Warp backend has been tested extensively against the native CUDA baseline. Below are representative residual magnitudes from the test suite (4096 random Gaussians, 128×128 image):

### Forward Pass

| Output | Max Absolute Difference | Notes |
|--------|------------------------|-------|
| `color` | ~4.8e-7 | Near machine epsilon (FP32) |
| `depth` | ~3.6e-7 | Near machine epsilon |
| `alpha` | ~3.0e-7 | Near machine epsilon |
| `radii` | Exact match | Integer values |
| `proj_2D` | ~3.8e-6 | Projection precision |
| `conic_2D` | ~1.5e-7 | Near machine epsilon |
| `conic_2D_inv` | ~0.031 | Matrix inversion amplifies errors |
| `gs_per_pixel` | Exact match | Integer values |
| `weight_per_gs_pixel` | ~1.1e-7 | Near machine epsilon |
| `x_mu` | ~1.9e-6 | Position precision |

### Backward Pass (precomputed colors)

| Gradient | Max Absolute Difference |
|----------|------------------------|
| `grad_means3D` | ~3.8e-5 |
| `grad_means2D` | ~5.7e-6 |
| `grad_colors` | ~3.8e-6 |
| `grad_opacity` | ~4.6e-5 |
| `grad_cov3D` | ~1.7e-5 |

### Backward Pass (scale + rotation mode)

| Gradient | Max Absolute Difference |
|----------|------------------------|
| `grad_means3D` | ~1.5e-5 |
| `grad_means2D` | ~1.3e-6 |
| `grad_colors` | ~7.6e-6 |
| `grad_opacity` | ~4.8e-6 |
| `grad_scales` | ~3.6e-6 |
| `grad_rotations` | ~1.1e-5 |


### Known Forward Mismatch: `conic_2D_inv`

The `conic_2D_inv` (inverse of the 2D covariance conic) has the largest forward residual (~0.031). This is caused by floating-point precision differences in the 2×2 matrix inversion path. The inversion amplifies small input differences, especially for near-singular covariance matrices (elongated Gaussians). Using the `scale + rotation` input path (rather than precomputed `cov3D`) can produce up to 38 mismatching elements. This residual has no observable impact on training convergence.

---

## Performance Characteristics

All benchmarks measured on **NVIDIA GeForce RTX 4060 Laptop GPU** (sm_89, 8 GiB, 24 SMs), Warp 1.12.0, PyTorch 2.7.0+cu126. Times are mean over 30 iterations (50 total, first 20 discarded for warmup).

### Summary Table

| Points | Resolution | num_rendered | Native FW | Warp FW | Ratio | Native BW | Warp BW | Ratio |
|--------|-----------|-------------|-----------|---------|-------|-----------|---------|-------|
| 4,096 | 128×128 | 2,059 | 0.53 ms | 1.54 ms | 2.9× | 1.25 ms | 2.29 ms | 1.8× |
| 16,384 | 128×128 | 8,254 | 0.57 ms | 1.97 ms | 3.4× | 1.73 ms | 2.36 ms | 1.4× |
| 65,536 | 256×256 | 33,292 | 11.56 ms | 2.43 ms | **0.21×** | 11.62 ms | 2.83 ms | **0.24×** |

### Analysis

- **Small scale (num_rendered < ~5K)**: Warp is **2–3× slower** than native CUDA. The overhead is dominated by:
  - Python-level dispatch and Warp launch overhead
  - Binning sort fixed costs (radix sort initialization)
  - Lack of `__shared__` memory cooperative fetch in render kernels
  - These fixed costs are amortized poorly when the compute workload is small.

- **Medium scale (num_rendered ~5K–20K)**: Warp is **1.4–2× slower**. The compute workload begins to dominate over fixed overhead, but the shared-memory gap in render kernels remains significant.

- **Large scale (num_rendered > ~30K)**: Warp can be **significantly faster** (demonstrated 4–5× faster at 65K points / 256×256). At this scale:
  - The native CUDA baseline's `num_rendered` is inflated by its isotropic AABB (square bounding), while Warp's tight AABB eliminates redundant tile assignments.
  - Render kernel compute dominates, and the per-pixel dispatch with tight AABB is competitive.
  - Warp's binning is more efficient due to fewer duplicated entries.

> **Note**: The 65,536-point / 256×256 benchmark shows a dramatic inversion because the native CUDA baseline generates many more tile assignments (isotropic radius), while Warp's tight AABB keeps `num_rendered` manageable. Larger resolutions and higher Gaussian counts will further favor the Warp backend.

### Warp Kernel Time Breakdown (65K points, 256×256)

| Stage | Time | Share |
|-------|------|-------|
| Preprocess | 1.14 ms | 21.6% |
| Binning | 1.44 ms | 27.3% |
| Render | 0.39 ms | 7.4% |
| Backward Render | 1.21 ms | 22.9% |
| (Backward Preprocess) | ~1.10 ms | 20.8% |

The binning stage (sort) is the single largest component. The render kernel itself is relatively fast due to tight AABB reducing per-tile workload.

---

## Known Limitations

### 1. No Shared Memory (Warp Limitation)

NVIDIA Warp does not expose CUDA `__shared__` memory. This prevents implementing the cooperative tile-level fetch pattern used in the CUDA baseline, where 256 threads in a tile collectively load Gaussian data from global memory to shared memory, then iterate over the shared buffer. Instead, each thread independently reads from global memory, causing:
- ~256× more global memory transactions per tile for the same data
- Poor memory coalescing (scattered reads by Gaussian index)
- NCU profiling shows **48.4% LG throttle (long scoreboard stall)** in the backward render kernel, directly caused by uncoalesced global memory reads

### 2. No Warp-Level Intrinsics

Warp does not expose CUDA warp-level primitives (`__shfl_sync`, `__ballot_sync`, `__any_sync`). This prevents:
- Warp-level reductions for gradient accumulation
- Warp-level early termination voting
- Cross-lane communication patterns used in advanced CUDA rasterizers

### 3. `tile_atomic_add` Scalar-Only

`wp.tile_atomic_add` only supports `float32` scalar operands. Vector/matrix-level atomic reductions require decomposition into individual scalar atomics, incurring excessive synchronization overhead. This was experimentally verified — a tile-reduced backward kernel (C2) was **2–3× slower** than the non-tiled version due to 10 scalar atomic operations per Gaussian.

### 4. Compile-Time Tile Shape

`BLOCK_X` and `BLOCK_Y` are defined as `wp.constant()` values (16×16). Changing tile shape requires modifying the source code and triggering Warp module re-compilation. The CUDA baseline also has fixed tile sizes, but the Warp JIT model makes this feel more rigid since constants are baked into the kernel at compile time.

### 5. First-Run JIT Compilation Overhead

The first call to any Warp kernel triggers JIT compilation of the entire module. On a typical system, this takes **~6–15 seconds** (dependent on the number of kernels and GPU). Subsequent calls use the Warp kernel cache and are near-instant.

### 6. Backward Non-Determinism

The backward render kernel accumulates gradients using `wp.atomic_add`, which is inherently non-deterministic when multiple threads write to the same address. This means:
- Two runs with identical inputs may produce slightly different gradient values
- The difference is typically within FP32 precision (< 1e-4 for most gradients)
- `conic_opacity` gradients can exhibit up to ~1.6e+05 max difference in extreme cases due to large gradient magnitudes
- This is consistent with the CUDA baseline's behavior (`atomicAdd` is also non-deterministic)

### 7. Python-Level Orchestration Overhead

Unlike the CUDA baseline where the entire pipeline is orchestrated in C++ with minimal Python interaction, the Warp backend uses Python to:
- Allocate intermediate tensors (via PyTorch)
- Launch individual Warp kernels sequentially
- Pass data between stages via Python variables

This introduces measurable fixed overhead (~0.5–1.0 ms per full forward + backward pass) that is significant at small scales but negligible at large scales.

### 8. Single Backward Mode

Only `backward_mode="manual"` is supported. The CUDA baseline's `autograd`-level differentiation is not applicable since Warp kernels are not natively integrated with PyTorch's autograd tape — the backward pass is explicitly coded with hand-derived gradients.

---

## Future Optimization Targets

The following are the most impactful potential improvements, roughly ordered by expected benefit:

### 1. Shared Memory Support (Pending Warp Feature)

If NVIDIA Warp adds `__shared__` memory support in a future release, the render and backward render kernels could be rewritten to use cooperative tile-level fetch, potentially closing the **2–3× gap** at small-to-medium scales. This is the single highest-impact optimization.

### 2. Compact Gather for Large Scenes

An experimental compact gather optimization (`_ENABLE_COMPACT_GATHER`) pre-copies scattered per-Gaussian data into a compact SoA buffer indexed by `point_list` order before the render kernel. This converts scattered global reads into sequential reads, improving memory coalescing. Profiling results:
- **65K points, 256×256**: Backward render **27% faster**, total forward+backward **16% faster**
- **4K–16K points**: **Slower** (48–135% regression) due to the copy overhead dominating

Currently disabled by default. Should be enabled selectively for large-scene training.

### 3. Runtime Tile Shape Adaptation

Currently `BLOCK_X=16, BLOCK_Y=16` is fixed at compile time. Allowing runtime selection of tile shapes (e.g., 8×8 for small images, 32×16 for wide images) could improve occupancy and reduce tile boundary overhead. Requires Warp to support dynamic kernel parameterization or template-like mechanisms.

### 4. TOP_K Externalization

The `TOP_K = 20` constant controls the maximum number of Gaussians considered per pixel before early termination. Externalizing this as a runtime parameter would allow:
- Lower TOP_K for faster training at acceptable quality loss
- Higher TOP_K for quality-critical rendering
- Adaptive TOP_K based on scene complexity

### 5. Warp-Level Primitives (Pending Warp Feature)

If warp-level intrinsics become available in Warp, per-warp reductions could replace `atomic_add` in the backward kernel, potentially eliminating the non-determinism issue and reducing LG throttle stalls.

---

## Acknowledgments

This Warp backend builds upon:

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis (INRIA, MPII). The original CUDA rasterizer is the reference implementation.
- [NVIDIA Warp](https://nvidia.github.io/warp/) — the Python framework for high-performance GPU simulation and compute.
- [PyTorch](https://pytorch.org/) — used for tensor management, autograd integration, and CUDA memory allocation.

---

## Citation

If you use the original 3D Gaussian Splatting in your research, please cite:

```bibtex
@article{kerbl3Dgaussians,
    author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
    title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
    journal   = {ACM Transactions on Graphics},
    number    = {4},
    volume    = {42},
    month     = {July},
    year      = {2023},
    url       = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
