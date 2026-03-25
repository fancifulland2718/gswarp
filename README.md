# gswarp

[中文](README_zh.md) · **English**

A **pure Python + NVIDIA Warp** reimplementation of the differentiable Gaussian rasterization pipeline. The original version was written in CUDA C++ for [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). This backend provides a **drop-in replacement** for the native CUDA rasterizer — no C++/CUDA compilation is required.

> **License**: This project inherits the [Gaussian-Splatting License](LICENSE) (Inria and MPII, for non-commercial research use only).

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Architecture Overview](#architecture-overview)
- [Differences from the CUDA Baseline](#differences-from-the-cuda-baseline)
- [Correctness](#correctness)
- [Performance Characteristics](#performance-characteristics)
- [Known Limitations](#known-limitations)
- [Future Optimization Directions](#future-optimization-directions)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Complete rasterization pipeline**: Preprocess → Binning → Forward render → Backward render, all implemented by Warp kernels.
- **Drop-in replacement**: The API is compatible with the native CUDA backend — you only need to change the import path to switch backends.
- **No compilation required**: Pure Python + Warp JIT. No `setup.py build_ext`, no CUDA toolkit headers, and no platform-specific build issues.
- **Spherical Harmonics**: Supports degrees 0–3, consistent with the original implementation.
- **Auto-tuning**: Occupancy-aware kernel block-dimension selection based on GPU SM architecture (Volta through Blackwell).
- **Multiple binning sort modes**: `warp_depth_stable_tile` (default, recommended), `warp_radix`, `torch`, `torch_count`.
- **Fused backward kernels**: Merges allocation and gradient-accumulation steps to reduce kernel launch overhead.
- **Tight AABB tile culling**: Uses per-axis 3σ bounding boxes for Gaussian-to-tile assignment (inspired by [Zhang et al., 2025](https://arxiv.org/abs/2601.19489)), reducing unnecessary tile overlap for elongated Gaussians.
- **Forward-state packing**: Efficient forward-state serialization for backward reuse, avoiding redundant recomputation.

---

## Requirements

| Component | Minimum Version | Tested Version |
|------|---------|---------|
| **Python** | 3.10+ | 3.10 |
| **NVIDIA GPU** | Compute Capability ≥ 7.0 (Volta) | RTX 4060 Laptop |
| **NVIDIA Driver** | Compatible with CUDA 12.x | 13.2 |
| **PyTorch** | 2.0+ (with CUDA support) | 2.7.0+cu126 |
| **NVIDIA Warp** | 1.12.0+ | 1.12.0 |

The SM-architecture auto-tuning table covers the following architectures:

| Architecture | Compute Capability |
|------|---------|
| Volta | 7.0 |
| Turing | 7.5 |
| Ampere (GA100) | 8.0 |
| Ampere (GA10x) | 8.6 |
| Ada Lovelace | 8.9 |
| Hopper | 9.0 |
| Blackwell | 10.0 |

GPUs outside this table still work correctly — the auto-tuning logic falls back to conservative default parameters.

---

## Installation

### 1. Install dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install warp-lang>=1.12.0
```

### 2. Clone and integrate

Add this repository as a submodule of a 3DGS project (or any project that uses `gswarp`):

```bash
git clone https://github.com/fancifulland2718/gswarp.git submodules/gswarp
```

### 3. (Optional) Build the native CUDA baseline for comparison

If you also need the native CUDA rasterizer (for A/B testing or fallback):

```bash
cd submodules/gswarp
pip install .
```

> **Note**: The Warp backend itself does **not** require `pip install .` or any native compilation. It can be run directly through Python imports.

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

# (Optional) Initialize runtime auto-tuning — detect the GPU and choose optimal parameters
initialize_runtime_tuning(device="cuda:0", verbose=True)

# Configure rasterization settings
raster_settings = GaussianRasterizationSettings(
    image_height=height,
    image_width=width,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg_color,          # [3] tensor, background color
    scale_modifier=1.0,
    viewmatrix=viewmatrix,     # [4, 4] world-to-camera matrix
    projmatrix=projmatrix,     # [4, 4] full projection matrix
    sh_degree=active_sh_degree,
    campos=campos,             # [3] camera position in world space
    prefiltered=False,
    # Warp-specific optional fields:
    backward_mode="manual",              # Only "manual" is supported
    binning_sort_mode="warp_depth_stable_tile",  # or: "warp_radix", "torch", "torch_count"
    auto_tune=True,
    auto_tune_verbose=True,
)

# Create the rasterizer and run the forward pass
rasterizer = GaussianRasterizer(raster_settings=raster_settings)
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(
    means3D=means3D,           # [N, 3]
    means2D=means2D,           # [N, 3] (screen space, receives gradients)
    opacities=opacities,       # [N, 1]
    shs=shs,                   # [N, K, 3] spherical-harmonic coefficients
    scales=scales,             # [N, 3]
    rotations=rotations,       # [N, 4] quaternions
)

# Backward propagation is handled automatically by PyTorch autograd
loss = compute_loss(color, target)
loss.backward()
```

### Switching from native CUDA to Warp

Replace your import statements:

```python
# Before (native CUDA):
from gswarp import GaussianRasterizationSettings, GaussianRasterizer

# After (Warp backend):
from gswarp.warp import GaussianRasterizationSettings, GaussianRasterizer
```

The Warp backend returns more forward outputs than the native version:

```python
# Native CUDA outputs:
color, radii = rasterizer(...)

# Warp backend outputs:
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(...)
```

---

## API Reference

### Core Functions

| Function | Description |
|------|------|
| `rasterize_gaussians(...)` | Full forward pass: preprocess + binning + rendering |
| `rasterize_gaussians_backward(...)` | Full backward pass: backward render + backward preprocess |
| `mark_visible(positions, viewmatrix, projmatrix)` | Returns the visibility mask of 3D positions |
| `preprocess_gaussians(...)` | Preprocess only (without rendering), useful for analysis |

### Runtime Configuration

| Function | Description |
|------|------|
| `initialize_runtime_tuning(device, verbose)` | One-time GPU detection and parameter tuning |
| `get_runtime_tuning_report(device)` | Returns the current tuning report (memory, tile size, sort mode) |
| `get_runtime_auto_tuning_config()` | Returns the auto-tuning switch status |
| `set_binning_sort_mode(mode)` | Sets the binning sort mode at runtime |
| `get_default_parameter_info()` | Returns compile-time constants (`TOP_K`, `BLOCK_X`, etc.) |
| `is_available()` | Checks whether Warp can be imported |

### GaussianRasterizationSettings Fields

| Field | Type | Description |
|------|------|------|
| `image_height` | `int` | Output image height in pixels |
| `image_width` | `int` | Output image width in pixels |
| `tanfovx` | `float` | tan(FoV_x / 2) |
| `tanfovy` | `float` | tan(FoV_y / 2) |
| `bg` | `Tensor[3]` | Background color |
| `scale_modifier` | `float` | Global scale multiplier |
| `viewmatrix` | `Tensor[4,4]` | World-to-camera transform matrix |
| `projmatrix` | `Tensor[4,4]` | Full projection matrix |
| `sh_degree` | `int` | Active SH degree (0–3) |
| `campos` | `Tensor[3]` | Camera position in world space |
| `prefiltered` | `bool` | Whether points are prefiltered |
| `backward_mode` | `str \| None` | `"manual"` (the only supported mode) |
| `binning_sort_mode` | `str \| None` | Binning sort algorithm |
| `auto_tune` | `bool` | Enables auto-tuning (default: `True`) |
| `auto_tune_verbose` | `bool` | Prints tuning information (default: `True`) |

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
┌─────────────────────────────────┐
│  3. FORWARD RENDER              │
│  - Per-pixel alpha blending     │
│  - Front-to-back compositing    │
│  - TOP_K early termination      │
│  - Color, depth, alpha outputs  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. BACKWARD RENDER             │
│  - Gradients w.r.t. conic,      │
│    opacity, color, and pos      │
│  - atomic_add accumulation      │
└─────────────────────────────────┘
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

The entire Warp backend is contained in a single Python file (about 4400 lines), including:

- all Warp kernel definitions (`@wp.kernel`)
- all Warp helper functions (`@wp.func`)
- runtime auto-tuning
- public API functions

This is intentional — Warp's JIT compilation model requires all kernel code and dependencies to remain within the same `wp.Module` scope. Splitting the code across multiple files would break Warp's ability to resolve `@wp.func` cross-references during JIT.

### Key Constants

| Constant | Value | Description |
|------|---|------|
| `TOP_K` | 20 | Maximum number of Gaussians per pixel (early termination) |
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

### 2. No Shared-Memory Cooperative Fetch

**CUDA baseline** uses `__shared__` memory for cooperative tile-level data fetching:

```c
// CUDA: forward.cu renderCUDA()
__shared__ int collected_id[BLOCK_SIZE];
__shared__ float2 collected_xy[BLOCK_SIZE];
__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

// All threads in the tile cooperatively load a batch of Gaussians from global
// memory into shared memory, and then iterate over that batch.
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
    // Read from shared memory — fast, broadcast to all threads in the tile
    float2 xy = collected_xy[j];
    float4 con_o = collected_conic_opacity[j];
    ...
}
```

In the **Warp backend**, using `__shared__` memory is currently relatively difficult. Each thread reads independently from global memory:

```python
# Warp: each thread reads independently
g_idx = point_list[tile_start + j]
xy_x = means2d_x[g_idx]
xy_y = means2d_y[g_idx]
con_x = conic_x[g_idx]
...
```

**Impact**:
- This is the **largest performance gap** between Warp and CUDA at small and medium scales. In CUDA's shared-memory pattern, a single load is shared by all 256 threads in a tile, whereas Warp issues 256 independent global reads per Gaussian.
- In large scenes (`num_rendered > ~30K`), blending computation and memory bandwidth dominate performance, so this gap becomes smaller.

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

### 5. Per-Pixel vs Per-Tile Dispatch

The CUDA baseline dispatches the render kernel on a 2D grid of tile blocks:

```c
dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
dim3 block(BLOCK_X, BLOCK_Y, 1);
```

The Warp backend dispatches in 1D, with one thread per pixel:

```python
wp.launch(kernel, dim=image_height * image_width, ...)
```

The tile index is derived inside the kernel using arithmetic on `wp.tid()`. The functionality is equivalent, but occupancy and scheduling characteristics differ.

---

## Correctness

The test configuration is as follows:

- `mode = sh_scale_rotation`
- `backward_mode="manual"`
- `binning_sort_mode="warp_depth_stable_tile"`
- `auto_tune=True`
- `auto_tune_verbose=True`)
- test scales: **4,096 / 16,384 / 65,536 / 262,144** particles

Correctness is still best analyzed in three layers: preprocess diagnostics, final rendered outputs, and backward gradients. Forward outputs still match exactly at the checked tolerance, the preprocess diagnostics also remain aligned, and the backward side still contains a small number of sparse outliers without showing a broader spread.

### Preprocess Diagnostics

Across the current four test scales, the active set and the key preprocess tensors (`radii`, `proj_2D`, `conic_2D`, `conic_2D_inv`) all align.

| Points | Native active | Warp active | Shared active | Shared `proj_2D` max-diff | Shared `conic_2D` max-diff | Shared `conic_2D_inv` max-diff |
|------|---------------|-------------|---------------|---------------------------|----------------------------|---------------------------------|
| 4,096 | 3,530 | 3,530 | 3,530 | 0.0 | 0.0 | 0.0 |
| 16,384 | 14,204 | 14,204 | 14,204 | 0.0 | 0.0 | 0.0 |
| 65,536 | 56,804 | 56,804 | 56,804 | 0.0 | 0.0 | 0.0 |
| 262,144 | 227,220 | 227,220 | 227,220 | 0.0 | 0.0 | 0.0 |

### Final Forward Pass (primary rendered outputs)

| Points | Resolution | `color` max-diff | `depth` max-diff | `alpha` max-diff |
|------|--------|--------------------|--------------------|--------------------|
| 4,096 | 128×128 | 0.0 | 0.0 | 0.0 |
| 16,384 | 128×128 | 0.0 | 0.0 | 0.0 |
| 65,536 | 256×256 | 0.0 | 0.0 | 0.0 |
| 262,144 | 384×384 | 0.0 | 0.0 | 0.0 |

Across all four scales, the public forward outputs are exactly aligned within the checked tolerance.

### Backward Pass (key gradients on the training path)

| Points | Gradient fields above threshold | `grad_means3D` mismatches / max-diff | `grad_shs` mismatches / max-diff | Other sparse fields |
|------|-------------------------------|--------------------------------------|----------------------------------|---------------------|
| 4,096 | `means3D` | 1 / 4.30e-2 | 0 / 2.44e-3 | none |
| 16,384 | none | 0 / 2.93e-3 | 0 / 1.95e-3 | none |
| 65,536 | `means3D`, `opacity`, `shs`, `rotations` | 3 / 5.47e-2 | 2 / 6.35e-3 | `opacity`: 2 / 9.38e-2; `rotations`: 1 / 1.17e-2 |
| 262,144 | `means3D`, `means2D`, `opacity`, `shs`, `scales`, `rotations` | 6 / 3.44e-1 | 46 / 5.47e-2 | `means2D`: 4 / 8.30e-3; `opacity`: 2 / 7.03e-2; `scales`: 3 / 5.31e-1; `rotations`: 4 / 9.28e-2 |

Conclusion:

- The preprocess active-set divergence still does not appear, and the preprocess diagnostics line up across all four scales.
- The backward result at 16K is fully back within threshold, while 4K / 65K / 262K still retain a small number of above-threshold outliers with low coverage.
- The main residual risk at the moment is still a handful of isolated backward outliers rather than any systematic forward or preprocess deviation.

---

## Performance Characteristics

The following data also comes from the **current code state**. The test platform is **NVIDIA GeForce RTX 4060 Laptop GPU** (sm_89, 8 GiB, 24 SMs), **Warp 1.12.0**, and **PyTorch 2.7.0+cu126**.

Methodology:

- **Steady-state runtime**: measured via the public API (`diff_gaussian_rasterization.GaussianRasterizer` and `diff_gaussian_rasterization.warp.GaussianRasterizer`) with dedicated warmup runs first, then a batched CUDA-event timing pass over the measured iterations; the main table reports the mean over the measured runs.
- **Memory usage**: after warmup, one forward stage and one backward stage are measured separately, recording the CUDA allocator peak increment (`peak_allocated_delta_mib`).
- **Stage timing / stage memory**: used only for hotspot analysis, measured diagnostically with internal `_warp_backend` helper functions; these stage-wise values are **not guaranteed** to sum strictly to public-API end-to-end time or peak memory item by item.

For the 4K / 16K / 65K / 262K cases, the evaluation uses **12+24 / 10+20 / 6+12 / 4+8** (warmup count + measured count), respectively.

### Public API Steady-State Runtime

| Points | Resolution | `num_rendered` | Native FW | Warp FW | FW ratio | Native BW | Warp BW | BW ratio |
|------|--------|----------------|----------|-----------|----------|----------|-----------|----------|
| 4,096 | 128×128 | 121,691 | 3.019 ms | 2.614 ms | **0.87×** | 2.954 ms | 2.615 ms | **0.89×** |
| 16,384 | 128×128 | 493,767 | 2.410 ms | 2.916 ms | 1.21× | 3.069 ms | 2.971 ms | **0.97×** |
| 65,536 | 256×256 | 7,497,291 | 6.925 ms | 7.176 ms | 1.04× | 2.031 ms | 2.155 ms | 1.06× |
| 262,144 | 384×384 | 65,904,035 | 60.030 ms | 57.033 ms | **0.95×** | 6.084 ms | 7.175 ms | 1.18× |

### Public API Peak Memory by Stage

| Points | Resolution | Native FW peak increment | Native BW peak increment | Warp FW peak increment | Warp BW peak increment |
|------|--------|------------------|------------------|-------------------|-------------------|
| 4,096 | 128×128 | 1.57 MiB | 4.41 MiB | 1.57 MiB | 4.42 MiB |
| 16,384 | 128×128 | 5.18 MiB | 17.63 MiB | 5.18 MiB | 17.69 MiB |
| 65,536 | 256×256 | 41.79 MiB | 70.57 MiB | 41.79 MiB | 70.82 MiB |
| 262,144 | 384×384 | 303.13 MiB | 283.00 MiB | 303.13 MiB | 284.00 MiB |

Public API peak memory remains effectively at parity with native CUDA. Even at 262K, the stage-wise differences still stay within roughly **1 MiB**.

### Internal Stage Hotspots

| Points | Resolution | Preprocess | Binning | Render | Backward Render | Selected Sort Mode |
|------|--------|--------|------|------|----------|--------------|
| 4,096 | 128×128 | 0.905 ms | 1.060 ms | 0.324 ms | 0.611 ms | `warp_depth_stable_tile` |
| 16,384 | 128×128 | 0.824 ms | 1.037 ms | 0.326 ms | 0.684 ms | `warp_depth_stable_tile` |
| 65,536 | 256×256 | 0.972 ms | 5.211 ms | 0.387 ms | 6.111 ms | `warp_depth_stable_tile` |
| 262,144 | 384×384 | 1.487 ms | 53.816 ms | 0.556 ms | 2.082 ms | `warp_depth_stable_tile` |

### Internal Stage Peak Memory (diagnostic only)

| Points | Resolution | Preprocess peak increment | Binning peak increment | Render peak increment | Backward Render peak increment |
|------|--------|----------------|--------------|--------------|------------------|
| 4,096 | 128×128 | 0.53 MiB | 0.00 MiB | 0.38 MiB | 0.16 MiB |
| 16,384 | 128×128 | 2.12 MiB | 0.00 MiB | 0.38 MiB | 0.62 MiB |
| 65,536 | 256×256 | 8.50 MiB | 0.00 MiB | 1.50 MiB | 2.50 MiB |
| 262,144 | 384×384 | 34.00 MiB | 0.00 MiB | 3.38 MiB | 10.00 MiB |

Looking at the internal stages, binning is still the main hotspot, but **binning adds almost no new peak allocation**; the allocator peak is mainly moved by preprocess and backward-render-related work.

---

## Known Limitations

### 1. No Shared Memory (Warp Limitation)

NVIDIA Warp lacks an explicit and flexible way to control shared memory. Alignment, padding, and construction of complex structures are difficult, so CUDA `__shared__` memory cannot be handled freely. This prevents the cooperative tile-level fetch pattern used by the CUDA baseline, where 256 threads in a tile collaboratively load Gaussian data from global memory into shared memory and then iterate over the shared buffer. Instead, each thread reads from global memory independently, causing:

- many more global-memory transactions per tile for the same data
- poor memory coalescing (scattered reads indexed by Gaussian)
- NCU profiling to show that **almost all kernels are bottlenecked by memory**

### 2. No Warp-Level Intrinsics

Warp does not expose CUDA warp-level intrinsics (`__shfl_sync`, `__ballot_sync`, `__any_sync`). The available synchronization mechanisms are relatively limited and not compatible with some advanced patterns. This prevents:

- warp-level reductions for gradient accumulation
- warp-level voting for early termination
- cross-lane communication patterns used in advanced CUDA rasterizers

### 3. `tile_atomic_add` Only Supports Scalars

`wp.tile_atomic_add` only supports a few scalar operands such as `float32`. Vector- and matrix-level atomic reductions must be decomposed into separate scalar atomic operations, which introduces too much synchronization overhead. This has been verified experimentally — the tile-reduced backward kernel (C2) is **2–3× slower** than the non-Warp version and requires 10 scalar atomic operations per Gaussian.

### 4. Compile-Time Tile Shape

`BLOCK_X` and `BLOCK_Y` are defined as `wp.constant()` values (16×16). Changing the tile shape requires modifying the source code and triggering Warp module recompilation. The CUDA baseline also uses fixed tile sizes, but Warp's JIT model makes this limitation more obvious because the constants are baked into the kernel at compile time.

### 5. First-Run JIT Compilation Overhead

The first call to any Warp kernel triggers JIT compilation of the whole module. On a typical system, this takes several seconds (depending on the number of kernels and the GPU). Subsequent calls use the Warp kernel cache and complete almost instantly.

### 6. Backward Non-Determinism

The backward render kernel uses `wp.atomic_add` to accumulate gradients. When multiple threads write to the same address, this is inherently non-deterministic. This means:

- two runs with identical inputs may produce slightly different gradient values
- the difference is usually within FP32 precision (most gradients < 1e-4)
- in extreme cases, `conic_opacity` gradients can reach max-diff values as high as ~1.6e+05 because the gradient magnitude itself is large
- this is consistent with the CUDA baseline (`atomicAdd` is also non-deterministic)

### 7. Python-Level Orchestration Overhead

Unlike the CUDA baseline, where the full pipeline is orchestrated in C++ with very little Python interaction, the Warp backend uses Python to:

- allocate intermediate tensors (through PyTorch)
- launch Warp kernels sequentially
- pass data between stages through Python variables

This introduces measurable fixed overhead. It is significant at small scales and negligible at large scales. **This may cause large fluctuations in Warp training throughput.**

### 8. Single Backward Mode

Only `backward_mode="manual"` is supported. The CUDA baseline's `autograd`-level differentiation is not applicable because Warp kernels are not natively integrated into PyTorch's autograd graph — the backward pass is explicitly encoded with manually derived gradients.

---

## Future Optimization Directions

The following are the most impactful potential improvements, roughly ordered by expected benefit:

### 1. Shared-Memory Support

If NVIDIA Warp adds richer `__shared__` memory support in a future release, the render and backward-render kernels could be rewritten into cooperative tile-level fetch kernels, potentially closing the **2–3× gap** at small and medium scales. This is the single most important optimization. It is still difficult to achieve on the current Warp version.

### 2. Runtime Tile-Shape Adaptation

Currently `BLOCK_X=16, BLOCK_Y=16` is fixed at compile time. Allowing runtime selection of tile shapes (for example, 8×8 for small images and 32×16 for wide images) could improve occupancy and reduce tile-boundary overhead. This would require Warp to support dynamic kernel parameterization or template-like mechanisms.

### 3. TOP_K Externalization

The `TOP_K = 20` constant controls the maximum number of Gaussians considered per pixel before early termination. Exposing it as a runtime parameter would allow:

- lowering TOP_K to speed up training at acceptable quality loss
- increasing TOP_K for quality-critical rendering
- adapting TOP_K based on scene complexity

### 4. Warp-Level Intrinsics (Pending Future Warp Features)

If warp-level intrinsics become available in Warp, per-warp reduction could replace `atomic_add` in the backward kernel, potentially removing the non-determinism issue and reducing LG-throttle stalls.

---

## Acknowledgments

This Warp backend builds upon the following projects:

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis (INRIA, MPII). The original CUDA rasterizer is the reference implementation.
- [Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction](https://arxiv.org/abs/2601.19489), by Ziyu Zhang, Tianle Liu, Diantao Tu, and Shuhan Shen. The tight AABB bounding-box technique used in this project is inspired by this work.
- [NVIDIA Warp](https://nvidia.github.io/warp/) — the Python framework for high-performance GPU simulation and compute.
- [PyTorch](https://pytorch.org/) — used for tensor management and CUDA memory allocation.


